import dgl
import dgl.function as fn
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from utils import *


class MultiHotEmbeddingLayer(nn.Module):
    def __init__(self, num_emb, emb_dim, base=2):
        super(MultiHotEmbeddingLayer, self).__init__()
        self.num_emb = num_emb
        enc_len = get_enc_len(num_emb - 1, base)
        self.encoding = nn.Embedding(num_emb, enc_len * base)
        self.embedding = nn.Parameter(torch.Tensor(enc_len * base, emb_dim))

        with torch.no_grad():
            self.encoding.weight.data.copy_(
                torch.from_numpy(int2multihot(np.arange(0, num_emb), enc_len, base)).float()
            )

            scale = 1 / (emb_dim * enc_len)**0.5
            nn.init.uniform_(self.embedding, -scale, scale)
        self.encoding.weight.requires_grad = False

    def forward(self, g, x):
        enc = self.encoding(x.squeeze())
        emb = torch.matmul(enc.view(-1, self.embedding.size(0)), self.embedding)
        return emb

    @property
    def weight(self):
        return torch.matmul(self.encoding.weight, self.embedding)


class EmbeddingLayer(nn.Module):
    def __init__(self, num_emb, emb_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_emb, emb_dim)
        scale = 1 / (emb_dim)**0.5
        nn.init.uniform_(self.embedding.weight, -scale, scale)

    def forward(self, g, x):
        return self.embedding(x.squeeze())

    @property
    def weight(self):
        return self.embedding.weight


class EmbeddingLayerAttri(nn.Module):
    def __init__(self, attri):
        super(EmbeddingLayerAttri, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(attri))

    def forward(self, g, x):
        return self.embedding(x.squeeze())

    @property
    def weight(self):
        return self.embedding.weight


class BaseModel(nn.Module):
    def __init__(
        self,
        node_attri,
        rel_attri,
        num_nodes,
        h_dim,
        out_dim,
        num_rels,
        num_hidden_layers=1,
        dropout=0,
        use_cuda=False
    ):
        super(BaseModel, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # create conjgcn layers
        self.build_model(node_attri, rel_attri)

    def build_model(self, node_attri, rel_attri):
        self.node_emb, self.rel_emb = self.build_input_layer(node_attri, rel_attri)
        self.layers = nn.ModuleList()
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self, node_attri, rel_attri):
        return None, None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        raise NotImplementedError


class RGCN(BaseModel):
    def build_input_layer(self, node_attri, rel_attri):
        if node_attri is not None:
            return EmbeddingLayerAttri(node_attri), None
        return EmbeddingLayer(self.num_nodes, self.h_dim), None

    def build_hidden_layer(self, idx):
        if idx == 0:
            in_dim = self.h_dim
        else:
            in_dim = self.out_dim
        if idx < self.num_hidden_layers - 1:
            act = nn.Tanh()
        else:
            act = None
        return RelGraphConv(
            in_dim,
            self.out_dim,
            self.num_rels,
            regularizer="basis",
            num_bases=-1,
            bias=True,
            activation=act,
            self_loop=True,
            dropout=self.dropout
        )

    def forward(self, g, h, r, norm):
        h = self.node_emb(g, h)
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class TrainModel(nn.Module):
    def __init__(
        self,
        node_attri,
        num_nodes,
        o_dim,
        num_rels,
        nlabel,
        num_hidden_layers=1,
        dropout=0,
        use_cuda=False,
        reg_param=0
    ):
        super(TrainModel, self).__init__()

        i_dim = o_dim if node_attri is None else node_attri.shape[1]
        self.model = RGCN(
            node_attri, None, num_nodes, i_dim, o_dim, num_rels * 2, num_hidden_layers, dropout, use_cuda
        )
        self.reg_param = reg_param

        if nlabel == 0:
            self.supervised = False
            self.w_relation = nn.Parameter(torch.Tensor(num_rels, o_dim))
            nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
        else:
            self.supervised = True
            self.node_fc = nn.Linear(o_dim, nlabel)
            nn.init.xavier_normal_(self.node_fc.weight, gain=nn.init.calculate_gain("sigmoid"))
            nn.init.zeros_(self.node_fc.bias)

        # self.edge_fc = nn.Linear(o_dim, num_rels * 2)
        self.edge_fc = nn.Linear(o_dim, o_dim)
        nn.init.xavier_normal_(self.edge_fc.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.edge_fc.bias)

    def calc_score(self, embedding, triplets):
        if isinstance(embedding, (tuple, list)):
            node_emb = embedding[0]
        else:
            node_emb = embedding
        s = node_emb[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = node_emb[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, edge_type, edge_norm):
        output = self.model.forward(g, h, edge_type, edge_norm)
        if self.supervised:
            if isinstance(output, (tuple, list)):
                pred = self.node_fc(output[0])
            else:
                pred = self.node_fc(output)
        else:
            pred = None

        return output, pred

    def unsupervised_regularization_loss(self, embedding, edge_type=None):
        reg = torch.mean(self.w_relation.pow(2))
        if isinstance(embedding, (tuple, list)):
            for emb in embedding:
                reg = reg + torch.mean(emb.pow(2))
        elif isinstance(embedding, torch.Tensor):
            reg = reg + torch.mean(embedding.pow(2))
        else:
            raise ValueError
        if edge_type is not None:
            if isinstance(embedding, (tuple, list)):
                for emb in embedding:
                    if emb.size(0) == edge_type.size(0):
                        mask = edge_type < self.w_relation.size(0)
                        # reg = reg + F.cross_entropy(self.edge_fc(emb[mask]), edge_type[mask])
                        emb_diff = self.edge_fc(emb[mask]) - torch.index_select(self.w_relation, 0, edge_type[mask])
                        reg = reg + torch.mean(torch.pow(emb_diff, 2))
            elif isinstance(embedding, torch.Tensor):
                if embedding.size(0) == edge_type.size(0):
                    mask = edge_type < self.w_relation.size(0)
                    # reg = reg + F.cross_entropy(self.edge_fc(embedding[mask]), edge_type[mask])
                    emb_diff = self.edge_fc(embedding[mask]) - torch.index_select(self.w_relation, 0, edge_type[mask])
                    reg = reg + torch.mean(torch.pow(emb_diff, 2))

        return reg

    def get_unsupervised_loss(self, g, embedding, edge_type, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.unsupervised_regularization_loss(embedding, edge_type=edge_type)
        return predict_loss + self.reg_param * reg_loss

    def supervised_regularization_loss(self, embedding, edge_type=None):
        return self.unsupervised_regularization_loss(embedding, edge_type=edge_type)

    def get_supervised_loss(self, g, embedding, edge_type, pred, matched_labels, matched_index, multi):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        if multi:
            predict_loss = F.binary_cross_entropy(torch.sigmoid(pred[matched_index]), matched_labels)
        else:
            predict_loss = F.nll_loss(F.log_softmax(pred[matched_index]), matched_labels)
        reg_loss = self.supervised_regularization_loss(embedding, edge_type=edge_type)
        return predict_loss + self.reg_param * reg_loss
