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


class CompGraphConv(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        self_loop=True,
        comp_opt="mult",
        bias=True,
        batch_norm=False,
        activation=None,
        dropout=0.0
    ):
        super(CompGraphConv, self).__init__()
        assert comp_opt in ["sub", "mult", "corr"]
        self.comp_opt = comp_opt

        self.in_weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.out_weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.rel_weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        if self_loop:
            self.num_rels = 3
            self.loop_weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
            self.loop_rel = nn.Parameter(torch.Tensor(1, input_dim))
        else:
            self.num_rels = 2
            self.register_parameter("loop_weight", None)
            self.register_parameter("loop_rel", None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.register_parameter("bias", None)
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None
        self.act = activation
        self.drop = nn.Dropout(dropout)

        # init
        nn.init.xavier_uniform_(self.in_weight)
        nn.init.xavier_uniform_(self.out_weight)
        nn.init.xavier_uniform_(self.rel_weight)
        if self_loop:
            nn.init.xavier_uniform_(self.loop_weight)
            nn.init.xavier_uniform_(self.loop_rel)
        if bias:
            nn.init.zeros_(self.bias)

        # register functions
        self.node_init_func = self._node_init_func
        self.edge_init_func = self._edge_init_func
        self.node_message_func = self._node_message_func
        self.node_reduce_func = fn.sum(msg="msg", out="agg")
        self.node_update_func = self._node_update_func
        self.edge_update_func = self._edge_update_func

    def _node_init_func(self, graph, node_feat):
        graph.ndata["h"] = node_feat

        return node_feat

    def _edge_init_func(self, graph, edge_feat, edge_norm=None):
        graph.edata["h"] = edge_feat

        if edge_norm is not None:
            graph.edata["norm"] = edge_norm

        return edge_feat

    def _comp_func(self, head, relation):
        if self.comp_opt == "sub":
            return head - relation
        elif self.comp_opt == "mult":
            return head * relation
        elif self.comp_opt == "corr":
            re_h, im_h = torch.chunk(rfft(head, dim=-1), chunks=2, dim=-1)
            # re_h, im_h = re_h.squeeze_(-1), im_h.squeeze_(-1)
            re_r, im_r = torch.chunk(rfft(relation, dim=-1), chunks=2, dim=-1)
            # re_r, im_r = re_r.squeeze_(-1), im_r.squeeze_(-1)
            o = torch.stack(complex_mul(*complex_conj(re_h, im_h), re_r, im_r), dim=-1)
            return irfft(o, dim=1, n=head.size(1)).squeeze(-1)
        else:
            raise NotImplementedError

    def _node_message_func(self, edges):
        comp = self._comp_func(edges.src["h"], edges.data["h"])
        msg = torch.matmul(comp, self.in_weight)

        if "is_rev" in edges.data:
            rev_msg = torch.matmul(comp, self.out_weight)
            mask = (~edges.data["is_rev"]).view(-1, 1)
            msg = msg.masked_fill(~(mask), 0.0) + rev_msg.masked_fill(mask, 0.0)

        if "norm" in edges.data:
            msg = msg * edges.data["norm"]

        return {"msg": msg}

    def _node_update_func(self, nodes):
        agg = nodes.data["agg"]

        if self.loop_weight is not None:
            loop_msg = self._comp_func(nodes.data["h"], self.loop_rel)
            loop_msg = torch.matmul(loop_msg, self.loop_weight)
            out = agg + loop_msg
            out = out * 0.3333333
        else:
            out = agg * 0.5

        if self.bias is not None:
            out = out + self.bias
        if self.bn is not None:
            out = self.bn(out)
        if self.act is not None:
            out = self.act(out)
        out = self.drop(out)

        return {"out": out}

    def _edge_update_func(self, edges):
        out = torch.matmul(edges.data["h"], self.rel_weight)

        return {"out": out}

    def forward(self, graph, node_feat, edge_feat, edge_norm=None):
        # g = graph.local_var()
        g = graph

        num_edges = g.num_edges()
        self.node_init_func(g, node_feat)
        self.edge_init_func(g, edge_feat, edge_norm)
        g.update_all(self.node_message_func, self.node_reduce_func, self.node_update_func)
        g.apply_edges(self.edge_update_func)

        if g.edata["out"].size(0) != num_edges:
            return g.ndata.pop("out"), g.edata.pop("out")[:num_edges]
        else:
            return g.ndata.pop("out"), g.edata.pop("out")

    def extra_repr(self):
        summary = [
            "in=%s, out=%s," % (self.input_dim, self.hidden_dim),
            "comp_opt=%s," % (self.comp_opt),
            "self_loop=%s, bias=%s," % (self.self_loop, self.bias is not None),
        ]

        return "\n".join(summary)


class CompGCN(BaseModel):
    def build_input_layer(self, node_attri, rel_attri):
        node_emb = None
        rel_emb = None
        if node_attri is not None:
            node_emb = EmbeddingLayerAttri(node_attri)
        else:
            node_emb = EmbeddingLayer(self.num_nodes, self.h_dim)
        if rel_attri is not None:
            rel_emb = EmbeddingLayerAttri(rel_attri)
        else:
            rel_emb = EmbeddingLayer(self.num_rels, self.h_dim)

        return node_emb, rel_emb

    def build_hidden_layer(self, idx):
        if idx == 0:
            in_dim = self.h_dim
        else:
            in_dim = self.out_dim
        if idx < self.num_hidden_layers - 1:
            act = nn.Tanh()
        else:
            act = None
        return CompGraphConv(
            in_dim,
            self.out_dim,
            comp_opt="corr",
            self_loop=True,
            activation=act,
            dropout=self.dropout
        )

    def forward(self, g, h, r, norm):
        h = self.node_emb(g, h)
        r = self.rel_emb(g, r)
        for layer in self.layers:
            h, r = layer(g, h, r, norm)
        return h, r


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
        self.model = CompGCN(
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
            nn.init.xavier_uniform_(self.node_fc.weight, gain=nn.init.calculate_gain("sigmoid"))
            nn.init.zeros_(self.node_fc.bias)

        # self.edge_fc = nn.Linear(o_dim, num_rels * 2)
        self.edge_fc = nn.Linear(o_dim, o_dim)
        nn.init.xavier_uniform_(self.edge_fc.weight, gain=nn.init.calculate_gain('sigmoid'))
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
