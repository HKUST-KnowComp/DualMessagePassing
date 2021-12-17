import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as tsp

from .basemodel import GraphAdjModelV2
from .container import *
# from ..constants import *
# from ..utils.act import map_activation_str_to_layer
# from ..utils.dl import batch_convert_len_to_mask, split_and_batchify_graph_feats
# from ..utils.init import init_weight, init_module
from constants import *
from utils.act import map_activation_str_to_layer
from utils.dl import batch_convert_len_to_mask, split_and_batchify_graph_feats
from utils.init import init_weight, init_module


class LRPLayer(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        lrp_seq_len=4,
        bias=True,
        act_func="relu",
        batch_norm=False,
        mlp=False,
        dropout=0.0
    ):
        super(LRPLayer, self).__init__()
        self.lrp_seq_len = lrp_seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight = nn.Parameter(th.Tensor(input_dim, hidden_dim, lrp_seq_len * lrp_seq_len))
        self.degnet_0 = nn.Linear(1, 2 * hidden_dim)
        self.degnet_1 = nn.Linear(2 * hidden_dim, hidden_dim)

        if bias:
            self.bias = nn.Parameter(th.Tensor(hidden_dim))
        else:
            self.register_parameter("bias", None)

        self.act = map_activation_str_to_layer(act_func)
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.register_parameter("bn", None)
        if mlp:
            self.mlp = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.register_parameter("mlp", None)

        self.drop = nn.Dropout(dropout)

        # init
        init_weight(self.weight, activation=act_func, init="uniform")
        init_module(self.degnet_0, activation=act_func, init="uniform")
        init_module(self.degnet_1, activation=act_func, init="uniform")
        if bias:
            nn.init.zeros_(self.bias)
        if mlp:
            init_module(self.mlp, activation=act_func, init="uniform")

    def forward(self, graph, node_feat, edge_feat, pooling_matrix, node_to_perm_matrix, edge_to_perm_matrix):
        node_out = tsp.mm(node_to_perm_matrix, node_feat) + tsp.mm(edge_to_perm_matrix, edge_feat)
        node_out = node_out.view(-1, self.lrp_seq_len * self.lrp_seq_len, self.input_dim)
        node_out = th.einsum('dab,bca->dc', node_out, self.weight)
        if self.bias is not None:
            node_out = node_out + self.bias
        node_out = self.act(node_out)
        # node_out = self.drop(node_out)
        node_out = tsp.mm(pooling_matrix, node_out)
        factor_degs = self.degnet_1(self.act(self.degnet_0(graph.ndata[INDEGREE].float().unsqueeze(1))))  #.squeeze()
        node_out = self.act(node_out * factor_degs)

        if self.bn is not None:
            node_out = self.bn(node_out)
        if self.mlp is not None:
            node_out = self.mlp(node_out)
            node_out = self.act(node_out)
        node_out = self.drop(node_out)
        edge_out = edge_feat

        return node_out, edge_out

    def extra_repr(self):
        summary = [
            "in=%s, out=%s" % (self.input_dim, self.hidden_dim),
            "lrp_seq_len=%s" % (self.lrp_seq_len),
        ]

        return "\n".join(summary)

    def get_output_dim(self):
        return self.hid_dim


class LRP(GraphAdjModelV2):
    def __init__(self, **kw):
        super(LRP, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
        lrp_seq_len = kw.get("lrp_seq_len", 4)
        batch_norm = kw.get("rep_lrp_batch_norm", False)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)

        lrp = ModuleList()
        for i in range(num_layers):
            lrp.add_module(
                "%s_lrp_(%d)" % (type, i),
                LRPLayer(
                    self.hid_dim, self.hid_dim,
                    lrp_seq_len=lrp_seq_len,
                    batch_norm=batch_norm,
                    act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"lrp": lrp})

    def get_pattern_rep(
        self,
        pattern,
        p_v_emb,
        p_e_emb,
        p_perm_pool,
        p_n_perm_matrix,
        p_e_perm_matrix,
        v_mask=None,
        e_mask=None
    ):
        if v_mask is not None:
            p_v_zero_mask = ~(v_mask)
            v_outputs = [p_v_emb.masked_fill(p_v_zero_mask, 0.0)]
        else:
            p_v_zero_mask = None
            v_outputs = [p_v_emb]

        if e_mask is not None:
            p_e_zero_mask = ~(e_mask)
            e_outputs = [p_e_emb.masked_fill(p_e_zero_mask, 0.0)]
        else:
            p_e_zero_mask = None
            e_outputs = [p_e_emb]

        for layer in self.p_rep_net["lrp"]:
            v, e = layer(pattern, v_outputs[-1], e_outputs[-1], p_perm_pool, p_n_perm_matrix, p_e_perm_matrix)
            if p_v_zero_mask is not None:
                v = v.masked_fill(p_v_zero_mask, 0.0)
            if p_e_zero_mask is not None:
                e = e.masked_fill(p_e_zero_mask, 0.0)
            if self.rep_residual and v_outputs[-1].size() == v.size() and e_outputs[-1].size() == e.size():
                v_outputs.append(v)
                e_outputs.append(e)
            else:
                v_outputs.append(v)
                e_outputs.append(e)

        return v_outputs[-1], e_outputs[-1]

    def get_graph_rep(
        self,
        graph,
        g_v_emb,
        g_e_emb,
        g_perm_pool,
        g_n_perm_matrix,
        g_e_perm_matrix,
        v_mask=None,
        e_mask=None,
        v_gate=None,
        e_gate=None
    ):
        if v_mask is not None or v_gate is not None:
            if v_gate is None:
                v_gate = v_mask.float()
            elif v_mask is not None:
                v_gate = v_mask.float() * v_gate
            v_outputs = [g_v_emb * v_gate]
        else:
            v_outputs = [g_v_emb]

        if e_mask is not None or e_gate is not None:
            if e_gate is None:
                e_gate = e_mask.float()
            elif e_mask is not None:
                e_gate = e_mask.float() * e_gate
            e_outputs = [g_e_emb * e_gate]
        else:
            e_outputs = [g_e_emb]

        for layer in self.g_rep_net["lrp"]:
            v, e = layer(graph, v_outputs[-1], e_outputs[-1], g_perm_pool, g_n_perm_matrix, g_e_perm_matrix)
            if v_gate is not None:
                v = v * v_gate
            if e_gate is not None:
                e = e * e_gate
            if self.rep_residual and v_outputs[-1].size() == v.size() and e_outputs[-1].size() == e.size():
                v_outputs.append(v)
                e_outputs.append(e)
            else:
                v_outputs.append(v)
                e_outputs.append(e)

        return v_outputs[-1], e_outputs[-1]

    def forward(
        self,
        pattern,
        p_perm_pool,
        p_n_perm_matrix,
        p_e_perm_matrix,
        graph,
        g_perm_pool,
        g_n_perm_matrix,
        g_e_perm_matrix
    ):
        bsz = pattern.batch_size
        p_v_len = pattern.batch_num_nodes()
        g_v_len = graph.batch_num_nodes()
        p_e_len = pattern.batch_num_edges()
        g_e_len = graph.batch_num_edges()

        p_v_mask = batch_convert_len_to_mask(p_v_len, pre_pad=True).view(bsz, -1, 1)
        g_v_mask = batch_convert_len_to_mask(g_v_len, pre_pad=True).view(bsz, -1, 1)
        p_e_mask = batch_convert_len_to_mask(p_e_len, pre_pad=True).view(bsz, -1, 1)
        g_e_mask = batch_convert_len_to_mask(g_e_len, pre_pad=True).view(bsz, -1, 1)
        vl_gate, el_gate = self.get_filter_gate(pattern, graph)

        p_enc = self.get_pattern_enc(pattern)
        p_v_emb, p_e_emb = self.get_pattern_emb(p_enc)
        p_v_rep, p_e_rep = self.get_pattern_rep(
            pattern, p_v_emb, p_e_emb,
            p_perm_pool,
            p_n_perm_matrix,
            p_e_perm_matrix
        )

        g_enc = self.get_graph_enc(graph)
        g_v_emb, g_e_emb = self.get_graph_emb(g_enc)
        g_v_rep, g_e_rep = self.get_graph_rep(
            graph, g_v_emb, g_e_emb,
            g_perm_pool,
            g_n_perm_matrix,
            g_e_perm_matrix,
            v_gate=vl_gate, e_gate=el_gate
        )

        if self.pred_with_deg:
            p_out_deg = pattern.out_degrees().float().view(-1, 1)
            p_in_deg = pattern.in_degrees().float().view(-1, 1)

            g_out_deg = graph.out_degrees().float().view(-1, 1)
            g_in_deg = graph.in_degrees().float().view(-1, 1)

        if self.node_pred:
            p_v_addfeat = []
            g_v_addfeat = []
            if self.pred_with_enc:
                p_v_addfeat.append(p_enc["v"])
                p_v_addfeat.append(p_enc["vl"])

                g_v_addfeat.append(g_enc["v"])
                g_v_addfeat.append(g_enc["vl"])

            if self.pred_with_deg:
                p_v_addfeat.append(p_out_deg)
                p_v_addfeat.append(p_in_deg)

                g_v_addfeat.append(g_out_deg)
                g_v_addfeat.append(g_in_deg)

            if len(p_v_addfeat) > 0:
                p_v_addfeat = th.cat(p_v_addfeat, dim=-1)
                p_v_addfeat = self.refine_node_weights(p_v_addfeat)
                p_v_output = th.cat([p_v_addfeat, p_v_rep], dim=-1)

                g_v_addfeat = th.cat(g_v_addfeat, dim=-1)
                g_v_addfeat = self.refine_node_weights(g_v_addfeat)
                g_v_output = th.cat([g_v_addfeat, g_v_rep], dim=-1)
            else:
                p_v_output = p_v_rep
                g_v_output = g_v_rep

            del p_v_addfeat, g_v_addfeat

            p_v_mask = self.refine_node_weights(p_v_mask)
            p_v_output = split_and_batchify_graph_feats(p_v_output, p_v_len, pre_pad=True)[0]
            p_v_output = p_v_output.masked_fill_(~(p_v_mask), 0)

            g_v_mask = self.refine_node_weights(g_v_mask)
            g_v_output = split_and_batchify_graph_feats(g_v_output, g_v_len, pre_pad=True)[0]
            g_v_output = g_v_output.masked_fill_(~(g_v_mask), 0)
        else:
            p_v_output = None
            g_v_output = None

        if self.edge_pred:
            p_u, p_v, p_e = pattern.all_edges(form="all", order="eid")
            g_u, g_v, g_e = graph.all_edges(form="all", order="eid")

            p_e_addfeat = []
            g_e_addfeat = []
            if self.pred_with_enc:
                p_e_addfeat.append(p_enc["v"][p_u])
                p_e_addfeat.append(p_enc["v"][p_v])
                p_e_addfeat.append(p_enc["vl"][p_u])
                p_e_addfeat.append(p_enc["el"][p_e])
                p_e_addfeat.append(p_enc["vl"][p_v])

                g_e_addfeat.append(g_enc["v"][g_u])
                g_e_addfeat.append(g_enc["v"][g_v])
                g_e_addfeat.append(g_enc["vl"][g_u])
                g_e_addfeat.append(g_enc["el"][g_e])
                g_e_addfeat.append(g_enc["vl"][g_v])

            if self.pred_with_deg:
                p_e_addfeat.append(p_out_deg[p_u])
                p_e_addfeat.append(p_in_deg[p_v])

                g_e_addfeat.append(g_out_deg[g_u])
                g_e_addfeat.append(g_in_deg[g_v])

            if len(p_e_addfeat) > 0:
                p_e_addfeat = th.cat(p_e_addfeat, dim=-1)
                p_e_addfeat = self.refine_edge_weights(p_e_addfeat)
                p_e_output = th.cat([p_e_addfeat, p_e_rep], dim=-1)

                g_e_addfeat = th.cat(g_e_addfeat, dim=-1)
                g_e_addfeat = self.refine_edge_weights(g_e_addfeat)
                g_e_output = th.cat([g_e_addfeat, g_e_rep], dim=-1)
            else:
                p_e_output = p_e_rep
                g_e_output = g_e_rep

            del p_e_addfeat, g_e_addfeat

            p_e_mask = self.refine_edge_weights(p_e_mask)
            p_e_output = split_and_batchify_graph_feats(p_e_output, p_e_len, pre_pad=True)[0]
            p_e_output = p_e_output.masked_fill(~(p_e_mask), 0)
            g_e_mask = self.refine_edge_weights(g_e_mask)
            g_e_output = split_and_batchify_graph_feats(g_e_output, g_e_len, pre_pad=True)[0]
            g_e_output = g_e_output.masked_fill(~(g_e_mask), 0)

        else:
            p_e_output = None
            g_e_output = None

        p_v_mask = p_v_mask.view(bsz, -1)
        p_e_mask = p_e_mask.view(bsz, -1)
        g_v_mask = g_v_mask.view(bsz, -1)
        g_e_mask = g_e_mask.view(bsz, -1)

        pred_c, (pred_v, pred_e) = self.get_subiso_pred(
            p_v_output, p_v_mask,
            p_e_output, p_e_mask,
            g_v_output, g_v_mask,
            g_e_output, g_e_mask
        )

        output = OutputDict(
            p_v_emb=p_v_emb,
            p_e_emb=p_e_emb,
            g_v_emb=g_v_emb,
            g_e_emb=g_e_emb,
            p_v_rep=p_v_rep,
            p_e_rep=p_e_rep,
            g_v_rep=g_v_rep,
            g_e_rep=g_e_rep,
            p_v_mask=p_v_mask,
            p_e_mask=p_e_mask,
            g_v_mask=g_v_mask,
            g_e_mask=g_e_mask,
            pred_c=pred_c,
            pred_v=pred_v,
            pred_e=pred_e,
        )

        return output
