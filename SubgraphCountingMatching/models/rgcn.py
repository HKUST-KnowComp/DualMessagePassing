import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import GraphAdjModel
from .container import *
# from ..constants import *
# from ..utils.act import map_activation_str_to_layer
# from ..utils.init import init_weight, init_module
from constants import *
from utils.act import map_activation_str_to_layer
from utils.init import init_weight, init_module


class RGCNLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_rels=1,
        regularizer="basis",
        num_bases=-1,
        edge_norm="in",
        self_loop=True,
        bias=True,
        batch_norm=False,
        act_func="relu",
        dropout=0.0,
    ):
        super(RGCNLayer, self).__init__()
        assert regularizer in ["none", "basis", "bdd"]
        assert edge_norm in ["none", "in", "both"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.regularizer = regularizer
        if regularizer == "none" or num_bases is None or num_bases > num_rels or num_bases <= 0:
            self.num_bases = num_rels
        else:
            self.num_bases = num_bases
        self.edge_norm = edge_norm
        if self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        else:
            self.register_parameter("loop_weight", None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(hidden_dim))
        else:
            self.register_parameter("bias", None)
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None
        self.act = map_activation_str_to_layer(act_func)
        self.drop = nn.Dropout(dropout)

        if regularizer == "none" or regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.input_dim, self.hidden_dim))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            else:
                self.register_parameter("w_comp", None)
        elif regularizer == "bdd":
            if input_dim % self.num_bases != 0 or hidden_dim % self.num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases (%d).' % self.num_bases)
            # add block diagonal weights
            submat_in = input_dim // self.num_bases
            submat_out = hidden_dim // self.num_bases

            # assuming input_dim and hidden_dim are both divisible by num_bases
            self.weight = nn.Parameter(th.Tensor(self.num_rels, self.num_bases * submat_in * submat_out))
            self.register_parameter("w_comp", None)
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")
        
        # init
        init_weight(self.weight, activation=act_func, init="uniform")
        if self.w_comp is not None:
            init_weight(self.w_comp, activation=act_func, init="uniform")
        if self_loop:
            init_weight(self.loop_weight, activation=act_func, init="uniform")
        nn.init.zeros_(self.bias)

        self.node_init_func = self._node_init_func
        self.edge_init_func = self._edge_init_func
        if regularizer == "none" or regularizer == "basis":
            self.node_message_func = self._basis_message_func
        elif regularizer == "bdd":
            self.node_message_func = self._bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")
        self.node_reduce_func = fn.sum(msg=NODEMSG, out=NODEAGG)
        self.node_update_func = self._node_update_func
        self.edge_update_func = None

    def _basis_message_func(self, edges):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases, self.input_dim * self.hidden_dim)
            weight = th.matmul(self.w_comp, weight).view(self.num_rels, self.input_dim, self.hidden_dim)
        else:
            weight = self.weight
        weight = weight.index_select(0, edges.data[EDGETYPE])
        msg = th.bmm(edges.src[NODEFEAT].unsqueeze(1), weight).squeeze(1)

        if self.edge_norm != "none":
            msg = msg * edges.data[NORM]
        return {NODEMSG: msg}

    def _bdd_message_func(self, edges):
        submat_in = self.input_dim // self.num_bases
        submat_out = self.hidden_dim // self.num_bases
        weight = self.weight.index_select(0, edges.data[EDGETYPE]).view(-1, submat_in, submat_out)
        msg = th.bmm(edges.src[NODEFEAT].view(-1, 1, submat_in), weight).view(-1, self.hidden_dim)

        if self.edge_norm != "none":
            msg = msg * edges.data[NORM]
        return {NODEMSG: msg}

    @property
    def self_loop(self):
        return hasattr(self, "loop_weight") and self.loop_weight is not None

    def _node_init_func(self, graph, node_feat=None):
        if node_feat is not None:
            graph.ndata[NODEFEAT] = node_feat

        if self.edge_norm == "in" or self.edge_norm == "both":
            if INDEGREE not in graph.ndata:
                in_deg = graph.ndata[INDEGREE] = graph.in_degrees()
            else:
                in_deg = graph.ndata[INDEGREE]
            if INNORM not in graph.ndata:
                if self.self_loop:
                    graph.ndata[INNORM] = (1.0 / (in_deg.float() + 1)).view(-1, 1)
                else:
                    graph.ndata[INNORM] = (1.0 / in_deg.float()).masked_fill_(in_deg == 0, 0.0).view(-1, 1)
        if self.edge_norm == "out" or self.edge_norm == "both":
            if OUTDEGREE not in graph.ndata:
                out_deg = graph.ndata[OUTDEGREE] = graph.out_degrees()
            else:
                out_deg = graph.ndata[OUTDEGREE]
            if OUTNORM not in graph.ndata:
                if self.self_loop:
                    graph.ndata[OUTNORM] = (1.0 / (out_deg.float() + 1)).view(-1, 1)
                else:
                    graph.ndata[OUTNORM] = (1.0 / out_deg.float()).masked_fill_(out_deg == 0, 0.0).view(-1, 1)

        return node_feat

    def _edge_init_func(self, graph, edge_type=None):
        if edge_type is not None:
            graph.edata[EDGETYPE] = edge_type

        if self.edge_norm == "in":
            graph.apply_edges(fn.CopyMessageFunction(fn.TargetCode.DST, INNORM, NORM))
        elif self.edge_norm == "out":
            # graph.apply_edges(fn.CopyMessageFunction(fn.TargetCode.SRC, OUTNORM, NORM))
            graph.apply_edges(fn.copy_u(OUTNORM, NORM))
        elif self.edge_norm == "both":
            graph.apply_edges(lambda edges: {NORM: (edges.src[OUTNORM] * edges.dst[INNORM]) ** 0.5})

        return edge_type
    
    def _node_update_func(self, nodes):
        agg = nodes.data[NODEAGG]

        if self.self_loop:
            loop_msg = th.matmul(nodes.data[NODEFEAT], self.loop_weight)
            if self.edge_norm == "in":
                out = agg + loop_msg * nodes.data[INNORM]
            elif self.edge_norm == "out":
                out = agg + loop_msg * nodes.data[OUTNORM]
            elif self.edge_norm == "both":
                out = agg + loop_msg * (nodes.data[INNORM] * nodes.data[OUTNORM])**0.5
            else:
                out = agg + loop_msg
        else:
            out = agg
        if self.bias is not None:
            out = out + self.bias
        if self.bn is not None:
            out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)

        return {NODEOUTPUT: out}

    def forward(self, g, node_feat, edge_type):
        self.node_init_func(g, node_feat)
        self.edge_init_func(g, edge_type)
        g.update_all(self.node_message_func, self.node_reduce_func, self.node_update_func)
        return g.ndata.pop(NODEOUTPUT), edge_type

    def get_output_dim(self):
        return self.hidden_dim

    def extra_repr(self):
        summary = [
            "in=%d, out=%d," % (self.input_dim, self.hidden_dim),
            "num_rels=%d, regularizer=%s, num_bases=%d," % (self.num_rels, self.regularizer, self.num_bases),
            "edge_norm=%s, self_loop=%s, bias=%s," % (self.edge_norm, self.self_loop, self.bias is not None),
        ]

        return "\n".join(summary)

    def get_output_dim(self):
        return self.hiden_dim


class RGCN(GraphAdjModel):
    def __init__(self, **kw):
        super(RGCN, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
            num_rels = self.max_ngel
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
            num_rels = self.max_npel
        edge_norm = kw.get("rep_rgcn_edge_norm", "in")
        batch_norm = kw.get("rep_rgcn_batch_norm", False)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)
        regularizer = kw.get("rep_rgcn_regularizer", "basis")
        num_bases = kw.get("rep_rgcn_num_bases", -1)

        rgcn = ModuleList()
        for i in range(num_layers):
            rgcn.add_module(
                "%s_rgcn_(%d)" % (type, i),
                RGCNLayer(
                    self.hid_dim,
                    self.hid_dim,
                    num_rels=num_rels,
                    regularizer=regularizer,
                    num_bases=num_bases,
                    edge_norm=edge_norm,
                    batch_norm=batch_norm,
                    act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"rgcn": rgcn})

    def get_pattern_rep(self, pattern, p_emb, mask=None):
        if mask is not None:
            p_zero_mask = ~(mask)
            outputs = [p_emb.masked_fill(p_zero_mask, 0.0)]
            etype = pattern.edata["label"]
            for layer in self.p_rep_net["rgcn"]:
                o, etype = layer(pattern, outputs[-1], etype)
                outputs.append(o.masked_fill(p_zero_mask, 0.0))
        else:
            outputs = [p_emb]
            etype = pattern.edata["label"]
            for layer in self.p_rep_net["rgcn"]:
                o, etype = layer(pattern, outputs[-1], etype)
                if self.rep_residual and outputs[-1].size() == o.size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)

        return outputs[-1]

    def get_graph_rep(self, graph, g_emb, mask=None, gate=None):
        if mask is None and gate is None:
            outputs = [g_emb]
            etype = graph.edata["label"]
            for layer in self.g_rep_net["rgcn"]:
                o, etype = layer(graph, outputs[-1], etype)
                if self.rep_residual and outputs[-1].size() == o.size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)
        else:
            if gate is None:
                gate = mask.float()
            elif mask is not None:
                gate = mask.float() * gate

            outputs = [g_emb * gate]
            etype = graph.edata["label"]
            for layer in self.g_rep_net["rgcn"]:
                o, etype = layer(graph, outputs[-1], etype)
                o = o * gate
                if self.rep_residual and outputs[-1].size() == o.size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)

        return outputs[-1]
