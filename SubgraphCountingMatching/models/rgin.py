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


class RGINLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_rels=1,
        regularizer="basis",
        num_bases=-1,
        num_mlp_layers=2,
        self_loop=True,
        bias=True,
        batch_norm=False,
        act_func="relu",
        dropout=0.0,
    ):
        super(RGINLayer, self).__init__()
        assert regularizer in ["none", "basis", "bdd"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.regularizer = regularizer
        if regularizer == "none" or num_bases is None or num_bases > num_rels or num_bases <= 0:
            self.num_bases = num_rels
        else:
            self.num_bases = num_bases
        if self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        else:
            self.register_parameter("loop_weight", None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(hidden_dim))
        else:
            self.register_parameter("bias", None)
        self.mlp = []
        for i in range(num_mlp_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            if i != num_mlp_layers - 1:
                if batch_norm:
                    self.mlp.append(nn.BatchNorm1d(hidden_dim))
                self.mlp.append(map_activation_str_to_layer(act_func))
        self.mlp = Sequential(*self.mlp)
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

        return {NODEMSG: msg}

    def _bdd_message_func(self, edges):
        submat_in = self.input_dim // self.num_bases
        submat_out = self.hidden_dim // self.num_bases
        weight = self.weight.index_select(0, edges.data[EDGETYPE]).view(-1, submat_in, submat_out)
        msg = th.bmm(edges.src[NODEFEAT].view(-1, 1, submat_in), weight).view(-1, self.hidden_dim)

        return {NODEMSG: msg}

    @property
    def self_loop(self):
        return hasattr(self, "loop_weight") and self.loop_weight is not None

    def _node_init_func(self, graph, node_feat=None):
        if node_feat is not None:
            graph.ndata[NODEFEAT] = node_feat
        return node_feat

    def _edge_init_func(self, graph, edge_type=None):
        if edge_type is not None:
            graph.edata[EDGETYPE] = edge_type

        return edge_type
    
    def _node_update_func(self, nodes):
        agg = nodes.data[NODEAGG]

        if self.self_loop:
            loop_msg = th.matmul(nodes.data[NODEFEAT], self.loop_weight)
            out = agg + loop_msg
        else:
            out = agg
        if self.bias is not None:
            out = out + self.bias
        if len(self.mlp) > 0:
            out = self.mlp(out)
        else:
            out = self.act(out)
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


class RGIN(GraphAdjModel):
    def __init__(self, **kw):
        super(RGIN, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
            num_rels = self.max_ngel
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
            num_rels = self.max_npel
        regularizer = kw.get("rep_rgin_regularizer", "basis")
        num_bases = kw.get("rep_rgin_num_bases", -1)
        num_mlp_layers = kw.get("rep_rgin_num_mlp_layers", 2)
        batch_norm = kw.get("rep_rgin_batch_norm", False)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)

        rgin = ModuleList()
        for i in range(num_layers):
            rgin.add_module(
                "%s_rgin_(%d)" % (type, i),
                RGINLayer(
                    self.hid_dim,
                    self.hid_dim,
                    num_rels=num_rels,
                    regularizer=regularizer,
                    num_bases=num_bases,
                    num_mlp_layers=num_mlp_layers,
                    batch_norm=batch_norm,
                    act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"rgin": rgin})

    def get_pattern_rep(self, pattern, p_emb, mask=None):
        if mask is not None:
            p_zero_mask = ~(mask)
            outputs = [p_emb.masked_fill(p_zero_mask, 0.0)]
            etype = pattern.edata["label"]
            for layer in self.p_rep_net["rgin"]:
                o, etype = layer(pattern, outputs[-1], etype)
                outputs.append(o.masked_fill(p_zero_mask, 0.0))
        else:
            outputs = [p_emb]
            etype = pattern.edata["label"]
            for layer in self.p_rep_net["rgin"]:
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
            for layer in self.g_rep_net["rgin"]:
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
            for layer in self.g_rep_net["rgin"]:
                o, etype = layer(graph, outputs[-1], etype)
                o = o * gate
                if self.rep_residual and outputs[-1].size() == o.size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)

        return outputs[-1]
