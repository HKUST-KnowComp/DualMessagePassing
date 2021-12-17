import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import GraphAdjModelV2
from .container import *
# from ..constants import *
# from ..utils.act import map_activation_str_to_layer
# from ..utils.init import init_weight, init_module
from constants import *
from utils.act import map_activation_str_to_layer
from utils.init import init_weight, init_module


class DMPLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        init_neigenv=4.0, # empirical value of triangles 
        init_eeigenv=4.0, # empirical value of triangles
        bias=True,
        num_mlp_layers=2,
        batch_norm=True,
        act_func="relu",
        dropout=0.0
    ):
        super(DMPLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.in_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.out_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.src_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.dst_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.nloop_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.eloop_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        if bias:
            self.nbias = nn.Parameter(th.Tensor(hidden_dim))
            self.ebias = nn.Parameter(th.Tensor(hidden_dim))
        else:
            self.register_parameter("nbias", None)
            self.register_parameter("ebias", None)
        self.nmlp = []
        for i in range(num_mlp_layers):
            self.nmlp.append(nn.Linear(hidden_dim, hidden_dim))
            if i != num_mlp_layers - 1:
                if batch_norm:
                    self.nmlp.append(nn.BatchNorm1d(hidden_dim))
                self.nmlp.append(map_activation_str_to_layer(act_func))
        self.nmlp = Sequential(*self.nmlp)
        self.emlp = []
        for i in range(num_mlp_layers):
            self.emlp.append(nn.Linear(hidden_dim, hidden_dim))
            if i != num_mlp_layers - 1:
                if batch_norm:
                    self.emlp.append(nn.BatchNorm1d(hidden_dim))
                self.emlp.append(map_activation_str_to_layer(act_func))
        self.emlp = Sequential(*self.emlp)
        self.act = map_activation_str_to_layer(act_func)
        self.drop = nn.Dropout(dropout)

        # init
        init_weight(self.in_weight, activation=act_func, init="uniform")
        init_weight(self.out_weight, activation=act_func, init="uniform")
        init_weight(self.src_weight, activation=act_func, init="uniform")
        init_weight(self.dst_weight, activation=act_func, init="uniform")
        init_weight(self.nloop_weight, activation=act_func, init="uniform")
        init_weight(self.eloop_weight, activation=act_func, init="uniform")
        for module in self.nmlp.modules():
            init_module(module, activation=act_func, init="uniform")
        for module in self.emlp.modules():
            init_module(module, activation=act_func, init="uniform")
        if bias:
            nn.init.zeros_(self.nbias)
            nn.init.zeros_(self.ebias)

        # reparamerization tricks
        with th.no_grad():
            self.in_weight.data.div_(init_neigenv)
            self.out_weight.data.div_(init_neigenv)
            self.nloop_weight.data.div_(init_neigenv)
            self.src_weight.data.div_(init_eeigenv)
            self.dst_weight.data.div_(init_eeigenv)
            self.eloop_weight.data.div_(init_eeigenv)

        # register functions
        self.node_init_func = self._node_init_func
        self.edge_init_func = self._edge_init_func
        self.node_message_func = self._node_message_func
        self.node_reduce_func = fn.sum(msg=NODEMSG, out=NODEAGG)
        self.node_update_func = self._node_update_func
        self.edge_update_func = self._edge_update_func

    def _node_init_func(self, graph, node_feat=None):
        if node_feat is not None:
            graph.ndata[NODEFEAT] = node_feat

        if OUTDEGREE not in graph.ndata:
            graph.ndata[OUTDEGREE] = graph.out_degrees()

        return graph.ndata[NODEFEAT]

    def _edge_init_func(self, graph, edge_feat=None):
        if edge_feat is not None:
            graph.edata[EDGEFEAT] = edge_feat

        return graph.edata[EDGEFEAT]

    def _node_message_func(self, edges):
        edge_msg = th.matmul(edges.dst[NODEFEAT], self.dst_weight) - th.matmul(edges.src[NODEFEAT], self.src_weight)
        node_msg = -th.matmul(edges.data[EDGEFEAT], self.in_weight)

        # no need to half them further
        if REVFLAG in edges.data:
            rmask = edges.data[REVFLAG].view(-1, 1)
            mask = ~(rmask)

            rev_edge_msg = th.matmul(edges.src[NODEFEAT], self.dst_weight) - th.matmul(edges.dst[NODEFEAT], self.src_weight)
            rev_node_msg = th.matmul(edges.data[EDGEFEAT], self.out_weight)

            edge_msg = edge_msg.masked_fill(rmask, 0.0) + rev_edge_msg.masked_fill(mask, 0.0)
            node_msg = node_msg.masked_fill(rmask, 0.0) + rev_node_msg.masked_fill(mask, 0.0)

        edges.data[EDGEAGG] = edge_msg
        return {NODEMSG: node_msg}

    def _node_update_func(self, nodes):
        agg = nodes.data[NODEAGG]
        out = th.matmul(nodes.data[NODEFEAT], self.nloop_weight) + agg
        if self.nbias is not None:
            out = out + self.nbias
        if len(self.nmlp) > 0:
            out = self.nmlp(out)
        else:
            out = self.act(out)
        out = self.drop(out)

        return {NODEOUTPUT: out}

    def _edge_update_func(self, edges):
        agg = edges.data[EDGEAGG]
        d = edges.dst[OUTDEGREE].unsqueeze(-1).float()
        d = (1 + d).log2() # avoid nan ...
        add = 2 * (1 + d) * th.matmul(edges.data[EDGEFEAT], (self.src_weight - self.dst_weight))
        out = th.matmul(edges.data[EDGEFEAT], self.eloop_weight) + add + agg
        if self.ebias is not None:
            out = out + self.ebias
        if len(self.emlp) > 0:
            out = self.emlp(out)
        else:
            out = self.act(out)
        out = self.drop(out)

        return {EDGEOUTPUT: out}

    def forward(self, graph, node_feat, edge_feat):
        # g = graph.local_var()
        g = graph
        self.node_init_func(g, node_feat)
        self.edge_init_func(g, edge_feat)
        g.update_all(self.node_message_func, self.node_reduce_func, self.node_update_func)
        g.apply_edges(self.edge_update_func)

        return g.ndata.pop(NODEOUTPUT), g.edata.pop(EDGEOUTPUT)

    def extra_repr(self):
        summary = [
            "in=%s, out=%s" % (self.input_dim, self.hidden_dim),
        ]

        return "\n".join(summary)

    def get_output_dim(self):
        return self.hidden_dim


class DMPNN(GraphAdjModelV2):
    def __init__(self, **kw):
        super(DMPNN, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
        init_neigenv = kw.get("init_neigenv", 4.0)
        init_eeigenv = kw.get("init_eeigenv", 4.0)
        num_mlp_layers = kw.get("rep_dmpnn_num_mlp_layers", 2)
        batch_norm = kw.get("rep_dmpnn_batch_norm", False)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)

        dmpnn = ModuleList()
        for i in range(num_layers):
            dmpnn.add_module(
                "%s_dmpnn_(%d)" % (type, i),
                DMPLayer(
                    self.hid_dim,
                    self.hid_dim,
                    init_neigenv=init_neigenv,
                    init_eeigenv=init_eeigenv,
                    num_mlp_layers=num_mlp_layers,
                    batch_norm=batch_norm,
                    act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"dmpnn": dmpnn})

    def get_pattern_rep(self, pattern, p_v_emb, p_e_emb, v_mask=None, e_mask=None):
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

        for layer in self.p_rep_net["dmpnn"]:
            v, e = layer(pattern, v_outputs[-1], e_outputs[-1])
            if p_v_zero_mask is not None:
                v = v.masked_fill(p_v_zero_mask, 0.0)
            if p_e_zero_mask is not None:
                e = e.masked_fill(p_e_zero_mask, 0.0)
            if self.rep_residual and v_outputs[-1].size() == v.size() and e_outputs[-1].size() == e.size():
                v_outputs.append(v_outputs[-1] + v)
                e_outputs.append(e_outputs[-1] + e)
            else:
                v_outputs.append(v)
                e_outputs.append(e)

        return v_outputs[-1], e_outputs[-1]

    def get_graph_rep(self, graph, g_v_emb, g_e_emb, v_mask=None, e_mask=None, v_gate=None, e_gate=None):
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

        for layer in self.g_rep_net["dmpnn"]:
            v, e = layer(graph, v_outputs[-1], e_outputs[-1])
            if v_gate is not None:
                v = v * v_gate
            if e_gate is not None:
                e = e * e_gate
            if self.rep_residual and v_outputs[-1].size() == v.size() and e_outputs[-1].size() == e.size():
                v_outputs.append(v_outputs[-1] + v)
                e_outputs.append(e_outputs[-1] + e)
            else:
                v_outputs.append(v)
                e_outputs.append(e)

        return v_outputs[-1], e_outputs[-1]
