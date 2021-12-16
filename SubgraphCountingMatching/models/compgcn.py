
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


if th.__version__ < "1.7.0":
    def rfft(input, n=None, dim=-1, norm=None):
        # no image part
        inp_dim = input.dim()

        if dim < 0:
            dim = inp_dim + dim
        if n is not None:
            diff = input.size(dim) - n
            if diff > 0:
                input = th.split(input, dim=dim, split_size_or_sections=(n, diff))[0]
            # else:
            #     sizes = tuple(input.size())
            #     padded = th.zeros((sizes[:dim] + (-diff, ) + sizes[(dim+1):]), dtype=input.dtype, device=input.device)
            #     input = th.cat([input, padded], dim=dim)
        else:
            n = input.size(dim) // 2 + 1
        if norm is None or norm == "backward":
            normalized = False
        elif norm == "forward":
            normalized = True
        else:
            raise ValueError

        if dim != inp_dim - 1:
            input = input.transpose(dim, inp_dim - 1)
        output = th.rfft(input, signal_ndim=1, normalized=normalized)
        if dim != inp_dim - 1:
            output = output.transpose(dim, inp_dim - 1)

        return output

    def irfft(input, n=None, dim=-1, norm=None):
        # calculate the dimension of the input and regard the last as the (real, image)
        inp_dim = input.dim()
        if input.size(-1) != 2:
            input = th.stack([input, th.zeros_like(input)], dim=-1)
        else:
            inp_dim -= 1

        if dim < 0:
            dim = inp_dim + dim
        if n is not None:
            diff = input.size(dim) - n
            if diff > 0:
                input = th.split(input, dim=dim, split_size_or_sections=(n, diff))[0]
            # else:
            #     sizes = tuple(input.size())
            #     padded = th.zeros((sizes[:dim] + (-diff, ) + sizes[(dim+1):]), dtype=input.dtype, device=input.device)
            #     input = th.cat([input, padded], dim=dim)
        else:
            n = 2 * (input.size(dim) - 1)
        if norm is None or norm == "backward":
            normalized = False
        elif norm == "forward":
            normalized = True
        else:
            raise ValueError

        if dim != inp_dim - 1:
            input = input.transpose(dim, inp_dim - 1)
        output = th.irfft(input, signal_ndim=1, normalized=normalized, signal_sizes=[n])
        if dim != inp_dim - 1:
            output = output.transpose(dim, inp_dim - 1)

        return output
else:
    def rfft(input, n=None, dim=None, norm=None):
        return th.view_as_real(th.fft.rfft(input, n=n, dim=dim, norm=norm))

    def irfft(input, n=None, dim=None, norm=None):
        if not th.is_complex(input) and input.size(-1) == 2:
            input = th.view_as_complex(input)
        return th.fft.irfft(input, n=n, dim=dim, norm=norm)


def complex_mul(re_x, im_x, re_y, im_y):
    return (re_x * re_y - im_x * im_y), (im_x * re_y + re_x * im_y)


def complex_conj(re_x, im_x):
    return re_x, -im_x


class CompGCNLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        self_loop=True,
        comp_opt="mult",
        edge_norm="both",
        bias=True,
        batch_norm=True,
        act_func="relu",
        dropout=0.0
    ):
        super(CompGCNLayer, self).__init__()
        assert edge_norm in ["none", "in", "out", "both"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_norm = edge_norm

        if self_loop:
            self.num_rels = 3
        else:
            self.num_rels = 2

        self.comp_opt = comp_opt

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

        self.in_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.out_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.rel_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        if self_loop:
            self.loop_rel = nn.Parameter(th.Tensor(1, input_dim))
        else:
            self.register_parameter("loop_rel", None)
        self.act = map_activation_str_to_layer(act_func)
        self.drop = nn.Dropout(dropout)

        # init
        init_weight(self.in_weight, activation=act_func, init="uniform")
        init_weight(self.out_weight, activation=act_func, init="uniform")
        init_weight(self.rel_weight, activation=act_func, init="uniform")
        if self_loop:
            init_weight(self.loop_weight, activation=act_func, init="uniform")
            init_weight(self.loop_rel, activation=act_func, init="uniform")
        if bias:
            nn.init.zeros_(self.bias)

        # register functions
        self.node_init_func = self._node_init_func
        self.edge_init_func = self._edge_init_func
        self.node_message_func = self._node_message_func
        self.node_reduce_func = fn.sum(msg=NODEMSG, out=NODEAGG)
        self.node_update_func = self._node_update_func
        self.edge_update_func = self._edge_update_func

    @property
    def self_loop(self):
        return hasattr(self, "loop_weight") and self.loop_weight is not None

    def _node_init_func(self, graph, node_feat=None):
        if node_feat is not None:
            graph.ndata[NODEFEAT] = node_feat

        if self.edge_norm == "in" or self.edge_norm == "both":
            if INNORM not in graph.ndata:
                if INDEGREE not in graph.ndata:
                    in_deg = graph.ndata[INDEGREE] = graph.in_degrees()
                else:
                    in_deg = graph.ndata[INDEGREE]
                if self.self_loop:
                    graph.ndata[INNORM] = (in_deg + 1).reciprocal().unsqueeze(-1)
                else:
                    graph.ndata[INNORM] = in_deg.reciprocal().masked_fill_(in_deg == 0, 1.0).unsqueeze(-1)
        if self.edge_norm == "out" or self.edge_norm == "both":
            if OUTNORM not in graph.ndata:
                if OUTDEGREE not in graph.ndata:
                    out_deg = graph.ndata[OUTDEGREE] = graph.out_degrees()
                else:
                    out_deg = graph.ndata[OUTDEGREE]
                if self.self_loop:
                    graph.ndata[OUTNORM] = (out_deg + 1).reciprocal().unsqueeze(-1)
                else:
                    graph.ndata[OUTNORM] = out_deg.reciprocal().masked_fill_(out_deg == 0, 1.0).unsqueeze(-1)

        return graph.ndata[NODEFEAT]

    def _edge_init_func(self, graph, edge_feat=None):
        if edge_feat is not None:
            graph.edata[EDGEFEAT] = edge_feat

        if self.edge_norm == "in":
            graph.apply_edges(fn.CopyMessageFunction(fn.TargetCode.DST, INNORM, NORM))
        elif self.edge_norm == "out":
            graph.apply_edges(fn.CopyMessageFunction(fn.TargetCode.SRC, OUTNORM, NORM))
        elif self.edge_norm == "both":
            graph.apply_edges(lambda edges: {NORM: (edges.src[OUTNORM] * edges.dst[INNORM])**0.5})

        return graph.edata[EDGEFEAT]

    def _comp_func(self, head, relation):
        if self.comp_opt == "sub":
            return head - relation
        elif self.comp_opt == "mult":
            return head * relation
        elif self.comp_opt == "corr":
            re_h, im_h = th.chunk(rfft(head, dim=-1), chunks=2, dim=-1)
            re_r, im_r = th.chunk(rfft(relation, dim=-1), chunks=2, dim=-1)
            o = th.stack(complex_mul(*complex_conj(re_h, im_h), re_r, im_r), dim=-1)
            return irfft(o, dim=1, n=head.size(1)).squeeze(-1)
        else:
            raise NotImplementedError

    def _node_message_func(self, edges):
        comp = self._comp_func(edges.src[NODEFEAT], edges.data[EDGEFEAT])
        msg = th.matmul(comp, self.in_weight)

        if REVFLAG in edges.data:
            rev_msg = th.matmul(comp, self.out_weight)
            mask = ~(edges.data[REVFLAG]).view(-1, 1)
            msg = msg.masked_fill(~(mask), 0.0) + rev_msg.masked_fill(mask, 0.0)

        if self.edge_norm != "none":
            msg = msg * edges.data[NORM]

        return {NODEMSG: msg}

    def _node_update_func(self, nodes):
        agg = nodes.data[NODEAGG]

        if self.self_loop:
            loop_msg = self._comp_func(nodes.data[NODEFEAT], self.loop_rel)
            loop_msg = th.matmul(loop_msg, self.loop_weight)
            out = agg + loop_msg
            out = out * 0.3333333
        else:
            out = agg * 0.5

        if self.bias is not None:
            out = out + self.bias
        if self.bn is not None:
            out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)

        return {NODEOUTPUT: out}

    def _edge_update_func(self, edges):
        out = th.matmul(edges.data[EDGEFEAT], self.rel_weight)

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
            "in=%s, out=%s," % (self.input_dim, self.hidden_dim),
            "comp_opt=%s," % (self.comp_opt),
            "edge_norm=%s, self_loop=%s, bias=%s," % (self.edge_norm, self.self_loop, self.bias is not None),
        ]

        return "\n".join(summary)

    def get_output_dim(self):
        return self.hidden_dim


class CompGCN(GraphAdjModelV2):
    def __init__(self, **kw):
        super(CompGCN, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
        comp_opt = kw.get("rep_compgcn_comp_opt", "mult")
        edge_norm = kw.get("rep_compgcn_edge_norm", "none")
        batch_norm = kw.get("rep_compgcn_batch_norm", False)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)

        compgcn = ModuleList()
        for i in range(num_layers):
            compgcn.add_module(
                "%s_compgcn_(%d)" % (type, i),
                CompGCNLayer(
                    self.hid_dim,
                    self.hid_dim,
                    comp_opt=comp_opt,
                    edge_norm=edge_norm,
                    batch_norm=batch_norm,
                    act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"compgcn": compgcn})

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

        for layer in self.p_rep_net["compgcn"]:
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

        for layer in self.g_rep_net["compgcn"]:
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
