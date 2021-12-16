import dgl.function as fn
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


class DMPLRPPoolLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        init_neigenv=4.0, # empirical value of triangles 
        init_eeigenv=4.0, # empirical value of triangles
        lrp_seq_len=4,
        bias=True,
        num_mlp_layers=2,
        batch_norm=True,
        act_func="relu",
        dropout=0.0
    ):
        super(DMPLRPPoolLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lrp_seq_len = lrp_seq_len
        self.num_rels = 3

        self.in_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.out_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.src_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.dst_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.nloop_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.eloop_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim))
        self.lrp_weight = nn.Parameter(th.Tensor(input_dim, hidden_dim, lrp_seq_len * lrp_seq_len))
        if bias:
            self.nbias = nn.Parameter(th.Tensor(hidden_dim))
            self.ebias = nn.Parameter(th.Tensor(hidden_dim))
            self.lrp_bias = nn.Parameter(th.Tensor(hidden_dim))
        else:
            self.register_parameter("nbias", None)
            self.register_parameter("ebias", None)
            self.register_parameter("lrp_bias", None)

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
        init_weight(self.lrp_weight, init="uniform")
        for module in self.nmlp.modules():
            init_module(module, activation=act_func, init="uniform")
        for module in self.emlp.modules():
            init_module(module, activation=act_func, init="uniform")
        if bias:
            nn.init.zeros_(self.nbias)
            nn.init.zeros_(self.ebias)
            nn.init.zeros_(self.lrp_bias)
        
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

    def forward(self, graph, node_feat, edge_feat, pooling_matrix, node_to_perm_matrix, edge_to_perm_matrix):
        # g = graph.local_var()
        g = graph
        self.node_init_func(g, node_feat)
        self.edge_init_func(g, edge_feat)
        g.update_all(self.node_message_func, self.node_reduce_func, self.node_update_func)
        g.apply_edges(self.edge_update_func)

        node_out, edge_out = g.ndata.pop(NODEOUTPUT), g.edata.pop(EDGEOUTPUT)

        node_out = tsp.mm(node_to_perm_matrix, node_out) + tsp.mm(edge_to_perm_matrix, edge_out)
        node_out = node_out.view(-1, self.lrp_seq_len * self.lrp_seq_len, self.input_dim)
        node_out = th.einsum('dab,bca->dc', node_out, self.lrp_weight)
        if self.lrp_bias is not None:
            node_out = node_out + self.lrp_bias
        node_out = tsp.mm(pooling_matrix, node_out)

        return node_out, edge_out, pooling_matrix, node_to_perm_matrix, edge_to_perm_matrix

    def extra_repr(self):
        summary = [
            "in=%s, out=%s" % (self.input_dim, self.hidden_dim),
            "lrp_seq_len=%s" % (self.lrp_seq_len),
        ]

        return "\n".join(summary)

    def get_output_dim(self):
        return self.hidden_dim


class DMPLRP(GraphAdjModelV2):
    def __init__(self, **kw):
        super(DMPLRP, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
        init_neigenv = kw.get("init_neigenv", 4.0)
        init_eeigenv = kw.get("init_eeigenv", 4.0)
        lrp_seq_len = kw.get("lrp_seq_len", 4)
        num_mlp_layers = kw.get("rep_dmpnn_num_mlp_layers", 2)
        batch_norm = kw.get("rep_dmpnn_batch_norm", False)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)

        DMPLRP = ModuleList()
        for i in range(num_layers):
            DMPLRP.add_module(
                "%s_DMPLRP_(%d)" % (type, i),
                DMPLRPPoolLayer(
                    self.hid_dim,
                    self.hid_dim,
                    init_neigenv=init_neigenv,
                    init_eeigenv=init_eeigenv,
                    lrp_seq_len=lrp_seq_len,
                    num_mlp_layers=num_mlp_layers,
                    batch_norm=batch_norm,
                    act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"DMPLRP": DMPLRP})

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

        for layer in self.p_rep_net["DMPLRP"]:
            v, e, p_perm_pool, p_n_perm_matrix, p_e_perm_matrix = layer(
                pattern, v_outputs[-1], e_outputs[-1], p_perm_pool, p_n_perm_matrix, p_e_perm_matrix
            )
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

        for layer in self.g_rep_net["DMPLRP"]:
            v, e, g_perm_pool, g_n_perm_matrix, g_e_perm_matrix = layer(
                graph, v_outputs[-1], e_outputs[-1], g_perm_pool, g_n_perm_matrix, g_e_perm_matrix
            )
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
