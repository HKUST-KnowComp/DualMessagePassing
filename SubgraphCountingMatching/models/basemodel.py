import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .container import *
from .embed import *
from .filter import *
from .pred import *
# from ..constants import *
# from ..utils.dl import batch_convert_len_to_mask, split_and_batchify_graph_feats, expand_dimensions
from constants import *
from utils.dl import batch_convert_len_to_mask, split_and_batchify_graph_feats, expand_dimensions

class BaseModel(nn.Module):
    """
    1. encoding
    2. filter + embedding
    3. representation
    4. interaction + prediction
    """
    def __init__(self, **kw):
        super(BaseModel, self).__init__()

        self.max_ngv = kw["max_ngv"]
        self.max_ngvl = kw["max_ngvl"]
        self.max_nge = kw["max_nge"]
        self.max_ngel = kw["max_ngel"]
        self.max_npv = kw["max_npv"]
        self.max_npvl = kw["max_npvl"]
        self.max_npe = kw["max_npe"]
        self.max_npel = kw["max_npel"]
        self.base = kw.get("base", 2)
        self.hid_dim = kw.get("hid_dim", 64)
        self.share_emb_net = kw.get("share_emb_net", True)
        self.share_enc_net = kw.get("share_enc_net", True)
        self.share_rep_net = kw.get("share_rep_net", True)
        self.rep_residual = kw.get("rep_residual", True)

        self.pred_with_enc = kw.get("pred_with_enc", False)
        self.pred_with_deg = kw.get("pred_with_deg", False)

        # create encoding layer
        self.g_enc_net = self.create_enc_net(type="graph", **kw)
        self.p_enc_net = self.create_enc_net(type="pattern", **kw)

        # create filter layers
        self.filter_net = self.create_filter_net(**kw)

        # create embedding layers
        self.g_emb_net = self.create_emb_net(type="graph", **kw)
        self.p_emb_net = self.create_emb_net(type="pattern", **kw)

        # create networks
        self.g_rep_net = self.create_rep_net(type="graph", **kw)
        self.p_rep_net = self.create_rep_net(type="pattern", **kw)

        # create predict layers
        self.pred_net = self.create_pred_net(**kw)

    def create_enc_net(self, type, **kw):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def create_filter_net(self, **kw):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def create_emb_net(self, type, **kw):
        emb_net = kw.get("emb_net", "Orthogonal")
        hid_dim = self.hid_dim

        if type == "graph":
            input_dims = self.get_graph_enc_dims()
        elif type == "pattern":
            input_dims = self.get_pattern_enc_dims()
        else:
            raise ValueError

        if emb_net == "Orthogonal":
            emb_net = OrderedDict({k: OrthogonalEmbedding(v, hid_dim) for k, v in input_dims.items()})
        elif emb_net == "Normal":
            emb_net = OrderedDict({k: NormalEmbedding(v, hid_dim) for k, v in input_dims.items()})
        elif emb_net == "Uniform":
            emb_net = OrderedDict({k: UniformEmbedding(v, hid_dim) for k, v in input_dims.items()})
        elif emb_net == "Equivariant":
            emb_net = OrderedDict({k: EquivariantEmbedding(v, hid_dim) for k, v in input_dims.items()})
        else:
            raise ValueError

        return ModuleDict(emb_net)

    def create_rep_net(self, type, **kw):
        # implement by subclasses
        raise NotImplementedError

    def create_pred_net(self, **kw):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_graph_enc_dims(self):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_pattern_enc_dims(self):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_graph_enc_dim(self):
        enc_dims = self.get_graph_enc_dims()
        return sum(enc_dims.values())

    def get_pattern_enc_dim(self):
        enc_dims = self.get_pattern_enc_dims()
        return sum(enc_dims.values())

    def get_rep_dim(self):
        rep_dim = self.hid_dim
        if self.pred_with_enc:
            rep_dim += self.get_graph_enc_dim()
        if self.pred_with_deg:
            rep_dim += 2
        return rep_dim

    def get_filter_gate(self, *args):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_pattern_enc(self, pattern):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_graph_enc(self, graph):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_pattern_emb(self, p_enc):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_graph_emb(self, g_enc):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def get_pattern_rep(self, p_emb, mask=None):
        # implement by subclasses
        raise NotImplementedError

    def get_graph_rep(self, g_emb, mask=None, gate=None):
        # implement by subclasses
        raise NotImplementedError

    def get_subiso_pred(self, *args):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def refine_node_weights(self, weights, use_max=False):
        return weights

    def refine_edge_weights(self, weights, use_max=False):
        return weights

    def forward(self, *args):
        # implement by EdgeSeqModel or GraphAdjModel
        raise NotImplementedError

    def expand(self, **kw):
        if "base" in kw and kw["base"] != self.base:
            raise ValueError
        kw = dict(kw)
        bak = dict()
        for k in ["max_npv", "max_npvl", "max_npe", "max_npel", "max_ngv", "max_ngvl", "max_nge", "max_ngel"]:
            bak[k] = getattr(self, k)
            setattr(self, k, max(kw.get(k, -1), bak[k]))

        try:
            # no need to expand encoding layer
            new_g_enc_net = self.create_enc_net(type="graph", **kw)
            del self.g_enc_net
            self.g_enc_net = new_g_enc_net

            if self.share_enc_net:
                self.p_enc_net = self.g_enc_net
            else:
                new_p_enc_net = self.create_enc_net(type="pattern", **kw)
                del self.p_enc_net
                self.p_enc_net = new_p_enc_net

            # expanding filter layers
            new_filter_net = self.create_filter_net(**kw)
            expand_dimensions(self.filter_net, new_filter_net, pre_pad=True)
            del self.filter_net
            self.filter_net = new_filter_net

            # expanding embedding layers
            new_g_emb_net = self.create_emb_net(type="graph", **kw)
            expand_dimensions(self.g_emb_net, new_g_emb_net, pre_pad=True)
            del self.g_emb_net
            self.g_emb_net = new_g_emb_net

            if self.share_emb_net:
                self.p_emb_net = self.g_emb_net
            else:
                new_p_emb_net = self.create_emb_net(type="pattern", **kw)
                expand_dimensions(self.p_emb_net, new_p_emb_net, pre_pad=True)
                del self.p_emb_net
                self.p_emb_net = new_p_emb_net

            # expanding predict layers
            if self.pred_with_enc:
                new_pred_net = self.create_pred_net(**kw)
                expand_dimensions(self.pred_net, new_pred_net, pre_pad=True)
                del self.pred_net
                self.pred_net = new_pred_net
        except Exception as e:
            for k, v in bak.items():
                setattr(self, k, v)
            # print(e)
            raise e


class EdgeSeqModel(BaseModel):
    def __init__(self, **kw):
        super(EdgeSeqModel, self).__init__(**kw)

    def create_enc_net(self, type, **kw):
        enc_net = kw.get("enc_net", "Multihot")

        if type == "graph":
            if enc_net == "Multihot":
                enc_net = OrderedDict(
                    {
                        "u": MultihotEmbedding(self.max_ngv, self.base),
                        "v": MultihotEmbedding(self.max_ngv, self.base),
                        "ul": MultihotEmbedding(self.max_ngvl, self.base),
                        "el": MultihotEmbedding(self.max_ngel, self.base),
                        "vl": MultihotEmbedding(self.max_ngvl, self.base),
                    }
                )
            elif enc_net == "Position":
                enc_net = OrderedDict(
                    {
                        "u": PositionEmbedding(get_enc_len(self.max_ngv-1, self.base) * self.base, self.max_ngv),
                        "v": PositionEmbedding(get_enc_len(self.max_ngv-1, self.base) * self.base, self.max_ngv),
                        "ul": PositionEmbedding(get_enc_len(self.max_ngvl-1, self.base) * self.base, self.max_ngvl),
                        "el": PositionEmbedding(get_enc_len(self.max_ngel-1, self.base) * self.base, self.max_ngel),
                        "vl": PositionEmbedding(get_enc_len(self.max_ngvl-1, self.base) * self.base, self.max_ngvl),
                    }
                )
            else:
                raise NotImplementedError
        elif type == "pattern":
            if self.share_enc_net:
                return self.g_enc_net
            else:
                if enc_net == "Multihot":
                    enc_net = OrderedDict(
                        {
                            "u": MultihotEmbedding(self.max_npv, self.base),
                            "v": MultihotEmbedding(self.max_npv, self.base),
                            "ul": MultihotEmbedding(self.max_npvl, self.base),
                            "el": MultihotEmbedding(self.max_npel, self.base),
                            "vl": MultihotEmbedding(self.max_npvl, self.base),
                        }
                    )
                elif enc_net == "Position":
                    enc_net = OrderedDict(
                        {
                            "u": PositionEmbedding(get_enc_len(self.max_npv - 1, self.base) * self.base, self.max_npv),
                            "v": PositionEmbedding(get_enc_len(self.max_npv - 1, self.base) * self.base, self.max_npv),
                            "ul": PositionEmbedding(get_enc_len(self.max_npvl - 1, self.base) * self.base, self.max_npvl),
                            "el": PositionEmbedding(get_enc_len(self.max_npel - 1, self.base) * self.base, self.max_npel),
                            "vl": PositionEmbedding(get_enc_len(self.max_npvl - 1, self.base) * self.base, self.max_npvl),
                        }
                    )
                else:
                    raise NotImplementedError
        else:
            raise ValueError

        for net in enc_net.values():
            net.weight.requires_grad = False
        
        return ModuleDict(enc_net)

    def create_filter_net(self, **kw):
        filter_net = kw.get("filter_net", "None")

        if filter_net == "None":
            return None
        elif filter_net == "ScalarFilter":
            return ModuleDict({k: ScalarFilter() for k in ["ul", "el", "vl"]})
        else:
            raise ValueError

    def create_pred_net(self, **kw):
        act_func = kw.get("pred_act_func", "relu")
        dropout = kw.get("pred_dropout", 0.0)
        pred_net = kw.get("pred_net", "SumPredictNet")
        hidden_dim = kw.get("pred_hid_dim", 64)
        return_weights = kw.get("pred_return_weights", "none")
        rep_dim = self.get_rep_dim()

        if pred_net == "MeanPredictNet":
            return MeanPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "SumPredictNet":
            return SumPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "MaxPredictNet":
            return MaxPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "MeanAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return MeanAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "SumAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return SumAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "MaxAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return MaxAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "MeanMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return MeanMemAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "SumMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return SumMemAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "MaxMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return MaxMemAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        elif pred_net == "DIAMNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            mem_init = kw.get("pred_mem_init", "mean")
            return DIAMNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                mem_init=mem_init,
                dropout=dropout,
                return_weights="edge" in return_weights
            )
        else:
            raise ValueError

    def get_graph_enc_dims(self):
        enc_dims = OrderedDict(
            {
                "u": get_enc_len(self.max_ngv - 1, self.base) * self.base,
                "v": get_enc_len(self.max_ngv - 1, self.base) * self.base,
                "ul": get_enc_len(self.max_ngvl - 1, self.base) * self.base,
                "el": get_enc_len(self.max_ngel - 1, self.base) * self.base,
                "vl": get_enc_len(self.max_ngvl - 1, self.base) * self.base,
            }
        )
        return enc_dims

    def get_pattern_enc_dims(self):
        if self.share_enc_net:
            return self.get_graph_enc_dims()
        else:
            enc_dims = OrderedDict(
                {
                    "u": get_enc_len(self.max_npv - 1, self.base) * self.base,
                    "v": get_enc_len(self.max_npv - 1, self.base) * self.base,
                    "ul": get_enc_len(self.max_npvl - 1, self.base) * self.base,
                    "el": get_enc_len(self.max_npel - 1, self.base) * self.base,
                    "vl": get_enc_len(self.max_npvl - 1, self.base) * self.base,
                }
            )
            return enc_dims

    def get_filter_gate(self, pattern, graph):
        if self.filter_net is None or len(self.filter_net) == 0:
            return None
        else:
            ul_gate = self.filter_net["ul"](pattern.ul, graph.ul)
            el_gate = self.filter_net["el"](pattern.el, graph.el)
            vl_gate = self.filter_net["vl"](pattern.vl, graph.vl)

            return ((ul_gate & vl_gate) & el_gate)

    def get_pattern_enc(self, pattern):
        pattern_enc = OrderedDict({
            "u": self.p_enc_net["u"](pattern.u),
            "v": self.p_enc_net["v"](pattern.v),
            "ul": self.p_enc_net["ul"](pattern.ul),
            "el": self.p_enc_net["el"](pattern.el),
            "vl": self.p_enc_net["vl"](pattern.vl)
        })

        return pattern_enc

    def get_graph_enc(self, graph):
        graph_enc = OrderedDict({
            "u": self.g_enc_net["u"](graph.u),
            "v": self.g_enc_net["v"](graph.v),
            "ul": self.g_enc_net["ul"](graph.ul),
            "el": self.g_enc_net["el"](graph.el),
            "vl": self.g_enc_net["vl"](graph.vl)
        })

        return graph_enc

    def get_pattern_emb(self, p_enc):
        emb = self.p_emb_net["u"](p_enc["u"]) + \
              self.p_emb_net["v"](p_enc["v"]) + \
              self.p_emb_net["ul"](p_enc["ul"]) + \
              self.p_emb_net["el"](p_enc["el"]) + \
              self.p_emb_net["vl"](p_enc["vl"])

        return emb

    def get_graph_emb(self, g_enc):
        emb = self.g_emb_net["u"](g_enc["u"]) + \
              self.g_emb_net["v"](g_enc["v"]) + \
              self.g_emb_net["ul"](g_enc["ul"]) + \
              self.g_emb_net["el"](g_enc["el"]) + \
              self.g_emb_net["vl"](g_enc["vl"])

        return emb

    def get_pattern_rep(self, p_emb, mask=None):
        # implement by subclasses
        raise NotImplementedError

    def get_graph_rep(self, g_emb, mask=None, gate=None):
        # implement by subclasses
        raise NotImplementedError

    def get_subiso_pred(self, p_e_rep, p_e_mask, g_e_rep, g_e_mask):
        pred_c, pred_w = self.pred_net(p_e_rep, p_e_mask, g_e_rep, g_e_mask)
        return pred_c, (None, pred_w)

    def forward(self, pattern, graph):
        p_e_len = pattern.batch_num_tuples()
        g_e_len = graph.batch_num_tuples()
        bsz = pattern.batch_size

        p_e_mask = batch_convert_len_to_mask(p_e_len, pre_pad=True).view(bsz, -1, 1)
        g_e_mask = batch_convert_len_to_mask(g_e_len, pre_pad=True).view(bsz, -1, 1)
        el_gate = self.get_filter_gate(pattern, graph).view(bsz, -1, 1).float()

        p_enc = self.get_pattern_enc(pattern)
        p_e_emb = self.get_pattern_emb(p_enc)
        p_e_rep = self.get_pattern_rep(p_e_emb, mask=p_e_mask)

        g_enc = self.get_graph_enc(graph)
        g_e_emb = self.get_graph_emb(g_enc)
        g_e_rep = self.get_graph_rep(g_e_emb, mask=g_e_mask, gate=el_gate)

        # handle the reversed edges
        if REVFLAG in pattern.tdata:
            p_e_mask = p_e_mask.masked_fill(pattern.tdata[REVFLAG].view(bsz, -1, 1), 0)
        if REVFLAG in graph.tdata:
            g_e_mask = g_e_mask.masked_fill(graph.tdata[REVFLAG].view(bsz, -1, 1), 0)

        p_e_addfeat = []
        g_e_addfeat = []
        if self.pred_with_enc:
            p_e_addfeat.append(p_enc["u"])
            p_e_addfeat.append(p_enc["v"])
            p_e_addfeat.append(p_enc["ul"])
            p_e_addfeat.append(p_enc["el"])
            p_e_addfeat.append(p_enc["vl"])

            g_e_addfeat.append(g_enc["u"])
            g_e_addfeat.append(g_enc["v"])
            g_e_addfeat.append(g_enc["ul"])
            g_e_addfeat.append(g_enc["el"])
            g_e_addfeat.append(g_enc["vl"])

        if self.pred_with_deg:
            num_p_v = pattern.batch_num_nodes().view(-1, 1)
            max_num_p_v = num_p_v.max()
            # prepadding
            p_v_shift = max_num_p_v - num_p_v
            p_out_deg = pattern.out_degrees().float()
            p_in_deg = pattern.in_degrees().float()
            p_e_addfeat.append(th.gather(p_out_deg, 1, pattern.u + p_v_shift).unsqueeze(-1))
            p_e_addfeat.append(th.gather(p_in_deg, 1, pattern.v + p_v_shift).unsqueeze(-1))

            num_g_v = graph.batch_num_nodes().view(-1, 1)
            max_num_g_v = num_g_v.max()
            # prepadding
            g_v_shift = max_num_g_v - num_g_v
            g_out_deg = graph.out_degrees().float()
            g_in_deg = graph.in_degrees().float()
            g_e_addfeat.append(th.gather(g_out_deg, 1, graph.u + g_v_shift).unsqueeze(-1))
            g_e_addfeat.append(th.gather(g_in_deg, 1, graph.v + g_v_shift).unsqueeze(-1))

        if len(p_e_addfeat) > 0:
            p_e_addfeat = th.cat(p_e_addfeat, dim=-1)
            p_e_addfeat = p_e_addfeat.masked_fill(~(p_e_mask), 0)
            p_e_addfeat = self.refine_edge_weights(p_e_addfeat)
            p_e_output = th.cat([p_e_addfeat, p_e_rep], dim=-1)

            g_e_addfeat = th.cat(g_e_addfeat, dim=-1)
            g_e_addfeat = g_e_addfeat.masked_fill(~(g_e_mask), 0)
            g_e_addfeat = self.refine_edge_weights(g_e_addfeat)
            g_e_output = th.cat([g_e_addfeat, g_e_rep], dim=-1)
        else:
            p_e_output = p_e_rep
            g_e_output = g_e_rep

        del p_e_addfeat, g_e_addfeat

        p_e_mask = self.refine_edge_weights(p_e_mask.float(), use_max=True).bool()
        g_e_mask = self.refine_edge_weights(g_e_mask.float(), use_max=True).bool()

        p_e_mask = p_e_mask.view(bsz, -1)
        g_e_mask = g_e_mask.view(bsz, -1)

        pred_c, (pred_v, pred_e) = self.get_subiso_pred(
            p_e_output, p_e_mask,
            g_e_output, g_e_mask
        )

        output = OutputDict(
            p_v_emb=None,
            p_e_emb=p_e_emb,
            g_v_emb=None,
            g_e_emb=g_e_emb,
            p_v_rep=None,
            p_e_rep=p_e_rep,
            g_v_rep=None,
            g_e_rep=g_e_rep,
            p_v_mask=None,
            p_e_mask=p_e_mask,
            g_v_mask=None,
            g_e_mask=g_e_mask,
            pred_c=pred_c,
            pred_v=pred_v,
            pred_e=pred_e,
        )

        return output


class GraphAdjModel(BaseModel):
    def __init__(self, **kw):
        self.add_node_id = kw.get("add_node_id", kw.get("gnn_add_node_id", False))
        super(GraphAdjModel, self).__init__(**kw)

    def create_enc_net(self, type, **kw):
        enc_net = kw.get("enc_net", "Multihot")

        if type == "graph":
            if enc_net == "Multihot":
                enc_net = OrderedDict({
                    "v": MultihotEmbedding(self.max_ngv, self.base),
                    "vl": MultihotEmbedding(self.max_ngvl, self.base),
                })
            elif enc_net == "Position":
                enc_net = OrderedDict({
                    "v": PositionEmbedding(get_enc_len(self.max_ngv - 1, self.base) * self.base, self.max_ngv),
                    "vl": PositionEmbedding(get_enc_len(self.max_ngvl - 1, self.base) * self.base, self.max_ngvl),
                })
            else:
                raise NotImplementedError
        elif type == "pattern":
            if self.share_enc_net:
                return self.g_enc_net
            else:
                if enc_net == "Multihot":
                    enc_net = OrderedDict({
                        "v": MultihotEmbedding(self.max_npv, self.base),
                        "vl": MultihotEmbedding(self.max_npvl, self.base),
                    })
                elif enc_net == "Position":
                    enc_net = OrderedDict({
                        "v": PositionEmbedding(get_enc_len(self.max_npv - 1, self.base) * self.base, self.max_npv),
                        "vl": PositionEmbedding(get_enc_len(self.max_npvl - 1, self.base) * self.base, self.max_npvl),
                    })
                else:
                    raise NotImplementedError
        else:
            raise ValueError

        for net in enc_net.values():
            net.weight.requires_grad = False

        return ModuleDict(enc_net)

    def create_filter_net(self, **kw):
        filter_net = kw.get("filter_net", "None")

        if filter_net == "None":
            return None
        elif filter_net == "ScalarFilter":
            return ModuleDict({"vl": ScalarFilter()})
        else:
            raise ValueError

    def create_pred_net(self, **kw):
        act_func = kw.get("pred_act_func", "relu")
        dropout = kw.get("pred_dropout", 0.0)
        pred_net = kw.get("pred_net", "SumPredictNet")
        hidden_dim = kw.get("pred_hid_dim", 64)
        return_weights = kw.get("pred_return_weights", "none")
        rep_dim = self.get_rep_dim()

        if pred_net == "MeanPredictNet":
            return MeanPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "SumPredictNet":
            return SumPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "MaxPredictNet":
            return MaxPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "MeanAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return MeanAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "SumAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return SumAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "MaxAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return MaxAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "MeanMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return MeanMemAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "SumMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return SumMemAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "MaxMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return MaxMemAttnPredictNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        elif pred_net == "DIAMNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            mem_init = kw.get("pred_mem_init", "mean")
            return DIAMNet(
                rep_dim,
                hidden_dim=hidden_dim,
                act_func=act_func,
                num_heads=num_heads,
                infer_steps=infer_steps,
                mem_len=mem_len,
                mem_init=mem_init,
                dropout=dropout,
                return_weights="node" in return_weights
            )
        else:
            raise ValueError

    def get_graph_enc_dims(self):
        enc_dims = OrderedDict({
            "v": get_enc_len(self.max_ngv - 1, self.base) * self.base,
            "vl": get_enc_len(self.max_ngvl - 1, self.base) * self.base
        })
        return enc_dims

    def get_pattern_enc_dims(self):
        if self.share_enc_net:
            return self.get_graph_enc_dims()
        else:
            enc_dims = OrderedDict({
                "v": get_enc_len(self.max_npv - 1, self.base) * self.base,
                "vl": get_enc_len(self.max_npvl - 1, self.base) * self.base
            })
            return enc_dims

    def get_filter_gate(self, pattern, graph):
        if self.filter_net is None or len(self.filter_net) == 0:
            return None
        else:
            bsz = pattern.batch_size
            p_e_len = pattern.batch_num_nodes()
            g_e_len = graph.batch_num_nodes()

            p_vl = split_and_batchify_graph_feats(pattern.ndata["label"].view(-1, 1), p_e_len, pre_pad=True)[0]
            g_vl = split_and_batchify_graph_feats(graph.ndata["label"].view(-1, 1), g_e_len, pre_pad=True)[0]
            vl_gate = self.filter_net["vl"](p_vl, g_vl)

            max_g_v_len = g_e_len.max()
            if bsz * max_g_v_len != graph.number_of_nodes():
                # vl_gate = vl_gate.view(-1, 1).masked_select(batch_convert_len_to_mask(g_e_len, pre_pad=True))
                vl_gate = th.cat([vl_gate[i, -g_e_len[i]:] for i in range(bsz)])
            vl_gate = vl_gate.view(-1, 1)
        return vl_gate

    def get_pattern_enc(self, pattern):
        pattern_enc = OrderedDict({
            "v": self.p_enc_net["v"](pattern.ndata["id"].view(-1)),
            "vl": self.p_enc_net["vl"](pattern.ndata["label"].view(-1))
        })
        return pattern_enc

    def get_graph_enc(self, graph):
        graph_enc = OrderedDict({
            "v": self.g_enc_net["v"](graph.ndata["id"].view(-1)),
            "vl": self.g_enc_net["vl"](graph.ndata["label"].view(-1))
        })
        return graph_enc

    def get_pattern_emb(self, p_enc):
        emb = self.p_emb_net["vl"](p_enc["vl"])
        if self.add_node_id:
            emb = emb + self.p_emb_net["v"](p_enc["v"])
        return emb

    def get_graph_emb(self, g_enc):
        emb = self.g_emb_net["vl"](g_enc["vl"])
        if self.add_node_id:
            emb = emb + self.g_emb_net["v"](g_enc["v"])
        return emb

    def get_pattern_rep(self, pattern, p_emb, mask=None):
        # implement by subclasses
        raise NotImplementedError

    def get_graph_rep(self, graph, g_emb, mask=None, gate=None):
        # implement by subclasses
        raise NotImplementedError

    def get_subiso_pred(self, p_v_rep, p_v_mask, g_v_rep, g_v_mask):
        v_pred_c, v_pred_w = self.pred_net(p_v_rep, p_v_mask, g_v_rep, g_v_mask)
        return v_pred_c, (v_pred_w, None)

    def forward(self, pattern, graph):
        bsz = pattern.batch_size
        p_v_len = pattern.batch_num_nodes()
        g_v_len = graph.batch_num_nodes()

        p_v_mask = batch_convert_len_to_mask(p_v_len, pre_pad=True).view(bsz, -1, 1)
        g_v_mask = batch_convert_len_to_mask(g_v_len, pre_pad=True).view(bsz, -1, 1)
        vl_gate = self.get_filter_gate(pattern, graph)

        p_enc = self.get_pattern_enc(pattern)
        p_v_emb = self.get_pattern_emb(p_enc)
        p_v_rep = self.get_pattern_rep(pattern, p_v_emb)

        g_enc = self.get_graph_enc(graph)
        g_v_emb = self.get_graph_emb(g_enc)
        g_v_rep = self.get_graph_rep(graph, g_v_emb, gate=vl_gate)

        p_v_addfeat = []
        g_v_addfeat = []
        if self.pred_with_enc:
            p_v_addfeat.append(p_enc["v"])
            p_v_addfeat.append(p_enc["vl"])

            g_v_addfeat.append(g_enc["v"])
            g_v_addfeat.append(g_enc["vl"])

        if self.pred_with_deg:
            p_out_deg = pattern.out_degrees().float().view(-1, 1)
            p_in_deg = pattern.in_degrees().float().view(-1, 1)
            p_v_addfeat.append(p_out_deg)
            p_v_addfeat.append(p_in_deg)

            g_out_deg = graph.out_degrees().float().view(-1, 1)
            g_in_deg = graph.in_degrees().float().view(-1, 1)
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

        p_v_mask = p_v_mask.view(bsz, -1)
        g_v_mask = g_v_mask.view(bsz, -1)

        pred_c, (pred_v, pred_e) = self.get_subiso_pred(
            p_v_output, p_v_mask,
            g_v_output, g_v_mask
        )

        output = OutputDict(
            p_v_emb=p_v_emb,
            p_e_emb=None,
            g_v_emb=g_v_emb,
            g_e_emb=None,
            p_v_rep=p_v_rep,
            p_e_rep=None,
            g_v_rep=g_v_rep,
            g_e_rep=None,
            p_v_mask=p_v_mask,
            p_e_mask=None,
            g_v_mask=g_v_mask,
            g_e_mask=None,
            pred_c=pred_c,
            pred_v=pred_v,
            pred_e=pred_e,
        )

        return output


class GraphAdjModelV2(BaseModel):
    def __init__(self, **kw):
        self.add_node_id = kw.get("add_node_id", kw.get("gnn_add_node_id", False))
        self.add_edge_id = kw.get("add_edge_id", kw.get("gnn_add_edge_id", False))
        self.node_pred = kw.get("node_pred", True)
        self.edge_pred = kw.get("edge_pred", True)
        super(GraphAdjModelV2, self).__init__(**kw)

    def create_enc_net(self, type, **kw):
        enc_net = kw.get("enc_net", "Multihot")

        if type == "graph":
            if enc_net == "Multihot":
                enc_net = OrderedDict({
                    "v": MultihotEmbedding(self.max_ngv, self.base),
                    "vl": MultihotEmbedding(self.max_ngvl, self.base),
                    "el": MultihotEmbedding(self.max_ngel, self.base)
                })
            elif enc_net == "Position":
                enc_net = OrderedDict({
                    "v": PositionEmbedding(get_enc_len(self.max_ngv - 1, self.base) * self.base, self.max_ngv),
                    "vl": PositionEmbedding(get_enc_len(self.max_ngvl - 1, self.base) * self.base, self.max_ngvl),
                    "el": PositionEmbedding(get_enc_len(self.max_ngel - 1, self.base) * self.base, self.max_ngel)
                })
            else:
                raise NotImplementedError
        elif type == "pattern":
            if self.share_enc_net:
                return self.g_enc_net
            else:
                if enc_net == "Multihot":
                    enc_net = OrderedDict({
                        "v": MultihotEmbedding(self.max_npv, self.base),
                        "vl": MultihotEmbedding(self.max_npvl, self.base),
                        "el": MultihotEmbedding(self.max_npel, self.base)
                    })
                elif enc_net == "Position":
                    enc_net = OrderedDict({
                        "v": PositionEmbedding(get_enc_len(self.max_npv - 1, self.base) * self.base, self.max_npv),
                        "vl": PositionEmbedding(get_enc_len(self.max_npvl - 1, self.base) * self.base, self.max_npvl),
                        "el": PositionEmbedding(get_enc_len(self.max_npel - 1, self.base) * self.base, self.max_npel)
                    })
                else:
                    raise NotImplementedError
        else:
            raise ValueError

        for net in enc_net.values():
            if net is not None:
                net.weight.requires_grad = False

        return ModuleDict(enc_net)

    def create_filter_net(self, **kw):
        filter_net = kw.get("filter_net", "None")

        if filter_net == "None":
            return None
        elif filter_net == "ScalarFilter":
            return ModuleDict({"vl": ScalarFilter(), "el": ScalarFilter()})
        else:
            raise ValueError

    def create_emb_net(self, type, **kw):
        emb_net = kw.get("emb_net", "Orthogonal")
        hid_dim = self.hid_dim

        if type == "graph":
            enc_dims = self.get_graph_enc_dims()
        elif type == "pattern":
            enc_dims = self.get_pattern_enc_dims()
        else:
            raise ValueError

        if emb_net == "Orthogonal":
            emb_net = OrderedDict({
                "v": OrthogonalEmbedding(enc_dims["v"], hid_dim),
                "vl": OrthogonalEmbedding(enc_dims["vl"], hid_dim),
                "el": OrthogonalEmbedding(enc_dims["el"], hid_dim)
            })
        elif emb_net == "Normal":
            emb_net = OrderedDict({
                "v": NormalEmbedding(enc_dims["v"], hid_dim),
                "vl": NormalEmbedding(enc_dims["vl"], hid_dim),
                "el": NormalEmbedding(enc_dims["el"], hid_dim)
            })
        elif emb_net == "Uniform":
            emb_net = OrderedDict({
                "v": UniformEmbedding(enc_dims["v"], hid_dim),
                "vl": UniformEmbedding(enc_dims["vl"], hid_dim),
                "el": UniformEmbedding(enc_dims["el"], hid_dim)
            })
        elif emb_net == "Equivariant":
            emb_net = OrderedDict({
                "v": EquivariantEmbedding(enc_dims["v"], hid_dim),
                "vl": EquivariantEmbedding(enc_dims["vl"], hid_dim),
                "el": EquivariantEmbedding(enc_dims["el"], hid_dim)
            })
        else:
            raise ValueError

        # rescale because of multi-hot or multi-sinusoid
        with th.no_grad():
            for net in emb_net:
                scale = enc_dims[net] // self.base
                emb_net[net].weight.div_(scale)

        return ModuleDict(emb_net)

    def create_pred_net(self, **kw):
        act_func = kw.get("pred_act_func", "relu")
        dropout = kw.get("pred_dropout", 0.0)
        pred_net = kw.get("pred_net", "SumPredictNet")
        hidden_dim = kw.get("pred_hid_dim", 64)
        return_weights = kw.get("pred_return_weights", "none")
        rep_v_dim, rep_e_dim = self.get_rep_dim()

        if pred_net == "MeanPredictNet":
            return ModuleDict(
                {
                    "v":
                        MeanPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        MeanPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "SumPredictNet":
            return ModuleDict(
                {
                    "v":
                        SumPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        SumPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "MaxPredictNet":
            return ModuleDict(
                {
                    "v":
                        MaxPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        MaxPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "MeanAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return ModuleDict(
                {
                    "v":
                        MeanAttnPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        MeanAttnPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "SumAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return ModuleDict(
                {
                    "v":
                        SumAttnPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        SumAttnPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "MaxAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            return ModuleDict(
                {
                    "v":
                        MaxAttnPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        MaxAttnPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "MeanMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return ModuleDict(
                {
                    "v":
                        MeanMemAttnPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        MeanMemAttnPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "SumMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return ModuleDict(
                {
                    "v":
                        SumMemAttnPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        SumMemAttnPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "MaxMemAttnPredictNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            return ModuleDict(
                {
                    "v":
                        MaxMemAttnPredictNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        MaxMemAttnPredictNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        elif pred_net == "DIAMNet":
            infer_steps = kw.get("pred_infer_steps", 1)
            num_heads = kw.get("pred_num_heads", 4)
            mem_len = kw.get("pred_mem_len", 4)
            mem_init = kw.get("pred_mem_init", "mean")
            return ModuleDict(
                {
                    "v":
                        DIAMNet(
                            rep_v_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            mem_init=mem_init,
                            dropout=dropout,
                            return_weights="node" in return_weights
                        ) if self.node_pred else None,
                    "e":
                        DIAMNet(
                            rep_e_dim,
                            hidden_dim=hidden_dim,
                            act_func=act_func,
                            num_heads=num_heads,
                            infer_steps=infer_steps,
                            mem_len=mem_len,
                            mem_init=mem_init,
                            dropout=dropout,
                            return_weights="edge" in return_weights
                        ) if self.edge_pred else None,
                }
            )
        else:
            raise ValueError

    def get_graph_enc_dims(self):
        enc_dims = OrderedDict({
            "v": get_enc_len(self.max_ngv - 1, self.base) * self.base,
            "vl": get_enc_len(self.max_ngvl - 1, self.base) * self.base,
            "el": get_enc_len(self.max_ngel - 1, self.base) * self.base,
        })
        return enc_dims

    def get_pattern_enc_dims(self):
        if self.share_enc_net:
            return self.get_graph_enc_dims()
        else:
            enc_dims = OrderedDict({
                "v": get_enc_len(self.max_npv - 1, self.base) * self.base,
                "vl": get_enc_len(self.max_npvl - 1, self.base) * self.base,
                "el": get_enc_len(self.max_npel - 1, self.base) * self.base,
            })
            return enc_dims

    def get_graph_enc_dim(self):
        enc_dims = self.get_graph_enc_dims()
        v_enc_dim = enc_dims["v"] + enc_dims["vl"]
        e_enc_dim = (enc_dims["v"] + enc_dims["vl"]) * 2 + enc_dims["el"]
        return v_enc_dim, e_enc_dim

    def get_pattern_enc_dim(self):
        enc_dims = self.get_pattern_enc_dims()
        v_enc_dim = enc_dims["v"] + enc_dims["vl"]
        e_enc_dim = (enc_dims["v"] + enc_dims["vl"]) * 2 + enc_dims["el"]
        return v_enc_dim, e_enc_dim

    def get_rep_dim(self):
        rep_v_dim, rep_e_dim = self.hid_dim, self.hid_dim
        if self.pred_with_enc:
            enc_v_dim, enc_e_dim = self.get_graph_enc_dim()
            rep_v_dim += enc_v_dim
            rep_e_dim += enc_e_dim
        if self.pred_with_deg:
            rep_v_dim += 2
            rep_e_dim += 2
        return rep_v_dim, rep_e_dim

    def get_filter_gate(self, pattern, graph):
        if self.filter_net is None or len(self.filter_net) == 0:
            return None, None
        else:
            bsz = pattern.batch_size
            p_v_len = pattern.batch_num_nodes()
            p_e_len = pattern.batch_num_edges()
            g_v_len = graph.batch_num_nodes()
            g_e_len = graph.batch_num_edges()

            p_vl = split_and_batchify_graph_feats(pattern.ndata["label"].view(-1, 1), p_v_len, pre_pad=True)[0]
            g_vl = split_and_batchify_graph_feats(graph.ndata["label"].view(-1, 1), g_v_len, pre_pad=True)[0]
            vl_gate = self.filter_net["vl"](p_vl, g_vl)

            max_g_v_len = g_v_len.max()
            if bsz * max_g_v_len != graph.number_of_nodes():
                # vl_gate = vl_gate.view(-1, 1).masked_select(batch_convert_len_to_mask(g_v_len, pre_pad=True))
                vl_gate = th.cat([vl_gate[i, -g_v_len[i]:] for i in range(bsz)])
            vl_gate = vl_gate.view(-1, 1)

            p_el = split_and_batchify_graph_feats(pattern.edata["label"].view(-1, 1), p_e_len, pre_pad=True)[0]
            g_el = split_and_batchify_graph_feats(graph.edata["label"].view(-1, 1), g_e_len, pre_pad=True)[0]
            el_gate = self.filter_net["el"](p_el, g_el)

            max_g_e_len = g_e_len.max()
            if bsz * max_g_e_len != graph.number_of_edges():
                # el_gate = el_gate.view(-1, 1).masked_select(batch_convert_len_to_mask(g_e_len, pre_pad=True))
                el_gate = th.cat([el_gate[i, -g_e_len[i]:] for i in range(bsz)])
            el_gate = el_gate.view(-1, 1)
        return vl_gate, el_gate

    def get_pattern_enc(self, pattern):
        pattern_enc = OrderedDict({
            "v": self.p_enc_net["v"](pattern.ndata["id"].view(-1)),
            "vl": self.p_enc_net["vl"](pattern.ndata["label"].view(-1)),
            "el": self.p_enc_net["el"](pattern.edata["label"].view(-1))
        })
        if self.add_edge_id:
            u, v = pattern.all_edges(form="uv", order="eid")
            pattern_enc["src"] = pattern_enc["v"][u]
            pattern_enc["dst"] = pattern_enc["v"][v]
        return pattern_enc

    def get_graph_enc(self, graph):
        graph_enc = OrderedDict({
            "v": self.g_enc_net["v"](graph.ndata["id"].view(-1)),
            "vl": self.g_enc_net["vl"](graph.ndata["label"].view(-1)),
            "el": self.g_enc_net["el"](graph.edata["label"].view(-1))
        })
        if self.add_edge_id:
            u, v = graph.all_edges(form="uv", order="eid")
            graph_enc["src"] = graph_enc["v"][u]
            graph_enc["dst"] = graph_enc["v"][v]
        return graph_enc

    def get_pattern_emb(self, p_enc):
        v_emb = self.p_emb_net["vl"](p_enc["vl"])
        if self.add_node_id:
            v_emb = v_emb + self.p_emb_net["v"](p_enc["v"])
        e_emb = self.p_emb_net["el"](p_enc["el"])
        if self.add_edge_id:
            e_emb = e_emb + self.p_emb_net["v"](p_enc["src"]) + self.p_emb_net["v"](p_enc["dst"])

        return v_emb, e_emb

    def get_graph_emb(self, g_enc):
        v_emb = self.g_emb_net["vl"](g_enc["vl"])
        if self.add_node_id:
            v_emb = v_emb + self.g_emb_net["v"](g_enc["v"])
        e_emb = self.g_emb_net["el"](g_enc["el"])
        if self.add_edge_id:
            e_emb = e_emb + self.g_emb_net["v"](g_enc["src"]) + self.g_emb_net["v"](g_enc["dst"])

        return v_emb, e_emb

    def get_pattern_rep(self, pattern, p_v_emb, p_e_emb, v_mask=None, e_mask=None):
        # implement by subclasses
        raise NotImplementedError

    def get_graph_rep(self, graph, g_v_emb, g_e_emb, v_mask=None, e_mask=None, v_gate=None, e_gate=None):
        # implement by subclasses
        raise NotImplementedError

    def get_subiso_pred(self, p_v_rep, p_v_mask, p_e_rep, p_e_mask, g_v_rep, g_v_mask, g_e_rep, g_e_mask):
        if self.node_pred:
            v_pred_c, v_pred_w = self.pred_net["v"](p_v_rep, p_v_mask, g_v_rep, g_v_mask)
        else:
            v_pred_c, v_pred_w = None, None
        if self.edge_pred:
            e_pred_c, e_pred_w = self.pred_net["e"](p_e_rep, p_e_mask, g_e_rep, g_e_mask)
        else:
            e_pred_c, e_pred_w = None, None
        if self.node_pred and self.edge_pred:
            g_v_len = g_v_mask.float().sum(dim=1).view(-1, 1)
            g_e_len = g_e_mask.float().sum(dim=1).view(-1, 1)
            g_len = g_v_len + g_e_len
            w1 = g_e_len / g_len
            w2 = g_v_len / g_len
            return w2 * v_pred_c + w1 * e_pred_c, (v_pred_w, e_pred_w)
        elif self.node_pred:
            return v_pred_c, (v_pred_w, e_pred_w)
        elif self.edge_pred:
            return e_pred_c, (v_pred_w, e_pred_w)
        else:
            raise ValueError

    def forward(self, pattern, graph):
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
        p_v_rep, p_e_rep = self.get_pattern_rep(pattern, p_v_emb, p_e_emb)

        g_enc = self.get_graph_enc(graph)
        g_v_emb, g_e_emb = self.get_graph_emb(g_enc)
        g_v_rep, g_e_rep = self.get_graph_rep(graph, g_v_emb, g_e_emb, v_gate=vl_gate, e_gate=el_gate)

        # handle the reversed edges
        if REVFLAG in pattern.edata:
            p_e_mask = p_e_mask.masked_fill(
                split_and_batchify_graph_feats(pattern.edata[REVFLAG], p_e_len, pre_pad=True)[0].view(bsz, -1, 1),
                0
            )
        if REVFLAG in graph.edata:
            g_e_mask = g_e_mask.masked_fill(
                split_and_batchify_graph_feats(graph.edata[REVFLAG], g_e_len, pre_pad=True)[0].view(bsz, -1, 1),
                0
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
	    g_v_output = None
	    g_e_output = None

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
