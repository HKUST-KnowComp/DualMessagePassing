import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import EdgeSeqModel
from .container import *
# from ..utils.act import map_activation_str_to_layer
# from ..utils.init import init_weight, init_module
from utils.act import map_activation_str_to_layer
from utils.init import init_weight, init_module


class RNNLayer(nn.Module):
    def __init__(self, rep_rnn_type, input_dim, hid_dim, layer_norm=False, bidirectional=False, dropout=0.0):
        super(RNNLayer, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        if rep_rnn_type == "LSTM":
            self.layer = nn.LSTM(
                input_dim, hid_dim//2 if bidirectional else hid_dim,
                bidirectional=bidirectional, batch_first=True
            )
        elif rep_rnn_type == "GRU":
            self.layer = nn.GRU(
                input_dim, hid_dim//2 if bidirectional else hid_dim,
                bidirectional=bidirectional, batch_first=True
            )
        elif rep_rnn_type == "RNN":
            self.layer = nn.RNN(
                input_dim, hid_dim//2 if bidirectional else hid_dim,
                bidirectional=bidirectional, batch_first=True
            )
        if layer_norm:
            self.ln = nn.LayerNorm(hid_dim)
        else:
            self.ln = None
        self.drop = nn.Dropout(dropout)

        # init
        init_module(self.layer)

    def forward(self, x):
        o = self.layer(x)[0]
        if self.ln is not None:
            o = self.ln(o)
        o = self.drop(o)

        return o

    def get_output_dim(self):
        return self.hid_dim - self.hid_dim % 2

    def extra_repr(self):
        ""


class RNN(EdgeSeqModel):
    def __init__(self, **kw):
        super(RNN, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
        rep_rnn_type = kw.get("rep_rnn_type", "LSTM")
        bidirectional = kw.get("rep_rnn_bidirectional", False)
        dropout = kw.get("rep_dropout", 0.0)

        rnn = ModuleList()
        for i in range(num_layers):
            rnn.add_module(
                "%s_rnn_(%d)" % (type, i),
                RNNLayer(
                    rep_rnn_type,
                    self.hid_dim, self.hid_dim,
                    bidirectional=bidirectional,
                    dropout=dropout
                )
            )

        return ModuleDict({"rnn": rnn})

    def get_pattern_rep(self, p_emb, mask=None):
        if mask is not None:
            p_zero_mask = ~(mask)
            outputs = [p_emb.masked_fill(p_zero_mask, 0.0)]
            for layer in self.p_rep_net["rnn"]:
                o = layer(outputs[-1])
                outputs.append(o.masked_fill(p_zero_mask, 0.0))
        else:
            outputs = [p_emb]
            for layer in self.p_rep_net["rnn"]:
                o = layer(outputs[-1])
                if self.rep_residual and outputs[-1].size() == o.size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)

        return outputs[-1]

    def get_graph_rep(self, g_emb, mask=None, gate=None):
        if mask is None and gate is None:
            outputs = [g_emb]
            for layer in self.g_rep_net["rnn"]:
                o = layer(outputs[-1])
                if self.rep_residual and outputs[-1].size() == o.size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)
        else:
            gate = ((mask.float() if mask is not None else 1) * (gate if gate is not None else 1))
            outputs = [g_emb * gate]
            for layer in self.g_rep_net["rnn"]:
                o = layer(outputs[-1])
                o = o * gate
                if self.rep_residual and outputs[-1].size() == o.size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)

        return outputs[-1]
