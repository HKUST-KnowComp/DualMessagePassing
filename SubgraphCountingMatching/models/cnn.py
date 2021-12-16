import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import EdgeSeqModel
from .container import *
# from ..utils.act import map_activation_str_to_layer
# from ..utils.init import init_weight, init_module
from utils.act import map_activation_str_to_layer
from utils.init import init_weight, init_module


class CNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=-1,
        stride=1,
        groups=1,
        dilation=1,
        batch_norm=True,
        act_func="relu",
        dropout=0.0
    ):
        super(CNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if padding == -1:
            padding = kernel_size//2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding,
            stride=stride, groups=groups, dilation=dilation
        )
        self.act = map_activation_str_to_layer(act_func, inplace=True)
        self.pool = nn.MaxPool1d(
            kernel_size=kernel_size//stride, stride=1, padding=padding
        )
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None
        self.drop = nn.Dropout(dropout)

        # init
        init_module(self.conv, init="normal", activation=act_func)

    def forward(self, x):
        o = self.conv(x)
        o = self.act(o)
        o = self.pool(o)
        if self.bn is not None:
            o = self.bn(o)
        o = self.drop(o)

        return o

    def get_output_dim(self):
        return self.out_channels

    def extra_repr(self):
        ""


class CNN(EdgeSeqModel):
    def __init__(self, **kw):
        super(CNN, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)

        batch_norm = kw.get("rep_cnn_batch_norm", True)
        kernel_sizes = kw.get("rep_cnn_kernel_sizes", 2)
        paddings = kw.get("rep_cnn_paddings", -1)
        strides = kw.get("rep_cnn_strides", 1)

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        if isinstance(paddings, int):
            paddings = [paddings] * num_layers
        if isinstance(strides, int):
            strides = [strides] * num_layers

        cnn = ModuleList()
        for i in range(num_layers):
            cnn.add_module(
                "%s_cnn_(%d)" % (type, i),
                CNNLayer(
                    self.hid_dim,
                    self.hid_dim,
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i],
                    stride=strides[i],
                    batch_norm=batch_norm,
                    act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"cnn": cnn})

    def get_pattern_rep(self, p_emb, mask=None):
        if mask is None:
            outputs = [p_emb.transpose(1, 2)]
            for layer in self.p_rep_net["cnn"]:
                o = layer(outputs[-1])
                if self.rep_residual and o.size() == outputs[-1].size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)
            for i in range(len(outputs)):
                outputs[i] = outputs[i].transpose(1, 2)
        else:
            gate = mask.float().transpose(1, 2)
            outputs = [p_emb.transpose(1, 2) * gate]
            for layer in self.p_rep_net["cnn"]:
                gate = F.max_pool1d(
                    gate,
                    kernel_size=layer.conv.kernel_size,
                    stride=layer.conv.stride,
                    padding=layer.conv.padding,
                    dilation=layer.conv.dilation
                )
                gate = F.max_pool1d(
                    gate,
                    kernel_size=layer.pool.kernel_size,
                    stride=layer.pool.stride,
                    padding=layer.pool.padding,
                    dilation=layer.pool.dilation
                )
                o = layer(outputs[-1])
                o = o * gate
                if self.rep_residual and o.size() == outputs[-1].size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)
            for i in range(len(outputs)):
                outputs[i] = outputs[i].transpose(1, 2)

        return outputs[-1]

    def get_graph_rep(self, g_emb, mask=None, gate=None):
        if mask is None and gate is None:
            outputs = [g_emb.transpose(1, 2)]
            for layer in self.g_rep_net["cnn"]:
                o = layer(outputs[-1])
                if self.rep_residual and o.size() == outputs[-1].size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)
            for i in range(len(outputs)):
                outputs[i] = outputs[i].transpose(1, 2)
        else:
            gate = ((mask.float() if mask is not None else 1) * (gate if gate is not None else 1)).transpose(1, 2)
            outputs = [g_emb.transpose(1, 2) * gate]
            for layer in self.g_rep_net["cnn"]:
                gate = F.max_pool1d(
                    gate,
                    kernel_size=layer.conv.kernel_size,
                    stride=layer.conv.stride,
                    padding=layer.conv.padding,
                    dilation=layer.conv.dilation
                )
                gate = F.max_pool1d(
                    gate,
                    kernel_size=layer.pool.kernel_size,
                    stride=layer.pool.stride,
                    padding=layer.pool.padding,
                    dilation=layer.pool.dilation
                )
                o = layer(outputs[-1])
                o = o * gate
                if self.rep_residual and o.size() == outputs[-1].size():
                    outputs.append(outputs[-1] + o)
                else:
                    outputs.append(o)
            for i in range(len(outputs)):
                outputs[i] = outputs[i].transpose(1, 2)

        return outputs[-1]

    def refine_edge_weights(self, weights, use_max=False):
        if weights is None:
            return None
        dim = weights.dim()
        dtype = weights.dtype
        if dim == 2:
            weights = weights.unsqueeze(-1)
        weights = weights.transpose(1, 2).float()
        if use_max:
            for layer in self.g_rep_net["cnn"]:
                if isinstance(layer, CNNLayer):
                    weights = F.max_pool1d(
                        weights,
                        kernel_size=layer.conv.kernel_size,
                        stride=layer.conv.stride,
                        padding=layer.conv.padding,
                        dilation=layer.conv.dilation,
                    )
                    weights = F.max_pool1d(
                        weights,
                        kernel_size=layer.pool.kernel_size,
                        stride=layer.pool.stride,
                        padding=layer.pool.padding,
                        dilation=layer.pool.dilation,
                    )
        else:
            for layer in self.g_rep_net["cnn"]:
                if isinstance(layer, CNNLayer):
                    weights = sum(layer.conv.kernel_size) * F.avg_pool1d(
                        weights,
                        kernel_size=layer.conv.kernel_size,
                        stride=layer.conv.stride,
                        padding=layer.conv.padding,
                        # dilation=layer.conv.dilation,
                    )
                    weights = F.max_pool1d(
                        weights,
                        kernel_size=layer.pool.kernel_size,
                        stride=layer.pool.stride,
                        padding=layer.pool.padding,
                        dilation=layer.pool.dilation,
                    )
        weights = weights.transpose(1, 2)
        if dim == 2:
            weights = weights.squeeze(-1)
        return weights.to(dtype)
