import csv
import igraph as ig
import json
import math
import numpy as np
import torch as th
import torch.nn as nn
import os
from argparse import Namespace
from collections import OrderedDict, namedtuple
from multiprocessing import Pool
from tqdm import tqdm

from .act import *


def calculate_gain(activation):
    if isinstance(activation, str):
        if activation in ["none", "maximum", "minimum"]:
            nonlinearity = "linear"
        elif activation in ["relu", "relu6", "elu", "selu", "celu", "gelu"]:
            nonlinearity = "relu"
        elif activation in ["leaky_relu", "prelu"]:
            nonlinearity = "leaky_relu"
        elif activation in ["softmax", "sparsemax", "gumbel_softmax"]:
            nonlinearity = "sigmoid"
        elif activation in ["sigmoid", "tanh"]:
            nonlinearity = activation
        else:
            raise NotImplementedError
    elif isinstance(activation, nn.Module):
        if isinstance(activation, (Identity, Maximum, Minimum)):
            nonlinearity = "linear"
        elif isinstance(activation, (ReLU, ReLU6, ELU, SELU, CELU, GELU)):
            nonlinearity = "relu"
        elif isinstance(activation, (LeakyReLU, PReLU)):
            nonlinearity = "leaky_relu"
        elif isinstance(activation, (Softmax, Sparsemax, GumbelSoftmax)):
            nonlinearity = "sigmoid"
        elif isinstance(activation, Sigmoid):
            nonlinearity = "sigmoid"
        elif isinstance(activation, Tanh):
            nonlinearity = "tanh"
        else:
            raise NotImplementedError
    else:
        raise ValueError

    return nn.init.calculate_gain(nonlinearity, LEAKY_RELU_A)


def calculate_fan_in_and_fan_out(x):
    if x.dim() < 2:
        x = x.unsqueeze(-1)
    num_input_fmaps = x.size(1)
    num_output_fmaps = x.size(0)
    receptive_field_size = 1
    if x.dim() > 2:
        receptive_field_size = x[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def zero_init(x, gain=1.0):
    return nn.init.zeros_(x)


def xavier_uniform_init(x, gain=1.0):
    fan_in, fan_out = calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = 1.7320508075688772 * std

    return nn.init.uniform_(x, -a, a)


def kaiming_normal_init(x, gain=1.0):
    fan_in, fan_out = calculate_fan_in_and_fan_out(x)
    std = gain / math.sqrt(fan_in)
    return nn.init.normal_(x, 0, std)


def orthogonal_init(x, gain=1.0):
    return nn.init.orthogonal_(x, gain=1.0)


def equivariant_init(x, gain=1.0):
    with th.no_grad():
        x_size = tuple(x.size())
        if len(x_size) == 1:
            kaiming_normal_init(x, gain=gain)
        elif len(x_size) == 2:
            kaiming_normal_init(x[0], gain=gain)
            vec = x[0]
            for i in range(1, x.size(0)):
                x[i].data.copy_(th.roll(vec, i, 0))
        else:
            x = x.view(x_size[:-2] + (-1, ))
            equivariant_init(x, gain=gain)
            x = x.view(x_size)
    return x


def identity_init(x, gain=1.0):
    with th.no_grad():
        x_size = tuple(x.size())
        if len(x_size) == 1:
            fan_in, fan_out = calculate_fan_in_and_fan_out(x)
            std = gain * (2.0 / float(fan_in + fan_out))
            nn.init.ones_(x)
            x += th.randn_like(x) * std**2
        elif len(x_size) == 2:
            fan_in, fan_out = calculate_fan_in_and_fan_out(x)
            std = gain * (2.0 / float(fan_in + fan_out))
            nn.init.eye_(x)
            x += th.randn_like(x) * std**2
        else:
            x = x.view(x_size[:-2] + (-1, ))
            identity_init(x, gain=gain)
            x = x.view(x_size)
    return x


def init_weight(x, activation="none", init="uniform"):
    gain = calculate_gain(activation)
    if init == "zero":
        init_func = zero_init
    elif init == "identity":
        init_func = identity_init
    elif init == "uniform":
        init_func = xavier_uniform_init
    elif init == "normal":
        init_func = kaiming_normal_init
    elif init == "orthogonal":
        init_func = orthogonal_init
    elif init == "equivariant":
        init_func = equivariant_init
    else:
        raise ValueError("init=%s is not supported now." % (init))

    if isinstance(x, th.Tensor):
        init_func(x, gain=gain)


def init_module(x, activation="none", init="uniform"):
    gain = calculate_gain(activation)
    if init == "zero":
        init_func = zero_init
    elif init == "identity":
        init_func = identity_init
    elif init == "uniform":
        init_func = xavier_uniform_init
    elif init == "normal":
        init_func = kaiming_normal_init
    elif init == "orthogonal":
        init_func = orthogonal_init
    elif init == "equivariant":
        init_func = equivariant_init
    else:
        raise ValueError("init=%s is not supported now." % (init))

    if isinstance(x, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init_func(x.weight, gain=gain)
        if hasattr(x, "bias") and x.bias is not None:
            nn.init.zeros_(x.bias)
    elif isinstance(x, nn.Embedding):
        with th.no_grad():
            if init == "uniform":
                nn.init.uniform_(x.weight, -1.0, 1.0)
            elif init == "normal":
                nn.init.normal_(x.weight, 0.0, 1.0)
            elif init == "orthogonal":
                nn.init.orthogonal_(x.weight, gain=math.sqrt(calculate_fan_in_and_fan_out(x.weight)[0]) * 1.0)
            elif init == "identity":
                nn.init.eye_(x.weight)
            elif init == "equivariant":
                nn.init.normal_(x.weight[0], 0.0, 1.0)
                vec = x.weight[0]
                for i in range(1, x.weight.size(0)):
                    x.weight[i].data.copy_(th.roll(vec, i, 0))
            if x.padding_idx is not None:
                x.weight[x.padding_idx].fill_(0)
    elif isinstance(x, nn.RNNBase):
        for layer_weights in x._all_weights:
            for w in layer_weights:
                if "weight" in w:
                    init_func(getattr(x, w))
                elif "bias" in w:
                    nn.init.zeros_(getattr(x, w))
    elif isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(x.weight)
        nn.init.zeros_(x.bias)


def change_dropout_rate(model, dropout):
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            child.p = dropout
        change_dropout_rate(child, dropout)
