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


def segment_data(data, max_len, pre_pad=False):
    pad_len = max_len - data.size(1) % max_len
    if pad_len != max_len:
        pad_size = list(data.size())
        pad_size[1] = pad_len
        zero_pad = th.zeros(pad_size, device=data.device, dtype=data.dtype, requires_grad=False)
        if pre_pad:
            data = th.cat([zero_pad, data], dim=1)
        else:
            data = th.cat([data, zero_pad], dim=1)
    return th.split(data, max_len, dim=1)


def segment_length(data_len, max_len):
    bsz = data_len.size(0)
    list_len = math.ceil(data_len.max().float() / max_len)
    segment_lens = th.arange(
        0, max_len * list_len, max_len, dtype=data_len.dtype, device=data_len.device, requires_grad=False
    ).view(1, list_len)
    diff = data_len.view(-1, 1) - segment_lens
    fill_max = diff > max_len
    fill_zero = diff < 0
    segment_lens = diff.masked_fill(fill_max, max_len)
    segment_lens.masked_fill_(fill_zero, 0)
    return th.split(segment_lens.view(bsz, -1), 1, dim=1)


def split_ids(x_ids):
    if x_ids[0] == x_ids[-1]:
        return th.LongTensor([x_ids.size(0)]).to(x_ids.device)
    diff = th.roll(x_ids, -1, 0) - x_ids
    return th.masked_select(th.arange(1, x_ids.size(0) + 1, device=x_ids.device), diff.bool())


def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes, pre_pad=False):
    bsz = graph_sizes.size(0)
    dtype, device = batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = graph_sizes.min(), graph_sizes.max()
    if min_size == max_size:
        feats = batched_graph_feats.view(bsz, max_size, -1)
        mask = th.ones((bsz, max_size), dtype=th.bool, device=device)
        return feats, mask
    else:
        feats = []
        mask = th.zeros((bsz, max_size), dtype=th.bool, device=device, requires_grad=False)

        graph_sizes_list = graph_sizes.view(-1).tolist()
        idx = 0
        if pre_pad:
            for i, l in enumerate(graph_sizes_list):
                if l < max_size:
                    feats.append(th.zeros((max_size - l, ) + batched_graph_feats.size()[1:], dtype=dtype, device=device))
                feats.append(batched_graph_feats[idx:idx + l])
                mask[i, -l:].fill_(1)
                idx += l
        else:
            for i, l in enumerate(graph_sizes_list):
                feats.append(batched_graph_feats[idx:idx + l])
                if l < max_size:
                    feats.append(th.zeros((max_size - l, ) + batched_graph_feats.size()[1:], dtype=dtype, device=device))
                mask[i, :l].fill_(1)
                idx += l
        feats = th.cat(feats, 0).view(bsz, max_size, -1)
    return feats, mask


def batch_convert_list_to_tensor(batch_list, max_seq_len=-1, pad_id=0, pre_pad=False):
    batch_tensor = [th.tensor(v) for v in batch_list]
    return batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=max_seq_len, pad_id=pad_id, pre_pad=pre_pad)


def batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=-1, pad_id=0, pre_pad=False):
    batch_lens = [len(v) for v in batch_tensor]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)

    result = th.empty(
        [len(batch_tensor), max_seq_len] + list(batch_tensor[0].size())[1:],
        dtype=batch_tensor[0].dtype,
        device=batch_tensor[0].device
    ).fill_(pad_id)
    for i, t in enumerate(batch_tensor):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            if pre_pad:
                result[i, -len_t:].data.copy_(t)
            else:
                result[i, :len_t].data.copy_(t)
        elif len_t == max_seq_len:
            result[i].data.copy_(t)
        else:
            result[i].data.copy_(t[:max_seq_len])
    return result


def batch_convert_len_to_mask(batch_lens, max_seq_len=-1, pre_pad=False):
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = th.ones(
        (len(batch_lens), max_seq_len),
        dtype=th.bool,
        device=batch_lens[0].device if isinstance(batch_lens[0], th.Tensor) else th.device("cpu")
    )
    if pre_pad:
        for i, l in enumerate(batch_lens):
            mask[i, :-l].fill_(0)
    else:
        for i, l in enumerate(batch_lens):
            mask[i, l:].fill_(0)
    return mask


def batch_convert_mask_to_start_and_end(mask):
    cumsum = mask.cumsum(dim=-1) * 2
    start_indices = cumsum.masked_fill(mask == 0, mask.size(-1)).min(dim=-1)[1]
    end_indices = cumsum.max(dim=-1)[1]

    return start_indices, end_indices


def convert_dgl_graph_to_edgeseq(graph, x_emb, x_len, e_emb):
    uid, vid, eid = graph.all_edges(form="all", order="srcdst")
    e = e_emb[eid]
    if x_emb is not None:
        u, v = x_emb[uid], x_emb[vid]
        e = th.cat([u, v, e], dim=1)
    e_len = th.tensor(graph.batch_num_edges, dtype=x_len.dtype, device=x_len.device).view(x_len.size())
    return e, e_len


def mask_seq_by_len(x, len_x):
    x_size = list(x.size())
    if x_size[1] == len_x.max():
        mask = batch_convert_len_to_mask(len_x)
        mask_size = x_size[0:2] + [1] * (len(x_size) - 2)
        x = x * mask.view(*mask_size)
    return x


def expand_dimensions(old_module, new_module, pre_pad=True):
    with th.no_grad():
        # if type(old_module) != type(new_module):
        #     raise ValueError("Error: the two input should have the same type.")
        if isinstance(old_module, th.Tensor) or isinstance(old_module, nn.Parameter):
            nn.init.zeros_(new_module)
            old_size = old_module.size()
            if pre_pad:
                if len(old_size) == 1:
                    new_module.data[-old_size[0]:].copy_(old_module)
                elif len(old_size) == 2:
                    new_module.data[-old_size[0]:, -old_size[1]:].copy_(old_module)
                elif len(old_size) == 3:
                    new_module.data[-old_size[0]:, -old_size[1]:, -old_size[2]:].copy_(old_module)
                elif len(old_size) == 4:
                    new_module.data[-old_size[0]:, -old_size[1]:, -old_size[2]:, -old_size[3]:].copy_(old_module)
                else:
                    raise NotImplementedError
            else:
                if len(old_size) == 1:
                    new_module.data[:old_size[0]].copy_(old_module)
                elif len(old_size) == 2:
                    new_module.data[:old_size[0], :old_size[1]].copy_(old_module)
                elif len(old_size) == 3:
                    new_module.data[:old_size[0], :old_size[1], :old_size[2]].copy_(old_module)
                elif len(old_size) == 4:
                    new_module.data[:old_size[0], :old_size[1], :old_size[2], :old_size[3]].copy_(old_module)
                else:
                    raise NotImplementedError
            return

        old_param_d = dict(old_module.named_parameters())
        for name, params in new_module.named_parameters():
            if name in old_param_d:
                expand_dimensions(old_param_d[name], params, pre_pad)
