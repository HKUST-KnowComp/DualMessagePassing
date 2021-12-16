import copy
import dgl
import os
import pickle
import numba
import numpy as np
import scipy.sparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as tsp
from collections import defaultdict, OrderedDict, Counter
from itertools import permutations, combinations

from torch.multiprocessing import Pool, Manager
from torch.utils.data import Dataset
from tqdm import tqdm
from constants import *
from utils import batch_convert_tensor_to_tensor


@numba.jit(numba.int64(numba.int64[:], numba.int64, numba.int64, numba.int64), nopython=True)
def long_item_bisect_left(array, x, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if array[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.jit(numba.int64(numba.int64[:, :], numba.int64[:], numba.int64, numba.int64), nopython=True)
def long_array_bisect_right(array, x, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        # lexicographical order
        smaller_than = False
        larger_than = False
        for i in range(len(array[mid])):
            if array[mid][i] > x[i]:
                larger_than = True
                break
            elif array[mid][i] < x[i]:
                smaller_than = True
                break
        if larger_than or not smaller_than:
            hi = mid
        else:
            lo = mid + 1
    return lo


@numba.jit(numba.int64[:](numba.int64, numba.int64[:, :]), nopython=True)
def compute_nodeseq_subisoweights(num_nodes, subisomorphisms):
    subiso_weights = np.zeros((num_nodes, ), dtype=np.int64)
    for subisomorphism in subisomorphisms:
        for g_u in subisomorphism:
            subiso_weights[g_u] += 1

    return subiso_weights


@numba.jit(numba.int64[:](
    numba.int64[:], numba.int64[:], numba.int64[:],
    numba.int64[:], numba.int64[:], numba.int64[:],
    numba.int64[:, :]), nopython=True
)
def compute_edgeseq_subisoweights(p_u, p_v, p_el, g_u, g_v, g_el, subisomorphisms):
    p_len = len(p_el)
    g_len = len(g_el)
    subiso_weights = np.zeros((g_len, ), dtype=np.int64)

    # numba does not support tuples as keys
    # hence, we use p_u * (max_v + 1) + p_v as the key
    max_v = max([p_u.max(), p_v.max(), g_u.max(), g_v.max()])
    mod = max_v + 1
    pattern_keys = p_u * mod + p_v
    pattern_elabels = dict()
    i = 0
    while i < p_len:
        j = i + 1
        while j < p_len and pattern_keys[i] == pattern_keys[j]:
            j += 1
        # p_els = np.copy(p_el[i:j])
        # p_els.sort()
        p_els = p_el[i:j]
        pattern_elabels[pattern_keys[i]] = p_els
        i = j

    for subisomorphism in subisomorphisms:
        for key in sorted(pattern_elabels.keys()):
            p_els = pattern_elabels[key]
            u, v = key // mod, key % mod
            u, v = subisomorphism[u], subisomorphism[v]
            u_i = long_item_bisect_left(g_u, u, 0, g_len)
            u_j = long_item_bisect_left(g_u, u + 1, 0, g_len)
            v_i = long_item_bisect_left(g_v, v, u_i, u_j)
            v_j = long_item_bisect_left(g_v, v + 1, v_i, u_j)
            # len_pels = len(p_els)
            for k in range(v_i, v_j):
                # e_k = long_item_bisect_left(p_els, g_el[k], 0, len_pels)
                # if e_k < p_els and g_el[k] == p_els[e_k]:
                #     subiso_weights[k] += 1
                for e in p_els:
                    if e == g_el[k]:
                        subiso_weights[k] += 1
    return subiso_weights


class EdgeSeq:
    def __init__(self, code=None):
        if code is None:
            self._batch_num_tuples = None
            self._number_of_tuples = 0
            self.tdata = {
                "u": th.zeros((0, ), dtype=th.long),
                "v": th.zeros((0, ), dtype=th.long),
                "ul": th.zeros((0, ), dtype=th.long),
                "el": th.zeros((0, ), dtype=th.long),
                "vl": th.zeros((0, ), dtype=th.long),
            }

        elif isinstance(code, EdgeSeq) or str(code.__class__).endswith("EdgeSeq'>"):
            self._batch_num_tuples = code._batch_num_tuples
            self._number_of_tuples = code._number_of_tuples
            self.tdata = code.tdata

        elif isinstance(code, (np.ndarray, th.Tensor)) and len(code.shape) in [2, 3]:
            if isinstance(code, np.ndarray):
                u, v, ul, el, vl = th.chunk(th.from_numpy(code), 5, dim=-1)
            else:
                u, v, ul, el, vl = th.chunk(code, 5, dim=-1)
            self.tdata = {
                "u": u.squeeze_(-1),
                "v": v.squeeze_(-1),
                "ul": ul.squeeze_(-1),
                "el": el.squeeze_(-1),
                "vl": vl.squeeze_(-1),
            }

            if len(code.shape) == 2:
                self._batch_num_tuples = None
                self._number_of_tuples = code.shape[0]
            elif len(code.shape) == 3:
                self._batch_num_tuples = [code.shape[1]] * code.shape[0]
                self._number_of_tuples = code.shape[0] * code.shape[1]

        elif isinstance(code, list):
            batch = EdgeSeq.batch(code)
            self._batch_num_tuples = batch._batch_num_tuples
            self._number_of_tuples = batch._number_of_tuples
            self.tdata = batch.tdata
        else:
            raise ValueError

    @property
    def batch_size(self):
        return 1 if self._batch_num_tuples is None else len(self._batch_num_tuples)

    @property
    def device(self):
        for v in self.tdata.values():
            if isinstance(v, th.Tensor):
                return v.device
        return th.device("cpu")

    @property
    def u(self):
        return self.tdata["u"]

    @property
    def src(self):
        return self.tdata["u"]

    @property
    def v(self):
        return self.tdata["v"]

    @property
    def dst(self):
        return self.tdata["v"]

    @property
    def ul(self):
        return self.tdata["ul"]

    @property
    def src_label(self):
        return self.tdata["ul"]

    @property
    def vl(self):
        return self.tdata["vl"]

    @property
    def dst_label(self):
        return self.tdata["vl"]

    @property
    def el(self):
        return self.tdata["el"]

    @property
    def edge_label(self):
        return self.tdata["el"]

    def batch_num_nodes(self):
        if self._batch_num_tuples is None:
            if self._number_of_tuples == 0:
                return th.zeros((1, ), dtype=th.long, device=self.device)
            return th.tensor([max(self.tdata["u"].max().item(), self.tdata["v"].max().item()) + 1], device=self.device)
        else:
            if self._number_of_tuples == 0:
                return th.zeros((len(self._batch_num_tuples), ), dtype=th.long, device=self.device)
            return th.stack([self.tdata["u"].max(dim=1)[0], self.tdata["v"].max(dim=1)[0]], dim=0).max(dim=0)[0] + 1

    def batch_num_tuples(self):
        if self._batch_num_tuples is None:
            return th.tensor([self._number_of_tuples], device=self.device)
        else:
            return th.tensor(self._batch_num_tuples, device=self.device)

    def number_of_nodes(self):
        if self._batch_num_tuples is None:
            if self._number_of_tuples == 0:
                return 0
            return max(self.tdata["u"].max().item(), self.tdata["v"].max().item()) + 1
        else:
            if self._number_of_tuples == 0:
                return 0
            max_uv = th.stack([self.tdata["u"].max(dim=1)[0], self.tdata["v"].max(dim=1)[0]], dim=0).max(dim=0)[0]
            return max_uv.sum().item() + self.batch_size

    def number_of_tuples(self):
        return self._number_of_tuples

    def in_degree(self, v):
        if INDEGREE not in self.tdata:
            if isinstance(v, int):
                if self._batch_num_tuples is None:
                    return (v == self.v).sum().item()
                else:
                    return (v == self.v).sum(dim=1)
            else:
                in_deg = self.in_degrees()
                return in_deg[..., v]
        else:
            in_deg = self.tdata[INDEGREE]
            if isinstance(v, int):
                if self._batch_num_tuples is None:
                    return in_deg[v].item()
                else:
                    return in_deg[:, v]
            else:
                return in_deg[..., v]

    def out_degree(self, u):
        if OUTDEGREE not in self.tdata:
            if isinstance(u, int):
                if self._batch_num_tuples is None:
                    return (u == self.u).sum().item()
                else:
                    return (u == self.u).sum(dim=1)
            else:
                out_deg = self.out_degrees()
                return out_deg[..., u]
        else:
            out_deg = self.tdata[OUTDEGREE]
            if isinstance(u, int):
                if self._batch_num_tuples is None:
                    return out_deg[u].item()
                else:
                    return out_deg[:, u]
            else:
                return out_deg[..., u]

    def in_degrees(self):
        if INDEGREE not in self.tdata:
            if self._batch_num_tuples is None:
                if self._number_of_tuples == 0:
                    in_deg = th.zeros((0, ), dtype=th.long)
                else:
                    in_deg = th.bincount(self.v, minlength=self.u.max().item() + 1)
            else:
                if self._number_of_tuples == 0:
                    in_deg = th.zeros((len(self._batch_num_tuples), 0), dtype=th.long)
                else:
                    v_lens = self._batch_num_tuples
                    num_us = (self.u.max(dim=1)[0] + 1).cpu().tolist()
                    in_deg = batch_convert_tensor_to_tensor(
                        [
                            th.bincount(self.v[i, -v_lens[i]:], minlength=num_us[i])
                            for i in range(self.batch_size)
                        ],
                        pre_pad=True
                    )
            self.tdata[INDEGREE] = in_deg
        else:
            in_deg = self.tdata[INDEGREE]
        return in_deg

    def out_degrees(self):
        if OUTDEGREE not in self.tdata:
            if self._batch_num_tuples is None:
                if self._number_of_tuples == 0:
                    out_deg = th.zeros((0, ), dtype=th.long)
                else:
                    out_deg = th.bincount(self.u, minlength=self.v.max().item() + 1)
            else:
                if self._number_of_tuples == 0:
                    out_deg = th.zeros((len(self._batch_num_tuples), 0), dtype=th.long)
                else:
                    u_lens = self._batch_num_tuples
                    num_vs = (self.v.max(dim=1)[0] + 1).cpu().tolist()
                    out_deg = batch_convert_tensor_to_tensor(
                        [
                            th.bincount(self.u[i, -u_lens[i]:], minlength=num_vs[i])
                            for i in range(self.batch_size)
                        ],
                        pre_pad=True
                    )
            self.tdata[OUTDEGREE] = out_deg
        else:
            out_deg = self.tdata[OUTDEGREE]
        return out_deg

    @staticmethod
    def subsequence(edgeseq, edges):
        edges = th.tensor(edges, dtype=th.long)
        edges.sort()

        subseq = EdgeSeq()
        for k, v in edgeseq.tdata.items():
            subseq.tdata[k] = v[edges].clone()

        if subseq._batch_num_tuples is None:
            subseq._batch_num_tuples = subseq._batch_num_tuples
            subseq._number_of_tuples = edges.shape[0]
        else:
            subseq._batch_num_tuples = []
            seq_len = max(edgeseq._batch_num_tuples)
            edges = edges.numpy()
            num_edges = len(edges)
            for num_tuples in subseq._batch_num_tuples:
                idx = long_array_bisect_right(edges, seq_len - num_tuples, 0, num_edges)
                subseq._batch_num_tuples.append(edges.shape[0] - idx)
            subseq._number_of_tuples = sum(subseq._number_of_tuples)

        return subseq

    def add_tuple(self, u, v, ul, el, vl, data=None):
        assert self._batch_num_tuples is None

        device = self.device

        with th.no_grad():
            # find the index
            idx = th.stack([self.tdata["u"], self.tdata["v"], self.tdata["el"]], dim=-1).numpy()
            idx = bisect_left(idx, np.array([u, v, el], dtype=idx.dtype))

            # update degrees
            if INDEGREE in self.tdata:
                diff = max(u, v) + 1 - self.tdata[INDEGREE].size(0)
                if diff > 0:
                    self.tdata[INDEGREE] = th.cat(
                        [
                            self.tdata[INDEGREE],
                            th.zeros((diff, ), dtype=self.tdata[INDEGREE].dtype, device=device)
                        ],
                        dim=0
                    )
                self.tdata[INDEGREE][v] += 1

            if OUTDEGREE in self.tdata:
                diff = max(u, v) + 1 - self.tdata[OUTDEGREE].size(0)
                if diff > 0:
                    self.tdata[OUTDEGREE] = th.cat(
                        [
                            self.tdata[OUTDEGREE],
                            th.zeros((diff, ), dtype=self.tdata[OUTDEGREE].dtype, device=device)
                        ],
                        dim=0
                    )
                self.tdata[OUTDEGREE][u] += 1

            # insert
            x = {
                "u": th.tensor([u], dtype=self.tdata["u"].dtype, device=device),
                "v": th.tensor([v], dtype=self.tdata["v"].dtype, device=device),
                "ul": th.tensor([ul], dtype=self.tdata["ul"].dtype, device=device),
                "el": th.tensor([el], dtype=self.tdata["el"].dtype, device=device),
                "vl": th.tensor([vl], dtype=self.tdata["vl"].dtype, device=device)
            }
            tdata_schema = self.get_tdata_schemes()
            for k in list(self.tdata.keys()):
                if k in x:
                    y = x[k]
                else:
                    if k == INDEGREE or k == OUTDEGREE:
                        continue
                    if data is not None and k in data:
                        y = data[k]
                    else:
                        y = th.zeros((1, ) + tdata_schema[k][0], dtype=tdata_schema[k][1], device=device)
                self.tdata[k] = th.cat([self.tdata[k][:idx], y, self.tdata[k][idx:]], dim=0)
            if data is not None:
                for k in data:
                    if k not in self.tdata:
                        y = data[k]
                        size = tuple(data[k].size())[1:]
                        self.tdata[k] = th.cat(
                            [
                                th.zeros((idx, ) + size, dtype=data[k].dtype),
                                y,
                                th.zeros((self._number_of_tuples-idx, ) + size, dtype=data[k].dtype)
                            ]
                        )
        self._number_of_tuples += 1

    def add_tuples(self, u, v, ul, el, vl, data=None):
        assert self._batch_num_tuples is None

        device = self.device

        with th.no_grad():
            x = {
                "u": u.to(device),
                "v": v.to(device),
                "ul": ul.to(device),
                "el": el.to(device),
                "vl": vl.to(device),
            }

            # update degrees
            if INDEGREE in self.tdata:
                diff = max(x["u"].max().item(), x["v"].max().item()) + 1 - self.tdata[INDEGREE].size(0)
                if diff > 0:
                    self.tdata[INDEGREE] = th.cat(
                        [
                            self.tdata[INDEGREE],
                            th.zeros((diff, ), dtype=self.tdata[INDEGREE].dtype, device=device)
                        ],
                        dim=0
                    )
                self.tdata[INDEGREE] += th.bincount(x["v"], minlength=self.tdata[INDEGREE].size(0))

            if OUTDEGREE in self.tdata:
                diff = max(x["u"].max().item(), x["v"].max().item()) + 1 - self.tdata[OUTDEGREE].size(0)
                if diff > 0:
                    self.tdata[OUTDEGREE] = th.cat(
                        [
                            self.tdata[OUTDEGREE],
                            th.zeros((diff, ), dtype=self.tdata[OUTDEGREE].dtype, device=device)
                        ],
                        dim=0
                    )
                self.tdata[OUTDEGREE] += th.bincount(x["u"], minlength=self.tdata[OUTDEGREE].size(0))

            num_new_tuples = x["u"].size(0)
            tdata_schema = self.get_tdata_schemes()
            for k in list(self.tdata.keys()):
                if k in x:
                    y = x[k]
                else:
                    if k == INDEGREE or k == OUTDEGREE:
                        continue
                    if data is not None and k in data:
                        y = data[k]
                    else:
                        y = th.zeros((num_new_tuples, ) + tdata_schema[k][0], dtype=tdata_schema[k][1], device=device)
                self.tdata[k] = th.cat([self.tdata[k], y], dim=0)
            if data is not None:
                for k in data:
                    if k not in self.tdata:
                        y = data[k]
                        size = tuple(data[k].size())[1:]
                        self.tdata[k] = th.cat(
                            [
                                th.zeros((self._number_of_tuples, ) + size, dtype=data[k].dtype),
                                y
                            ]
                        )

            idx = th.stack([self.tdata["u"], self.tdata["v"], self.tdata["el"]], dim=-1).cpu().numpy()
            idx = idx.view(
                [("u", idx.dtype), ("v", idx.dtype), ("el", idx.dtype)]
            ).argsort(axis=0, order=["u", "v", "el"]).squeeze(-1)
            for k in list(self.tdata.keys()):
                if k == INDEGREE or k == OUTDEGREE:
                    continue
                self.tdata[k] = self.tdata[k][idx]

        self._number_of_tuples += num_new_tuples

    def __len__(self):
        return self._number_of_tuples

    @property
    def device(self):
        for v in self.tdata.values():
            if isinstance(v, th.Tensor):
                return v.device
        return th.device("cpu")

    @staticmethod
    def from_graph(graph):
        class_name = str(graph.__class__)

        if class_name in ["<class 'dataset.Graph'>", "<class 'Graph'>"]:
            if graph.batch_size > 1:
                graphs = Graph.unbatch(graph)
                return EdgeSeq.batch([EdgeSeq.from_graph(g) for g in graphs])
            else:
                nids = graph.ndata[NODEID]
                nlabels = graph.ndata[NODELABEL]
                elabels = graph.edata[EDGELABEL]
                uid, vid, eid = graph.all_edges(form="all", order="srcdst")
                code = th.stack([nids[uid], nids[vid], nlabels[uid], elabels[eid], nlabels[vid]], dim=-1).numpy()
                code.view(
                    [("u", code.dtype), ("v", code.dtype), ("ul", code.dtype), ("el", code.dtype), ("vl", code.dtype)]
                ).sort(axis=0, order=["u", "v", "ul", "el", "vl"])

                return EdgeSeq(code)

        elif class_name in ["<class 'dgl.heterograph.DGLHeteroGraph'>", "<class 'dgl.graph.DGLGraph'>"]:
            import dgl
            if graph.batch_size > 1:
                graphs = dgl.unbatch(graph)
                return EdgeSeq.batch([EdgeSeq.from_graph(g) for g in graphs])
            else:
                nids = graph.ndata[NODEID]
                nlabels = graph.ndata[NODELABEL]
                elabels = graph.edata[EDGELABEL]
                uid, vid, eid = graph.all_edges(form="all", order="srcdst")
                code = th.stack([nids[uid], nids[vid], nlabels[uid], elabels[eid], nlabels[vid]], dim=-1).numpy()
                code.view(
                    [("u", code.dtype), ("v", code.dtype), ("ul", code.dtype), ("el", code.dtype), ("vl", code.dtype)]
                ).sort(axis=0, order=["u", "v", "ul", "el", "vl"])

                return EdgeSeq(code)

        elif class_name == "<class 'igraph.Graph'>":
            vids = graph.vs[NODEID]
            vlabels = graph.vs[NODELABEL]
            code = list()
            for edge in graph.es:
                v, u = edge.tuple
                code.append((vids[v], vids[u], vlabels[v], edge[EDGELABEL], vlabels[u]))
            code = np.array(code, dtype=np.int64)
            code.view(
                [("u", code.dtype), ("v", code.dtype), ("ul", code.dtype), ("el", code.dtype), ("vl", code.dtype)]
            ).sort(axis=0, order=["u", "v", "ul", "el", "vl"])

            return EdgeSeq(code)

        else:
            raise ValueError

    def to_graph(self):
        if self._batch_num_tuples is None:
            nid2nlabel = dict(zip(self.tdata["u"].numpy(), self.tdata["ul"].numpy()))
            nid2nlabel.update(dict(zip(self.tdata["v"].numpy(), self.tdata["vl"].numpy())))
            num_nodes = len(nid2nlabel)
            nidx2nid = list(nid2nlabel.keys())
            nid2nidx = dict(zip(nidx2nid, range(num_nodes)))
            eidx2elabel = dict(zip(range(self._number_of_tuples), self.tdata["el"].numpy()))
            num_edges = len(eidx2elabel)
            graph = Graph()
            graph.add_nodes(num_nodes)
            graph.ndata[NODEID] = th.LongTensor(list(nid2nlabel.keys()))
            graph.ndata[NODELABEL] = th.LongTensor(list(nid2nlabel.values()))
            graph.add_edges(
                th.tensor([nid2nidx[u] for u in self.tdata["u"].numpy()]),
                th.tensor([nid2nidx[v] for v in self.tdata["v"].numpy()])
            )
            graph.edata[EDGEID] = th.LongTensor(list(eidx2elabel.keys()))
            graph.edata[EDGELABEL] = th.LongTensor(list(eidx2elabel.values()))
            same = (((th.roll(self.tdata["u"], 1, 0) - self.tdata["u"]) == 0) & \
                ((th.roll(self.tdata["v"], 1, 0) - self.tdata["v"]) == 0))
            key = th.zeros((len(eidx2elabel), ), dtype=th.long)
            while same.max().item():
                key += same
                same = (same & (th.roll(same, 1, 0)))
            graph.edata["key"] = key
            graph._batch_num_nodes = None
            graph._number_of_nodes = len(nid2nlabel)
            graph._batch_num_edges = None
            graph._number_of_edges = len(eidx2elabel)

            return graph

        else:
            graphs = []
            for i in range(self.batch_size):
                num_tuples = self._batch_num_tuples[i]
                nid2nlabel = dict(
                    zip(self.tdata["u"][i, -num_tuples:].numpy(), self.tdata["ul"][i, -num_tuples:].numpy())
                )
                nid2nlabel.update(dict(
                    zip(self.tdata["v"][i, -num_tuples:].numpy(), self.tdata["vl"][i, -num_tuples:].numpy()))
                )
                num_nodes = len(nid2nlabel)
                nidx2nid = list(nid2nlabel.keys())
                nid2nidx = dict(zip(nidx2nid, range(num_nodes)))
                eidx2elabel = dict(zip(range(num_tuples), self.tdata["el"][i, -num_tuples:].numpy()))
                num_edges = len(eidx2elabel)
                graph = Graph()
                graph.add_nodes(num_nodes)
                graph.ndata[NODEID] = th.LongTensor(list(nid2nlabel.keys()))
                graph.ndata[NODELABEL] = th.LongTensor(list(nid2nlabel.values()))
                graph.add_edges(
                    th.tensor([nid2nidx[u] for u in self.tdata["u"][i, -num_tuples:].numpy()]),
                    th.tensor([nid2nidx[v] for v in self.tdata["v"][i, -num_tuples:].numpy()])
                )
                graph.edata[EDGEID] = th.LongTensor(list(eidx2elabel.keys()))
                graph.edata[EDGELABEL] = th.LongTensor(list(eidx2elabel.values()))
                same = (((th.roll(self.tdata["u"][i, -num_tuples:], 1, 0) - self.tdata["u"][i, -num_tuples:]) == 0) & \
                    ((th.roll(self.tdata["v"][i, -num_tuples:], 1, 0) - self.tdata["v"][i, -num_tuples:]) == 0))
                key = th.zeros((len(eidx2elabel), ), dtype=th.long)
                while same.max().item():
                    key += same
                    same = (same & (th.roll(same, 1, 0)))
                graph.edata["key"] = key
                graph._batch_num_nodes = None
                graph._number_of_nodes = len(nid2nlabel)
                graph._batch_num_edges = None
                graph._number_of_edges = len(eidx2elabel)
                graphs.append(graph)

            return Graph.batch(graphs)

    @staticmethod
    def batch(data, pre_pad=True):
        assert isinstance(data, list) and len(data) > 0

        for i in range(len(data)):
            if not isinstance(data[i], EdgeSeq):
                data[i] = EdgeSeq(data[i])

        batch = EdgeSeq()

        _batch_num_tuples = []
        tdata_schemes = OrderedDict(
            {
                "u": (tuple(), th.long),
                "v": (tuple(), th.long),
                "ul": (tuple(), th.long),
                "el": (tuple(), th.long),
                "vl": (tuple(), th.long),
            }
        )
        for x in data:
            _batch_num_tuples.append(x.batch_num_tuples())
            for k, v in x.get_tdata_schemes().items():
                if k == INDEGREE or k == OUTDEGREE:
                    continue
                if k not in tdata_schemes:
                    tdata_schemes[k] = v
                elif v != tdata_schemes[k]:
                    raise ValueError
        _batch_num_tuples = th.cat(_batch_num_tuples, dim=0)
        padded_len = _batch_num_tuples.max().item()
        _batch_num_tuples = _batch_num_tuples
        for k, v in tdata_schemes.items():
            batch.tdata[k] = th.zeros((_batch_num_tuples.size(0), padded_len) + v[0], dtype=v[1])
        idx = 0
        if pre_pad:
            for x in data:
                if x._batch_num_tuples is None:
                    for k in tdata_schemes:
                        if k not in x.tdata:
                            continue
                        x_len = x.tdata[k].shape[0]
                        if x_len > 0 and k in x.tdata:
                            batch.tdata[k][idx, -x_len:].data.copy_(x.tdata[k])
                    idx += 1
                else:
                    for i in range(len(x._batch_num_tuples)):
                        for k in tdata_schemes:
                            if k not in x.tdata:
                                continue
                            x_len = x.tdata[k].shape[1]
                            if x_len > 0 and k in x.tdata:
                                batch.tdata[k][idx, -x_len:].data.copy_(x.tdata[k][i])
                        idx += 1
        else:
            for x in data:
                if x._batch_num_tuples is None:
                    for k in tdata_schemes:
                        if k not in x.tdata:
                            continue
                        x_len = x.tdata[k].shape[0]
                        if x_len > 0 and k in x.tdata:
                            batch.tdata[k][idx, :x_len].data.copy_(x.tdata[k])
                    idx += 1
                else:
                    for i in range(len(x._batch_num_tuples)):
                        for k in tdata_schemes:
                            if k not in x.tdata:
                                continue
                            x_len = x.tdata[k].shape[1]
                            if x_len > 0 and k in x.tdata:
                                batch.tdata[k][idx, :x_len].data.copy_(x.tdata[k][i])
                        idx += 1
        batch._number_of_tuples = _batch_num_tuples.sum().item()
        batch._batch_num_tuples = _batch_num_tuples.tolist()

        return batch

    @staticmethod
    def unbatch(batch):
        if batch._batch_num_tuples is None:
            return [batch]

        else:
            num_used_tuples = 0
            edgeseqs = []
            for i in range(batch.batch_size):
                n = batch._batch_num_tuples[i]
                edgeseq = EdgeSeq()
                edgeseq._number_of_tuples = n
                for k, v in batch.tdata.items():
                    edgeseq.tdata[k] = v[i, -n:]
                edgeseqs.append(edgeseq)
                num_used_tuples += n

            return edgeseqs

    def copy(self, deep=False):
        x = EdgeSeq()
        if deep:
            x._batch_num_tuples = copy.deepcopy(self._batch_num_tuples)
            x._number_of_tuples = self._number_of_tuples
            x.tdata = {k: v.clone() for k, v in self.tdata.items()}
        else:
            x._batch_num_tuples = copy.copy(self._batch_num_tuples)
            x._number_of_tuples = self._number_of_tuples
            x.tdata = copy.copy(self.tdata)

        return x

    def to(self, device):
        if device is None or self.device == device:
            return self

        x = EdgeSeq()
        x._batch_num_tuples = self._batch_num_tuples
        x._number_of_tuples = self._number_of_tuples
        x.tdata = {k: v.to(device) for k, v in self.tdata.items()}

        return x

    def get_tdata_schemes(self):
        tdata_schemes = OrderedDict()
        if self._batch_num_tuples is None:
            for k, v in self.tdata.items():
                tdata_schemes[k] = (v.shape[1:], v.dtype)
        else:
            for k, v in self.tdata.items():
                tdata_schemes[k] = (v.shape[2:], v.dtype)
        return tdata_schemes

    def __repr__(self):
        tdata_schemes = self.get_tdata_schemes()
        tdata_schemes = ["{}: Scheme(shape={}, dtype={})".format(k, *v) for k, v in tdata_schemes.items()]
        return "EdgeSeq(num_tuples={},\n        tdata_schemes={})".format(
            self._number_of_tuples, "{" + ", ".join(tdata_schemes) + "}"
        )


class EdgeSeqDataset(Dataset):
    def __init__(self, data=None, cache=None, num_workers=1, share_memory=False):
        super(EdgeSeqDataset, self).__init__()

        if num_workers <= 0:
            num_workers = os.cpu_count()

        if data:
            self.data = EdgeSeqDataset.preprocess_batch(
                data, cache, num_workers=num_workers, share_memory=share_memory, use_tqdm=True
            )
        else:
            self.data = list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, filename):
        cache = defaultdict(list)
        for x in self.data:
            for k in list(x.keys()):
                if k.startswith("_"):
                    cache[k].append(x.pop(k))
        with open(filename, "wb") as f:
            try:
                th.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL, _use_new_zipfile_serialization=False)
            except TypeError:
                th.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        if len(cache) > 0:
            keys = cache.keys()
            for i in range(len(self.data)):
                for k in keys:
                    self.data[i][k] = cache[k][i]

    def load(self, filename):
        with open(filename, "rb") as f:
            data = th.load(f)
        del self.data
        self.data = data

        return self

    @staticmethod
    def preprocess(x, cache=None):
        if cache is None:
            pattern = EdgeSeq.from_graph(x["pattern"])
            graph = EdgeSeq.from_graph(x["graph"])
        else:
            p_id, g_id = x["id"].split("-")
            pattern = cache.get(p_id, None)
            if pattern is None:
                pattern = cache[p_id] = EdgeSeq.from_graph(x["pattern"])
            graph = cache.get(g_id, None)
            if graph is None:
                graph = cache[g_id] = EdgeSeq.from_graph(x["graph"])

        subisomorphisms = x["subisomorphisms"]
        if isinstance(subisomorphisms, list):
            subisomorphisms = th.tensor(subisomorphisms, dtype=th.int64)
        elif isinstance(subisomorphisms, np.ndarray):
            subisomorphisms = th.from_numpy(subisomorphisms)
        elif isinstance(subisomorphisms, th.Tensor):
            subisomorphisms = subisomorphisms
        subisomorphisms = subisomorphisms.view(-1, x["pattern"].vcount())

        x = {
            "id": x["id"],
            "pattern": pattern,
            "graph": graph,
            "counts": x["counts"],
            "subisomorphisms": subisomorphisms
        }
        return x

    @staticmethod
    def preprocess_batch(data, cache=None, num_workers=1, share_memory=False, use_tqdm=False):
        with Manager() as manager:
            d = list()
            if share_memory:
                if num_workers == 1:
                    if cache is None:
                        cache = dict()
                else:
                    if cache is None:
                        cache = manager.dict()
                    elif isinstance(cache, dict):
                        cache = manager.dict(cache)
            else:
                cache = None

            if num_workers == 1:
                if use_tqdm:
                    data = tqdm(data)
                for x in data:
                    d.append(EdgeSeqDataset.preprocess(x, cache))
            else:
                with Pool(num_workers) as pool:
                    pool_results = []
                    for x in data:
                        pool_results.append(pool.apply_async(EdgeSeqDataset.preprocess, args=(x, cache)))
                    pool.close()

                if use_tqdm:
                    pool_results = tqdm(pool_results)
                for x in pool_results:
                    d.append(x.get())

        return d

    @staticmethod
    def calculate_node_weights(x):
        g_ids = set(x["graph"].tdata["u"].numpy()) | set(x["graph"].tdata["v"].numpy())
        if x["counts"] == 0:
            node_weights = th.zeros((len(g_ids), ), dtype=th.long)
        else:
            node_weights = th.from_numpy(
                compute_nodeseq_subisoweights(len(g_ids), x["subisomorphisms"].numpy())
            )

        return node_weights

    @staticmethod
    def calculate_edge_weights(x):
        if x["counts"] == 0:
            edge_weights = th.zeros((x["graph"].number_of_tuples(), ), dtype=th.long)
        else:
            p_u, p_v = x["pattern"].tdata["u"].numpy(), x["pattern"].tdata["v"].numpy()
            p_ids = set(p_u)
            p_ids.update(p_v)
            p_id2idx = np.zeros((max(p_ids) + 1,), dtype=np.int64)
            p_id2idx.fill(-1)
            for i, j in enumerate(sorted(p_ids)):
                p_id2idx[j] = i
            p_u, p_v = p_id2idx[p_u], p_id2idx[p_v]

            g_u, g_v = x["graph"].tdata["u"].numpy(), x["graph"].tdata["v"].numpy()
            g_ids = set(g_u)
            g_ids.update(g_v)
            g_id2idx = np.zeros((max(g_ids) + 1,), dtype=np.int64)
            g_id2idx.fill(-1)
            for i, j in enumerate(sorted(g_ids)):
                g_id2idx[j] = i
            g_u, g_v = g_id2idx[g_u], g_id2idx[g_v]

            edge_weights = th.from_numpy(
                compute_edgeseq_subisoweights(
                    p_u, p_v, x["pattern"].tdata["el"].numpy(),
                    g_u, g_v, x["graph"].tdata["el"].numpy(),
                    x["subisomorphisms"].numpy()
                )
            )

        return edge_weights

    @staticmethod
    def add_reversed_edges(x, max_npe, max_npel, max_nge, max_ngel, cache=None):
        p_id, g_id = x["id"].split("-")
        if cache is not None and p_id in cache:
            x["pattern"] = cache[p_id]
        else:
            if REVFLAG not in x["pattern"].tdata:
                x["pattern"].add_tuples(
                    x["pattern"].tdata["v"],
                    x["pattern"].tdata["u"],
                    x["pattern"].tdata["vl"],
                    x["pattern"].tdata["el"] + max_npel,
                    x["pattern"].tdata["ul"],
                    data={
                        REVFLAG: th.ones((len(x["pattern"]),), dtype=th.bool)
                    }
                )
            if cache is not None:
                cache[p_id] = x["pattern"]
        if cache is not None and g_id in cache:
            x["graph"] = cache[g_id]
        else:
            if REVFLAG not in x["graph"].tdata:
                x["graph"].add_tuples(
                    x["graph"].tdata["v"],
                    x["graph"].tdata["u"],
                    x["graph"].tdata["vl"],
                    x["graph"].tdata["el"] + max_ngel,
                    x["graph"].tdata["ul"],
                    data={
                        REVFLAG: th.ones((len(x["graph"]),), dtype=th.bool)
                    }
                )
            if cache is not None:
                cache[g_id] = x["graph"]

        if "_edge_weights" in x and x["_edge_weights"].size(0) < x["graph"].number_of_tuples():
            new_edge_weights = th.zeros((x["graph"].number_of_tuples(),), dtype=th.long)
            if x["counts"] > 0:
                g_u, g_v = x["graph"].tdata["u"], x["graph"].tdata["v"]
                g_el = x["graph"].tdata["el"]
                ogn_mask = ~(x["graph"].tdata[REVFLAG])
                assert x["_edge_weights"].size(0) == ogn_mask.sum().item()
                edge_weight_mapping = dict()
                for u, v, el, ew in zip(g_u[ogn_mask].numpy(), g_v[ogn_mask].numpy(), g_el[ogn_mask].numpy(), x["_edge_weights"].numpy()):
                    edge_weight_mapping[(u, v, el)] = ew
                    edge_weight_mapping[(v, u, el + max_ngel)] = ew
                new_edge_weights = th.tensor([edge_weight_mapping[k] for k in zip(g_u.numpy(), g_v.numpy(), g_el.numpy())], dtype=th.long)
            x["_edge_weights"] = new_edge_weights
        return x

    @staticmethod
    def add_reversed_edges_batch(
            data, max_npe, max_npel, max_nge, max_ngel, num_workers=1, share_memory=False, use_tqdm=False
    ):
        with Manager() as manager:
            d = list()
            if share_memory:
                if num_workers == 1:
                    cache = dict()
                else:
                    cache = manager.dict()
            else:
                cache = None

            if num_workers == 1:
                if use_tqdm:
                    data = tqdm(data)
                for x in data:
                    d.append(EdgeSeqDataset.add_reversed_edges(x, max_npe, max_npel, max_nge, max_ngel, cache))
            else:
                with Pool(num_workers) as pool:
                    pool_results = []
                    for x in data:
                        pool_results.append(
                            pool.apply_async(
                                EdgeSeqDataset.add_reversed_edges,
                                args=(x, max_npe, max_npel, max_nge, max_ngel, cache)
                            )
                        )
                    pool.close()

                if use_tqdm:
                    pool_results = tqdm(pool_results)
                for x in pool_results:
                    d.append(x.get())

        return d

    @staticmethod
    def batchify(batch, return_weights=None, num_workers=1):
        bsz = len(batch)

        _id = [x["id"] for x in batch]
        pattern = EdgeSeq.batch([EdgeSeq(x["pattern"]) for x in batch], pre_pad=True)
        graph = EdgeSeq.batch([EdgeSeq(x["graph"]) for x in batch], pre_pad=True)
        counts = th.tensor([x["counts"] for x in batch], dtype=th.int64)

        node_weights, edge_weights = None, None
        if return_weights is None:
            return _id, pattern, graph, counts, (node_weights, edge_weights)

        if isinstance(return_weights, str):
            return_weights = return_weights.split(",")

        if "node" in return_weights:
            node_weights = list()
            for x in batch:
                if "_node_weights" not in x:
                    x["_node_weights"] = EdgeSeqDataset.calculate_node_weights(x)
                node_weights.append(x["_node_weights"])
            node_weights = batch_convert_tensor_to_tensor(node_weights, pre_pad=True)

        if "edge" in return_weights:
            edge_weights = list()
            for x in batch:
                if "_edge_weights" not in x:
                    x["_edge_weights"] = EdgeSeqDataset.calculate_edge_weights(x)
                edge_weights.append(x["_edge_weights"])
            edge_weights = batch_convert_tensor_to_tensor(edge_weights, pre_pad=True)

        return _id, pattern, graph, counts, (node_weights, edge_weights)


class Graph(dgl.DGLGraph):
    def __init__(self, graph=None, *args):

        if graph is None:
            super(Graph, self).__init__(multigraph=True)
            self.readonly(True)
            self.ndata[NODEID] = th.zeros((0, ), dtype=th.long)
            self.ndata[NODELABEL] = th.zeros((0, ), dtype=th.long)
            self.edata[EDGEID] = th.zeros((0, ), dtype=th.long)
            self.edata[EDGELABEL] = th.zeros((0, ), dtype=th.long)
        else:
            class_name = str(graph.__class__)

            if class_name in ["<class 'dataset.Graph'>", "<class 'Graph'>"]:
                super(Graph, self).__init__(
                    graph._graph, node_frames=graph._node_frames, edge_frames=graph._edge_frames, multigraph=True
                )

            elif class_name in ["<class 'dgl.heterograph.DGLHeteroGraph'>", "<class 'dgl.graph.DGLGraph'>"]:
                super(Graph, self).__init__(
                    graph._graph, node_frames=graph._node_frames, edge_frames=graph._edge_frames, multigraph=True
                )

                if NODEID not in graph.ndata:
                    self.ndata[NODEID] = th.cat([th.arange(n) for n in self.batch_num_nodes()])
                if NODELABEL not in graph.ndata:
                    self.ndata[NODELABEL] = th.zeros((self.number_of_nodes(), ), dtype=th.long)

                if EDGEID not in graph.edata:
                    self.edata[EDGEID] = th.cat([th.arange(n) for n in self.batch_num_edges()])
                if EDGELABEL not in graph.edata:
                    self.edata[EDGELABEL] = th.zeros((self.number_of_edges(), ), dtype=th.long)

            elif class_name == "<class 'dgl.heterograph_index.HeteroGraphIndex'>":
                super(Graph, self).__init__(graph, *args, multigraph=True)

                if NODEID not in self.ndata:
                    self.ndata[NODEID] = th.arange(self.number_of_nodes())
                if NODELABEL not in self.ndata:
                    self.ndata[NODELABEL] = th.zeros((self.number_of_nodes(), ), dtype=th.long)
                if EDGEID not in self.edata:
                    self.edata[EDGEID] = th.arange(self.number_of_edges())
                if EDGELABEL not in self.edata:
                    self.edata[EDGELABEL] = th.zeros((self.number_of_edges(), ), dtype=th.long)

            elif class_name == "<class 'igraph.Graph'>":
                super(Graph, self).__init__(multigraph=True)
                self.add_nodes(graph.vcount())
                edges = graph.get_edgelist()
                self.add_edges([e[0] for e in edges], [e[1] for e in edges])
                self.readonly(True)

                self._batch_num_nodes = None
                self._batch_num_edges = None
                for k in graph.vertex_attributes():
                    v = graph.vs[k]
                    if isinstance(v, np.ndarray):
                        self.ndata[k] = th.from_numpy(v)
                    elif isinstance(v, th.Tensor):
                        self.ndata[k] = v
                    elif isinstance(v, list):
                        self.ndata[k] = th.tensor(v)
                if NODEID not in self.ndata:
                    self.ndata[NODEID] = th.arange(self.number_of_nodes())
                if NODELABEL not in self.ndata:
                    self.ndata[NODELABEL] = th.zeros((self.number_of_nodes(), ), dtype=th.long)

                for k in graph.edge_attributes():
                    v = graph.es[k]
                    if isinstance(v, np.ndarray):
                        self.edata[k] = th.from_numpy(v)
                    elif isinstance(v, th.Tensor):
                        self.edata[k] = v
                    elif isinstance(v, list):
                        self.edata[k] = th.tensor(v)
                if NODEID not in self.edata:
                    self.edata[EDGEID] = th.arange(self.number_of_edges())
                if NODELABEL not in self.edata:
                    self.edata[EDGELABEL] = th.zeros((self.number_of_edges(), ), dtype=th.long)

            else:
                raise ValueError

    @property
    def u(self):
        return self.all_edges(form="uv", order="eid")[0]

    @property
    def src(self):
        return self.u

    @property
    def v(self):
        return self.all_edges(form="uv", order="eid")[1]

    @property
    def dst(self):
        return self.v

    @property
    def ul(self):
        return self.ndata[NODELABEL][self.u]

    @property
    def src_label(self):
        return self.ul

    @property
    def vl(self):
        return self.ndata[NODELABEL][self.v]

    @property
    def dst_label(self):
        return self.vl

    @property
    def el(self):
        return self.edata[EDGELABEL]

    @property
    def edge_label(self):
        return self.el

    def batch_num_nodes(self, *args):
        if dgl.__version__ < "0.5.0":
            if self._batch_num_nodes is None:
                return th.tensor([self.number_of_nodes()])
            else:
                return th.tensor(self._batch_num_nodes)
        else:
            return super(Graph, self).batch_num_nodes(*args)

    def batch_num_edges(self, *args):
        if dgl.__version__ < "0.5.0":
            if self._batch_num_edges is None:
                return th.tensor([self.number_of_edges()])
            else:
                return th.tensor(self._batch_num_edges)
        else:
            return super(Graph, self).batch_num_edges(*args)

    def in_degree(self, v):
        if INDEGREE not in self.ndata:
            if isinstance(v, int):
                return (v == self.v).sum().item()
            else:
                in_deg = self.in_degrees()
                return in_deg[v]
        else:
            in_deg = self.ndata[INDEGREE]
            if isinstance(v, int):
                return in_deg[v].item()
            else:
                return in_deg[v]

    def out_degree(self, u):
        if OUTDEGREE not in self.ndata:
            if isinstance(u, int):
                return (u == self.u).sum().item()
            else:
                out_deg = self.out_degrees()
                return out_deg[u]
        else:
            out_deg = self.ndata[OUTDEGREE]
            if isinstance(u, int):
                return out_deg[u].item()
            else:
                return out_deg[u]

    def in_degrees(self):
        if INDEGREE not in self.ndata:
            in_deg = super(Graph, self).in_degrees()
            self.ndata[INDEGREE] = in_deg
        else:
            in_deg = self.ndata[INDEGREE]
        return in_deg

    def out_degrees(self):
        if OUTDEGREE not in self.ndata:
            out_deg = super(Graph, self).out_degrees()
            self.ndata[OUTDEGREE] = out_deg
        else:
            out_deg = self.ndata[OUTDEGREE]
        return out_deg

    def add_nodes(self, num_new_nodes, **kw):
        # _batch_num_nodes will be changed after calling dgl.batch...
        # assert self._batch_num_nodes is None
        assert self.batch_size == 1

        num_nodes = self.ndata[NODEID].size(0) if NODEID in self.ndata else 0
        data = kw.get("data", dict())
        if NODEID not in data:
            data[NODEID] = th.arange(num_nodes, num_nodes + num_new_nodes)

        super(Graph, self).add_nodes(num_new_nodes, **kw)

    def add_edge(self, u, v, **kw):
        # the recent version DGL will call add_edges so that we migrate to it
        data = kw.get("data", dict())
        for k in list(data.keys()):
            v = data[k]
            if isinstance(v, int):
                data[k] = th.tensor([v], dtype=th.long)
            elif isinstance(v, float):
                data[k] = th.tensor([v], dtype=th.float)
        self.add_edges([u], [v], **kw)

    def add_edges(self, u, v, **kw):
        # _batch_num_edges will be changed after calling dgl.batch...
        # assert self._batch_num_edges is None
        assert self.batch_size == 1

        if isinstance(u, list):
            u = th.LongTensor(u)
        elif isinstance(u, np.ndarray):
            u = th.from_numpy(u)
        elif isinstance(u, th.Tensor):
            pass
        else:
            raise ValueError
        if isinstance(v, list):
            v = th.LongTensor(v)
        elif isinstance(v, np.ndarray):
            v = th.from_numpy(v)
        elif isinstance(v, th.Tensor):
            pass
        else:
            raise ValueError

        num_edges = self.edata[EDGEID].size(0) if EDGEID in self.edata else 0
        num_new_edges = len(u)
        data = kw.get("data", dict())
        if EDGEID not in data:
            data[EDGEID] = th.arange(num_edges, num_edges + num_new_edges)
        super(Graph, self).add_edges(u, v, **kw)
        if INDEGREE in self.ndata:
            self.ndata[INDEGREE] += th.bincount(v, minlength=self.ndata[INDEGREE].size(0))

        if OUTDEGREE in self.ndata:
            self.ndata[OUTDEGREE] += th.bincount(u, minlength=self.ndata[OUTDEGREE].size(0))

    def __len__(self):
        return self.number_of_nodes()

    @property
    def device(self):
        for v in self.ndata.values():
            if isinstance(v, th.Tensor):
                return v.device
        for v in self.edata.values():
            if isinstance(v, th.Tensor):
                return v.device
        return th.device("cpu")

    @staticmethod
    def from_edgeseq(edgeseq):

        if isinstance(edgeseq, (np.ndarray, th.Tensor)):
            edgeseq = EdgeSeq(edgeseq)

        return edgeseq.to_graph()

    def to_edgeseq(self):

        return EdgeSeq.from_graph(self)

    @staticmethod
    def batch(data):
        assert isinstance(data, list) and len(data) > 0

        for i in range(len(data)):
            if not isinstance(data[i], Graph):
                data[i] = Graph(data[i])

        return dgl.batch(data)

    @staticmethod
    def unbatch(batch):
        return dgl.unbatch(batch)

    def copy(self, deep=False):
        if deep:
            x = self.clone()
        else:
            x = self.local_var()

        return x

    def to(self, device):
        if device is None or self.device == device:
            return self
        if dgl.__version__ < "0.5.0":
            x = self.local_var()
            x.to(device)
            return x
        else:
            return self.to(device)

    def get_ndata_schemes(self):
        ndata_schemes = OrderedDict()
        for k, v in self.ndata.items():
            ndata_schemes[k] = (v.shape[1:], v.dtype)
        return ndata_schemes

    def get_edata_schemes(self):
        edata_schemes = OrderedDict()
        for k, v in self.edata.items():
            edata_schemes[k] = (v.shape[1:], v.dtype)
        return edata_schemes

    def __repr__(self):
        ndata_schemes = self.get_ndata_schemes()
        ndata_schemes = ["{}: Scheme(shape={}, dtype={})".format(k, *v) for k, v in ndata_schemes.items()]
        edata_schemes = self.get_edata_schemes()
        edata_schemes = ["{}: Scheme(shape={}, dtype={})".format(k, *v) for k, v in edata_schemes.items()]

        return "Graph(num_nodes={}, num_edges={},\n      ndata_schemes={}\n      edata_schemes={}\n)".format(
            self.number_of_nodes(), self.number_of_edges(), "{" + ", ".join(ndata_schemes) + "}",
            "{" + ", ".join(edata_schemes) + "}"
        )


class GraphAdjDataset(Dataset):
    def __init__(self, data=None, cache=None, num_workers=1, share_memory=False):
        super(GraphAdjDataset, self).__init__()

        if num_workers <= 0:
            num_workers = os.cpu_count()

        if data:
            self.data = GraphAdjDataset.preprocess_batch(
                data, cache, num_workers=num_workers, share_memory=share_memory, use_tqdm=True
            )
        else:
            self.data = list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, filename):
        cache = defaultdict(list)
        for x in self.data:
            for k in list(x.keys()):
                if k.startswith("_"):
                    cache[k].append(x.pop(k))
        with open(filename, "wb") as f:
            try:
                th.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL, _use_new_zipfile_serialization=False)
            except TypeError:
                th.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        if len(cache) > 0:
            keys = cache.keys()
            for i in range(len(self.data)):
                for k in keys:
                    self.data[i][k] = cache[k][i]

    def load(self, filename):
        with open(filename, "rb") as f:
            data = th.load(f)
        del self.data
        self.data = data

        return self

    @staticmethod
    def preprocess(x, cache=None):
        p_id, g_id = x["id"].split("-")
        if cache is None:
            pattern = Graph(x["pattern"])
            graph = Graph(x["graph"])
        else:
            if p_id in cache:
                pattern = cache[p_id]
            else:
                cache[p_id] = pattern = Graph(x["pattern"])
            if g_id in cache:
                graph = cache[g_id]
            else:
                cache[g_id] = graph = Graph(x["graph"])

        subisomorphisms = x["subisomorphisms"]
        if isinstance(subisomorphisms, list):
            subisomorphisms = th.tensor(subisomorphisms, dtype=th.int64)
        elif isinstance(subisomorphisms, np.ndarray):
            subisomorphisms = th.from_numpy(subisomorphisms)
        elif isinstance(subisomorphisms, th.Tensor):
            pass

        x = {
            "id": x["id"],
            "pattern": pattern,
            "graph": graph,
            "counts": x["counts"],
            "subisomorphisms": subisomorphisms
        }

        return x

    @staticmethod
    def preprocess_batch(data, cache=None, num_workers=1, share_memory=False, use_tqdm=False):
        with Manager() as manager:
            d = list()
            if share_memory:
                if num_workers == 1:
                    if cache is None:
                        cache = dict()
                else:
                    if cache is None:
                        cache = manager.dict()
                    elif isinstance(cache, dict):
                        cache = manager.dict(cache)
            else:
                cache = None

            if num_workers == 1:
                if use_tqdm:
                    data = tqdm(data)
                for x in data:
                    d.append(GraphAdjDataset.preprocess(x, cache))
            else:
                with Pool(num_workers) as pool:
                    pool_results = []
                    for x in data:
                        pool_results.append(pool.apply_async(GraphAdjDataset.preprocess, args=(x, cache)))
                    pool.close()

                if use_tqdm:
                    pool_results = tqdm(pool_results)
                for x in pool_results:
                    d.append(x.get())

        return d

    @staticmethod
    def calculate_node_weights(x):
        if x["counts"] == 0:
            node_weights = th.zeros((x["graph"].number_of_nodes(),), dtype=th.long)
        else:
            node_weights = th.from_numpy(
                compute_nodeseq_subisoweights(x["graph"].number_of_nodes(), x["subisomorphisms"].numpy())
            )

        return node_weights

    @staticmethod
    def calculate_edge_weights(x):
        if x["counts"] == 0:
            edge_weights = th.zeros((x["graph"].number_of_edges(),), dtype=th.long)
        else:
            p_uid, p_vid, p_eid = x["pattern"].all_edges(form="all", order="eid")
            p_elabel = x["pattern"].edata[EDGELABEL][p_eid]
            g_uid, g_vid, g_eid = x["graph"].all_edges(form="all", order="srcdst")
            g_elabel = x["graph"].edata[EDGELABEL][g_eid]
            edge_weights = th.zeros(g_eid.size(0), dtype=th.long)
            edge_weights[g_eid] = th.from_numpy(
                compute_edgeseq_subisoweights(
                    p_uid.numpy(), p_vid.numpy(), p_elabel.numpy(),
                    g_uid.numpy(), g_vid.numpy(), g_elabel.numpy(),
                    x["subisomorphisms"].numpy()
                )
            )

        return edge_weights

    @staticmethod
    def add_reversed_edges(x, max_npe, max_npel, max_nge, max_ngel, cache=None):
        p_id, g_id = x["id"].split("-")
        if cache is not None and p_id in cache:
            x["pattern"] = cache[p_id]
        else:
            if REVFLAG not in x["pattern"].edata:
                num_pe = x["pattern"].number_of_edges()
                u, v = x["pattern"].all_edges(form="uv", order="eid")
                x["pattern"].add_edges(
                    v,
                    u,
                    data={
                        EDGEID: th.arange(max_npe, max_npe + num_pe),
                        EDGELABEL: x["pattern"].edata[EDGELABEL] + max_npel,
                        REVFLAG: th.ones((num_pe, ), dtype=th.bool)
                    }
                )
            if cache is not None:
                cache[p_id] = x["pattern"]
        if cache is not None and g_id in cache:
            x["graph"] = cache[g_id]
        else:
            if REVFLAG not in x["graph"].edata:
                num_ge = x["graph"].number_of_edges()
                u, v = x["graph"].all_edges(form="uv", order="eid")
                x["graph"].add_edges(
                    v,
                    u,
                    data={
                        EDGEID: th.arange(max_nge, max_nge + num_ge),
                        EDGELABEL: x["graph"].edata[EDGELABEL] + max_ngel,
                        REVFLAG: th.ones((num_ge, ), dtype=th.bool)
                    }
                )
            if cache is not None:
                cache[g_id] = x["graph"]

        if "_edge_weights" in x and x["_edge_weights"].size(0) < x["graph"].number_of_edges():
            x["_edge_weights"] = x["_edge_weights"].repeat(2)
            assert x["_edge_weights"].size(0) == x["graph"].number_of_edges()
        return x

    @staticmethod
    def add_reversed_edges_batch(
        data, max_npe, max_npel, max_nge, max_ngel, num_workers=1, share_memory=False, use_tqdm=False
    ):
        with Manager() as manager:
            d = list()
            if share_memory:
                if num_workers == 1:
                    cache = dict()
                else:
                    cache = manager.dict()
            else:
                cache = None

            if num_workers == 1:
                if use_tqdm:
                    data = tqdm(data)
                for x in data:
                    d.append(GraphAdjDataset.add_reversed_edges(x, max_npe, max_npel, max_nge, max_ngel, cache))
            else:
                with Pool(num_workers) as pool:
                    pool_results = []
                    for x in data:
                        pool_results.append(
                            pool.apply_async(
                                GraphAdjDataset.add_reversed_edges,
                                args=(x, max_npe, max_npel, max_nge, max_ngel, cache)
                            )
                        )
                    pool.close()

                if use_tqdm:
                    pool_results = tqdm(pool_results)
                for x in pool_results:
                    d.append(x.get())

        return d


    @staticmethod
    def batchify(batch, return_weights=None, num_workers=1):
        bsz = len(batch)

        _id = [x["id"] for x in batch]
        pattern = Graph.batch([x["pattern"] for x in batch])
        graph = Graph.batch([x["graph"] for x in batch])
        counts = th.tensor([x["counts"] for x in batch], dtype=th.int64)

        node_weights, edge_weights = None, None
        if return_weights is None:
            return _id, pattern, graph, counts, (node_weights, edge_weights)

        if isinstance(return_weights, str):
            return_weights = return_weights.split(",")

        if "node" in return_weights:
            node_weights = list()
            for x in batch:
                if "_node_weights" not in x:
                    x["_node_weights"] = GraphAdjDataset.calculate_node_weights(x)
                node_weights.append(x["_node_weights"])
            node_weights = batch_convert_tensor_to_tensor(node_weights, pre_pad=True)

        if "edge" in return_weights:
            edge_weights = list()
            for x in batch:
                if "_edge_weights" not in x:
                    x["_edge_weights"] = GraphAdjDataset.calculate_edge_weights(x)
                edge_weights.append(x["_edge_weights"])
            edge_weights = batch_convert_tensor_to_tensor(edge_weights, pre_pad=True)

        return _id, pattern, graph, counts, (node_weights, edge_weights)


class LRPDataset(Dataset):
    seq_len = 4

    def __init__(self, graphadj_dataset=None, cache=None, num_workers=1, share_memory=False):
        super(LRPDataset, self).__init__()

        if num_workers <= 0:
            num_workers = os.cpu_count()

        if graphadj_dataset is None:
            self.data = []
        else:
            self.data = LRPDataset.preprocess_batch(
                graphadj_dataset, cache=cache, num_workers=num_workers, share_memory=share_memory, use_tqdm=True
            )

    @staticmethod
    def preprocess(x, cache=None):
        if cache is None:
            pattern_lrp_egonet_seq = LRPDataset.graph_to_egonet_seq(x["pattern"])
            graph_lrp_egonet_seq = LRPDataset.graph_to_egonet_seq(x["graph"])
        else:
            p_id, g_id = x["id"].split("-")
            pattern_lrp_egonet_seq = cache.get(p_id, None)
            if pattern_lrp_egonet_seq is None:
                pattern_lrp_egonet_seq = cache[p_id] = LRPDataset.graph_to_egonet_seq(x["pattern"])
            graph_lrp_egonet_seq = cache.get(g_id, None)
            if graph_lrp_egonet_seq is None:
                graph_lrp_egonet_seq = cache[g_id] = LRPDataset.graph_to_egonet_seq(x["graph"])

        x = {
            "id": x["id"],
            "pattern": x["pattern"],
            "graph": x["graph"],
            "pattern_lrp_egonet_seq": pattern_lrp_egonet_seq,
            "graph_lrp_egonet_seq": graph_lrp_egonet_seq,
            "counts": x["counts"],
            "subisomorphisms": x["subisomorphisms"]
        }
        return x

    @staticmethod
    def preprocess_batch(data, cache=None, num_workers=1, share_memory=False, use_tqdm=False):
        with Manager() as manager:
            d = list()
            if share_memory:
                if num_workers == 1:
                    if cache is None:
                        cache = dict()
                else:
                    if cache is None:
                        cache = manager.dict()
                    elif isinstance(cache, dict):
                        cache = manager.dict(cache)
            else:
                cache = None

            if num_workers == 1:
                if use_tqdm:
                    data = tqdm(data)
                for x in data:
                    d.append(LRPDataset.preprocess(x, cache))
            else:
                with Pool(num_workers) as pool:
                    pool_results = []
                    for x in data:
                        pool_results.append(pool.apply_async(LRPDataset.preprocess, args=(x, cache)))
                    pool.close()

                if use_tqdm:
                    pool_results = tqdm(pool_results)
                for x in pool_results:
                    d.append(x.get())

        return d

    def load(self, filename):
        with open(filename, "rb") as f:
            data = th.load(f)
        del self.data
        self.data = data

        return self

    def save(self, filename):
        cache = defaultdict(list)
        for x in self.data:
            for k in list(x.keys()):
                if k.startswith("_"):
                    cache[k].append(x.pop(k))
        with open(filename, "wb") as f:
            try:
                th.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL, _use_new_zipfile_serialization=False)
            except TypeError:
                th.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        if len(cache) > 0:
            keys = cache.keys()
            for i in range(len(self.data)):
                for k in keys:
                    self.data[i][k] = cache[k][i]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

    @staticmethod
    def generate_neighbour_perms(adj_m, start_node):
        adjlist = adj_m.indices[adj_m.indptr[start_node]:adj_m.indptr[start_node+1]]
        nei_len = LRPDataset.seq_len - 1
        all_perm = permutations(adjlist, min(nei_len, len(adjlist)))
        # all_perm = combinations(adjlist, min(nei_len, len(adjlist)))
        return tuple((start_node,) + p for p in all_perm)

    @staticmethod
    def convert_seq_to_ind(eid_map, perm):
        dim_dict = {node: i for i, node in enumerate(perm)}
        node_to_perm_row = np.arange(len(perm), dtype=np.int16) * (1 + LRPDataset.seq_len)
        node_to_perm_col = np.array(perm, dtype=np.int16)
        src = np.repeat(perm, len(perm))
        dst = np.tile(perm, len(perm))
        # src = np.repeat(perm[:1], len(perm) - 1)
        # dst = np.asarray(perm[1:])
        edge_to_perm_row = []
        edge_to_perm_col = []
        for uv in zip(src, dst):
            if uv in eid_map:
                edge_to_perm_row.append(dim_dict[uv[0]] * LRPDataset.seq_len + dim_dict[uv[1]])
                edge_to_perm_col.append(eid_map[uv])
        edge_to_perm_row = np.array(edge_to_perm_row, dtype=np.int16)
        edge_to_perm_col = np.array(edge_to_perm_col, dtype=np.int16)

        return (node_to_perm_row, node_to_perm_col, edge_to_perm_row, edge_to_perm_col)

    @staticmethod
    def graph_to_egonet_seq(graph):
        num_of_nodes = graph.number_of_nodes()
        egonet_seq_graph = []
        src, dst, eid = graph.all_edges(form="all", order="eid")
        # adj_m = graph.adjacency_matrix_scipy() # too slow
        if REVFLAG in graph.edata:
            non_rev_mask = ~(graph.edata[REVFLAG])
            src, dst, eid = src[non_rev_mask], dst[non_rev_mask], eid[non_rev_mask]
        src, dst, eid = src.numpy(), dst.numpy(), eid.numpy()
        adj_m = scipy.sparse.csr_matrix(
            (np.ones((len(eid),), dtype=np.int8), (src, dst)),
            shape=(num_of_nodes, num_of_nodes)
        )
        eid_map = {(u, v): e for u, v, e in zip(src, dst, eid)}
        for i in range(num_of_nodes):
            perms = LRPDataset.generate_neighbour_perms(adj_m, start_node=i)
            egonet_seq = tuple(LRPDataset.convert_seq_to_ind(eid_map, perm) for perm in perms)
            egonet_seq_graph.append(egonet_seq)
        return tuple(egonet_seq_graph)

    @staticmethod
    def build_perm_pooling_matrix(split_list, pooling="sum"):
        dim0, dim1 = split_list.shape[0], split_list.sum()
        row = np.concatenate([np.repeat(np.int64(i), c) for i, c in enumerate(split_list)])
        col = np.arange(dim1, dtype=np.int64)
        if pooling == "mean":
            data = th.cat([(th.tensor([1.0], dtype=th.float32)/c).repeat(c) for i, c in enumerate(split_list)])
        else:
            data = th.ones((dim1,), dtype=th.float32)
        pooling_matrix = tsp.FloatTensor(
            th.stack([th.from_numpy(row), th.from_numpy(col)]),
            data,
            size=(dim0, dim1)
        )

        return pooling_matrix

    @staticmethod
    def build_batch_graph_to_perm_matrices(graphs, lrp_egonet):
        batch_num_nodes = np.array([g.number_of_nodes() for g in graphs], dtype=np.int64)
        batch_num_edges = np.array([g.number_of_edges() for g in graphs], dtype=np.int64)

        node_to_perm_row = []
        node_to_perm_col = []
        edge_to_perm_row = []
        edge_to_perm_col = []

        sum_row_number = 0
        sum_ncol_number = 0
        sum_ecol_number = 0
        out_len = LRPDataset.seq_len * LRPDataset.seq_len
        for i, g_egonet in enumerate(lrp_egonet):
            for n_egonet in g_egonet:
                for perm_ind in n_egonet:
                    node_to_perm_row.append(perm_ind[0].astype(np.int64) + sum_row_number)
                    node_to_perm_col.append(perm_ind[1].astype(np.int64) + sum_ncol_number)
                    edge_to_perm_row.append(perm_ind[2].astype(np.int64) + sum_row_number)
                    edge_to_perm_col.append(perm_ind[3].astype(np.int64) + sum_ecol_number)
                    sum_row_number += out_len
            sum_ncol_number += batch_num_nodes[i]
            sum_ecol_number += batch_num_edges[i]
        node_to_perm_row = np.concatenate(node_to_perm_row)
        node_to_perm_col = np.concatenate(node_to_perm_col)
        edge_to_perm_row = np.concatenate(edge_to_perm_row)
        edge_to_perm_col = np.concatenate(edge_to_perm_col)

        ndim0 = sum_row_number
        ndim1 = batch_num_nodes.sum()
        edim0 = sum_row_number
        edim1 = batch_num_edges.sum()

        node_to_perm_matrix = tsp.FloatTensor(
            th.stack([th.from_numpy(node_to_perm_row), th.from_numpy(node_to_perm_col)]),
            th.ones((len(node_to_perm_row)), dtype=th.float32),
            size=(ndim0, ndim1)
        )
        edge_to_perm_matrix = tsp.FloatTensor(
            th.stack([th.from_numpy(edge_to_perm_row), th.from_numpy(edge_to_perm_col)]),
            th.ones((len(edge_to_perm_row)), dtype=th.float32),
            size=(edim0, edim1)
        )
        return node_to_perm_matrix, edge_to_perm_matrix

    @staticmethod
    def batchify(batch, return_weights="none", num_workers=1):
        _id = [x["id"] for x in batch]
        pattern = [x["pattern"] for x in batch]
        pattern_lrp_egonet_seq = [x["pattern_lrp_egonet_seq"] for x in batch]
        graph = [x["graph"] for x in batch]
        graph_lrp_egonet_seq = [x["graph_lrp_egonet_seq"] for x in batch]
        counts = th.tensor([x["counts"] for x in batch], dtype=th.int64)

        p_split_list = np.asarray([len(node) for seq in pattern_lrp_egonet_seq for node in seq], dtype=np.int64)
        p_perm_pool = LRPDataset.build_perm_pooling_matrix(p_split_list, "mean")
        pattern_to_perm_matrices = LRPDataset.build_batch_graph_to_perm_matrices(pattern, pattern_lrp_egonet_seq)
        g_split_list = np.asarray([len(node) for seq in graph_lrp_egonet_seq for node in seq], dtype=np.int64)
        g_perm_pool = LRPDataset.build_perm_pooling_matrix(g_split_list, "mean")
        graph_to_perm_matrices = LRPDataset.build_batch_graph_to_perm_matrices(graph, graph_lrp_egonet_seq)

        pattern = dgl.batch(pattern)
        graph = dgl.batch(graph)

        if isinstance(return_weights, str):
            return_weights = return_weights.split(",")

        if "node" in return_weights:
            node_weights = list()
            for x in batch:
                if "_node_weights" not in x:
                    x["_node_weights"] = GraphAdjDataset.calculate_node_weights(x)
                node_weights.append(x["_node_weights"])
            node_weights = batch_convert_tensor_to_tensor(node_weights, pre_pad=True)
        else:
            node_weights = None
        if "edge" in return_weights:
            edge_weights = list()
            for x in batch:
                if "_edge_weights" not in x:
                    x["_edge_weights"] = GraphAdjDataset.calculate_edge_weights(x)
                edge_weights.append(x["_edge_weights"])
            edge_weights = batch_convert_tensor_to_tensor(edge_weights, pre_pad=True)
        else:
            edge_weights = None

        return _id, pattern, p_perm_pool, pattern_to_perm_matrices, \
            graph, g_perm_pool, graph_to_perm_matrices, \
            counts, (node_weights, edge_weights)
