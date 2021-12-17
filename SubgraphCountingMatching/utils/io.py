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

csv.field_size_limit(500 * 1024 * 1024)


def get_subdirs(dirpath, leaf_only=True):
    subdirs = list()
    is_leaf = True
    for filename in os.listdir(dirpath):
        filename = os.path.join(dirpath, filename)
        if os.path.isdir(filename):
            is_leaf = False
            subdirs.extend(get_subdirs(filename, leaf_only=leaf_only))
    if not leaf_only or is_leaf:
        subdirs.append(dirpath)
    return subdirs


def get_files(dirpath):
    files = list()
    for filename in os.listdir(dirpath):
        filename = os.path.join(dirpath, filename)
        if os.path.isdir(filename):
            files.extend(get_files(filename))
        else:
            files.append(filename)
    return files


def _read_graphs_from_dir(dirpath):
    graphs = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".gml":
                continue
            try:
                graph = ig.read(os.path.join(dirpath, filename))
                graph.vs["id"] = list(map(int, graph.vs["id"]))
                graph.vs["label"] = list(map(int, graph.vs["label"]))
                graph.es["label"] = list(map(int, graph.es["label"]))
                graph.es["key"] = list(map(int, graph.es["key"]))
                graphs[names[0]] = graph
            except BaseException as e:
                print(e)
                break
    return graphs


def read_graphs_from_dir(dirpath, num_workers=4):
    graphs = dict()
    subdirs = get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()

        for subdir, x in tqdm(results):
            x = x.get()
            graphs[os.path.basename(subdir)] = x
    dirpath = os.path.basename(dirpath)
    if dirpath in graphs and (dirpath == "graphs" or "G_" not in dirpath):
        graphs.update(graphs.pop(dirpath))
    return graphs


def read_patterns_from_dir(dirpath, num_workers=4):
    patterns = dict()
    subdirs = get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()

        for subdir, x in tqdm(results):
            x = x.get()
            patterns.update(x)
    dirpath = os.path.basename(dirpath)
    if dirpath in patterns and (dirpath == "patterns" or "P_" not in dirpath):
        patterns.update(patterns.pop(dirpath))
    return patterns


def _read_metadata_from_csv(csv_file):
    meta = dict()
    try:
        with open(csv_file, "r", newline="") as f:
            csv_reader = csv.reader(f, delimiter=",")
            header = next(csv_reader)
            gid_idx = header.index("g_id")
            cnt_idx = header.index("counts")
            iso_idx = header.index("subisomorphisms")
            for row in csv_reader:
                meta[row[gid_idx]] = {
                    "counts": int(row[cnt_idx]),
                    "subisomorphisms": np.asarray(eval(row[iso_idx]), dtype=np.int64)
                }
    except BaseException as e:
        print(csv_file, e)
    return meta


def read_metadata_from_dir(dirpath, num_workers=4):
    meta = dict()
    files = get_files(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for filename in files:
            if filename.endswith(".csv"):
                results.append(
                    (
                        os.path.splitext(os.path.basename(filename))[0],
                        pool.apply_async(_read_metadata_from_csv, args=(filename, ))
                    )
                )
        pool.close()

        for p_id, x in tqdm(results):
            x = x.get()
            if p_id not in meta:
                meta[p_id] = x
            else:
                meta[p_id].update(x)
    dirpath = os.path.basename(dirpath)
    if dirpath in meta and dirpath == "metadata":
        meta.update(meta.pop(dirpath))
    return meta


def load_data(pattern_dir, graph_dir, metadata_dir, num_workers=4):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)
    meta = read_metadata_from_dir(metadata_dir, num_workers=num_workers)

    if os.path.exists(os.path.join(metadata_dir, "train.txt")):
        train_indices = set([int(x) for x in open(os.path.join(metadata_dir, "train.txt"))])
    else:
        train_indices = None
    if os.path.exists(os.path.join(metadata_dir, "dev.txt")):
        dev_indices = set([int(x) for x in open(os.path.join(metadata_dir, "dev.txt"))])
    else:
        dev_indices = None
    if os.path.exists(os.path.join(metadata_dir, "test.txt")):
        test_indices = set([int(x) for x in open(os.path.join(metadata_dir, "test.txt"))])
    else:
        test_indices = None

    train_data, dev_data, test_data = list(), list(), list()
    shared_graph = True
    for p, pattern in patterns.items():
        # each pattern corresponds to specific graphs
        if p in graphs:
            shared_graph = False
            for g, graph in graphs[p].items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                x["subisomorphisms"] = meta[p][g]["subisomorphisms"]
                x["counts"] = meta[p][g]["counts"]

                g_idx = int(g.rsplit("_", 1)[-1])
                if train_indices is not None:
                    if g_idx in train_indices:
                        train_data.append(x)
                elif g_idx % 10 > 1:
                    train_data.append(x)
                if dev_indices is not None:
                    if g_idx in dev_indices:
                        dev_data.append(x)
                elif g_idx % 10 == 0:
                    dev_data.append(x)
                if test_indices is not None:
                    if g_idx in test_indices:
                        test_data.append(x)
                elif g_idx % 10 == 1:
                    test_data.append(x)
        # patterns share graphs
        else:
            for g, graph in graphs.items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                x["subisomorphisms"] = meta[p][g]["subisomorphisms"]
                x["counts"] = meta[p][g]["counts"]

                g_idx = int(g.rsplit("_", 1)[-1])
                if train_indices is not None:
                    if g_idx in train_indices:
                        train_data.append(x)
                elif g_idx % 3 > 1:
                    train_data.append(x)
                if dev_indices is not None:
                    if g_idx in dev_indices:
                        dev_data.append(x)
                elif g_idx % 3 == 0:
                    dev_data.append(x)
                if test_indices is not None:
                    if g_idx in test_indices:
                        test_data.append(x)
                elif g_idx % 3 == 1:
                    test_data.append(x)

    return OrderedDict({"train": train_data, "dev": dev_data, "test": test_data}), shared_graph


def str2value(x):
    try:
        return eval(x)
    except:
        return x


def str2bool(x):
    x = x.lower()
    return x == "true" or x == "yes"


def str2list(x):
    results = []
    for x in x.split(","):
        x = x.strip()
        if x == "" or x == "null":
            continue
        try:
            x = str2value(x)
        except:
            pass
        results.append(x)
    return results


def load_config(path, as_dict=True):
    with open(path, "r") as f:
        config = json.load(f)
        if not as_dict:
            config = namedtuple("config", config.keys())(*config.values())
    return config


def save_config(config, path):
    if isinstance(config, dict):
        pass
    elif isinstance(config, Namespace):
        config = vars(config)
    else:
        try:
            config = config._as_dict()
        except BaseException as e:
            raise e

    with open(path, "w") as f:
        json.dump(config, f)


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray, th.Tensor)):
            return obj.tolist()
        else:
            return super(TensorEncoder, self).default(obj)


def load_results(path):
    with open(path, "w") as f:
        results = json.load(f)
    return results


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, cls=TensorEncoder)
