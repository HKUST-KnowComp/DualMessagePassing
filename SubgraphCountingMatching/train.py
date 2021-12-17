import datetime
import gc
import math
import numpy as np
import os
import pickle
import random
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from itertools import chain
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from constants import *
from dataset import *

from torch.optim import AdamW

from config import get_train_config
from utils.graph import compute_norm, compute_largest_eigenvalues, convert_to_dual_graph, get_dual_subisomorphisms
from utils.log import init_logger, close_logger, generate_log_line, generate_best_line, get_best_epochs
from utils.io import load_data, load_config, save_config, save_results
from utils.scheduler import map_scheduler_str_to_scheduler
from utils.sampler import BucketSampler, CircurriculumSampler
from utils.anneal import anneal_fn
from utils.cyclical import cyclical_fn
from models import *

warnings.filterwarnings("ignore")


def process_model_config(config):
    model_config = deepcopy(config)

    # for reversed edges:
    # the number of edges becomes double
    # the number of edge labels becomes double
    if config["add_rev"]:
        model_config["max_nge"] *= 2
        model_config["max_ngel"] *= 2
        model_config["max_npe"] *= 2
        model_config["max_npel"] *= 2

    if config["convert_dual"]:
        max_ngv = model_config["max_ngv"]
        max_npv = model_config["max_npv"]
        avg_gd = math.ceil(model_config["max_nge"] / model_config["max_ngv"])
        avg_pd = math.ceil(model_config["max_npe"] / model_config["max_npv"])

        model_config["max_ngv"] = model_config["max_nge"]
        model_config["max_nge"] = (avg_gd * avg_gd) * max_ngv // 2 - max_ngv
        model_config["max_npv"] = model_config["max_npe"]
        model_config["max_npe"] = (avg_pd * avg_pd) * max_npv // 2 - max_npv
        model_config["max_ngvl"] = model_config["max_ngel"]
        model_config["max_ngel"] = model_config["max_ngvl"]
        model_config["max_npvl"] = model_config["max_npel"]
        model_config["max_npel"] = model_config["max_npvl"]

    return model_config


def build_model(config, **kw):
    if config["rep_net"] == "CNN":
        model = CNN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "RNN":
        model = RNN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "TXL":
        model = TransformerXL(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "RGCN":
        model = RGCN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "RGIN":
        model = RGIN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "CompGCN":
        model = CompGCN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "DMPNN":
        model = DMPNN(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "LRP":
        model = LRP(pred_return_weights=config["match_weights"], **config, **kw)
    elif config["rep_net"] == "DMPLRP":
        model = DMPLRP(pred_return_weights=config["match_weights"], **config, **kw)
    return model


def load_model(path, **kw):
    # get config
    if os.path.exists(os.path.join(path, "config.json")):
        config = load_config(os.path.join(path, "config.json"))
    else:
        raise ValueError("config.json is not found.")

    # get the best epoch
    if os.path.exists(os.path.join(path, "log.txt")):
        best_epochs = get_best_epochs(os.path.join(path, "log.txt"))
    else:
        raise FileNotFoundError("log.txt is not found.")

    model = build_model(process_model_config(config), **kw)
    model.load_state_dict(
        th.load(
            os.path.join(path, "epoch%d.pt" % (best_epochs["eval-" + config["eval_metric"]]["dev"][0])),
            map_location=th.device("cpu")
        )
    )

    return model, best_epochs


def load_edgeseq_datasets(pattern_dir, graph_dir, metadata_dir, save_data_dir=None, num_workers=1, logger=None):
    if save_data_dir and all(
        [
            os.path.exists(os.path.join(save_data_dir, "train_edgeseq_dataset.pt")),
            os.path.exists(os.path.join(save_data_dir, "dev_edgeseq_dataset.pt")),
            os.path.exists(os.path.join(save_data_dir, "test_edgeseq_dataset.pt"))
        ]
    ):
        if logger:
            logger.info("loading datasets from {}".format(save_data_dir))
        datasets = OrderedDict()
        datasets["train"] = EdgeSeqDataset().load(os.path.join(save_data_dir, "train_edgeseq_dataset.pt"))
        if logger:
            logger.info("{:8d} training data have been loaded".format(len(datasets["train"])))
        datasets["dev"] = EdgeSeqDataset().load(os.path.join(save_data_dir, "dev_edgeseq_dataset.pt"))
        if logger:
            logger.info("{:8d} dev data have been loaded".format(len(datasets["dev"])))
        datasets["test"] = EdgeSeqDataset().load(os.path.join(save_data_dir, "test_edgeseq_dataset.pt"))
        if logger:
            logger.info("{:8d} test data have been loaded".format(len(datasets["test"])))

    else:
        if logger:
            logger.info("loading datasets from {}, {}, and {}".format(pattern_dir, graph_dir, metadata_dir))
        data, shared_graph = load_data(
            pattern_dir=pattern_dir,
            graph_dir=graph_dir,
            metadata_dir=metadata_dir,
            num_workers=num_workers
        )
        cache = dict() if shared_graph else None
        datasets = OrderedDict()
        datasets["train"] = EdgeSeqDataset(
            data["train"],
            cache=cache,
            num_workers=num_workers,
            share_memory=shared_graph
        )
        data.pop("train")
        if logger:
            logger.info("{:8d} training data have been loaded".format(len(datasets["train"])))
        datasets["dev"] = EdgeSeqDataset(
            data["dev"],
            cache=cache,
            num_workers=num_workers,
            share_memory=shared_graph
        )
        data.pop("dev")
        if logger:
            logger.info("{:8d} dev data have been loaded".format(len(datasets["dev"])))
        datasets["test"] = EdgeSeqDataset(
            data["test"],
            cache=cache,
            num_workers=num_workers,
            share_memory=shared_graph
        )
        data.pop("test")
        if logger:
            logger.info("{:8d} test data have been loaded".format(len(datasets["test"])))
        del cache
        del data
        gc.collect()

        if save_data_dir:
            datasets["train"].save(os.path.join(save_data_dir, "train_edgeseq_dataset.pt"))
            datasets["dev"].save(os.path.join(save_data_dir, "dev_edgeseq_dataset.pt"))
            datasets["test"].save(os.path.join(save_data_dir, "test_edgeseq_dataset.pt"))

    return datasets


def load_graphadj_datasets(pattern_dir, graph_dir, metadata_dir, save_data_dir=None, num_workers=1, logger=None):
    if save_data_dir and all(
        [
            os.path.exists(os.path.join(save_data_dir, "train_graphadj_dataset.pt")),
            os.path.exists(os.path.join(save_data_dir, "dev_graphadj_dataset.pt")),
            os.path.exists(os.path.join(save_data_dir, "test_graphadj_dataset.pt"))
        ]
    ):
        datasets = OrderedDict()
        datasets["train"] = GraphAdjDataset().load(os.path.join(save_data_dir, "train_graphadj_dataset.pt"))
        if logger:
            logger.info("{:8d} training data have been loaded".format(len(datasets["train"])))
        datasets["dev"] = GraphAdjDataset().load(os.path.join(save_data_dir, "dev_graphadj_dataset.pt"))
        if logger:
            logger.info("{:8d} dev data have been loaded".format(len(datasets["dev"])))
        datasets["test"] = GraphAdjDataset().load(os.path.join(save_data_dir, "test_graphadj_dataset.pt"))
        if logger:
            logger.info("{:8d} test data have been loaded".format(len(datasets["test"])))

    else:
        data, shared_graph = load_data(
            pattern_dir=pattern_dir,
            graph_dir=graph_dir,
            metadata_dir=metadata_dir,
            num_workers=num_workers
        )
        datasets = OrderedDict()
        cache = dict() if shared_graph else None
        datasets["train"] = GraphAdjDataset(
            data["train"],
            cache=cache,
            num_workers=num_workers,
            share_memory=shared_graph
        )
        data.pop("train")
        if logger:
            logger.info("{:8d} training data have been loaded".format(len(datasets["train"])))
        datasets["dev"] = GraphAdjDataset(
            data["dev"],
            cache=cache,
            num_workers=num_workers,
            share_memory=shared_graph
        )
        data.pop("dev")
        if logger:
            logger.info("{:8d} dev data have been loaded".format(len(datasets["dev"])))
        datasets["test"] = GraphAdjDataset(
            data["test"],
            cache=cache,
            num_workers=num_workers,
            share_memory=shared_graph
        )
        data.pop("test")
        if logger:
            logger.info("{:8d} test data have been loaded".format(len(datasets["test"])))
        del cache
        del data
        gc.collect()

        if save_data_dir:
            datasets["train"].save(os.path.join(save_data_dir, "train_graphadj_dataset.pt"))
            datasets["dev"].save(os.path.join(save_data_dir, "dev_graphadj_dataset.pt"))
            datasets["test"].save(os.path.join(save_data_dir, "test_graphadj_dataset.pt"))

    return datasets


def remove_loops(dataset):
    if isinstance(dataset, EdgeSeqDataset):
        for x in dataset:
            u, v = x["pattern"].u, x["pattern"].v
            nonloopmask = (u != v)
            for k, v in x["pattern"].tdata.items():
                x["pattern"].tdata[k] = v[nonloopmask]
            u, v = x["graph"].u, x["graph"].v
            nonloopmask = (u != v)
            for k, v in x["graph"].tdata.items():
                x["graph"].tdata[k] = v[nonloopmask]
    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            u, v, e = x["pattern"].all_edges(form="all", order="eid")
            loopmask = (u == v)
            x["pattern"].remove_edges(e[loopmask])
            u, v, e = x["graph"].all_edges(form="all", order="eid")
            loopmask = (u == v)
            x["graph"].remove_edges(e[loopmask])


def add_reversed_edges(dataset, max_npe, max_npel, max_nge, max_ngel):
    if isinstance(dataset, EdgeSeqDataset):
        for x in dataset:
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
    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            if REVFLAG not in x["graph"].edata:
                num_ge = x["graph"].number_of_edges()
                u, v = x["graph"].all_edges(form="uv", order="eid")
                eid = th.arange(max_nge, max_nge + num_ge)
                x["graph"].add_edges(
                    v,
                    u,
                    data={
                        EDGEID: eid,
                        EDGELABEL: x["graph"].edata[EDGELABEL] + max_ngel,
                        REVFLAG: th.ones((num_ge,), dtype=th.bool)
                    }
                )

            if REVFLAG not in x["pattern"].edata:
                num_pe = x["pattern"].number_of_edges()
                u, v = x["pattern"].all_edges(form="uv", order="eid")
                eid = th.arange(max_npe, max_npe + num_pe)
                x["pattern"].add_edges(
                    v,
                    u,
                    data={
                        EDGEID: eid,
                        EDGELABEL: x["pattern"].edata[EDGELABEL] + max_npel,
                        REVFLAG: th.ones((num_pe,), dtype=th.bool)
                    }
                )


def calculate_degrees(dataset):
    if isinstance(dataset, EdgeSeqDataset):
        for x in dataset:
            if INDEGREE not in x["pattern"].tdata:
                x["pattern"].tdata[INDEGREE] = x["pattern"].in_degrees()
            if OUTDEGREE not in x["pattern"].tdata:
                x["pattern"].tdata[OUTDEGREE] = x["pattern"].out_degrees()
            if INDEGREE not in x["graph"].tdata:
                x["graph"].tdata[INDEGREE] = x["graph"].in_degrees()
            if OUTDEGREE not in x["graph"].tdata:
                x["graph"].tdata[OUTDEGREE] = x["graph"].out_degrees()
    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            if INDEGREE not in x["pattern"].ndata:
                x["pattern"].ndata[INDEGREE] = x["pattern"].in_degrees()
            if OUTDEGREE not in x["pattern"].ndata:
                x["pattern"].ndata[OUTDEGREE] = x["pattern"].out_degrees()
            if INDEGREE not in x["graph"].ndata:
                x["graph"].ndata[INDEGREE] = x["graph"].in_degrees()
            if OUTDEGREE not in x["graph"].ndata:
                x["graph"].ndata[OUTDEGREE] = x["graph"].out_degrees()


def calculate_norms(dataset, self_loop=True):
    if isinstance(dataset, EdgeSeqDataset):
        pass
    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            if NORM not in x["pattern"].ndata or NORM not in x["pattern"].edata:
                node_norm, edge_norm = compute_norm(x["pattern"], self_loop)
                x["pattern"].ndata[NORM] = node_norm
                x["pattern"].edata[NORM] = edge_norm
            if NORM not in x["graph"].ndata or NORM not in x["graph"].edata:
                node_norm, edge_norm = compute_norm(x["graph"], self_loop)
                x["graph"].ndata[NORM] = node_norm
                x["graph"].edata[NORM] = edge_norm


def calculate_eigenvalues(dataset):
    if isinstance(dataset, EdgeSeqDataset):
        pass
    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            if NODEEIGENV not in x["pattern"].ndata or EDGEEIGENV not in x["pattern"].edata:
                node_eigenv, edge_eigenv = compute_largest_eigenvalues(x["pattern"])
                x["pattern"].ndata[NODEEIGENV] = th.clamp_min(node_eigenv, 1.0).repeat(x["pattern"].number_of_nodes()).unsqueeze(-1)
                x["pattern"].edata[EDGEEIGENV] = th.clamp_min(edge_eigenv, 1.0).repeat(x["pattern"].number_of_edges()).unsqueeze(-1)
            if NODEEIGENV not in x["graph"].ndata or EDGEEIGENV not in x["graph"].edata:
                node_eigenv, edge_eigenv = compute_largest_eigenvalues(x["graph"])
                x["graph"].ndata[NODEEIGENV] = th.clamp_min(node_eigenv, 1.0).repeat(x["graph"].number_of_nodes()).unsqueeze(-1)
                x["graph"].edata[EDGEEIGENV] = th.clamp_min(edge_eigenv, 1.0).repeat(x["graph"].number_of_edges()).unsqueeze(-1)


def convert_to_dual_data(dataset):
    if isinstance(dataset, EdgeSeqDataset):
        for x in dataset:
            p = x["pattern"].to_graph()
            g = x["graph"].to_graph()
            conj_p = convert_to_dual_graph(p)
            conj_g = convert_to_dual_graph(g)
            # find the corresponding edge isomorphisms
            if x["counts"] > 0 and p.number_of_edges() > 0:
                p_uid, p_vid, p_eid = p.all_edges(form="all", order="eid")
                p_elabel = p.edata[EDGELABEL][p_eid]
                p_uid, p_vid, p_eid, p_elabel = p_uid.numpy(), p_vid.numpy(), p_eid.numpy(), p_elabel.numpy()
                g_uid, g_vid, g_eid = g.all_edges(form="all", order="srcdst")
                g_elabel = g.edata[EDGELABEL][g_eid]
                g_uid, g_vid, g_eid, g_elabel = g_uid.numpy(), g_vid.numpy(), g_eid.numpy(), g_elabel.numpy()

                conj_subisomorphisms = get_dual_subisomorphisms(
                    p_uid, p_vid, p_elabel, g_uid, g_vid, g_elabel,
                    x["subisomorphisms"].numpy())
                conj_subisos = []
                for i in range(len(conj_subisomorphisms)):
                    conj_subiso = np.empty((p_eid.shape[0],), dtype=np.int64)
                    conj_subiso.fill(-1)
                    conj_subiso = g_eid[conj_subisomorphisms[i]]
                    mask = conj_subiso >= 0
                    conj_subisos.append((conj_subiso)[mask])
                conj_subisos = th.from_numpy(np.vstack(conj_subisos))
            else:
                conj_subisos = th.zeros((0,), dtype=th.long)
            x["pattern"] = conj_p.to_edgeseq()
            x["graph"] = conj_g.to_edgeseq()
            x["subisomorphisms"] = conj_subisos
            x["counts"] = len(conj_subisos)

    elif isinstance(dataset, GraphAdjDataset):
        for x in dataset:
            conj_p = convert_to_dual_graph(x["pattern"])
            conj_g = convert_to_dual_graph(x["graph"])
            # find the corresponding edge isomorphisms
            if x["counts"] > 0 and x["pattern"].number_of_edges() > 0:
                p_uid, p_vid, p_eid = x["pattern"].all_edges(form="all", order="eid")
                p_elabel = x["pattern"].edata[EDGELABEL][p_eid]
                p_uid, p_vid, p_eid, p_elabel = p_uid.numpy(), p_vid.numpy(), p_eid.numpy(), p_elabel.numpy()
                g_uid, g_vid, g_eid = x["graph"].all_edges(form="all", order="srcdst")
                g_elabel = x["graph"].edata[EDGELABEL][g_eid]
                g_uid, g_vid, g_eid, g_elabel = g_uid.numpy(), g_vid.numpy(), g_eid.numpy(), g_elabel.numpy()

                conj_subisomorphisms = get_dual_subisomorphisms(
                    p_uid, p_vid, p_elabel, g_uid, g_vid, g_elabel,
                    x["subisomorphisms"].numpy())
                conj_subisos = []
                for i in range(len(conj_subisomorphisms)):
                    conj_subiso = np.empty((p_eid.shape[0], ), dtype=np.int64)
                    conj_subiso.fill(-1)
                    conj_subiso = g_eid[conj_subisomorphisms[i]]
                    mask = conj_subiso >= 0
                    conj_subisos.append((conj_subiso)[mask])
                conj_subisos = th.from_numpy(np.vstack(conj_subisos))
            else:
                conj_subisos = th.zeros((0,), dtype=th.long)
            x["pattern"] = conj_p
            x["graph"] = conj_g
            x["subisomorphisms"] = conj_subisos
            x["counts"] = len(conj_subisos)


def train_epoch(model, optimizer, scheduler, data_type, data_loader, device, config, epoch, logger=None, writer=None):
    is_graph = not isinstance(data_loader.dataset, EdgeSeqDataset)
    epoch_steps = len(data_loader)
    total_steps = config["train_epochs"] * epoch_steps
    total_eval_metric = 0
    total_rep_reg = 0
    total_match_v_loss = 0
    total_match_e_loss = 0
    total_match_v_reg = 0
    total_match_e_reg = 0
    total_bp_loss = 0
    total_cnt = EPS

    if config["eval_metric"] == "MAE":
        eval_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["eval_metric"] == "MSE":
        eval_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["eval_metric"] == "SMSE":
        eval_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config["eval_metric"] == "AUC":
        eval_crit = lambda pred, target: roc_auc_score(
            target.cpu().numpy() > 0,
            F.relu(pred).detach().cpu().numpy())
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    else:
        raise NotImplementedError

    model.train()

    for batch_id, batch in enumerate(data_loader):
        if len(batch) == 5:
            ids, pattern, graph, counts, (node_weights, edge_weights) = batch
            p_perm_pool, p_n_perm_matrix, p_e_perm_matrix = None, None, None
            g_perm_pool, g_n_perm_matrix, g_e_perm_matrix = None, None, None
        elif len(batch) == 9:
            ids, pattern, p_perm_pool, (p_n_perm_matrix, p_e_perm_matrix), \
                graph, g_perm_pool, (g_n_perm_matrix, g_e_perm_matrix), \
                counts, (node_weights, edge_weights) = batch
        else:
            raise NotImplementedError
        bsz = counts.shape[0]
        total_cnt += bsz
        step = epoch * epoch_steps + batch_id
        lr = scheduler.get_last_lr()[0] if scheduler is not None else config["lr"]
        if isinstance(config["neg_pred_slp"], (int, float)):
            neg_slp = float(config["neg_pred_slp"])
        elif config["neg_pred_slp"].startswith("anneal_"):
            neg_slp, init_slp, final_slp = config["neg_pred_slp"].rsplit("$", 3)
            neg_slp = anneal_fn(
                neg_slp[7:],
                step,
                num_init_steps=0,
                num_anneal_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_slp),
                value2=float(final_slp)
            )
        elif config["neg_pred_slp"].startswith("cyclical_"):
            neg_slp, init_slp, final_slp = config["neg_pred_slp"].rsplit("$", 3)
            neg_slp = cyclical_fn(
                neg_slp[9:],
                step,
                num_init_steps=0,
                num_cyclical_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_slp),
                value2=float(final_slp)
            )
        else:
            raise ValueError
        if isinstance(config["match_loss_w"], (int, float)):
            match_loss_w = float(config["match_loss_w"])
        elif config["match_loss_w"].startswith("anneal_"):
            match_loss_w, init_loss_w, final_loss_w = config["match_loss_w"].rsplit("$", 3)
            match_loss_w = anneal_fn(
                match_loss_w[7:],
                step,
                num_init_steps=0,
                num_anneal_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_loss_w),
                value2=float(final_loss_w)
            )
        elif config["match_loss_w"].startswith("cyclical_"):
            match_loss_w, init_loss_w, final_loss_w = config["match_loss_w"].rsplit("$", 3)
            match_loss_w = cyclical_fn(
                match_loss_w[9:],
                step,
                num_init_steps=0,
                num_cyclical_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_loss_w),
                value2=float(final_loss_w)
            )
        else:
            raise ValueError
        if isinstance(config["match_reg_w"], (int, float)):
            match_reg_w = float(config["match_reg_w"])
        elif config["match_reg_w"].startswith("anneal_"):
            match_reg_w, init_reg_w, final_reg_w = config["match_reg_w"].rsplit("$", 3)
            match_reg_w = anneal_fn(
                match_reg_w[7:],
                step,
                num_init_steps=0,
                num_anneal_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_reg_w),
                value2=float(final_reg_w)
            )
        elif config["match_reg_w"].startswith("cyclical_"):
            match_reg_w, init_reg_w, final_reg_w = config["match_reg_w"].rsplit("$", 3)
            match_reg_w = cyclical_fn(
                match_reg_w[9:],
                step,
                num_init_steps=0,
                num_cyclical_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_reg_w),
                value2=float(final_reg_w)
            )
        else:
            raise ValueError
        if isinstance(config["rep_reg_w"], (int, float)):
            rep_reg_w = float(config["rep_reg_w"])
        elif config["rep_reg_w"].startswith("anneal_"):
            rep_reg_w, init_reg_w, final_reg_w = config["rep_reg_w"].rsplit("$", 3)
            rep_reg_w = anneal_fn(
                rep_reg_w[7:],
                step,
                num_init_steps=0,
                num_anneal_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_reg_w),
                value2=float(final_reg_w)
            )
        elif config["rep_reg_w"].startswith("cyclical_"):
            rep_reg_w, init_reg_w, final_reg_w = config["rep_reg_w"].rsplit("$", 3)
            rep_reg_w = cyclical_fn(
                rep_reg_w[9:],
                step,
                num_init_steps=0,
                num_cyclical_steps=total_steps,
                num_cycles=NUM_CYCLES,
                value1=float(init_reg_w),
                value2=float(final_reg_w)
            )
        else:
            raise ValueError

        pattern = pattern.to(device)
        graph = graph.to(device)
        counts = counts.float().unsqueeze(-1).to(device)

        if p_perm_pool is None:
            # pred_c, (pred_v, pred_e), ((p_v_mask, p_e_mask), (g_v_mask, g_e_mask)) = model(pattern, graph)
            output = model(pattern, graph)
        else:
            p_perm_pool = p_perm_pool.to(device)
            p_n_perm_matrix = p_n_perm_matrix.to(device)
            p_e_perm_matrix = p_e_perm_matrix.to(device)
            g_perm_pool = g_perm_pool.to(device)
            g_n_perm_matrix = g_n_perm_matrix.to(device)
            g_e_perm_matrix = g_e_perm_matrix.to(device)
            # pred_c, (pred_v, pred_e), ((p_v_mask, p_e_mask), (g_v_mask, g_e_mask)) = model(
            #     pattern, p_perm_pool, p_n_perm_matrix, p_e_perm_matrix,
            #     graph, g_perm_pool, g_n_perm_matrix, g_e_perm_matrix
            # )
            output = model(
                pattern, p_perm_pool, p_n_perm_matrix, p_e_perm_matrix,
                graph, g_perm_pool, g_n_perm_matrix, g_e_perm_matrix
            )

        eval_metric = eval_crit(output["pred_c"], counts)
        bp_loss = bp_crit(output["pred_c"], counts, neg_slp)

        if node_weights is not None and output["pred_v"] is not None and is_graph:
            with th.no_grad():
                node_weights = node_weights.to(device)
                node_weights = model.refine_node_weights(node_weights.float())
                node_weights.masked_fill_(~(output["g_v_mask"]), 0)
                output["pred_v"].masked_fill_(~(output["g_v_mask"]), 0)
            match_v_loss = (bp_crit(output["pred_v"], node_weights, neg_slp)) * output["pred_v"].size(1)
            match_v_reg = (bp_crit(F.relu(output["pred_v"] - output["pred_c"]), th.zeros_like(output["pred_v"]), 0)) * output["pred_v"].size(1)
        else:
            match_v_loss = th.tensor([0.0], device=device)
            match_v_reg = th.tensor([0.0], device=device)
        if edge_weights is not None and output["pred_e"] is not None:
            with th.no_grad():
                edge_weights = edge_weights.to(device)
                edge_weights = model.refine_edge_weights(edge_weights.float())
                edge_weights.masked_fill_(~(output["g_e_mask"]), 0)
                output["pred_e"].masked_fill_(~(output["g_e_mask"]), 0)
            match_e_loss = (bp_crit(output["pred_e"], edge_weights, neg_slp)) * output["pred_e"].size(1)
            match_e_reg = (bp_crit(F.relu(output["pred_e"] - output["pred_c"]), th.zeros_like(output["pred_e"]), 0)) * output["pred_e"].size(1)
        else:
            match_e_loss = th.tensor([0.0], device=device)
            match_e_reg = th.tensor([0.0], device=device)
        rep_reg = th.tensor([0.0], device=device)
        if output["p_v_rep"] is not None:
            rep_reg = rep_reg + bp_crit(output["p_v_rep"], th.zeros_like(output["p_v_rep"]), 1) * output["p_v_rep"].size(1)
        if output["p_e_rep"] is not None:
            rep_reg = rep_reg + bp_crit(output["p_e_rep"], th.zeros_like(output["p_e_rep"]), 1) * output["p_e_rep"].size(1)
        if output["g_v_rep"] is not None:
            rep_reg = rep_reg + bp_crit(output["g_v_rep"], th.zeros_like(output["g_v_rep"]), 1) * output["g_v_rep"].size(1)
        if output["g_e_rep"] is not None:
            rep_reg = rep_reg + bp_crit(output["g_e_rep"], th.zeros_like(output["g_e_rep"]), 1) * output["g_e_rep"].size(1)

        bp_loss = bp_loss + rep_reg_w * rep_reg
        bp_loss = bp_loss + match_loss_w * (match_v_loss + match_e_loss)
        bp_loss = bp_loss + match_reg_w * (match_v_reg + match_e_reg)

        eval_metric_item = eval_metric.item()
        bp_loss_item = bp_loss.item()
        match_v_loss_item = match_v_loss.item()
        match_e_loss_item = match_e_loss.item()
        match_v_reg_item = match_v_reg.item()
        match_e_reg_item = match_e_reg.item()
        rep_reg_item = rep_reg.item()
        total_eval_metric += eval_metric_item * bsz
        total_bp_loss += bp_loss_item * bsz
        total_match_v_loss += match_v_loss_item * bsz
        total_match_e_loss += match_e_loss_item * bsz
        total_match_v_reg += match_v_reg_item * bsz
        total_match_e_reg += match_e_reg_item * bsz
        total_rep_reg += rep_reg_item * bsz

        bp_loss.backward()
        if config["train_grad_steps"] < 2 or \
           (batch_id % config["train_grad_steps"] == 0 or batch_id == epoch_steps - 1):
            if config["max_grad_norm"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        if writer:
            writer.add_scalar(
                "%s/eval-%s" % (data_type, config["eval_metric"]), eval_metric_item, step
            )
            writer.add_scalar(
                "%s/train-%s" % (data_type, config["bp_loss"]), bp_loss_item, step
            )
            writer.add_scalar(
                "train/lr", lr, step
            )
            writer.add_scalar(
                "train/neg_slp", neg_slp, step
            )
            writer.add_scalar(
                "train/match_loss_w", match_loss_w, step
            )
            writer.add_scalar(
                "train/match_v_loss", match_v_loss_item, step
            )
            writer.add_scalar(
                "train/match_e_loss", match_e_loss_item, step
            )
            writer.add_scalar(
                "train/match_reg_w", match_reg_w, step
            )
            writer.add_scalar(
                "train/match_v_reg", match_v_reg_item, step
            )
            writer.add_scalar(
                "train/match_e_reg", match_e_reg_item, step
            )
            writer.add_scalar(
                "train/rep_reg_w", rep_reg_w, step
            )
            writer.add_scalar(
                "train/rep_reg", rep_reg_item, step
            )

        if logger and (batch_id % config["train_log_steps"] == 0 or batch_id == epoch_steps - 1):
            ind = np.arange(bsz)
            np.random.shuffle(ind)
            ind = ind[:5]
            logger.info(
                generate_log_line(
                    data_type,
                    epoch=epoch,
                    total_epochs=config["train_epochs"],
                    step=batch_id,
                    total_steps=epoch_steps,
                    **{
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "eval-%s" % (config["eval_metric"]):
                            eval_metric_item,
                        "train-%s" % (config["bp_loss"]):
                            "{:6.3f}".format(bp_loss_item),
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "lr":
                            "{:.6f}".format(lr),
                        "neg_slp":
                            "{:.6f}".format(match_loss_w),
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "match_loss_w":
                            "{:.6f}".format(match_loss_w),
                        "match_v_loss":
                            "{:6.3f}".format(match_v_loss_item),
                        "match_e_loss":
                            "{:6.3f}".format(match_e_loss_item),
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "match_reg_w":
                            "{:.6f}".format(match_reg_w),
                        "match_v_reg":
                            "{:6.3f}".format(match_v_reg_item),
                        "match_e_reg":
                            "{:6.3f}".format(match_e_reg_item),
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "rep_reg_w":
                            "{:.6f}".format(rep_reg_w),
                        "rep_reg":
                            "{:6.3f}".format(rep_reg_item),
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "gold":
                            ", ".join(["{:+4.3f}".format(x) for x in counts[ind].detach().cpu().view(-1).numpy()]),
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "pred":
                            ", ".join(["{:+4.3f}".format(x) for x in output["pred_c"][ind].detach().cpu().view(-1).numpy()])
                    }
                )
            )

        # del pred_c, pred_v, pred_e
        # del p_v_mask, p_e_mask, g_v_mask, g_e_mask
        del output
        del batch
        del ids, pattern, graph, counts, node_weights, edge_weights
        del eval_metric
        del rep_reg
        del match_v_loss, match_e_loss
        del match_v_reg, match_e_reg
        del bp_loss

    epoch_avg_eval_metric = total_eval_metric / total_cnt
    epoch_avg_bp_loss = total_bp_loss / total_cnt
    epoch_avg_match_v_loss = total_match_v_loss / total_cnt
    epoch_avg_match_e_loss = total_match_e_loss / total_cnt
    epoch_avg_match_v_reg = total_match_v_reg / total_cnt
    epoch_avg_match_e_reg = total_match_e_reg / total_cnt
    epoch_avg_rep_reg = total_rep_reg / total_cnt

    if writer:
        writer.add_scalar(
            "%s/eval-%s-epoch" % (data_type, config["eval_metric"]), epoch_avg_eval_metric, epoch
        )
        writer.add_scalar(
            "%s/train-%s-epoch" % (data_type, config["bp_loss"]), epoch_avg_bp_loss, epoch
        )

    if logger:
        logger.info("-" * 80)
        logger.info(
            generate_log_line(
                data_type,
                epoch=epoch,
                total_epochs=config["train_epochs"],
                **{
                    "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                    "eval-%s-epoch" % (config["eval_metric"]):
                        "{:.6f}".format(epoch_avg_eval_metric),
                    "train-%s-epoch" % (config["bp_loss"]):
                        "{:6.3f}".format(epoch_avg_bp_loss),
                    "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                    "lr-last":
                        "{:.6f}".format(lr),
                    "neg_slp-last":
                        "{:.6f}".format(neg_slp),
                    "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                    "match_loss_w-last":
                        "{:.6f}".format(match_loss_w),
                    "match_v_loss":
                        "{:6.3f}".format(epoch_avg_match_v_loss),
                    "match_e_loss":
                        "{:6.3f}".format(epoch_avg_match_e_loss),
                    "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                    "match_reg_w-last":
                        "{:.6f}".format(match_reg_w),
                    "match_v_reg":
                        "{:6.3f}".format(epoch_avg_match_v_reg),
                    "match_e_reg":
                        "{:6.3f}".format(epoch_avg_match_e_reg),
                    "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                    "rep_reg_w-last":
                        "{:.6f}".format(rep_reg_w),
                    "rep_reg":
                        "{:6.3f}".format(epoch_avg_rep_reg),
                }
            )
        )

    gc.collect()
    return epoch_avg_eval_metric, epoch_avg_bp_loss


def evaluate_epoch(model, data_type, data_loader, device, config, epoch, logger=None, writer=None):
    is_graph = not isinstance(data_loader.dataset, EdgeSeqDataset)
    epoch_steps = len(data_loader)
    total_eval_metric = 0
    total_cnt = EPS

    evaluate_results = {
        "data": {
            "id": list(),
            "counts": list(),
            "node_weights": list(),
            "edge_weights": list()
        },
        "prediction": {
            "pred_c": list(),
            "pred_v": list(),
            "pred_e": list()
        },
        "error": {
            "AE": list(),
            "SE": list(),
            "NED": list(),
            "EED": list(),
            "MAE": INF,
            "MSE": INF,
            "RMSE": INF,
            "AUC": 0.0,
            "MNED": INF,
            "MEED": INF
        },
        "time": {
            "avg": list(),
            "total": 0.0
        }
    }

    if config["eval_metric"] == "MAE":
        eval_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["eval_metric"] == "MSE":
        eval_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["eval_metric"] == "SMSE":
        eval_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config["eval_metric"] == "AUC":
        eval_crit = lambda pred, target: roc_auc_score(
            target.cpu().numpy() > 0,
            F.relu(pred).detach().cpu().numpy())
    else:
        raise NotImplementedError

    model.eval()

    with th.no_grad():
        for batch_id, batch in enumerate(data_loader):
            if len(batch) == 5:
                ids, pattern, graph, counts, (node_weights, edge_weights) = batch
                p_perm_pool, p_n_perm_matrix, p_e_perm_matrix = None, None, None
                g_perm_pool, g_n_perm_matrix, g_e_perm_matrix = None, None, None
            elif len(batch) == 9:
                ids, pattern, p_perm_pool, (p_n_perm_matrix, p_e_perm_matrix), \
                    graph, g_perm_pool, (g_n_perm_matrix, g_e_perm_matrix), \
                    counts, (node_weights, edge_weights) = batch
            else:
                raise NotImplementedError

            bsz = counts.shape[0]
            total_cnt += bsz
            step = epoch * epoch_steps + batch_id

            evaluate_results["data"]["id"].append(ids)

            pattern = pattern.to(device)
            graph = graph.to(device)
            counts = counts.float().unsqueeze(-1).to(device)

            if p_perm_pool is None:
                st = time.time()
                # pred_c, (pred_v, pred_e), ((p_v_mask, p_e_mask), (g_v_mask, g_e_mask)) = model(pattern, graph)
                output = model(pattern, graph)
                et = time.time()
            else:
                p_perm_pool = p_perm_pool.to(device)
                p_n_perm_matrix = p_n_perm_matrix.to(device)
                p_e_perm_matrix = p_e_perm_matrix.to(device)
                g_perm_pool = g_perm_pool.to(device)
                g_n_perm_matrix = g_n_perm_matrix.to(device)
                g_e_perm_matrix = g_e_perm_matrix.to(device)
                st = time.time()
                # pred_c, (pred_v, pred_e), ((p_v_mask, p_e_mask), (g_v_mask, g_e_mask)) = model(
                #     pattern, p_perm_pool, p_n_perm_matrix, p_e_perm_matrix,
                #     graph, g_perm_pool, g_n_perm_matrix, g_e_perm_matrix
                # )
                output = model(
                    pattern, p_perm_pool, p_n_perm_matrix, p_e_perm_matrix,
                    graph, g_perm_pool, g_n_perm_matrix, g_e_perm_matrix
                )
                et = time.time()

            avg_t = (et - st) / (bsz)
            evaluate_results["time"]["avg"].append(th.tensor([avg_t]).repeat(bsz))

            if node_weights is not None and output["pred_v"] is not None and is_graph:
                node_weights = node_weights.to(device)
                node_weights = model.refine_node_weights(node_weights.float())
                for i in range(bsz):
                    evaluate_results["data"]["node_weights"].append(
                        th.masked_select(node_weights[i], output["g_v_mask"][i])
                    )
                    evaluate_results["prediction"]["pred_v"].append(
                        th.masked_select(output["pred_v"][i], output["g_v_mask"][i])
                    )
                ned = F.l1_loss(F.relu(output["pred_v"]), node_weights, reduction="none").sum(dim=1).detach()
                evaluate_results["error"]["NED"].append(ned.view(-1))
            else:
                evaluate_results["error"]["NED"].append(th.tensor([0.0]).repeat(bsz))

            if edge_weights is not None and output["pred_e"] is not None:
                edge_weights = edge_weights.to(device)
                edge_weights = model.refine_edge_weights(edge_weights.float())
                for i in range(bsz):
                    evaluate_results["data"]["edge_weights"].append(
                        th.masked_select(edge_weights[i], output["g_e_mask"][i])
                    )
                    evaluate_results["prediction"]["pred_e"].append(
                        th.masked_select(output["pred_e"][i], output["g_e_mask"][i])
                    )
                eed = F.l1_loss(F.relu(output["pred_e"]), edge_weights, reduction="none").sum(dim=1).detach()
                evaluate_results["error"]["EED"].append(eed.view(-1))
            else:
                evaluate_results["error"]["EED"].append(th.tensor([0.0]).repeat(bsz))

            evaluate_results["data"]["counts"].append(counts.view(-1))
            evaluate_results["prediction"]["pred_c"].append(output["pred_c"].view(-1))
            eval_metric = eval_crit(output["pred_c"], counts)
            eval_metric_item = eval_metric.item()
            total_eval_metric += eval_metric_item * bsz

            if writer:
                writer.add_scalar(
                    "%s/eval-%s" % (data_type, config["eval_metric"]), eval_metric_item, step
                )

        evaluate_results["data"]["id"] = list(chain.from_iterable(evaluate_results["data"]["id"]))
        evaluate_results["data"]["counts"] = th.cat(evaluate_results["data"]["counts"], dim=0)
        evaluate_results["prediction"]["pred_c"] = th.cat(evaluate_results["prediction"]["pred_c"], dim=0)
        evaluate_results["time"]["avg"] = th.cat(evaluate_results["time"]["avg"], dim=0)
        if len(evaluate_results["data"]["node_weights"]) > 0:
            evaluate_results["data"]["node_weights"] = th.cat(evaluate_results["data"]["node_weights"], dim=0)
        if len(evaluate_results["data"]["edge_weights"]) > 0:
            evaluate_results["data"]["edge_weights"] = th.cat(evaluate_results["data"]["edge_weights"], dim=0)
        if len(evaluate_results["prediction"]["pred_v"]) > 0:
            evaluate_results["prediction"]["pred_v"] = th.cat(evaluate_results["prediction"]["pred_v"], dim=0)
        if len(evaluate_results["prediction"]["pred_e"]) > 0:
            evaluate_results["prediction"]["pred_e"] = th.cat(evaluate_results["prediction"]["pred_e"], dim=0)
        if len(evaluate_results["error"]["NED"]) > 0:
            evaluate_results["error"]["NED"] = th.cat(evaluate_results["error"]["NED"], dim=0)
        if len(evaluate_results["error"]["EED"]) > 0:
            evaluate_results["error"]["EED"] = th.cat(evaluate_results["error"]["EED"], dim=0)

        epoch_avg_eval_metric = total_eval_metric / total_cnt

        pred_c = F.relu(evaluate_results["prediction"]["pred_c"])
        counts = evaluate_results["data"]["counts"]

        ae = F.l1_loss(pred_c, counts, reduction="none")
        se = F.mse_loss(pred_c, counts, reduction="none")
        evaluate_results["error"]["AE"] = ae.view(-1)
        evaluate_results["error"]["SE"] = se.view(-1)
        evaluate_results["error"]["MAE"] = evaluate_results["error"]["AE"].mean().item()
        evaluate_results["error"]["MSE"] = evaluate_results["error"]["SE"].mean().item()
        evaluate_results["error"]["RMSE"] = evaluate_results["error"]["MSE"]**0.5
        evaluate_results["error"]["MNED"] = evaluate_results["error"]["NED"].mean().item()
        evaluate_results["error"]["MEED"] = evaluate_results["error"]["EED"].mean().item()
        evaluate_results["error"]["AUC"] = roc_auc_score((counts > 0).cpu().numpy(), (pred_c > 0).cpu().numpy())
        evaluate_results["time"]["total"] = evaluate_results["time"]["avg"].sum().item()
        epoch_avg_eval_metric = total_eval_metric / total_cnt

        if writer:
            writer.add_scalar("%s/eval-%s-epoch" % (data_type, config["eval_metric"]), epoch_avg_eval_metric, epoch)
            writer.add_scalar("%s/eval-MAE-epoch" % (data_type), evaluate_results["error"]["MAE"] , epoch)
            writer.add_scalar("%s/eval-MSE-epoch" % (data_type), evaluate_results["error"]["MSE"], epoch)
            writer.add_scalar("%s/eval-RMSE-epoch" % (data_type), evaluate_results["error"]["RMSE"], epoch)
            writer.add_scalar("%s/eval-AUC-epoch" % (data_type), evaluate_results["error"]["AUC"], epoch)
            writer.add_scalar("%s/eval-MNED-epoch" % (data_type), evaluate_results["error"]["MNED"], epoch)
            writer.add_scalar("%s/eval-MEED-epoch" % (data_type), evaluate_results["error"]["MEED"], epoch)

        if logger:
            logger.info("-" * 80)
            logger.info(
                generate_log_line(
                    data_type,
                    epoch=epoch,
                    total_epochs=config["train_epochs"],
                    **{
                        "eval-%s-epoch" % (config["eval_metric"]):
                            "{:.6f}".format(epoch_avg_eval_metric),
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "eval-MAE-epoch":
                            evaluate_results["error"]["MAE"],
                        "eval-MSE-epoch":
                            evaluate_results["error"]["MSE"],
                        "eval-RMSE-epoch":
                            evaluate_results["error"]["RMSE"],
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) +
                        "eval-AUC-epoch":
                            evaluate_results["error"]["AUC"],
                        "eval-MNED-epoch":
                            evaluate_results["error"]["MNED"],
                        "eval-MEED-epoch":
                            evaluate_results["error"]["MEED"]
                    }
                )
            )

        # del pred_c, pred_v, pred_e
        # del p_v_mask, p_e_mask, g_v_mask, g_e_mask
        del output
        del batch
        del ids, pattern, graph, counts, node_weights, edge_weights
        del eval_metric

    gc.collect()
    return epoch_avg_eval_metric, evaluate_results


if __name__ == "__main__":
    config = get_train_config()

    random.seed(config["seed"])
    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = "%s_%s_%s" % (config["rep_net"], config["pred_net"], ts)
    config["save_model_dir"] = os.path.join(config["save_model_dir"], model_name)

    os.makedirs(config["save_model_dir"], exist_ok=True)
    if config["save_data_dir"]:
        os.makedirs(config["save_data_dir"], exist_ok=True)

    # set device
    if isinstance(config["gpu_id"], int) and config["gpu_id"] >= 0:
        device = th.device("cuda:%d" % (config["gpu_id"]))
    else:
        device = th.device("cpu")

    # set logger and writer
    logger = init_logger(log_file=os.path.join(config["save_model_dir"], "log.txt"), log_tag=config["rep_net"])
    writer = SummaryWriter(config["save_model_dir"])
    save_config(config, os.path.join(config["save_model_dir"], "config.json"))

    # load data
    if config["rep_net"] in ["CNN", "RNN", "TXL"]:
        datasets = load_edgeseq_datasets(
            pattern_dir=config["pattern_dir"],
            graph_dir=config["graph_dir"],
            metadata_dir=config["metadata_dir"],
            save_data_dir=config["save_data_dir"],
            num_workers=config["num_workers"],
            logger=logger
        )
    else:
        datasets = load_graphadj_datasets(
            pattern_dir=config["pattern_dir"],
            graph_dir=config["graph_dir"],
            metadata_dir=config["metadata_dir"],
            save_data_dir=config["save_data_dir"],
            num_workers=config["num_workers"],
            logger=logger
        )

    # remove loops
    if "withoutloop" in config["metadata_dir"] or "withoutloop" in config["save_data_dir"]:
        for data_type in datasets:
            remove_loops(datasets[data_type])

    max_ngv = config["max_ngv"]
    max_nge = config["max_nge"]
    max_ngvl = config["max_ngvl"]
    max_ngel = config["max_ngel"]
    if config["share_emb_net"]:
        max_npv = max_ngv
        max_npe = max_nge
        max_npvl = max_ngvl
        max_npel = max_ngel
    else:
        max_npv = config["max_npv"]
        max_npe = config["max_npe"]
        max_npvl = config["max_npvl"]
        max_npel = config["max_npel"]

    # compute the p_len and g_len for original data
    for data_type in datasets:
        if isinstance(datasets[data_type], EdgeSeqDataset):
            for x in datasets[data_type]:
                x["g_len"] = len(x["graph"])
                x["p_len"] = len(x["pattern"])
        elif isinstance(datasets[data_type], GraphAdjDataset):
            for x in datasets[data_type]:
                x["g_len"] = len(x["graph"])
                x["p_len"] = len(x["pattern"])
                if NODEID not in x["graph"].ndata:
                    x["graph"].ndata[NODEID] = th.arange(x["graph"].number_of_nodes())
                if EDGEID not in x["graph"].edata:
                    x["graph"].edata[EDGEID] = th.arange(x["graph"].number_of_edges())
                if NODEID not in x["pattern"].ndata:
                    x["pattern"].ndata[NODEID] = th.arange(x["pattern"].number_of_nodes())
                if EDGEID not in x["pattern"].edata:
                    x["pattern"].edata[EDGEID] = th.arange(x["pattern"].number_of_edges())

    # add E reversed edges
    if config["add_rev"]:
        if logger:
            logger.info("adding reversed edges...")
        for data_type in datasets:
            add_reversed_edges(datasets[data_type], max_npe, max_npel, max_nge, max_ngel)
        max_npe *= 2
        max_npel *= 2
        max_nge *= 2
        max_ngel *= 2
    
    # convert graphs to conj_graphs
    if config["convert_dual"]:
        if logger:
            logger.info("converting dual graphs and isomorphisms...")
        for data_type in datasets:
            convert_to_dual_data(datasets[data_type])
        avg_gd = math.ceil(max_nge / max_ngv)
        avg_pd = math.ceil(max_npe / max_npv)

        max_ngv, max_nge = max_nge, (avg_gd * avg_gd) * max_ngv - max_ngv
        max_npv, max_npe = max_npe, (avg_pd * avg_pd) * max_npv - max_npv
        max_ngvl, max_ngel = max_ngel, max_ngvl
        max_npvl, max_npel = max_npel, max_npvl

    # calculate the degrees, norms, and lambdas
    max_neigenv = 4.0
    max_eeigenv = 4.0
    if logger:
        logger.info("calculating degress...")
    for data_type in datasets:
        calculate_degrees(datasets[data_type])
        if isinstance(datasets[data_type], GraphAdjDataset):
            # calculate_norms(datasets[data_type], self_loop=True) # models handle norms
            calculate_eigenvalues(datasets[data_type])
            for x in datasets[data_type]:
                max_neigenv = max(max_neigenv, x["pattern"].ndata[NODEEIGENV][0].item())
                max_eeigenv = max(max_eeigenv, x["pattern"].edata[EDGEEIGENV][0].item())

    if config["rep_net"].endswith("LRP"):
        lrp_datasets = OrderedDict()
        share_memory = "small" not in config["graph_dir"]
        cache = dict() if share_memory else None
        for data_type in datasets:
            LRPDataset.seq_len = config["lrp_seq_len"]
            lrp_datasets[data_type] = LRPDataset(
                datasets[data_type],
                cache=cache,
                num_workers=config["num_workers"],
                share_memory=share_memory
            )
            for x in lrp_datasets[data_type]:
                x["g_len"] = len(x["graph"])
                x["p_len"] = len(x["pattern"])
        del cache
        del datasets
        gc.collect()
        datasets = lrp_datasets

    # create/load model
    if config["load_model_dir"]:
        model, best_epochs = load_model(config["load_model_dir"], init_neigenv=max_neigenv, init_eeigenv=max_eeigenv)
        for metric, epochs in best_epochs.items():
            for data_type in epochs:
                logger.info(
                    generate_best_line(
                        data_type,
                        epochs[data_type][0],
                        epochs[data_type][0],
                        **{
                            metric: "{:.3f}".format(epochs[data_type][1])
                        }
                    )
                )
        model.expand(pred_return_weights=config["match_weights"], **process_model_config(config))
    else:
        model = build_model(process_model_config(config), init_neigenv=max_neigenv, init_eeigenv=max_eeigenv)
    model = model.to(device)
    logger.info(model)
    logger.info("number of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], amsgrad=True)
    optimizer.zero_grad()
    # 6% for warmup
    num_warmup_steps = int(
        len(datasets["train"]) / config["train_batch_size"] * 0.5 *
        min(config["train_epochs"] * 0.06, config["early_stop_rounds"])
    )
    num_schedule_steps = int(
        len(datasets["train"]) / config["train_batch_size"] * config["train_epochs"]
    )
    min_percent = max(1e-3, config["weight_decay"])
    # 6% for minimum
    if min_percent > EPS:
        num_schedule_steps -= num_warmup_steps
    # 20000 as one cycle
    num_cycles = max(1, num_schedule_steps / 20000)
    scheduler = map_scheduler_str_to_scheduler(
        config["scheduler"],
        num_warmup_steps=num_warmup_steps,
        num_schedule_steps=num_schedule_steps,
        num_cycles=num_cycles,
        min_percent=min_percent
    )
    scheduler.set_optimizer(optimizer)

    # set records
    train_bp_losses = list()
    best_bp_epoch = -1
    eval_metrics = {"train": list(), "dev": list(), "test": list()}
    best_eval_epochs = {"train": -1, "dev": -1, "test": -1}

    logger.info("-" * 80)
    for epoch in range(config["train_epochs"]):
        for data_type, dataset in datasets.items():
            if data_type == "train":
                tmp = dataset.__class__()
                tmp.data = list(dataset.data)
                np.random.shuffle(tmp.data)
                tmp.data = tmp.data[:math.ceil(len(dataset.data)*config["train_ratio"])]
                dataset = tmp
                # circurriculum learning
                sampler = CircurriculumSampler(
                    dataset,
                    learning_by=["p_len", "g_len"],
                    used_ratio=min(1.0, 0.5 + epoch / min(config["train_epochs"] * 0.06, config["early_stop_rounds"])),
                    batch_size=config["train_batch_size"],
                    group_by=["g_len", "p_len"],
                    shuffle=True,
                    seed=config["seed"],
                    drop_last=False
                )
                sampler.set_epoch(epoch)
                data_loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=partial(
                        dataset.batchify,
                        return_weights=config["match_weights"]
                    ),
                )
                eval_metric, bp_loss = train_epoch(
                    model, optimizer, scheduler, data_type, data_loader, device, config, epoch, logger, writer
                )

                train_bp_losses.append(bp_loss)
                if best_bp_epoch == -1 or bp_loss <= train_bp_losses[best_bp_epoch]:
                    best_bp_epoch = epoch
            else:
                sampler = BucketSampler(
                    dataset,
                    group_by=["g_len", "p_len"],
                    batch_size=config["eval_batch_size"],
                    shuffle=False,
                    seed=config["seed"],
                    drop_last=False
                )
                sampler.set_epoch(epoch)
                data_loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=partial(dataset.batchify, return_weights=config["match_weights"]),
                )
                eval_metric, eval_results = evaluate_epoch(
                    model, data_type, data_loader, device, config, epoch, logger, writer
                )
                save_results(
                    eval_results, os.path.join(config["save_model_dir"], "%s_results%d.json" % (data_type, epoch))
                )

            eval_metrics[data_type].append(eval_metric)

            # update best evaluation results
            best_eval_epoch = best_eval_epochs[data_type]
            if best_eval_epoch == -1:
                best_eval_epochs[data_type] = epoch
            elif config["eval_metric"] == "AUC":
                if eval_metric >= eval_metrics[data_type][best_eval_epoch]:
                    best_eval_epochs[data_type] = epoch
            else:
                if eval_metric <= eval_metrics[data_type][best_eval_epoch]:
                    best_eval_epochs[data_type] = epoch
            best_eval_epoch = best_eval_epochs[data_type]

            if data_type != "train" and best_eval_epoch == epoch:
                checkpoint_file = os.path.join(config["save_model_dir"], "epoch%d.pt" % (epoch))
                if not os.path.exists(checkpoint_file):
                    if isinstance(model, nn.DataParallel):
                        th.save(model.module.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)
                    else:
                        th.save(model.state_dict(), checkpoint_file)

        logger.info("=" * 80)
        if best_bp_epoch == epoch:
            logger.info(
                generate_best_line(
                    "train",
                    best_bp_epoch,
                    config["train_epochs"],
                    **{
                        "train-" + config["bp_loss"]: "{:.3f}".format(train_bp_losses[best_bp_epoch])
                    }
                )
            )
        for data_type, best_eval_epoch in best_eval_epochs.items():
            if best_eval_epoch == epoch:
                logger.info(
                    generate_best_line(
                        data_type,
                        best_eval_epoch,
                        config["train_epochs"],
                        **{
                            "eval-" + config["eval_metric"]: "{:.3f}".format(eval_metrics[data_type][best_eval_epoch])
                        }
                    )
                )
        logger.info("=" * 80)

        if config["early_stop_rounds"] > 0 and (
            epoch - best_bp_epoch > config["early_stop_rounds"] and
            epoch - best_eval_epochs["dev"] > config["early_stop_rounds"]
        ):
            break

    logger.info("=" * 80)
    logger.info(
        generate_best_line(
            "train",
            best_bp_epoch,
            config["train_epochs"],
            **{
                config["bp_loss"]: "{:.3f}".format(train_bp_losses[best_bp_epoch])
            }
        )
    )
    for data_type, best_eval_epoch in best_eval_epochs.items():
        logger.info(
            generate_best_line(
                data_type,
                best_eval_epoch,
                config["train_epochs"],
                **{
                    config["eval_metric"]: "{:.3f}".format(eval_metrics[data_type][best_eval_epoch])
                }
            )
        )
    logger.info("=" * 80)

    close_logger(logger)
