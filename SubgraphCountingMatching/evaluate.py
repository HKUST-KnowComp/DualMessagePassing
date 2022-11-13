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

from config import get_eval_config
from utils.graph import compute_norm, compute_largest_eigenvalues, convert_to_dual_graph, get_dual_subisomorphisms
from utils.log import init_logger, close_logger, generate_log_line, generate_best_line, get_best_epochs
from utils.io import load_data, load_config, save_config, save_results
from utils.scheduler import map_scheduler_str_to_scheduler
from utils.sampler import BucketSampler, CircurriculumSampler
from utils.anneal import anneal_fn
from utils.cyclical import cyclical_fn
from models import *

from train import process_model_config, load_model
from train import load_edgeseq_datasets, load_graphadj_datasets
from train import remove_loops, add_reversed_edges, convert_to_dual_data
from train import calculate_degrees, calculate_norms, calculate_eigenvalues
from train import evaluate_epoch

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    config = get_eval_config()

    random.seed(config["seed"])
    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not config["load_model_dir"] or not os.path.exists(config["load_model_dir"]):
        raise FileNotFoundError
    model_name = "_".join(os.path.split(config["load_model_dir"])[1].split("_")[:2])

    if config["save_data_dir"]:
        os.makedirs(config["save_data_dir"], exist_ok=True)

    # set device
    if isinstance(config["gpu_id"], int) and config["gpu_id"] >= 0:
        device = th.device("cuda:%d" % (config["gpu_id"]))
    else:
        device = th.device("cpu")

    # set logger and writer
    logger = init_logger(log_file=os.path.join(config["load_model_dir"], "eval_log_%s.txt" % (ts)), log_tag=model_name)
    logger.info("evaluation config: ", str(config))

    # create/load model
    model, best_epochs = load_model(config["load_model_dir"])
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
    model_config = load_config(os.path.join(config["load_model_dir"], "config.json"), as_dict=True)
    for k, v in model_config.items():
        if k not in config:
            config[k] = v
    model.expand(pred_return_weights=config["match_weights"], **process_model_config(config))
    model = model.to(device)
    logger.info(model)
    logger.info("number of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

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
    if logger:
        logger.info("calculating degress...")
    for data_type in datasets:
        calculate_degrees(datasets[data_type])
        calculate_norms(datasets[data_type], self_loop=True)
        calculate_eigenvalues(datasets[data_type])

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

    # set records
    eval_metrics = {"train": None, "dev": None, "test": None}

    logger.info("-" * 80)
    for data_type, dataset in datasets.items():
        sampler = BucketSampler(
            dataset,
            group_by=["g_len", "p_len"],
            batch_size=config["eval_batch_size"],
            shuffle=False,
            seed=config["seed"],
            drop_last=False
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=partial(dataset.batchify, return_weights=config["match_weights"]),
        )
        eval_metric, eval_results = evaluate_epoch(
            model, data_type, data_loader, device, config, 0, logger, None
        )
        save_results(
            eval_results, os.path.join(config["load_model_dir"], "eval_%s_results_%s.json" % (data_type, ts))
        )

        eval_metrics[data_type] = eval_metric

    for data_type in eval_metrics:
        if eval_metrics[data_type] is not None:
            logger.info(
                generate_best_line(
                    data_type,
                    0,
                    0,
                    **{
                        "eval-" + config["eval_metric"]: "{:.3f}".format(eval_metrics[data_type])
                    }
                )
            )
    logger.info("=" * 80)

    close_logger(logger)
