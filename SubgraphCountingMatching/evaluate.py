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

warnings.filterwarnings("ignore")


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
                pred_c, (pred_v, pred_e), ((p_v_mask, p_e_mask), (g_v_mask, g_e_mask)) = model(pattern, graph)
                et = time.time()
            else:
                p_perm_pool = p_perm_pool.to(device)
                p_n_perm_matrix = p_n_perm_matrix.to(device)
                p_e_perm_matrix = p_e_perm_matrix.to(device)
                g_perm_pool = g_perm_pool.to(device)
                g_n_perm_matrix = g_n_perm_matrix.to(device)
                g_e_perm_matrix = g_e_perm_matrix.to(device)
                st = time.time()
                pred_c, (pred_v, pred_e), ((p_v_mask, p_e_mask), (g_v_mask, g_e_mask)) = model(
                    pattern, p_perm_pool, p_n_perm_matrix, p_e_perm_matrix,
                    graph, g_perm_pool, g_n_perm_matrix, g_e_perm_matrix
                )
                et = time.time()

            avg_t = (et - st) / (bsz)
            evaluate_results["time"]["avg"].append(th.tensor([avg_t]).repeat(bsz))

            if node_weights is not None and pred_v is not None and is_graph:
                node_weights = node_weights.to(device)
                node_weights = model.refine_node_weights(node_weights.float())
                for i in range(bsz):
                    evaluate_results["data"]["node_weights"].append(
                        th.masked_select(node_weights[i], g_v_mask[i])
                    )
                    evaluate_results["prediction"]["pred_v"].append(
                        th.masked_select(pred_v[i], g_v_mask[i])
                    )
                ned = F.l1_loss(F.relu(pred_v), node_weights, reduction="none").sum(dim=1).detach()
                evaluate_results["error"]["NED"].append(ned.view(-1))
            else:
                evaluate_results["error"]["NED"].append(th.tensor([0.0]).repeat(bsz))

            if edge_weights is not None and pred_e is not None:
                edge_weights = edge_weights.to(device)
                edge_weights = model.refine_edge_weights(edge_weights.float())
                for i in range(bsz):
                    evaluate_results["data"]["edge_weights"].append(
                        th.masked_select(edge_weights[i], g_e_mask[i])
                    )
                    evaluate_results["prediction"]["pred_e"].append(
                        th.masked_select(pred_e[i], g_e_mask[i])
                    )
                eed = F.l1_loss(F.relu(pred_e), edge_weights, reduction="none").sum(dim=1).detach()
                evaluate_results["error"]["EED"].append(eed.view(-1))
            else:
                evaluate_results["error"]["EED"].append(th.tensor([0.0]).repeat(bsz))

            evaluate_results["data"]["counts"].append(counts.view(-1))
            evaluate_results["prediction"]["pred_c"].append(pred_c.view(-1))
            eval_metric = eval_crit(pred_c, counts)
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
                            epoch_avg_eval_metric,
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

        del pred_c, pred_v, pred_e
        del p_v_mask, p_e_mask, g_v_mask, g_e_mask
        del batch
        del ids, pattern, graph, counts, node_weights, edge_weights
        del eval_metric

    gc.collect()
    return epoch_avg_eval_metric, evaluate_results


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
