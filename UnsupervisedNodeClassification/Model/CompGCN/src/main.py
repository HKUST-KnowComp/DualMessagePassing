
import argparse
import dgl
import math
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import utils
from model import *


np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)


class CosineWarmupRestartScheduler(LambdaLR):
    def __init__(
        self,
        num_warmup_steps=600,
        num_schedule_steps=10000,
        num_cycles=2,
        min_percent=1e-3
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.num_cycles = num_cycles
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(CosineWarmupRestartScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        if progress >= 1.0:
            return self.min_percent
        return max(self.min_percent, 0.5 * (1.0 + math.cos(math.pi * ((float(self.num_cycles) * progress) % 1.0))))


def main(args):

    # load graph data
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "start loading...", flush=True)
    if args.supervised == "True":
        train_pool, train_labels, nlabels, multi = utils.load_label(args.label)
        train_data, num_nodes, num_rels, train_indices, ntrain, node_attri = utils.load_supervised(
            args, args.link, args.node, train_pool
        )
    elif args.supervised == "False":
        train_data, num_nodes, num_rels, node_attri = utils.load_unsupervised(args, args.link, args.node)
        nlabels = 0
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "finish loading...", flush=True)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    print("check 1", flush=True)
    # create model
    model = TrainModel(
        node_attri,
        num_nodes,
        args.n_hidden,
        num_rels,
        nlabels,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        use_cuda=use_cuda,
        reg_param=args.regularization
    )
    print("check 2", flush=True)
    if use_cuda:
        model.to("cuda:%d" % (args.gpu))
    print("check 3", flush=True)
    """
    # build adj list and calculate degrees for sampling
    degrees = utils.get_adj_and_degrees(num_nodes, train_data)
    """
    # build graph
    graph = utils.build_graph_from_triplets(num_nodes, num_rels, train_data)
    graph.ndata[dgl.NID] = torch.arange(num_nodes, dtype=torch.long)
    graph.edata[dgl.EID] = torch.arange(len(train_data) * 2, dtype=torch.long)
    seed_nodes = list()
    if os.path.exists(args.node.replace("node.dat", "seed_node.dat")):
        with open(args.node.replace("node.dat", "seed_node.dat"), "r") as f:
            for line in f:
                seed_nodes.append(int(line))
    seed_nodes = set(seed_nodes)
    if len(seed_nodes) > 0:
        dataloader = torch.utils.data.DataLoader(
            np.array([x for x in train_data if x[0] in seed_nodes or x[2] in seed_nodes]),
            batch_size=args.graph_batch_size, shuffle=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.graph_batch_size, shuffle=True
        )
    print("check 4", flush=True)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs * len(dataloader), eta_min=3e-6)
    optimizer.zero_grad()
    scheduler.step(0)

    # training loop
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "start training...", flush=True)
    model.train()
    prev_loss = np.float32("inf")
    for epoch in range(args.n_epochs):
        losses = []
        for batch in tqdm(dataloader):
            # perform edge neighborhood sampling to generate training graph and data
            if args.supervised == "True":
                subg, samples, matched_labels, matched_index = \
                    utils.generate_sampled_graph_and_labels_supervised(
                        graph, batch, args.sampler, args.sample_depth, args.sample_width,
                        args.graph_split_size,
                        train_indices, train_labels, multi, nlabels, ntrain,
                        if_train=True, label_batch_size=args.label_batch_size
                    )
                if multi:
                    matched_labels = torch.from_numpy(matched_labels).float()
                else:
                    matched_labels = torch.from_numpy(matched_labels).long()
                if use_cuda:
                    matched_labels = matched_labels.to("cuda:%d" % (args.gpu))
            elif args.supervised == "False":
                subg, samples, labels = \
                    utils.generate_sampled_graph_and_labels_unsupervised(
                        graph, batch, args.sampler, args.sample_depth, args.sample_width,
                        args.graph_split_size, args.negative_sample
                    )
                samples = torch.from_numpy(samples)
                labels = torch.from_numpy(labels)
                if use_cuda:
                    samples = samples.to("cuda:%d" % (args.gpu))
                    labels = labels.to("cuda:%d" % (args.gpu))
            else:
                raise ValueError

            # calculate norms and eigenvalues of the subgraph
            edge_norm = utils.compute_edgenorm(subg)
            if use_cuda:
                subg = subg.to("cuda:%d" % (args.gpu))
                edge_norm = edge_norm.to("cuda:%d" % (args.gpu))
            edge_type = subg.edata["type"]

            embed, pred = model(subg, h=subg.ndata[dgl.NID], edge_type=edge_type, edge_norm=edge_norm)

            if args.supervised == "True":
                loss = model.get_supervised_loss(subg, embed, edge_type, pred, matched_labels, matched_index, multi)
            elif args.supervised == "False":
                loss = model.get_unsupervised_loss(subg, embed, edge_type, samples, labels)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        loss = sum(losses) / len(losses)

        print(
            time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) +
            "Epoch {:05d} | Loss {:.4f}".format(epoch, loss),
            flush=True
        )
        if loss > prev_loss:
            break
        prev_loss = loss

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "training done", flush=True)

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "start output...", flush=True)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.graph_batch_size * 4, shuffle=False)
    model.eval()
    with torch.no_grad():
        node_emb, node_sampled = model.model.node_emb.weight.detach().cpu().clone(), set()
        for batch in tqdm(dataloader):
            subg, samples, labels = \
                utils.generate_sampled_graph_and_labels_unsupervised(
                    graph, batch, args.sampler, args.sample_depth, args.sample_width,
                    args.graph_split_size, args.negative_sample
                )

            # calculate norms and eigenvalues of the subgraph
            edge_norm = utils.compute_edgenorm(subg)
            nid = subg.ndata[dgl.NID]
            coef = (subg.ndata["in_deg"].float() + 1) / (graph.ndata["in_deg"][nid].float() + 1)
            coef = coef.view(-1, 1)
            if use_cuda:
                subg = subg.to("cuda:%d" % (args.gpu))
                edge_norm = edge_norm.to("cuda:%d" % (args.gpu))
            edge_type = subg.edata["type"]

            embed, pred = model(subg, h=subg.ndata[dgl.NID], edge_type=edge_type, edge_norm=edge_norm)

            node_emb[nid] = node_emb[nid] * (1 - coef) + embed[0].detach().cpu() * coef
            # node_emb[nid].data.copy_(embed[0].detach().cpu())
            node_sampled.update(nid.numpy())

    print("{:5}% node embeddings are saved.".format(len(node_sampled) * 100 / num_nodes))
    if len(seed_nodes) > 0:
        seed_nodes = np.array(sorted(seed_nodes))
        utils.save(args, node_emb[seed_nodes].numpy(), index=seed_nodes)
    else:
        utils.save(args, node_emb.numpy())

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CompGCN")
    parser.add_argument(
        "--link", type=str, required=True,
        help="dataset to use"
    )
    parser.add_argument(
        "--node", type=str, required=True,
        help="dataset to use"
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="dataset to use"
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output embedding file"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2,
        help="dropout probability"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=50,
        help="number of hidden units"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="gpu"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="learning rate"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2,
        help="number of propagation rounds"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=2000,
        help="number of minimum training epochs"
    )
    parser.add_argument(
        "--regularization", type=float, default=0.01,
        help="regularization weight"
    )
    parser.add_argument(
        "--grad-norm", type=float, default=1.0,
        help="norm to clip gradient to"
    )
    parser.add_argument(
        "--label-batch-size", type=int, default=512
    )
    parser.add_argument(
        "--graph-batch-size", type=int, default=20000,
        help="number of edges to sample in each iteration"
    )
    parser.add_argument(
        "--graph-split-size", type=float, default=0.5,
        help="portion of edges used as positive sample"
    )
    parser.add_argument(
        "--negative-sample", type=int, default=5,
        help="number of negative samples per positive sample"
    )
    parser.add_argument(
        "--sampler", type=str, default="neighbor",
        help="type of subgraph sampler: neighbor or randomwalk"
    )
    parser.add_argument(
        "--sample-depth", type=int, default=6
    )
    parser.add_argument(
        "--sample-width", type=int, default=128
    )
    parser.add_argument(
        "--attributed", type=str, default="False"
    )
    parser.add_argument(
        "--supervised", type=str, default="False"
    )

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
