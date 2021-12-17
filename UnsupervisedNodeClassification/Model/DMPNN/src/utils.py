import dgl
from numpy.random.mtrand import seed
import torch
import numpy as np
import numba
from collections import defaultdict


if torch.__version__ < "1.7.0":

    def rfft(input, n=None, dim=-1, norm=None):
        # no image part
        inp_dim = input.dim()

        if dim < 0:
            dim = inp_dim + dim
        if n is not None:
            diff = input.size(dim) - n
            if diff > 0:
                input = torch.split(input, dim=dim, split_size_or_sections=(n, diff))[0]
            # else:
            #     sizes = tuple(input.size())
            #     padded = torch.zeros((sizes[:dim] + (-diff, ) + sizes[(dim+1):]), dtype=input.dtype, device=input.device)
            #     input = torch.cat([input, padded], dim=dim)
        else:
            n = input.size(dim) // 2 + 1
        if norm is None or norm == "backward":
            normalized = False
        elif norm == "forward":
            normalized = True
        else:
            raise ValueError

        if dim != inp_dim - 1:
            input = input.transpose(dim, inp_dim - 1)
        output = torch.rfft(input, signal_ndim=1, normalized=normalized)
        if dim != inp_dim - 1:
            output = output.transpose(dim, inp_dim - 1)

        return output

    def irfft(input, n=None, dim=-1, norm=None):
        # calculate the dimension of the input and regard the last as the (real, image)
        inp_dim = input.dim()
        if input.size(-1) != 2:
            input = torch.stack([input, torch.zeros_like(input)], dim=-1)
        else:
            inp_dim -= 1

        if dim < 0:
            dim = inp_dim + dim
        if n is not None:
            diff = input.size(dim) - n
            if diff > 0:
                input = torch.split(input, dim=dim, split_size_or_sections=(n, diff))[0]
            # else:
            #     sizes = tuple(input.size())
            #     padded = torch.zeros((sizes[:dim] + (-diff, ) + sizes[(dim+1):]), dtype=input.dtype, device=input.device)
            #     input = torch.cat([input, padded], dim=dim)
        else:
            n = 2 * (input.size(dim) - 1)
        if norm is None or norm == "backward":
            normalized = False
        elif norm == "forward":
            normalized = True
        else:
            raise ValueError

        if dim != inp_dim - 1:
            input = input.transpose(dim, inp_dim - 1)
        output = torch.irfft(input, signal_ndim=1, normalized=normalized, signal_sizes=[n])
        if dim != inp_dim - 1:
            output = output.transpose(dim, inp_dim - 1)

        return output
else:

    def rfft(input, n=None, dim=None, norm=None):
        return torch.view_as_real(torch.fft.rfft(input, n=n, dim=dim, norm=norm))

    def irfft(input, n=None, dim=None, norm=None):
        if not torch.is_complex(input) and input.size(-1) == 2:
            input = torch.view_as_complex(input)
        return torch.fft.irfft(input, n=n, dim=dim, norm=norm)


def complex_mul(re_x, im_x, re_y, im_y):
    return (re_x * re_y - im_x * im_y), (im_x * re_y + re_x * im_y)


def complex_conj(re_x, im_x):
    return re_x, -im_x



def uniform_choice_int(N, n):
    return np.random.randint(0, N, size=(n,))


def uniform_choice(array, n):
    index = np.random.randint(0, len(array), size=(n,))
    return array[index]


@numba.jit(numba.int64[:](numba.int64[:], numba.int64), nopython=True)
def _get_enc_len(x, base=10):
    lens = np.zeros((len(x), ), dtype=np.int64)
    for i, n in enumerate(x):
        cnt = 0
        while n > 0:
            n = n // base
            cnt += 1
        lens[i] = cnt

    return lens


def get_enc_len(x, base=10):
    if isinstance(x, int):
        return _get_enc_len(np.array([x], dtype=np.int64), base)[0]
    elif isinstance(x, float):
        return _get_enc_len(np.array([int(x)], dtype=np.int64), base)[0]
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x.astype(np.int64)
    x_shape = x.shape

    return _get_enc_len(x.reshape(-1), base).reshape(*x_shape)


@numba.jit(
    numba.int64[:, :](numba.int64[:], numba.int64, numba.int64),
    nopython=True,
    nogil=True
)
def _int2multihot(x, len_x, base):
    rep = np.zeros((len(x), len_x * base), dtype=np.int64)
    for i, n in enumerate(x):
        n = n % base**len_x
        idx = (len_x - 1) * base
        while n:
            rep[i, idx + n % base] = 1
            n = n // base
            idx -= base
        while idx >= 0:
            rep[i, idx] = 1
            idx -= base
    return rep


def int2multihot(x, len_x, base=10):
    if isinstance(x, int):
        return _int2multihot(np.array([x], dtype=np.int64), len_x, base)[0]
    elif isinstance(x, float):
        return _int2multihot(np.array([int(x)], dtype=np.int64), len_x, base)[0]
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x.astype(np.int64)

    return _int2multihot(x, len_x, base)



def load_supervised(args, link, node, train_pool):
    num_nodes, num_rels, train_data = 0, 0, []
    train_indices = defaultdict(list)
    with open(link, "r") as file:
        for index, line in enumerate(file):
            if index == 0:
                num_nodes, num_rels = line[:-1].split(" ")
                num_nodes, num_rels = int(num_nodes), int(num_rels)
                print(f"#nodes: {num_nodes}, #relations: {num_rels}")
            else:
                line = np.array(line[:-1].split(" ")).astype(np.int64)
                train_data.append(line)
                if line[0] in train_pool:
                    train_indices[line[0]].append(index - 1)
                if line[-1] in train_pool:
                    train_indices[line[-1]].append(index - 1)

    if args.attributed == "True":
        node_attri = {}
        with open(node, "r") as file:
            for line in file:
                line = line[:-1].split("\t")
                node_attri[int(line[0])] = np.array(line[1].split(",")).astype(np.float32)
        return np.array(train_data), num_nodes, num_rels, train_indices, len(train_indices), np.array(
            [node_attri[k] for k in range(len(node_attri))]
        ).astype(np.float32)
    elif args.attributed == "False":
        return np.array(train_data), num_nodes, num_rels, train_indices, len(train_indices), None


def load_label(train_label):
    train_pool, train_labels, all_labels, multi = set(), {}, set(), False
    with open(train_label, "r") as file:
        for line in file:
            node, label = line[:-1].split("\t")
            node = int(node)
            train_pool.add(node)
            if multi or "," in label:
                multi = True
                label = np.array(label.split(",")).astype(np.int64)
                for each in label:
                    all_labels.add(label)
                train_labels[node] = label
            else:
                label = int(label)
                train_labels[node] = label
                all_labels.add(label)

    return train_pool, train_labels, len(all_labels), multi


def load_unsupervised(args, link, node):
    num_nodes, num_rels, train_data = 0, 0, []
    with open(link, "r") as file:
        for index, line in enumerate(file):
            if index == 0:
                num_nodes, num_rels = line[:-1].split(" ")
                num_nodes, num_rels = int(num_nodes), int(num_rels)
                print(f"#nodes: {num_nodes}, #relations: {num_rels}")
            else:
                line = np.array(line[:-1].split(" ")).astype(np.int64)
                train_data.append(line)

    if args.attributed == "True":
        node_attri = {}
        with open(node, "r") as file:
            for line in file:
                line = line[:-1].split("\t")
                node_attri[int(line[0])] = np.array(line[1].split(",")).astype(np.float32)
        return np.array(train_data), num_nodes, num_rels, np.array([node_attri[k] for k in range(len(node_attri))]
                                                                  ).astype(np.float32)
    elif args.attributed == "False":
        return np.array(train_data), num_nodes, num_rels, None


def save(args, embs, index=None):
    with open(f"{args.output}", "w") as file:
        file.write(str(args))
        file.write("\n")
        if index is None:
            for n, emb in enumerate(embs):
                file.write(f"{n}\t")
                file.write(" ".join(emb.astype(str)))
                file.write("\n")
        else:
            for n, emb in zip(index, embs):
                file.write(f"{n}\t")
                file.write(" ".join(emb.astype(str)))
                file.write("\n")

    return


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################


def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    degrees = np.zeros(num_nodes).astype(np.int64)
    for i, triplet in enumerate(triplets):
        degrees[triplet[0]] += 1
        degrees[triplet[2]] += 1

    return degrees


def sample_subgraph_by_randomwalks(graph, seed_nodes, depth=2, width=10):
    if isinstance(seed_nodes, torch.Tensor):
        nodes = [seed_nodes]
        seed_nodes = set(seed_nodes.numpy())
    elif isinstance(seed_nodes, np.ndarray):
        nodes = [torch.from_numpy(seed_nodes)]
        seed_nodes = set(seed_nodes)
    else:
        nodes = [torch.tensor(seed_nodes)]
        seed_nodes = set(seed_nodes)

    for i in range(width - 1):
        traces, types = dgl.sampling.random_walk(graph, nodes[0], length=depth)
        nodes.append(dgl.sampling.pack_traces(traces, types)[0])
    nodes = torch.unique(torch.cat(nodes))
    subg = dgl.sampling.sample_neighbors(
        graph, nodes, width, edge_dir="in", copy_ndata=True, copy_edata=True
    )

    # remove nodes
    in_deg = subg.in_degrees().float()
    out_deg = subg.out_degrees().float()
    deg = in_deg + out_deg
    del_nids = torch.LongTensor(sorted(set(subg.ndata[dgl.NID][deg == 0].numpy()) - seed_nodes))
    subg.remove_nodes(del_nids)

    # del deg and norm
    for k in ["in_deg", "out_deg", "norm"]:
        if k in subg.ndata:
            subg.ndata.pop(k)
    for k in ["in_deg", "out_deg", "norm"]:
        if k in subg.edata:
            subg.edata.pop(k)
    return subg


def sample_subgraph_by_neighbors(graph, seed_nodes, depth=2, width=10):
    if isinstance(seed_nodes, torch.Tensor):
        nodes = seed_nodes
        seed_nodes = set(seed_nodes.numpy())
    elif isinstance(seed_nodes, np.ndarray):
        nodes = torch.from_numpy(seed_nodes)
        seed_nodes = set(seed_nodes)
    else:
        nodes = torch.tensor(seed_nodes)
        seed_nodes = set(seed_nodes)
    for i in range(depth - 1):
        subg = dgl.sampling.sample_neighbors(
            graph, nodes, width, edge_dir="in", copy_ndata=True, copy_edata=False
        )
        mask = (subg.ndata["out_deg"] > 0)
        nodes = torch.unique(torch.cat([nodes, subg.ndata[dgl.NID][mask]]))
    subg = dgl.sampling.sample_neighbors(
        graph, nodes, width, edge_dir="in", copy_ndata=True, copy_edata=True
    )

    # remove nodes
    in_deg = subg.in_degrees().float()
    out_deg = subg.out_degrees().float()
    deg = in_deg + out_deg
    del_nids = torch.LongTensor(sorted(set(subg.ndata[dgl.NID][deg == 0].numpy()) - seed_nodes))
    subg.remove_nodes(del_nids)

    # del deg and norm
    for k in ["in_deg", "out_deg", "norm"]:
        if k in subg.ndata:
            subg.ndata.pop(k)
    for k in ["in_deg", "out_deg", "norm"]:
        if k in subg.edata:
            subg.edata.pop(k)
    return subg


def generate_sampled_graph_and_labels_supervised(
    graph,
    edges,
    sampler,
    sample_depth,
    sample_width,
    split_size,
    train_indices,
    train_labels,
    multi,
    nlabel,
    ntrain,
    if_train=True,
    label_batch_size=512,
    batch_index=0
):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    labeled_samples, sampled_nodes = labeled_edges_sampling(edges, train_indices, ntrain, if_train, label_batch_size, batch_index)

    seed_nodes = np.unique(np.concatenate([edges[:, 0], edges[:, 2], labeled_samples[:, 0], labeled_samples[:, 2]]))
    if multi:
        matched_labels, matched_index = correct_order_multi(seed_nodes, sampled_nodes, train_labels, nlabel)
    else:
        matched_labels, matched_index = correct_order_single(seed_nodes, sampled_nodes, train_labels)

    seed_nodes = np.unique(np.concatenate([edges[:, 0], edges[:, 2], labeled_samples[:, 0], labeled_samples[:, 2]]))

    if sampler == "neighbor":
        subg = sample_subgraph_by_neighbors(graph, torch.from_numpy(seed_nodes), sample_depth, sample_width)
    elif sampler == "randomwalk":
        subg = sample_subgraph_by_randomwalks(graph, torch.from_numpy(seed_nodes), sample_depth, sample_width)

    samples = np.concatenate([edges, labeled_samples])
    samples[:, 0] = convert_subgraph_nids(samples[:, 0], subg.ndata[dgl.NID].numpy())
    samples[:, 2] = convert_subgraph_nids(samples[:, 2], subg.ndata[dgl.NID].numpy())

    # randomly delete edges from subgraphs
    if split_size < 1.0:
        del_eids = np.unique(uniform_choice_int(subg.number_of_edges(), int(subg.number_of_edges() * (1 - split_size))))
        subg.remove_edges(del_eids)

    return subg, samples, matched_labels, matched_index


def generate_sampled_graph_and_labels_unsupervised(
    graph,
    edges,
    sampler,
    sample_depth,
    sample_width,
    split_size,
    negative_rate
):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # negative sampling
    neg_samples = negative_sampling(edges, graph.number_of_nodes(), negative_rate)

    seed_nodes = np.unique(np.concatenate([edges[:, 0], edges[:, 2], neg_samples[:, 0], neg_samples[:, 2]]))
    
    if sampler == "neighbor":
        subg = sample_subgraph_by_neighbors(graph, torch.from_numpy(seed_nodes), sample_depth, sample_width)
    elif sampler == "randomwalk":
        subg = sample_subgraph_by_randomwalks(graph, torch.from_numpy(seed_nodes), sample_depth, sample_width)

    samples = np.concatenate([edges, neg_samples])
    samples[:, 0] = convert_subgraph_nids(samples[:, 0], subg.ndata[dgl.NID].numpy())
    samples[:, 2] = convert_subgraph_nids(samples[:, 2], subg.ndata[dgl.NID].numpy())

    # randomly delete edges from subgraphs
    if split_size < 1.0:
        del_eids = np.unique(uniform_choice_int(subg.number_of_edges(), int(subg.number_of_edges() * (1 - split_size))))
        subg.remove_edges(del_eids)

    labels = np.zeros((len(samples)), dtype=np.float32)
    labels[:len(edges)].fill(1.0)

    return subg, samples, labels


def compute_edgenorm(g, norm="in"):
    if "in_deg" not in g.ndata:
        g.ndata["in_deg"] = g.in_degrees()
    if "out_deg" not in g.ndata:
        g.ndata["out_deg"] = g.out_degrees()
    in_deg = g.ndata["in_deg"].float()
    out_deg = g.ndata["out_deg"].float()
    u, v = g.all_edges(form="uv", order="eid")
    if norm == "in":
        norm = in_deg[v].reciprocal().unsqueeze(-1)
    elif norm == "out":
        norm = out_deg[u].reciprocal().unsqueeze(-1)
    elif norm == "both":
        norm = torch.pow(out_deg[u] * in_deg[v], 0.5).reciprocal().unsqueeze(-1)
    norm.masked_fill_(torch.isnan(norm), norm.min())
    norm.masked_fill_(torch.isinf(norm), norm.min())
    return norm


def compute_largest_eigenvalues(g):
    if "in_deg" not in g.ndata:
        g.ndata["in_deg"] = g.in_degrees()
    if "out_deg" not in g.ndata:
        g.ndata["out_deg"] = g.out_degrees()
    u, v = g.all_edges(form="uv", order="eid")
    in_deg = g.ndata["in_deg"].float()
    out_deg = g.ndata["out_deg"].float()
    max_nd = (out_deg[u] + in_deg[v]).max()
    max_ed = (in_deg[u] + out_deg[v]).max()

    node_eigenv = max_nd
    edge_eigenv = max_ed

    return node_eigenv, edge_eigenv


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    triplets = triplets.copy()
    triplets.view(
        [("src", triplets.dtype), ("rel", triplets.dtype), ("dst", triplets.dtype)]
    ).sort(axis=0, order=["src", "dst", "rel"])
    g.add_edges(triplets[:, 0], triplets[:, 2])
    g.add_edges(triplets[:, 2], triplets[:, 0])
    rel = np.concatenate([triplets[:, 1], triplets[:, 1] + num_rels])
    g.edata["type"] = torch.from_numpy(rel).long()
    g.edata["norm"] = compute_edgenorm(g)
    return g


def labeled_edges_sampling(edges, train_indices, ntrain, if_train, label_batch_size, batch_index=0):
    if if_train:
        sampled_index = set(uniform_choice_int(ntrain, label_batch_size))
    else:
        sampled_index = set(
            np.arange(batch_index * label_batch_size, min(ntrain, (batch_index + 1) * label_batch_size))
        )

    new_edges, sampled_nodes = [], set()
    for index, (labeled_node, node_edges) in enumerate(train_indices.items()):
        if index in sampled_index:
            sampled_nodes.add(labeled_node)
            new_edges.append(np.array(node_edges))
    new_edges = np.unique(np.concatenate(new_edges))

    return new_edges, sampled_nodes


@numba.jit(
    nopython=True
)
def correct_order_single(node_id, sampled_nodes, train_labels):
    matched_labels, matched_index = [], []
    for index, each in enumerate(node_id):
        if each in sampled_nodes:
            matched_labels.append(train_labels[each])
            matched_index.append(index)

    return np.array(matched_labels), np.array(matched_index)

@numba.jit(
    nopython=True
)
def correct_order_multi(node_id, sampled_nodes, train_labels, nlabel):
    matched_labels, matched_index = [], []
    for index, each in enumerate(node_id):
        if each in sampled_nodes:
            curr_label = np.zeros(nlabel, dtype=np.int64)
            curr_label[train_labels[each]] = 1
            matched_labels.append(curr_label)
            matched_index.append(index)

    return np.array(matched_labels), np.array(matched_index)


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    # 0~i-1, i+1~N-1
    values = np.random.randint(num_entity - 1, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj] + (values[subj] >= neg_samples[subj, 0])
    neg_samples[obj, 2] = values[obj] + (values[obj] >= neg_samples[obj, 2])

    return neg_samples


@numba.jit(
    numba.int64[:](numba.int64[:], numba.int64[:]), nopython=True
)
def convert_subgraph_nids(ori_nids, subg_nids):
    rev_map = dict()
    for i, j in enumerate(subg_nids):
        rev_map[j] = i
    mapped_nids = np.zeros(ori_nids.shape, dtype=np.int64)
    for i, j in enumerate(ori_nids):
        mapped_nids[i] = rev_map[j]
    return mapped_nids
