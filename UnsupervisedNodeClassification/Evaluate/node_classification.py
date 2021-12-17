import warnings
import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from utils import SingleLabelBinarySeachCV, MultiLabelBinarySeachCV


seed = 1
max_iter = 300
np.random.seed(seed)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def nc_evaluate(dataset, supervised, label_file_path, label_test_path, emb_dict):

    if supervised == 'True':
        if dataset == 'Yelp':
            return semisupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict)
        elif dataset == 'DBLP' or dataset == 'Freebase' or dataset == 'PubMed':
            return semisupervised_single_class_single_label(label_file_path, label_test_path, emb_dict)
    elif supervised == 'False':
        if dataset == 'Yelp':
            return unsupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict)
        elif dataset == 'DBLP' or dataset == 'Freebase' or dataset == 'PubMed':
            return unsupervised_single_class_single_label(label_file_path, label_test_path, emb_dict)


def semisupervised_single_class_single_label(label_file_path, label_test_path, emb_dict):

    train_labels, train_embeddings = [], []
    with open(label_file_path, 'r') as label_file:
        for line in label_file:
            index, _, _, label = line.strip().split('\t')
            train_labels.append(label)
            train_embeddings.append(emb_dict[index])
    train_labels, train_embeddings = np.array(train_labels).astype(int), np.array(train_embeddings)

    test_labels, test_embeddings = [], []
    with open(label_test_path, 'r') as label_file:
        for line in label_file:
            index, _, _, label = line.strip().split('\t')
            test_labels.append(label)
            test_embeddings.append(emb_dict[index])
    test_labels, test_embeddings = np.array(test_labels).astype(int), np.array(test_embeddings)

    c = SingleLabelBinarySeachCV(train_embeddings, train_labels, multi_class="ovr")
    clf = LinearSVC(random_state=seed, max_iter=max_iter, multi_class="ovr", C=c)
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)

    macro = f1_score(test_labels, preds, average='macro')
    micro = f1_score(test_labels, preds, average='micro')

    return macro, micro


def unsupervised_single_class_single_label(label_file_path, label_test_path, emb_dict):

    labels, embeddings = [], []
    for file_path in [label_file_path, label_test_path]:
        with open(file_path, 'r') as label_file:
            for line in label_file:
                index, _, _, label = line.strip().split('\t')
                labels.append(label)
                embeddings.append(emb_dict[index])
    labels, embeddings = np.array(labels).astype(int), np.array(embeddings)

    macro, micro = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(embeddings, labels):
        c = SingleLabelBinarySeachCV(embeddings[train_idx], labels[train_idx], multi_class="ovr")
        clf = LinearSVC(random_state=seed, max_iter=max_iter, multi_class="ovr", C=c)
        clf.fit(embeddings[train_idx], labels[train_idx])
        preds = clf.predict(embeddings[test_idx])

        macro.append(f1_score(labels[test_idx], preds, average='macro'))
        micro.append(f1_score(labels[test_idx], preds, average='micro'))
    print(macro)
    print(micro)
    return np.mean(macro), np.mean(micro)


def semisupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict):

    train_embs = []
    with open(label_file_path, 'r') as label_file:
        for line in label_file:
            index, _, nclass, label = line.strip().split('\t')
            train_embs.append(emb_dict[index])
    train_embs = np.array(train_embs)

    test_embs = []
    with open(label_test_path, 'r') as label_file:
        for line in label_file:
            index, _, nclass, label = line.strip().split('\t')
            test_embs.append(emb_dict[index])
    test_embs = np.array(test_embs)

    label_count, labled_node_index = 0, 0
    binary_labels, label_dict, = [], {}
    with open(label_file_path, 'r') as label_file:
        for line in label_file:
            index, _, nclass, label = line.strip().split('\t')
            for each in label.split(','):
                if (nclass, each) not in label_dict:
                    label_dict[(nclass, each)] = label_count
                    label_count += 1
                    binary_labels.append(np.zeros(len(train_embs), dtype=np.bool8))
                binary_labels[label_dict[(nclass, each)]][labled_node_index] = 1
            labled_node_index += 1
    train_labels = np.vstack(binary_labels)

    label_count, labled_node_index = 0, 0
    binary_labels, label_dict, = [], {}
    with open(label_test_path, 'r') as label_file:
        for line in label_file:
            index, _, nclass, label = line.strip().split('\t')
            for each in label.split(','):
                if (nclass, each) not in label_dict:
                    label_dict[(nclass, each)] = label_count
                    label_count += 1
                    binary_labels.append(np.zeros(len(test_embs), dtype=np.bool8))
                binary_labels[label_dict[(nclass, each)]][labled_node_index] = 1
            labled_node_index += 1
    test_labels = np.vstack(binary_labels)

    weights, total_scores = [], []
    for ntype, (train_label, test_label) in enumerate(zip(train_labels, test_labels)):
        c = MultiLabelBinarySeachCV(train_embs, train_label, multi_class="crammer_singer")
        clf = LinearSVC(random_state=seed, max_iter=max_iter, C=c, multi_class="crammer_singer")
        clf.fit(train_embs, train_label)
        preds = clf.predict(test_embs)
        scores = f1_score(test_label, preds, average='binary')
        weights.append(sum(test_label))
        total_scores.append(scores)

    macro = sum(total_scores) / len(total_scores)
    micro = sum([score * weight for score, weight in zip(total_scores, weights)]) / sum(weights)

    return macro, micro


def unsupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict):
    embs = []
    for file_path in [label_file_path, label_test_path]:
        with open(file_path, 'r') as label_file:
            for line in label_file:
                index, _, nclass, label = line.strip().split('\t')
                embs.append(emb_dict[index])
    embs = np.array(embs)

    label_count, labled_node_index = 0, 0
    binary_labels, label_dict, = [], {}
    for file_path in [label_file_path, label_test_path]:
        with open(file_path, 'r') as label_file:
            for line in label_file:
                index, _, nclass, label = line.strip().split('\t')
                for each in label.split(','):
                    if (nclass, each) not in label_dict:
                        label_dict[(nclass, each)] = label_count
                        label_count += 1
                        binary_labels.append(np.zeros(len(embs), dtype=np.int32))
                    binary_labels[label_dict[(nclass, each)]][labled_node_index] = 1
                labled_node_index += 1
    binary_labels = np.vstack(binary_labels)

    cs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(embs, binary_labels[binary_labels.sum(axis=1).argmax()]):
        c = MultiLabelBinarySeachCV(embs[train_idx], binary_labels[:, train_idx], multi_class="crammer_singer")
        cs.append(c)
    weights, total_scores = [], []
    for ntype, binary_label in enumerate(binary_labels):
        scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for nsplit, train_idx, test_idx in enumerate(skf.split(embs, binary_label)):
            clf = LinearSVC(random_state=seed, max_iter=max_iter, C=cs[nsplit], multi_class="crammer_singer")
            clf.fit(embs[train_idx], binary_label[train_idx])
            preds = clf.predict(embs[test_idx])
            scores.append(f1_score(binary_label[test_idx], preds, average='binary'))

        weights.append(sum(binary_label))
        total_scores.append(sum(scores) / 5)

    macro = sum(total_scores) / len(total_scores)
    micro = sum([score * weight for score, weight in zip(total_scores, weights)]) / sum(weights)

    return macro, micro
