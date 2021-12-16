import warnings
import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

seed = 1
max_iter = 100
np.random.seed(seed)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def SingleLabelBinarySeachCV(data, labels, multi_class="ovr"):
    best_c = 1.0
    c0 = np.power(10.0, -(labels.max() - labels.min() + 1))
    c1 = 1 / c0
    cnt = 0
    max_cnt = 2 * (labels.max() - labels.min() + 1) - 1
    while cnt < max_cnt and np.abs(c0 - c1) > 1e-10:
        np.random.seed(cnt)
        index = np.random.choice(len(data), size=(int(len(data) * (cnt+1) / max_cnt), ), replace=False)
        cur_data, cur_labels = data[index], labels[index]
        clf0 = LinearSVC(random_state=seed, max_iter=int(max_iter * (cnt+1) / max_cnt), multi_class=multi_class, C=c0)
        clf0.fit(cur_data, cur_labels)
        preds0 = clf0.predict(cur_data)
        macro0 = f1_score(cur_labels, preds0, average='macro')
        micro0 = f1_score(cur_labels, preds0, average='micro')

        clf1 = LinearSVC(random_state=seed, max_iter=int(max_iter * (cnt+1) / max_cnt), multi_class=multi_class, C=c1)
        clf1.fit(cur_data, cur_labels)
        preds1 = clf1.predict(cur_data)
        macro1 = f1_score(cur_labels, preds1, average='macro')
        micro1 = f1_score(cur_labels, preds1, average='micro')

        if macro0 + micro0 > macro1 + micro1:
            best_c = c0
            c1 /= 10
        else:
            best_c = c1
            c0 *= 10
        cnt += 1
    return best_c


def MultiLabelBinarySeachCV(data, labels, multi_class="crammer_singer"):
    best_c = 1.0
    c0 = np.power(10.0, -len(labels))
    c1 = 1 / c0
    cnt = 0
    max_cnt = 2 * len(labels) - 1
    while cnt < max_cnt and np.abs(c0 - c1) > 1e-10:
        np.random.seed(cnt)
        index = np.random.choice(len(data), size=(int(len(data) * (cnt+1) / max_cnt), ), replace=False)
        cur_data, cur_labels = data[index], labels[:, index]
        weights0 = np.zeros((len(cur_data)), dtype=np.float32)
        scores0 = np.zeros((len(cur_data)), dtype=np.float32)
        for ntype, nlabels in enumerate(cur_labels):
            clf0 = LinearSVC(random_state=seed, max_iter=int(max_iter * (cnt+1) / max_cnt), multi_class="crammer_singer", C=c0)
            clf0.fit(cur_data, nlabels)
            preds0 = clf0.predict(cur_data)
            scores0[ntype] = f1_score(nlabels, preds0, average='binary')
            weights0[ntype] = sum(nlabels)
        macro0 = scores0.sum() / scores0.shape[0]
        micro0 = (scores0 * weights0 / weights0.sum()).sum()

        weights1 = np.zeros((len(cur_data)), dtype=np.float32)
        scores1 = np.zeros((len(cur_data)), dtype=np.float32)
        for ntype, nlabels in enumerate(cur_labels):
            clf1 = LinearSVC(random_state=seed, max_iter=int(max_iter * (cnt+1) / max_cnt), multi_class="crammer_singer", C=c1)
            clf1.fit(cur_data, nlabels)
            preds1 = clf1.predict(cur_data)
            scores1[ntype] = f1_score(nlabels, preds1, average='binary')
            weights1[ntype] = sum(nlabels)
        macro1 = scores1.sum() / scores1.shape[0]
        micro1 = (scores1 * weights1 / weights1.sum()).sum()

        if macro0 + micro0 > macro1 + micro1:
            best_c = c0
            c1 /= 10
        else:
            best_c = c1
            c0 *= 10
        cnt += 1
    return best_c
