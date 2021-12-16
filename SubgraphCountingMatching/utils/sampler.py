import copy
import math
import numpy as np
import torch as th

from collections import OrderedDict
from torch.utils.data import Sampler


class BucketSampler(Sampler):
    def __init__(
        self,
        dataset,
        group_by,
        batch_size,
        shuffle=False,
        seed=0,
        drop_last=False
    ):
        super(BucketSampler, self).__init__(dataset)
        self.dataset = dataset
        if isinstance(group_by, str):
            group_by = [group_by]
        self.group_by = group_by
        self.cache = OrderedDict()
        for attr in group_by:
            self.cache[attr] = th.tensor([x[attr] for x in dataset], dtype=th.float32)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        if drop_last:
            self.num_samples = math.ceil((len(self.dataset) - self.batch_size) / self.batch_size) * self.batch_size
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.batch_size) * self.batch_size
        self.total_size = self.num_samples

    def __iter__(self):
        rng = th.Generator()
        rng.manual_seed(self.seed + self.epoch)
        array = th.stack(list(self.cache.values()), dim=-1)

        if not self.drop_last:
            ind = th.arange(len(self.dataset))
            padding_size = self.total_size - len(self.dataset)
            while padding_size > len(array):
                ind = th.cat([ind, ind], dim=0)
                padding_size -= len(array)
            if padding_size > 0:
                ind = th.cat([ind, th.randperm(len(self.dataset))[:padding_size]], dim=0)
            array = array[ind]
        else:
            ind = th.arange(self.total_size)
            array = array[:self.total_size]
        assert len(array) == self.total_size

        rand = th.rand((self.total_size, 1), generator=rng)
        array = th.cat([array, rand], dim=-1)
        array = array.numpy().view(list(zip(list(self.cache.keys()) + ["rand"],
                                            [np.float32] * (len(self.cache) + 1)))).flatten()
        indices = np.argsort(array, axis=0, order=self.group_by)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        if self.shuffle:
            indices = th.randperm(len(batches), generator=rng)
            batches = batches[indices]

        batch_idx = 0
        while batch_idx < len(batches):
            yield ind[batches[batch_idx]]
            batch_idx += 1

    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch=-1):
        if epoch == -1:
            self.epoch += 1
        else:
            self.epoch = epoch


class CircurriculumSampler(BucketSampler):
    def __init__(
        self,
        dataset,
        learning_by,
        used_ratio,
        batch_size,
        group_by=None,
        shuffle=False,
        seed=0,
        drop_last=False
    ):
        if isinstance(learning_by, str):
            learning_by = [learning_by]
        if isinstance(group_by, str):
            group_by = [group_by]
        elif group_by is None:
            group_by = learning_by
        super(CircurriculumSampler, self).__init__(dataset, group_by, batch_size, shuffle, seed, drop_last)
        self.learning_by = learning_by
        for attr in learning_by:
            if attr not in self.cache:
                self.cache[attr] = th.tensor([x[attr] for x in dataset], dtype=th.float32)

        self.used_ratio = used_ratio

    def __iter__(self):
        rng = th.Generator()
        rng.manual_seed(self.seed + self.epoch)
        array = th.stack(list(self.cache.values()), dim=-1)

        if not self.drop_last:
            ind = th.arange(len(self.dataset))
            padding_size = self.total_size - len(self.dataset)
            while padding_size > len(array):
                ind = th.cat([ind, ind], dim=0)
                padding_size -= len(array)
            if padding_size > 0:
                ind = th.cat([ind, th.randperm(len(self.dataset))[:padding_size]], dim=0)
            array = array[ind]
        else:
            ind = th.arange(self.total_size)
            array = array[:self.total_size]
        assert len(array) == self.total_size

        rand = th.rand((self.total_size, 1), generator=rng)
        array = th.cat([array, rand], dim=-1)
        array = array.numpy().view(list(zip(list(self.cache.keys()) + ["rand"],
                                            [np.float32] * (len(self.cache) + 1)))).flatten()

        if self.learning_by == self.group_by or self.learning_by == self.group_by[:len(self.learning_by)]:
            group_indices = np.argsort(array, axis=0, order=self.group_by)
            indices = group_indices[:math.ceil(self.used_ratio * len(group_indices))]
        else:
            learn_indices = np.argsort(array, axis=0, order=self.learning_by)
            learn_indices = learn_indices[:int(self.used_ratio * len(learn_indices))]
            indices = np.argsort(array[learn_indices], axis=0, order=self.group_by)

        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        if self.shuffle:
            batches = [batches[i] for i in th.randperm(len(batches), generator=rng).tolist()]

        batch_idx = 0
        while batch_idx < len(batches):
            yield ind[batches[batch_idx]]
            batch_idx += 1
