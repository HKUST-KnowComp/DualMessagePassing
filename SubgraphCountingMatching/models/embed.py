
import numba
import numpy as np
import torch as th
import torch.nn as nn


@numba.jit(numba.int64[:](numba.int64[:], numba.int64), nopython=True)
def _get_enc_len(x, base=10):
    lens = np.zeros((len(x), ), dtype=np.int64)
    for i, n in enumerate(x):
        cnt = 0
        while n > 0:
            n = n // base
            cnt += 1
        # avoid 0 length
        if cnt == 0:
            cnt = 1
        lens[i] = cnt
    return lens


def get_enc_len(x, base=10):
    if isinstance(x, int):
        return _get_enc_len(np.array([x], dtype=np.int64), base)[0]
    elif isinstance(x, float):
        return _get_enc_len(np.array([int(x)], dtype=np.int64), base)[0]
    if isinstance(x, th.Tensor):
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
def _int2anybase(x, len_x, base):
    numbers = np.zeros((len(x), len_x), dtype=np.int64)
    for i, n in enumerate(x):
        n = n % base**len_x
        idx = len_x - 1
        while n:
            numbers[i, idx] = n % base
            n = n // base
            idx -= 1

    return numbers


def int2anybase(x, len_x, base=10):
    if isinstance(x, int):
        return _int2anybase(np.array([x], dtype=np.int64), len_x, base)[0]
    elif isinstance(x, float):
        return _int2anybase(np.array([int(x)], dtype=np.int64), len_x, base)[0]
    if isinstance(x, th.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x.astype(np.int64)

    return _int2anybase(x, len_x, base)


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
    if isinstance(x, th.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x.astype(np.int64)

    return _int2multihot(x, len_x, base)



class Embedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kw):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, **kw)

    def forward(self, x):
        if x.dtype == th.long:
            emb = super(Embedding, self).forward(x)
        elif x.dtype == th.float and x.size(-1) == self.num_embeddings:
            x_size = x.size()
            emb = th.matmul(x.view(-1, x_size[-1]), self.weight)
            emb = emb.view(x_size[:-1] + (self.embedding_dim, ))
        else:
            raise NotImplementedError
        return emb

    def get_output_dim(self):
        return self.embedding_dim


class NormalEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kw):
        super(NormalEmbedding, self).__init__(num_embeddings, embedding_dim, **kw)

        # init
        nn.init.normal_(self.weight, 0.0, 1.0)
        if self.padding_idx is not None:
            with th.no_grad():
                self.weight[self.padding_idx].fill_(0)


class UniformEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kw):
        super(UniformEmbedding, self).__init__(num_embeddings, embedding_dim, **kw)

        # init
        nn.init.uniform_(self.weight, -1.0, 1.0)
        if self.padding_idx is not None:
            with th.no_grad():
                self.weight[self.padding_idx].fill_(0)


class OrthogonalEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kw):
        super(OrthogonalEmbedding, self).__init__(num_embeddings, embedding_dim, **kw)

        # init
        nn.init.orthogonal_(self.weight)
        if self.padding_idx is not None:
            with th.no_grad():
                self.weight[self.padding_idx].fill_(0)


"""
Ravanbakhsh, S.; Schneider, J.; and Poczos, B.
Equivariance Through Parameter-Sharing.
In Proceedings of International Conference on Machine Learning, volume 70, of JMLR: W&CP, August 2017.
"""
class EquivariantEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kw):
        super(EquivariantEmbedding, self).__init__(num_embeddings, embedding_dim, **kw)

        self.row_vec = nn.Parameter(th.Tensor(self.embedding_dim, ))

        # init
        self.allow_forward = True
        nn.init.normal_(self.row_vec, 0.0, 1.0)
        with th.no_grad():
            for i in range(num_embeddings):
                self.weight[i].data.copy_(th.roll(self.row_vec, i, 0))

    def forward(self, x):
        if not self.allow_forward:
            with th.no_grad():
                for i in range(self.num_embeddings):
                    self.weight[i] = th.roll(self.row_vec, i, 0)
            self.allow_forward = True

        if x.dtype == th.long:
            emb = super(EquivariantEmbedding, self).forward(x)
        elif x.dtype == th.float and x.size(-1) == self.num_embeddings:
            x_size = x.size()
            emb = th.mm(x.view(-1, x_size[-1]), self.weight)
            emb = emb.view(x_size[:-1] + (self.embedding_dim, ))
        else:
            raise NotImplementedError
        return emb

    def backward(self, x):
        self.allow_forward = False
        return super(EquivariantEmbedding, self).backward(x)


class MultihotEmbedding(Embedding):
    def __init__(self, max_n=1024, base=2):
        self.max_n = max_n
        self.base = base

        enc_len = get_enc_len(max_n-1, base)
        super(MultihotEmbedding, self).__init__(max_n, 2*enc_len)
        with th.no_grad():
            self.weight.data.copy_(th.from_numpy(int2multihot(np.arange(0, max_n), enc_len, base)).float())

    def extra_repr(self):
        return "base=%d, max_n=%d, enc_dim=%d" % (self.base, self.max_n, self.weight.shape[1])


class PositionEmbedding(Embedding):
    def __init__(self, embedding_dim, max_len=512, scale=1):

        freq_seq = th.arange(0, embedding_dim, 2.0, dtype=th.float)
        inv_freq = th.pow(10000, (freq_seq / embedding_dim)).reciprocal()
        sinusoid_inp = th.ger(th.arange(0, max_len, 1.0), inv_freq)
        super(PositionEmbedding, self).__init__(max_len, embedding_dim)
        with th.no_grad():
            self.weight.data.copy_(th.cat([th.sin(sinusoid_inp), th.cos(sinusoid_inp)], dim=-1) * scale)

    def extra_repr(self):
        return "embedding_dim=%d, max_len=%d" % (self.weight.shape[1], self.weight.shape[0])
