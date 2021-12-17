import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# from ..constants import _INF
# from ..utils.act import map_activation_str_to_layer
# from ..utils.dl import batch_convert_mask_to_start_and_end
# from ..utils.init import init_weight, init_module
from constants import _INF
from utils.act import map_activation_str_to_layer
from utils.dl import batch_convert_mask_to_start_and_end
from utils.init import init_weight, init_module


class PredictNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(PredictNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.act = map_activation_str_to_layer(act_func)
        self.drop = nn.Dropout(dropout)
        self.p_fc = nn.Linear(input_dim, hidden_dim)
        self.g_fc = nn.Linear(input_dim, hidden_dim)

        self.pred_fc1 = nn.Linear(hidden_dim * 4 + 4, hidden_dim)
        self.pred_fc2 = nn.Linear(hidden_dim + 4, 1)

        if return_weights:
            self.weight_fc1 = nn.Linear(hidden_dim * 4 + 2, hidden_dim)
            self.weight_fc2 = nn.Linear(hidden_dim + 2, 1)
        else:
            self.weight_fc1 = None
            self.weight_fc2 = None

        # init
        init_module(self.p_fc, activation=act_func, init="normal")
        init_module(self.g_fc, activation=act_func, init="normal")
        init_module(self.pred_fc1, activation=act_func, init="normal")
        init_module(self.pred_fc2, activation=act_func, init="zero")
        if return_weights:
            init_module(self.weight_fc1, activation=act_func, init="normal")
            init_module(self.weight_fc2, activation=act_func, init="zero")

    def init_pattern(self, p_rep, p_mask=None):
        """
        input: bsz x plen x input_dim
        ouput: bsz x plen x hid_dim
        """
        p = self.p_fc(p_rep)

        return p

    def agg_pattern(self, p_rep, p_mask=None):
        """
        input: bsz x plen x hid_dim
        ouput: bsz x hid_dim
        """
        return self.agg_graph(p_rep, p_mask)

    def init_graph(self, g_rep, g_mask=None):
        """
        input: bsz x glen x input_dim
        ouput: bsz x glen x hid_dim
        """
        g = self.g_fc(g_rep)

        return g

    def agg_graph(self, g_rep, g_mask=None):
        """
        input: bsz x glen x hid_dim
        ouput: bsz x hid_dim
        """
        raise NotImplementedError

    def forward(self, p_rep, p_mask, g_rep, g_mask):
        # p_rep can be [bsz x p_len x dim] or [bsz x dim]
        # g_rep must be [bsz x g_len x dim]
        bsz = p_mask.size(0)
        p_len = p_mask.size(1)
        g_len = g_mask.size(1)

        pl = p_mask.float().sum(dim=1).view(bsz, 1)
        pl_inv = 1.0 / pl
        gl = g_mask.float().sum(dim=1).view(bsz, 1)
        gl_inv = 1.0 / gl

        # bsz x g_len x dim
        if p_rep.dim() == 2:
            p = p_rep.unsqueeze(1).expand(bsz, g_len, -1)
        elif p_rep.dim() == 3:
            p = self.init_pattern(p_rep, p_mask)
            p = self.drop(p)
            p = self.agg_pattern(p, p_mask)
            p = p.unsqueeze(1).expand(bsz, g_len, -1)
        else:
            raise ValueError

        # bsz x g_len x dim
        g = self.init_graph(g_rep, g_mask)
        g = self.drop(g)

        if self.weight_fc1 is not None:
            # bsz x g_len x dim
            w = th.cat(
                [
                    p, g, g - p, g * p,
                    pl.expand(bsz, g_len).unsqueeze(-1),
                    pl_inv.expand(bsz, g_len).unsqueeze(-1)
                ],
                dim=2
            )
            w = self.weight_fc1(w)
            w = self.act(w)
            w = self.weight_fc2(
                th.cat(
                    [
                        w,
                        pl.expand(bsz, g_len).unsqueeze(-1),
                        pl_inv.expand(bsz, g_len).unsqueeze(-1)
                    ],
                    dim=2
                )
            )
            w = w.squeeze_(-1)
        else:
            w = None

        # bsz x 1 x dim
        p = p[:, 0, :]
        g = self.agg_graph(g)

        # bsz x 1 x dim
        y = th.cat(
            [
                p, g, g - p, g * p,
                pl, gl, pl_inv, gl_inv
            ],
            dim=1
        )
        y = self.pred_fc1(y)
        y = self.act(y)
        y = self.pred_fc2(th.cat([y, pl, gl, pl_inv, gl_inv], dim=1))

        return y, w


class BasePoolPredictNet(PredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(BasePoolPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )


class MeanPredictNet(BasePoolPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(MeanPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.mean(g_rep, dim=1)


class SumPredictNet(BasePoolPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(SumPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.sum(g_rep, dim=1)


class MaxPredictNet(BasePoolPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(MaxPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.max(g_rep, dim=1)[0]


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        hidden_dim,
        num_heads=1,
        scale=1,
        score_func="softmax",
        add_zero_attn=False,
        add_gate=False,
        add_residual=False,
        pre_lnorm=False,
        post_lnorm=False,
        dropout=0.0
    ):
        super(Attention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim

        if (hidden_dim == -1 and query_dim % num_heads != 0):
            raise ValueError(
                "Error: query_dim (%d) %% num_heads (%d) should be 0." % (query_dim, num_heads)
            )
        if (hidden_dim == -1 and key_dim % num_heads != 0):
            raise ValueError(
                "Error: key_dim (%d) %% num_heads (%d) should be 0." % (query_dim, num_heads)
            )
        if (hidden_dim != -1 and hidden_dim % num_heads != 0):
            raise ValueError(
                "Error: hidden_dim (%d) %% num_heads (%d) should be 0." % (query_dim, num_heads)
            )
        self.num_heads = num_heads
        self.scale = scale
        self.add_zero_attn = add_zero_attn
        self.add_residual = add_residual
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm
        self.drop = nn.Dropout(dropout)

        if hidden_dim != -1:
            self.weight_v = nn.Parameter(th.Tensor(value_dim, hidden_dim))
            self.weight_o = nn.Parameter(th.Tensor(hidden_dim, query_dim))
        else:
            self.register_parameter("weight_v", None)
            self.register_parameter("weight_o", None)

        self.score_act = map_activation_str_to_layer(score_func)
        if hasattr(self.score_act, "dim"):
            setattr(self.score_act, "dim", 2)  # not the last dimension
        if add_gate:
            self.g_net = nn.Linear(query_dim * 2, query_dim, bias=True)
        else:
            self.g_net = None
        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
            self.o_layer_norm = None
        if post_lnorm:
            self.q_layer_norm = None
            self.k_layer_norm = None
            self.v_layer_norm = None
            self.o_layer_norm = nn.LayerNorm(query_dim)

        # init
        if hidden_dim != -1:
            init_weight(self.weight_v, init="normal")
            init_weight(self.weight_o, init="normal")
        if add_gate:
            init_weight(self.g_net, init="normal")
            nn.init.constant_(self.g_net.bias, 1.0)

        if pre_lnorm:
            init_weight(self.q_layer_norm)
            init_weight(self.k_layer_norm)
            init_weight(self.v_layer_norm)
        if post_lnorm:
            init_weight(self.o_layer_norm)

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        bsz = query.size(0)
        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        original_query = query

        if self.add_zero_attn:
            key = th.cat(
                [
                    key,
                    th.zeros((bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device)
                ],
                dim=1
            )
            value = th.cat(
                [
                    value,
                    th.zeros((bsz, 1) + value.size()[2:], dtype=value.dtype, device=value.device)
                ],
                dim=1
            )
            if key_mask is not None:
                key_mask = th.cat(
                    [
                        key_mask,
                        th.ones((bsz, 1), dtype=key_mask.dtype, device=key_mask.device)
                    ],
                    dim=1
                )

        if self.pre_lnorm:
            # layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        attn_score = self.compute_score(query, key, key_mask)
        attn_score = self.drop(attn_score)

        # [bsz x qlen x klen x num_heads] x [bsz x klen x num_heads x head_dim] -> [bsz x qlen x num_heads x head_dim]
        if self.weight_v is not None:
            value = th.matmul(value, self.weight_v)
        value = value.view(bsz, vlen, self.num_heads, -1)
        attn_vec = th.einsum("bijn,bjnd->bind", (attn_score, value))
        attn_vec = attn_vec.contiguous().view(bsz, qlen, -1)

        if query_mask is not None:
            while query_mask.dim() < attn_vec.dim():
                query_mask = query_mask.unsqueeze(-1)
            attn_vec.masked_fill_(query_mask == 0, 0)

        if self.weight_o is not None:
            attn_vec = th.matmul(attn_vec, self.weight_o)
        attn_vec = self.drop(attn_vec)

        if self.g_net is not None:
            g = F.sigmoid(self.g_net(th.cat([original_query, attn_vec], dim=-1)))
            attn_out = g * original_query + (1 - g) * attn_vec
        else:
            attn_out = attn_vec

        if self.add_residual:
            attn_out = original_query + attn_out

        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out

    def compute_score(self, query, key, key_mask=None):
        """
        param: query [bsz * qlen * qdim]
        param: key [bsz * klen * kdim]
        param: key_mask [bsz * klen] or [bsz * 1 * klen]
        return: attn_score [bsz * qlen * klen * num_heads]
        """
        raise NotImplementedError

    def get_output_dim(self):
        return self.query_dim


"""
Effective Approaches to Attention-based Neural Machine Translation
Minh-Thang Luong, Hieu Pham, Christopher D. Manning
https://arxiv.org/abs/1508.04025
"""
class DotAttention(Attention):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        hidden_dim,
        num_heads=1,
        scale=1,
        score_func="softmax",
        add_zero_attn=False,
        add_gate=False,
        add_residual=False,
        pre_lnorm=False,
        post_lnorm=False,
        dropout=0.0
    ):
        super(DotAttention, self).__init__(
            query_dim,
            key_dim,
            value_dim,
            hidden_dim,
            num_heads=num_heads,
            scale=scale,
            score_func=score_func,
            add_zero_attn=add_zero_attn,
            add_gate=add_gate,
            add_residual=add_residual,
            pre_lnorm=pre_lnorm,
            post_lnorm=post_lnorm,
            dropout=dropout
        )

        if hidden_dim == -1 and \
           (query_dim != key_dim or query_dim != value_dim or key_dim != value_dim):
            raise ValueError(
                "Error: when hidden_dim equals 1, we need the query, key, and value have the same dimension!"
            )

        if hidden_dim != -1:
            self.weight_q = nn.Parameter(th.Tensor(query_dim, hidden_dim))
            self.weight_k = nn.Parameter(th.Tensor(key_dim, hidden_dim))
        else:
            self.register_parameter("weight_q", None)
            self.register_parameter("weight_k", None)

        if hidden_dim != -1:
            init_weight(self.weight_q, init="normal")
            init_weight(self.weight_k, init="normal")

    def compute_score(self, query, key, key_mask=None):
        bsz = query.size(0)
        qlen, klen = query.size(1), key.size(1)

        if self.weight_q is not None:
            query = th.matmul(query, self.weight_q)
            key = th.matmul(key, self.weight_k)

        query = query.view(bsz, qlen, self.num_heads, -1)
        key = key.view(bsz, klen, self.num_heads, -1)

        # [bsz x qlen x klen x num_heads]
        attn_score = th.einsum("bind,bjnd->bijn", (query, key))
        attn_score.mul_(self.scale)

        if key_mask is not None:
            if key_mask.dim() < attn_score.dim():
                key_mask = key_mask.unsqueeze(-1)
            while key_mask.dim() < attn_score.dim():
                key_mask = key_mask.unsqueeze(1)
            attn_score.masked_fill_(key_mask == 0, _INF)

        # [bsz x qlen x klen x num_heads]
        if self.score_act is not None:
            attn_score = self.score_act(attn_score)

        return attn_score


class BaseAttnPredictNet(PredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(BaseAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

        self.num_heads = num_heads
        self.infer_steps = infer_steps

        self.p_attn = DotAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1 / (hidden_dim / num_heads)**0.5,
            score_func="sparsemax",
            add_gate=True,
            add_residual=False,
            pre_lnorm=False,
            post_lnorm=False
        )
        self.g_attn = DotAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1 / (hidden_dim / num_heads)**0.5,
            score_func="sparsemax",
            add_gate=True,
            add_residual=False,
            pre_lnorm=False,
            post_lnorm=False
        )

        # make the attention should prefer to output the original
        for param in self.p_attn.parameters():
            if param.requires_grad:
                init_weight(param, init="identity")
        for param in self.g_attn.parameters():
            if param.requires_grad:
                init_weight(param, init="identity")

    def infer_fn(self, g, p, g_mask, p_mask):
        g = self.p_attn(g, p, p, query_mask=g_mask, key_mask=p_mask)
        g = self.g_attn(g, g, g, query_mask=g_mask, key_mask=g_mask)

        return g

    def forward(self, p_rep, p_mask, g_rep, g_mask):
        g = g_rep
        for i in range(self.infer_steps):
            g = self.infer_fn(g, p_rep, g_mask, p_mask)

        return super(BaseAttnPredictNet, self).forward(p_rep, p_mask, g, g_mask)


class MeanAttnPredictNet(BaseAttnPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(MeanAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            infer_steps=infer_steps,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.mean(g_rep, dim=1)


class SumAttnPredictNet(BaseAttnPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(SumAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            infer_steps=infer_steps,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.sum(g_rep, dim=1)


class MaxAttnPredictNet(BaseAttnPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(MaxAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            infer_steps=infer_steps,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.max(g_rep, dim=1)[0]


"""
Neural Subgraph Isomorphism Counting
Xin Liu, Haojie Pan, Mutian He, Yangqiu Song, Xin Jiang, Lifeng Shang
https://arxiv.org/abs/1912.11589
"""
def init_mem(x, x_mask=None, mem_len=4, mem_init="mean", attn=None, lstm=None):
    assert mem_init in [
        "mean", "sum", "max",
        "attn", "lstm",
        "circular_mean", "circular_sum", "circular_max",
        "circular_attn", "circular_lstm"
    ]

    bsz, seq_len, hidden_dim = x.size()

    if mem_init.startswith("circular"):
        pad_len = math.ceil((seq_len + 1) / 2) - 1
        x = F.pad(x.transpose(1, 2), pad=(0, pad_len), mode="circular").transpose(1, 2)
        if x_mask is not None:
            x_mask = F.pad(x_mask.unsqueeze(1), pad=(0, pad_len), mode="circular").squeeze(1)
        seq_len += pad_len

    if seq_len <= mem_len:
        if mem_init.endswith("mean") or mem_init.endswith("max") or mem_init.endswith("sum"):
            mem = th.cat(
                [
                    th.zeros((bsz, mem_len-seq_len, x.size(2)), dtype=x.dtype, device=x.device),
                    x
                ],
                dim=1
            )
        elif mem_init.endswith("attn"):
            mem = list()
            hidden_dim = attn.query_dim
            mem.append(th.zeros((bsz, mem_len-seq_len, hidden_dim), dtype=x.dtype, device=x.device))
            for i in range(seq_len):
                m = x[:, i:i+1]
                mk = x_mask[:, i:i+1] if mk is not None else None
                if attn:
                    h = attn(m, m, m, query_mask=mk, key_mask=mk)
                else:
                    attn_score = th.einsum("bid,bjd->bij", (m, m))
                    attn_score.mul_(1 / hidden_dim**0.5)
                    attn_score.masked_fill_(mk.unsqueeze(1) == 0, _INF)
                    attn_score = F.softmax(attn_score, dim=-1)
                    h = th.einsum("bij,bjd->bid", (attn_score, m))
                mem.append(h)
            mem = th.cat(mem, dim=1)
        elif mem_init.endswith("lstm"):
            mem = list()
            hidden_dim = lstm.hidden_size * lstm.num_layers
            if lstm.bidirectional:
                hidden_dim *= 2
            mem.append(th.zeros((bsz, mem_len - seq_len, hidden_dim), dtype=x.dtype, device=x.device))
            for i in range(seq_len):
                hx = None
                m = x[:, i:i+1]
                _, hx = lstm(m, hx)
                mem.append(hx[0].view(bsz, 1, -1))
            mem = th.cat(mem, dim=1)
        if x_mask is not None:
            mem_mask = th.cat(
                [
                    th.zeros((bsz, mem_len-seq_len), dtype=x_mask.dtype, device=x_mask.device),
                    x_mask
                ],
                dim=1
            )
        else:
            mem_mask = None
    else:
        stride = seq_len // mem_len
        kernel_size = seq_len - (mem_len - 1) * stride

        if mem_init.endswith("mean"):
            mem = F.avg_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=stride).transpose(1, 2)
        elif mem_init.endswith("max"):
            mem = F.max_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=stride).transpose(1, 2)
        elif mem_init.endswith("sum"):
            mem = F.avg_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=stride).transpose(1, 2) * kernel_size
        elif mem_init.endswith("attn"):
            # split and self attention
            mem = list()
            hidden_dim = attn.query_dim
            for i in range(0, seq_len - kernel_size + 1, stride):
                j = i + kernel_size
                m = x[:, i:j]
                mk = x_mask[:, i:j] if x_mask is not None else None
                if attn:
                    h = attn(m.mean(dim=-2, keepdim=True), m, m, query_mask=None, key_mask=mk)
                else:
                    attn_score = th.einsum("bid,bjd->bij", (m.mean(dim=-2, keepdim=True), m))
                    attn_score.mul_(1 / hidden_dim**0.5)
                    attn_score.masked_fill_(mk.unsqueeze(1) == 0, _INF)
                    attn_score = F.softmax(attn_score, dim=-1)
                    h = th.einsum("bij,bjd->bid", (attn_score, m))
                mem.append(h)
            mem = th.cat(mem, dim=1)
        elif mem_init.endswith("lstm"):
            mem = list()
            for i in range(0, seq_len - kernel_size + 1, stride):
                hx = None
                j = i + kernel_size
                m = x[:, i:j]
                _, hx = lstm(m, hx)
                mem.append(hx[0].view(bsz, 1, -1))
            mem = th.cat(mem, dim=1)

        if x_mask is not None:
            mem_mask = F.max_pool1d(
                x_mask.float().unsqueeze(1),
                kernel_size=kernel_size,
                stride=stride
            ).squeeze(1).bool()
        else:
            mem_mask = None

    return mem, mem_mask


class MemDotAttention(DotAttention):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        mem_dim,
        hidden_dim,
        num_heads=1,
        scale=1,
        mem_len=1,
        mem_init="mean",
        score_func="softmax",
        add_zero_attn=False,
        add_gate=False,
        add_residual=False,
        pre_lnorm=False,
        post_lnorm=False,
        dropout=0.0
    ):
        super(MemDotAttention, self).__init__(
            query_dim,
            mem_dim,
            mem_dim,
            hidden_dim,
            num_heads=num_heads,
            scale=scale,
            score_func=score_func,
            add_zero_attn=add_zero_attn,
            add_gate=add_gate,
            add_residual=add_residual,
            pre_lnorm=pre_lnorm,
            post_lnorm=post_lnorm,
            dropout=dropout
        )

        self.mem_len = mem_len
        self.mem_init = mem_init

        # key_dim -> mem_dim
        self.proj_k = nn.Linear(key_dim, mem_dim)
        # value_dim -> mem_dim
        self.proj_v = nn.Linear(value_dim, mem_dim)

        self.attn = None
        self.lstm = None
        if mem_init.endswith("attn"):
            self.attn = DotAttention(
                query_dim=mem_dim,
                key_dim=mem_dim,
                value_dim=mem_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                scale=1 / (hidden_dim/num_heads)**0.5,
                score_func=score_func,
                add_zero_attn=add_zero_attn,
                add_gate=False,
                add_residual=False,
                pre_lnorm=pre_lnorm,
                post_lnorm=post_lnorm,
                dropout=dropout
            )
        elif mem_init.endswith("lstm"):
            self.lstm = nn.LSTM(mem_dim, mem_dim, batch_first=True)
        else:
            pass

        # init
        init_weight(self.proj_k, init="normal")
        init_weight(self.proj_v, init="normal")
        if self.lstm is not None:
            init_weight(self.lstm)

    def init_memory(self, x, x_mask=None):
        bsz = x.size(0)

        if x_mask is not None:
            mem = list()
            mem_mask = list()
            start, end = batch_convert_mask_to_start_and_end(x_mask)
            for i in range(bsz):
                m, mk = init_mem(
                    x[i:i + 1, start[i].item():end[i].item()],
                    x_mask[i:i + 1, start[i].item():end[i].item()],
                    mem_len=self.mem_len,
                    mem_init=self.mem_init,
                    attn=self.attn,
                    lstm=self.lstm
                )
                mem.append(m)
                mem_mask.append(mk)
            mem = th.cat(mem, dim=0)
            mem_mask = th.cat(mem_mask, dim=0)
        else:
            mem, mem_mask = init_mem(
                x,
                None,
                mem_len=self.mem_len,
                mem_init=self.mem_init,
                attn=self.attn,
                lstm=self.lstm
            )
        return mem, mem_mask

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        mem_k, mem_k_mask = self.init_memory(self.proj_k(key), key_mask)
        mem_v, mem_v_mask = self.init_memory(self.proj_v(value), key_mask)

        return super(MemDotAttention, self).forward(query, mem_k, mem_v, query_mask, mem_k_mask)


class BaseMemAttnPredictNet(PredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        mem_len=4,
        mem_init="mean",
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(BaseMemAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

        self.mem_len = mem_len
        self.mem_init = mem_init
        self.num_heads = num_heads
        self.infer_steps = infer_steps

        self.p_attn = MemDotAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            mem_dim=hidden_dim,
            hidden_dim=hidden_dim,
            mem_len=self.mem_len,
            mem_init=self.mem_init,
            num_heads=num_heads,
            score_func="sparsemax",
            add_gate=True,
            pre_lnorm=True
        )
        self.g_attn = MemDotAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            mem_dim=hidden_dim,
            hidden_dim=hidden_dim,
            mem_len=self.mem_len,
            mem_init=self.mem_init,
            num_heads=num_heads,
            score_func="sparsemax",
            add_gate=True,
            pre_lnorm=True
        )

        # make the attention should prefer to output the original
        for param in self.p_attn.parameters():
            if param.requires_grad:
                init_weight(param, init="identity")
        for param in self.g_attn.parameters():
            if param.requires_grad:
                init_weight(param, init="identity")

    def infer_fn(self, g, p, g_mask, p_mask):
        g = self.p_attn(g, p, p, query_mask=g_mask, key_mask=p_mask)
        g = self.g_attn(g, g, g, query_mask=g_mask, key_mask=g_mask)

        return g

    def forward(self, p_rep, p_mask, g_rep, g_mask):
        g = g_rep
        for i in range(self.infer_steps):
            g = self.infer_fn(g, p_rep, g_mask, p_mask)

        return super(BaseMemAttnPredictNet, self).forward(p_rep, p_mask, g, g_mask)


class MeanMemAttnPredictNet(BaseMemAttnPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        mem_len=4,
        mem_init="mean",
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(MeanMemAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            infer_steps=infer_steps,
            mem_len=mem_len,
            mem_init=mem_init,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.mean(g_rep, dim=1)


class SumMemAttnPredictNet(BaseMemAttnPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        mem_len=4,
        mem_init="sum",
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(SumMemAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            infer_steps=infer_steps,
            mem_len=mem_len,
            mem_init=mem_init,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.sum(g_rep, dim=1)


class MaxMemAttnPredictNet(BaseMemAttnPredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        max_seq_len=512,
        infer_steps=3,
        mem_len=4,
        mem_init="max",
        act_func="relu",
        dropout=0.0,
        return_weights=False
    ):
        super(MaxMemAttnPredictNet, self).__init__(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            infer_steps=infer_steps,
            mem_len=mem_len,
            mem_init=mem_init,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

    def agg_graph(self, g_rep, g_mask=None):
        return th.max(g_rep, dim=1)[0]


class DIAMNet(PredictNet):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        act_func="relu",
        num_heads=1,
        max_seq_len=512,
        infer_steps=1,
        mem_len=4,
        mem_init="mean",
        dropout=0.0,
        return_weights=False
    ):
        super(DIAMNet, self).__init__(
            input_dim,
            hidden_dim,
            act_func=act_func,
            dropout=dropout,
            return_weights=return_weights
        )

        self.num_heads = num_heads
        self.infer_steps = infer_steps
        self.mem_len = mem_len
        self.mem_init = mem_init
        mem_dim = hidden_dim

        # input_dim -> mem_dim
        if mem_init.endswith("attn"):
            self.mem_layer = DotAttention(
                query_dim=mem_dim,
                key_dim=input_dim,
                value_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                scale=1 / (hidden_dim / num_heads)**0.5,
                score_func="sparsemax",
                add_gate=True,
                add_residual=False,
                pre_lnorm=False,
                post_lnorm=False
            )
        elif mem_init.endswith("lstm"):
            self.mem_layer = nn.LSTM(input_dim, mem_dim, batch_first=True)
        else:
            self.mem_layer = nn.Linear(input_dim, mem_dim)

        # mem_dim -> mem_dim
        self.p_attn = DotAttention(
            query_dim=mem_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1 / (hidden_dim / num_heads)**0.5,
            score_func="sparsemax",
            add_gate=True,
            add_residual=False,
            pre_lnorm=False,
            post_lnorm=False
        )
        self.g_attn = DotAttention(
            query_dim=mem_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1 / (hidden_dim / num_heads)**0.5,
            score_func="sparsemax",
            add_gate=True,
            add_residual=False,
            pre_lnorm=False,
            post_lnorm=False
        )
        self.m_attn = DotAttention(
            query_dim=hidden_dim,
            key_dim=mem_dim,
            value_dim=mem_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1 / (hidden_dim / num_heads)**0.5,
            score_func="sparsemax",
            add_gate=True,
            add_residual=False,
            pre_lnorm=False,
            post_lnorm=False
        )

        if return_weights:
            del self.weight_fc1
            self.weight_fc1 = nn.Linear(mem_dim * mem_len + 2, hidden_dim)

        del self.pred_fc1
        self.pred_fc1 = nn.Linear(mem_dim * mem_len + 4, hidden_dim)

        # init
        init_weight(self.pred_fc1, activation=act_func, init="normal")
        if return_weights:
            init_weight(self.weight_fc1, activation=act_func, init="normal")

        # make the attention should prefer to output the original
        for param in self.p_attn.parameters():
            if param.requires_grad:
                init_weight(param, init="identity")
        for param in self.g_attn.parameters():
            if param.requires_grad:
                init_weight(param, init="identity")
        for param in self.m_attn.parameters():
            if param.requires_grad:
                init_weight(param, init="identity")

    def init_pattern(self, p_rep, p_mask, memory, memory_mask):
        """
        input: bsz x plen x input_dim
        ouput: bsz x plen x hid_dim
        """
        p = self.p_fc(p_rep)
        p = self.m_attn(p, memory, memory, p_mask, memory_mask)

        return p

    def agg_pattern(self, p_rep, p_mask=None):
        """
        input: bsz x plen x hid_dim
        ouput: bsz x hid_dim
        """
        if self.mem_init == "max" or self.mem_init == "circular_max":
            return th.max(p_rep, dim=1)[0]
        elif self.mem_init == "sum" or self.mem_init == "circular_sum":
            return th.sum(p_rep, dim=1)
        else:
            return th.mean(p_rep, dim=1)

    def init_graph(self, g_rep, g_mask, memory, memory_mask):
        g = self.g_fc(g_rep)
        g = self.m_attn(g, memory, memory, g_mask, memory_mask)

        return g

    def init_memory(self, x, x_mask=None):
        bsz = x.size(0)
        if x_mask is not None:
            # bucket x by lens
            start, end = batch_convert_mask_to_start_and_end(x_mask)
            end = end + 1

            if th.all(start == start[0]):
                ind = th.argsort(end, dim=0)
                ind_reversed = th.argsort(ind, dim=0)
                x = x[ind]
                x_mask = x_mask[ind]
                start = start[ind]
                end = end[ind]
                bucket_indices = [0] + th.arange(bsz)[(th.roll(end, 1, 0) - end) < 0].tolist() + [bsz]
            elif th.all(end == end[0]):
                ind = th.argsort(start, dim=0)
                ind_reversed = th.argsort(ind, dim=0)
                x = x[ind]
                x_mask = x_mask[ind]
                start = start[ind]
                end = end[ind]
                bucket_indices = [0] + th.arange(bsz)[(th.roll(start, 1, 0) - start) < 0].tolist() + [bsz]
            else:
                ind = None
                ind_reversed = None
                bucket_indices = list(range(bsz + 1))

            start = start.tolist()
            end = end.tolist()
        else:
            ind = None
            ind_reversed = None
            start, end = [0], [x.size(1)]
            bucket_indices = [0, bsz]

        mem = list()
        mem_mask = list()
        if self.mem_init.endswith("attn"):
            for i in range(len(bucket_indices) - 1):
                j = bucket_indices[i]
                m, mk = init_mem(
                    x[bucket_indices[i]:bucket_indices[i + 1], start[j]:end[j]],
                    x_mask[bucket_indices[i]:bucket_indices[i + 1], start[j]:end[j]] if x_mask is not None else None,
                    mem_len=self.mem_len,
                    mem_init=self.mem_init,
                    attn=self.mem_layer
                )
                mem.append(m)
                mem_mask.append(mk)
        elif self.mem_init.endswith("lstm"):
            for i in range(len(bucket_indices) - 1):
                j = bucket_indices[i]
                m, mk = init_mem(
                    x[bucket_indices[i]:bucket_indices[i + 1], start[j]:end[j]],
                    x_mask[bucket_indices[i]:bucket_indices[i + 1], start[j]:end[j]] if x_mask is not None else None,
                    mem_len=self.mem_len,
                    mem_init=self.mem_init,
                    lstm=self.mem_layer
                )
                mem.append(m)
                mem_mask.append(mk)
        else:
            for i in range(len(bucket_indices) - 1):
                j = bucket_indices[i]
                m, mk = init_mem(
                    x[bucket_indices[i]:bucket_indices[i + 1], start[j]:end[j]],
                    x_mask[bucket_indices[i]:bucket_indices[i + 1], start[j]:end[j]] if x_mask is not None else None,
                    mem_len=self.mem_len,
                    mem_init=self.mem_init,
                )
                mem.append(self.mem_layer(m))
                mem_mask.append(mk)
        mem = th.cat(mem, dim=0)
        mem_mask = th.cat(mem_mask, dim=0) if x_mask is not None else None

        if ind_reversed is not None:
            mem = mem[ind_reversed]
            mem_mask = mem_mask[ind_reversed]

        return mem, mem_mask

    def infer_fn(self, m, p, g, m_mask, p_mask, g_mask):
        m = self.p_attn(m, p, p, query_mask=m_mask, key_mask=p_mask)
        m = self.g_attn(m, g, g, query_mask=m_mask, key_mask=g_mask)

        return m

    def forward(self, p_rep, p_mask, g_rep, g_mask):
        bsz = p_mask.size(0)
        p_len = p_mask.size(1)
        pl = p_mask.float().sum(dim=1).view(bsz, 1)
        pl_inv = 1.0 / pl
        g_len = g_mask.size(1)
        gl = g_mask.float().sum(dim=1).view(bsz, 1)
        gl_inv = 1.0 / gl

        # bsz x m_len x dim
        m, m_mask = self.init_memory(g_rep, g_mask)
        for i in range(self.infer_steps):
            m = self.infer_fn(m=m, g=g_rep, p=p_rep, m_mask=m_mask, g_mask=g_mask, p_mask=p_mask)

        if self.weight_fc1 is not None:
            p = self.init_pattern(p_rep, p_mask, m, m_mask)
            p = self.drop(p)
            p = self.agg_pattern(p, p_mask)
            p = p.unsqueeze(1).expand(bsz, g_len, -1)

            g = self.init_graph(g_rep, g_mask, m, m_mask)
            g = self.drop(g)

            # bsz x g_len x dim
            w = th.cat(
                [
                    p, g, g - p, g * p,
                    pl.expand(bsz, g_len).unsqueeze(-1),
                    pl_inv.expand(bsz, g_len).unsqueeze(-1)
                ],
                dim=2
            )
            w = self.weight_fc1(w)
            w = self.act(w)
            w = self.weight_fc2(
                th.cat(
                    [
                        w,
                        pl.expand(-1, g_len).unsqueeze(-1),
                        pl_inv.expand(-1, g_len).unsqueeze(-1)
                    ],
                    dim=2
                )
            )
            w = w.squeeze_(-1)
        else:
            w = None

        # bsz x (m_len x dim)
        m = m.view(bsz, -1)

        # bsz x dim
        y = th.cat([m, pl, gl, pl_inv, gl_inv], dim=1)
        y = self.pred_fc1(y)
        y = self.act(y)
        y = self.pred_fc2(th.cat([y, pl, gl, pl_inv, gl_inv], dim=1))

        return y, w
