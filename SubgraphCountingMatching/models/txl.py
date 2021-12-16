import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import EdgeSeqModel
from .container import *
from .embed import PositionEmbedding
# from ..constants import _INF
# from ..utils.act import map_activation_str_to_layer
# from ..utils.init import init_weight, init_module
# from ..utils.dl import segment_data
from constants import _INF
from utils.act import map_activation_str_to_layer
from utils.init import init_weight, init_module
from utils.dl import segment_data


class TXLFF(nn.Module):
    def __init__(self, input_dim, hid_dim, act_func="relu", dropout=0.0, pre_lnorm=False):
        super(TXLFF, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.pre_lnorm = pre_lnorm

        self.layer1 = nn.Linear(input_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, input_dim)
        self.act = map_activation_str_to_layer(act_func, inplace=True)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

        # init
        init_module(self.layer1, activation=act_func, init="normal")
        init_module(self.layer2, init="normal")

    def forward(self, x):
        if self.pre_lnorm:
            x = self.layer_norm(x)

        o = self.layer1(x)
        o = self.act(o)
        o = self.drop(o)
        o = self.layer2(o)
        o = self.drop(o)

        # residual connection
        o = o + x

        if not self.pre_lnorm:
            o = self.layer_norm(o)

        return o

    def get_output_dim(self):
        return self.input_dim

    def extra_repr(self):
        ""


class TXLAttn(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        num_heads=1,
        dropout=0.0,
        seg_len=None,
        ext_len=None,
        mem_len=None,
        pre_lnorm=False
    ):
        super(TXLAttn, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.scale = 1 / ((hid_dim/num_heads) ** 0.5)
        self.pre_lnorm = pre_lnorm
        self.drop = nn.Dropout(dropout)

        self.q_net = nn.Linear(input_dim, hid_dim, bias=False)
        self.k_net = nn.Linear(input_dim, hid_dim, bias=False)
        self.v_net = nn.Linear(input_dim, hid_dim, bias=False)
        self.r_net = nn.Linear(input_dim, hid_dim, bias=False)
        self.o_net = nn.Linear(input_dim, hid_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

        # init
        init_module(self.q_net, init="normal")
        init_module(self.k_net, init="normal")
        init_module(self.v_net, init="normal")
        init_module(self.r_net, init="normal")
        init_module(self.o_net, init="normal")

    def rel_shift(self, x, zero_triu=False):
        # x: bsz x qlen x klen x dim
        x_size = list(x.size())
        # pad: bsz x qlen x 1 x dim
        zero_pad = th.zeros(x_size[:2] + [1, x_size[-1]], device=x.device, dtype=x.dtype)
        x_padded = th.cat([zero_pad, x], dim=2)
        x_padded = x_padded.view(x_size[0], x_size[2]+1, x_size[1], x_size[3])
        x = x_padded[:, 1:].view(*x_size)

        if zero_triu:
            ones = th.ones((x.size(1), x.size(2)), device=x.device, dtype=x.dtype)
            x = x * th.tril(ones, diagonal=x.size(2) - x.size(1)).unsqueeze(0).unsqueeze(-1)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # r: [bsz, klen, hid_dim], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_r_bias: [klen, n_head], used for term D
        bsz, qlen = w.size(0), w.size(1)

        original_w = w

        if mems is not None:
            c = th.cat([mems, w], dim=1)
        else:
            c = w
        klen = c.size(1)
            
        if self.pre_lnorm:
            ##### layer normalization
            w = self.layer_norm(w)
            c = self.layer_norm(c)
        
        r_head_k = self.r_net(r)
        w_head_q = self.q_net(w)
        w_head_k = self.k_net(c)
        w_head_v = self.v_net(c)

        r_head_k = r_head_k.view(klen, self.num_heads, -1)      # [klen x num_heads x dim]
        w_head_q = w_head_q.view(bsz, qlen, self.num_heads, -1) # [bsz x qlen x num_heads x dim]
        w_head_k = w_head_k.view(bsz, klen, self.num_heads, -1) # [bsz x klen x num_heads x dim]
        w_head_v = w_head_v.view(bsz, klen, self.num_heads, -1) # [bsz x klen x num_heads x dim]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # [bsz x qlen x num_heads x dim]
        AC = th.einsum("bind,bjnd->bijn", (rw_head_q, w_head_k))             # [bsz x qlen x klen x num_heads]

        rr_head_q = w_head_q + r_r_bias                                         # [bsz x qlen x num_heads x dim]
        BD = th.einsum("bind,jnd->bijn", (rr_head_q, r_head_k))              # [bsz x qlen x klen x num_heads]
        BD = self.rel_shift(BD)

        # [bsz x qlen x klen x num_heads]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [bsz x klen] -> [bsz x qlen x klen x num_heads]
                attn_score = attn_score.masked_fill_(~(attn_mask).unsqueeze(1).unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3:
                # [bsz x qlen x klen] -> [bsz x qlen x klen x num_heads]
                attn_score = attn_score.masked_fill_(~(attn_mask).unsqueeze(-1), _INF)

        # [bsz x qlen x klen x num_heads]
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.drop(attn_prob)

        attn_vec = th.einsum("bijn,bjnd->bind", (attn_prob, w_head_v))

        # [bsz x qlen x hid_dim]
        attn_vec = attn_vec.contiguous().view(bsz, qlen, self.hid_dim)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        # residual connection
        output = attn_out + original_w
        
        if not self.pre_lnorm:
            output = self.layer_norm(output)

        return output

    def get_output_dim(self):
        return self.input_dim

    def extra_repr(self):
        ""


class TXLLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, num_heads, dropout=0.0, act_func="relu", pre_lnorm=False):
        super(TXLLayer, self).__init__()

        self.attn = TXLAttn(
            input_dim, hid_dim, num_heads=num_heads,
            dropout=dropout, pre_lnorm=pre_lnorm
        )
        self.ff = TXLFF(
            input_dim, hid_dim, act_func=act_func,
            dropout=dropout, pre_lnorm=pre_lnorm
        )

    def forward(self, x, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        output = self.attn(
            x, r, r_w_bias, r_r_bias,
            attn_mask=attn_mask, mems=mems
        )
        output = self.ff(output)

        return output

    def get_output_dim(self):
        return self.ff.get_output_dim()


class TransformerXL(EdgeSeqModel):
    def __init__(self, **kw):
        self.seg_len = kw.get("seg_len", 64)
        self.ext_len = kw.get("seg_len", 0)
        self.mem_len = kw.get("mem_len", 64)
        self.clamp_len = kw.get("clamp_len", -1)
        self.max_len = self.seg_len + self.ext_len + self.mem_len
        self.same_length = False
        self.num_heads = kw.get("num_heads", 1)
        super(TransformerXL, self).__init__(**kw)

        self.pos_emb = PositionEmbedding(self.hid_dim, max_len=max(self.clamp_len, self.max_len))
        self.pos_emb.weight.requires_grad = False
        self.drop = nn.Dropout(kw.get("rep_dropout", 0.0))

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
        num_heads = self.num_heads
        dropout = kw.get("rep_dropout", 0.0)
        act_func = kw.get("rep_act_func", "relu")
        pre_lnorm = kw.get("pre_lnorm", False)

        txl = ModuleList()
        for i in range(num_layers):
            txl.add_module(
                "%s_txl_(%d)" % (type, i),
                TXLLayer(
                    self.hid_dim, self.hid_dim, num_heads=num_heads,
                    dropout=dropout, act_func=act_func, pre_lnorm=pre_lnorm
                )
            )
        
        r_w_bias = nn.Parameter(th.Tensor(num_heads, self.hid_dim//num_heads))
        r_r_bias = nn.Parameter(th.Tensor(num_heads, self.hid_dim//num_heads))

        # init
        init_weight(r_w_bias, init="normal")
        init_weight(r_r_bias, init="normal")

        return MixtureDict({"txl": txl, "r_w_bias": r_w_bias, "r_r_bias": r_r_bias})
        
    def init_mems(self, num_layers, x):
        if self.mem_len > 0:
            mems = []
            for i in range(num_layers+1):
                empty = th.empty((x.size(0), 0, self.hid_dim), dtype=x.dtype, device=x.device)
                mems.append(empty)

            return mems
        else:
            return None

    def update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        new_mems = []
        end_idx = mlen + max(0, qlen - 0 - self.ext_len)
        beg_idx = max(0, end_idx - self.mem_len)
        for i in range(len(hids)):
            if mems is None or mlen == 0:
                new_mems.append(hids[i][:, beg_idx:end_idx].detach())
            else:
                cat = th.cat([mems[i], hids[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())
        return new_mems

    def _forward(self, x, x_mask, net, attn_mask=None, mems=None):
        bsz, qlen = x.size(0), x.size(1)
        txl, r_w_bias, r_r_bias = net["txl"], net["r_w_bias"], net["r_r_bias"]

        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen

        pos_seq = th.arange(klen-1, -1, -1.0, dtype=th.long, device=x.device)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        r = self.pos_emb(pos_seq).unsqueeze(0)

        x = self.drop(x)
        r = self.drop(r)

        x_zero_mask = ~(x_mask)

        outputs = [x]
        for i, layer in enumerate(txl):
            m = None if mems is None else mems[i]
            o = layer(outputs[-1], r, r_w_bias, r_r_bias, attn_mask=attn_mask, mems=m).masked_fill(x_zero_mask, 0.0)
            o = o.masked_fill(x_zero_mask, 0.0)
            # we do not use residual connections here because we use connections in each transformer layer
            outputs.append(o)

        new_mems = self.update_mems(outputs, mems, mlen, qlen)

        return outputs[-1], new_mems

    def encoder_forward(self, enc_inp, enc_mask, enc_net, mems=None):
        bsz, qlen = enc_inp.size(0), enc_inp.size(1)
        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen
        enc_attn_mask = th.ones((qlen, klen), dtype=th.bool, device=enc_inp.device).unsqueeze(0)

        return self._forward(enc_inp, enc_mask, enc_net, attn_mask=enc_attn_mask, mems=mems)
    
    def decoder_forward(self, dec_inp, dec_mask, dec_net, mems=None):
        bsz, qlen = dec_inp.size(0), dec_inp.size(1)
        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen
        
        ones = th.ones((qlen, klen), dtype=th.bool, device=dec_inp.device)
        if self.same_length:
            mask_len = klen - self.tgt_mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (1 - (th.triu(ones, diagonal=1+mlen) + th.tril(ones, -mask_shift_len))).unsqueeze(0)
        else:
            dec_attn_mask = (1 - th.triu(ones, diagonal=1+mlen)).unsqueeze(0)
        return self._forward(dec_inp, dec_mask, dec_net, attn_mask=dec_attn_mask, mems=mems)

    def get_pattern_rep(self, p_emb, mask=None):
        if mask is None:
            mask = th.ones(p_emb.size()[:2] + (1,), dtype=th.bool, device=p_emb.device)
        else:
            p_emb = p_emb.masked_fill(~(mask), 0.0)

        p_segments = segment_data(p_emb, self.seg_len)
        p_segmasks = segment_data(mask, self.seg_len, pre_pad=True)
        p_outputs = list()
        for i, (p_seg, p_mask) in enumerate(zip(p_segments, p_segmasks)):
            if i == 0:
                p_mems = self.init_mems(len(self.p_rep_net), p_seg)
            p_output, p_mems = self.encoder_forward(p_seg, p_mask, self.p_rep_net, mems=p_mems)
            p_outputs.append(p_output)
        p_output = th.cat(p_outputs, dim=1)[:, -p_emb.size(1):]

        return p_output

    def get_graph_rep(self, g_emb, mask=None, gate=None):
        if mask is None:
            mask = th.ones(g_emb.size()[:2] + (1,), dtype=th.bool, device=g_emb.device)
        else:
            g_emb = g_emb.masked_fill(~(mask), 0.0)
        if gate is not None:
            g_emb = g_emb * gate

        g_segments = segment_data(g_emb, self.seg_len)
        g_segmasks = segment_data(mask, self.seg_len, pre_pad=True)
        g_outputs = list()
        for i, (g_seg, g_mask) in enumerate(zip(g_segments, g_segmasks)):
            if i == 0:
                g_mems = self.init_mems(len(self.g_rep_net), g_seg)
            g_output, g_mems = self.encoder_forward(g_seg, g_mask, self.g_rep_net, mems=g_mems)
            g_outputs.append(g_output)
        g_output = th.cat(g_outputs, dim=1)[:, -g_emb.size(1):]

        if gate is not None:
            g_output = g_output * gate

        return g_output
