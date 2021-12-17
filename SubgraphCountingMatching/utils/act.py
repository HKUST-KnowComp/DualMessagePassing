import torch as th
import torch.nn as nn
import math

from torch.nn.functional import sigmoid
from torch.nn.modules.activation import Sigmoid

from torch.nn.functional import softmax
from torch.nn.modules.activation import Softmax

from torch.nn.functional import tanh
from torch.nn.modules.activation import Tanh

from torch.nn.functional import relu
from torch.nn.modules.activation import ReLU

from torch.nn.functional import relu6
from torch.nn.modules.activation import ReLU6

from torch.nn.functional import leaky_relu
from torch.nn.modules.activation import LeakyReLU

from torch.nn.functional import prelu
from torch.nn.modules.activation import PReLU


LEAKY_RELU_A = 1 / 5.5
PI = 3.141592653589793


class Identity(nn.Module):
    def forward(self, x):
        return x

try:
    from torch.nn.functional import elu
except ImportError:

    def elu(x, alpha=1.0, inplace=False):
        if inplace:
            neg_x = th.clamp(alpha * (th.exp(x.clone()) - 1), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            return x
        else:
            return th.clamp(x, min=0) + th.clamp(alpha * (th.exp(x) - 1), max=0)


try:
    from torch.nn.modules.activation import ELU
except ImportError:

    class ELU(nn.Module):
        def __init__(self, alpha=1.0, inplace=False):
            super(ELU, self).__init__()
            self.alpha = alpha
            self.inplace = inplace

        def forward(self, x):
            return elu(x, self.alpha, self.inplace)

        def extra_repr(self):
            return "alpha={}{}".format(self.alpha, ", inplace=True" if self.inplace else "")


"""
Gaussian Error Linear Units (GELUs)
Dan Hendrycks, Kevin Gimpel
https://arxiv.org/abs/1606.08415
"""
try:
    from torch.nn.functional import gelu
except ImportError:

    def gelu(x):
        return 0.5 * x * (1 + th.tanh(math.sqrt(2 / PI) * (x + 0.044715 * th.pow(x, 3))))


try:
    from torch.nn.modules.activation import GELU
except ImportError:

    class GELU(nn.Module):
        def __init__(self):
            super(GELU, self).__init__()

        def forward(self, x):
            return gelu(x)


"""
Self-Normalizing Neural Networks
Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
https://arxiv.org/abs/1706.02515
"""
try:
    from torch.nn.functional import selu
except ImportError:

    def selu(x, inplace=False):
        if inplace:
            neg_x = th.clamp(1.0507009873554804934193349852946 * (th.exp(x.clone()) - 1), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            x *= 1.6732632423543772848170429916717
            return x
        else:
            return 1.6732632423543772848170429916717 * (
                th.clamp(x, min=0) + th.clamp(1.0507009873554804934193349852946 * (th.exp(x) - 1), max=0)
            )


try:
    from torch.nn.modules.activation import SELU
except ImportError:

    class SELU(nn.Module):
        def __init__(self, inplace=False):
            super(SELU, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            return selu(x, self.inplace)

        def extra_repr(self):
            return "inplace=True" if self.inplace else ""


"""
Continuously Differentiable Exponential Linear Units
Jonathan T. Barron
https://arxiv.org/abs/1704.07483
"""
try:
    from torch.nn.functional import celu
except ImportError:

    def celu(x, alpha=1.0, inplace=False):
        if inplace:
            neg_x = th.clamp(alpha * (th.exp(x.clone() / alpha) - 1), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            return x
        else:
            return th.clamp(x, min=0) + th.clamp(alpha * (th.exp(x / alpha) - 1), max=0)


try:
    from torch.nn.modules.activation import CELU
except ImportError:

    class CELU(nn.Module):
        def __init__(self, alpha=1.0, inplace=False):
            super(CELU, self).__init__()
            self.alpha = alpha
            self.inplace = inplace

        def forward(self, x):
            return celu(x, self.alpha)

        def extra_repr(self):
            return "alpha={}{}".format(self.alpha, ", inplace=True" if self.inplace else "")


"""
Continuously Differentiable Exponential Linear Units
Jonathan T. Barron
https://arxiv.org/abs/1704.07483
"""
try:
    from torch.nn.functional import rrelu
except ImportError:

    def rrelu(x, lower=1 / 8, upper=1 / 3, training=False, inplace=False):
        a = th.rand(1, device=x.device) * (1 / 3 - 1 / 8) + 1 / 8
        if inplace:
            neg_x = th.clamp(a * x.clone(), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            return x
        else:
            return th.clamp(x, min=0) + th.clamp(a * x, max=0)


try:
    from torch.nn.modules.activation import RReLU
except ImportError:

    class RReLU(nn.Module):
        def __init__(self, lower=1 / 8, upper=1 / 3, inplace=False):
            super(RReLU, self).__init__()
            self.lower = lower
            self.upper = upper
            self.inplace = inplace

        def forward(self, x):
            return rrelu(x, self.lower, self.upper, self.training, self.inplace)

        def extra_repr(self):
            return "lower={}, upper={}{}".format(self.lower, self.upper, ", inplace=True" if self.inplace else "")


"""
From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
André F. T. Martins, Ramón Fernandez Astudillo
http://arxiv.org/abs/1602.02068
"""


def sparsemax(x, dim=-1):
    # Sparsemax currently only handles 2-dim tensors,
    # so we reshape to a convenient shape and reshape back after sparsemax
    x = x.transpose(0, dim)
    original_size = x.size()
    x = x.reshape(x.size(0), -1)
    x = x.transpose(0, 1)
    dim = 1

    number_of_logits = x.size(dim)

    # Translate input by max for numerical stability
    x = x - th.max(x, dim=dim, keepdim=True)[0].expand_as(x)

    # Sort input in descending order.
    # (NOTE: Can be replaced with linear time selection method described here:
    # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
    zs = th.sort(x, dim=dim, descending=True)[0]
    range = th.arange(start=1, end=number_of_logits + 1, step=1, device=x.device, dtype=x.dtype).view(1, -1)
    range = range.expand_as(zs)

    # Determine sparsity of projection
    bound = 1 + range * zs
    cumulative_sum_zs = th.cumsum(zs, dim)
    is_gt = th.gt(bound, cumulative_sum_zs).type(x.type())
    k = th.max(is_gt * range, dim, keepdim=True)[0]

    # Compute threshold function
    zs_sparse = is_gt * zs

    # Compute taus
    taus = (th.sum(zs_sparse, dim, keepdim=True) - 1) / k
    taus = taus.expand_as(x)

    # Sparsemax
    output = th.max(th.zeros_like(x), x - taus)

    # Reshape back to original shape
    output = output.transpose(0, 1)
    output = output.reshape(original_size)
    output = output.transpose(0, dim)

    return output


class Sparsemax(nn.Module):
    """Sparsemax function."""
    def __init__(self, dim=-1):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = dim

    def forward(self, x):
        """Forward function.
        Args:
            x (th.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            th.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        x = x.transpose(0, self.dim)
        original_size = x.size()
        x = x.reshape(x.size(0), -1)
        x = x.transpose(0, 1)
        dim = 1

        number_of_logits = x.size(dim)

        # Translate input by max for numerical stability
        x = x - th.max(x, dim=dim, keepdim=True)[0].expand_as(x)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = th.sort(x, dim=dim, descending=True)[0]
        rg = th.arange(start=1, end=number_of_logits + 1, step=1, device=x.device, dtype=x.dtype).view(1, -1)
        rg = rg.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + rg * zs
        cumulative_sum_zs = th.cumsum(zs, dim)
        is_gt = th.gt(bound, cumulative_sum_zs).type(x.type())
        k = th.max(is_gt * rg, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (th.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(x)

        # Sparsemax
        self.output = th.max(th.zeros_like(x), x - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, x):
        """Backward function."""
        dim = 1

        nonzeros = th.ne(self.output, 0)
        x_sum = th.sum(x * nonzeros, dim=dim) / th.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (x - x_sum.expand_as(x))

        return self.grad_input

    def extra_repr(self):
        return "dim={}".format(self.dim)


"""
The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
Chris J. Maddison, Andriy Mnih, Yee Whye Teh
https://arxiv.org/abs/1611.00712
"""
try:
    from torch.nn.functional import gumbel_softmax
except ImportError:

    def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
        # Gumbel(logits,tau)
        gumbels = -th.empty_like(logits, memory_format=th.legacy_contiguous_format).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = th.zeros_like(logits, memory_format=th.legacy_contiguous_format).scatter_(dim, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            return y_soft


class GumbelSoftmax(nn.Module):
    def __init__(self, tau=1.0, hard=False, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.dim = dim
        self.hard = hard

    def forward(self, x):
        return gumbel_softmax(x, self.tau, self.hard, self.dim)

    def extra_repr(self):
        return "tau={}, hard={}, dim={}".format(self.tau, self.hard, self.dim)


"""
Maximum
"""
def maximum(x, dim=-1, scale_up=False, inplace=False):
    if inplace:
        x_ = x.clone()
        max_x = th.max(x_, dim=dim, keepdim=True)[0]
        max_mask = x_ == max_x
        x.masked_fill_(max_mask == 0, 0.0)
        if scale_up:
            x_sum = th.sum(x_, dim=dim, keepdim=True)
            max_sum = th.sum(x, dim=dim, keepdim=True)
            scale = x_sum / max_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            x *= scale
        return x
    else:
        max_x = th.max(x, dim=dim, keepdim=True)[0]
        max_mask = x == max_x
        masked_x = x * max_mask.float()
        if scale_up:
            x_sum = th.sum(x, dim=dim, keepdim=True)
            max_sum = th.sum(masked_x, dim=dim, keepdim=True)
            scale = x_sum / max_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            masked_x = masked_x * scale
        return masked_x


class Maximum(nn.Module):
    def __init__(self, dim=-1, scale_up=False, inplace=False):
        super(Maximum, self).__init__()
        self.dim = dim
        self.scale_up = scale_up
        self.inplace = inplace

    def forward(self, x):
        return maximum(x, self.dim, self.scale_up, self.inplace)

    def extra_repr(self):
        return "dim={}, scale_up={}{}".format(self.dim, self.scale_up, ", inplace=True" if self.inplace else "")


"""
Minimum
"""
def minimum(x, dim=-1, scale_up=False, inplace=False):
    if inplace:
        x_ = x.clone()
        min_x = th.min(x_, dim=dim, keepdim=True)[0]
        min_mask = x_ == min_x
        x.masked_fill_(min_mask == 0, 0.0)
        if scale_up:
            x_sum = th.sum(x_, dim=dim, keepdim=True)
            min_sum = th.sum(x, dim=dim, keepdim=True)
            scale = x_sum / min_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            x *= scale
        return x
    else:
        min_x = th.min(x, dim=dim, keepdim=True)[0]
        min_mask = x == min_x
        masked_x = x * min_mask.float()
        if scale_up:
            x_sum = th.sum(x, dim=dim, keepdim=True)
            min_sum = th.sum(masked_x, dim=dim, keepdim=True)
            scale = x_sum / min_sum
            scale.masked_fill_(scale.isnan(), 0.0)
            masked_x = masked_x * scale
        return masked_x


class Minimum(nn.Module):
    def __init__(self, dim=-1, scale_up=False, inplace=False):
        super(Minimum, self).__init__()
        self.dim = dim
        self.scale_up = scale_up
        self.inplace = inplace

    def forward(self, x):
        return minimum(x, self.dim, self.scale_up, self.inplace)

    def extra_repr(self):
        return "dim={}, scale_up={}{}".format(self.dim, self.scale_up, ", inplace=True" if self.inplace else "")


supported_act_funcs = {
    "none": Identity(),
    "softmax": Softmax(dim=-1),
    "sparsemax": Sparsemax(dim=-1),
    "gumbel_softmax": GumbelSoftmax(dim=-1),
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "relu": ReLU(),
    "relu6": ReLU6(),
    "leaky_relu": LeakyReLU(negative_slope=LEAKY_RELU_A),
    "prelu": PReLU(init=LEAKY_RELU_A),
    "elu": ELU(),
    "celu": CELU(),
    "selu": SELU(),
    "gelu": GELU(),
    "maximum": Maximum(dim=-1),
    "minimum": Minimum(dim=-1)
}


def map_activation_str_to_layer(act_func, **kw):
    if act_func not in supported_act_funcs:
        print(act_func)
        raise NotImplementedError

    act = supported_act_funcs[act_func]
    for k, v in kw.items():
        if hasattr(act, k):
            try:
                setattr(act, k, v)
            except:
                pass
    return act
