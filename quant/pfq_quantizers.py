import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def ste_round(x):
    return (x - x.detach()) + torch.round(x.detach())

class LSQActivation(nn.Module):
    def __init__(self, bit: int, per_channel: bool = False, ch_axis: int = 1, init_s: float = None):
        super().__init__()
        assert 2 <= bit <= 8
        self.bit = bit
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.Qn = -(2**(bit-1))
        self.Qp = (2**(bit-1)) - 1
        self.s = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("_init", torch.tensor(0, dtype=torch.uint8))
        self.init_s = init_s

    @torch.no_grad()
    def _init_from_x(self, x):
        if self.per_channel:
            reduce_dims = [d for d in range(x.ndim) if d != self.ch_axis]
            s_val = 2 * x.abs().mean(dim=reduce_dims, keepdim=True) / math.sqrt(self.Qp)
        else:
            s_val = 2 * x.abs().mean() / math.sqrt(self.Qp)
        self.s.data.copy_(torch.tensor(self.init_s, device=x.device, dtype=x.dtype) if self.init_s else s_val)
        self._init.fill_(1)

    def forward(self, x):
        if self.training and self._init.item() == 0:
            self._init_from_x(x)
        s = self.s
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        s = (s - s.detach()) * g + s.detach()
        q = torch.clamp(ste_round(x / (s + 1e-12)), self.Qn, self.Qp)
        return q * s

class AdaRoundWeight(nn.Module):
    def __init__(self, w: torch.Tensor, bit: int, per_channel: bool = True, ch_axis: int = 0):
        super().__init__()
        assert 2 <= bit <= 8
        self.bit = bit
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.Qn = -(2**(bit-1))
        self.Qp = (2**(bit-1)) - 1
        self.register_buffer("sw", torch.zeros(1))
        self.v = nn.Parameter(torch.zeros_like(w))
        self.register_buffer("_init", torch.tensor(0, dtype=torch.uint8))
        self._init_from_w(w)

    @torch.no_grad()
    def _init_from_w(self, w):
        if self.per_channel and w.ndim >= 2:
            reduce_dims = [d for d in range(w.ndim) if d != self.ch_axis]
            s = 2 * w.abs().mean(dim=reduce_dims, keepdim=True) / math.sqrt(self.Qp)
        else:
            s = 2 * w.abs().mean() / math.sqrt(self.Qp)
        self.sw.data = s
        frac = (w / (self.sw + 1e-12)) - torch.floor(w / (self.sw + 1e-12))
        self.v.data.copy_(torch.logit(torch.clamp(frac, 1e-4, 1-1e-4)))
        self._init.fill_(1)

    def forward(self, w):
        sw = self.sw
        # Use STE for the sigmoid to ensure proper quantization
        sigmoid_v = torch.sigmoid(self.v)
        # Apply STE to make the rounding differentiable
        rounded_sigmoid = (sigmoid_v - sigmoid_v.detach()) + torch.round(sigmoid_v.detach())
        q = torch.clamp(torch.floor(w / (sw + 1e-12)) + rounded_sigmoid, self.Qn, self.Qp)
        return q * sw
