"""Causal temporal convolutional network (TCN), PyTorch version.

Each output minute sees only that minute and earlier ones (left-padded dilated
convolutions). Two outputs per minute: a rain/no-rain logit and a rain rate [mm/h].

head="free":    the rate is whatever the network says (softplus, never negative).
head="physics": the network outputs a correction in dB (wet antenna + baseline
                drift, bounded to -1..4 dB) that is subtracted from the excess loss; the ITU-R P.838
                power law then turns the rest into mm/h. The AI learns the
                corrections; the physics does the conversion.

The MindSpore twin is tcn_ms.py; layer names match so weights can be copied
across (tcn_ms.load_torch_state).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.conv1(F.pad(x, (self.pad, 0))))
        h = self.drop(h)
        h = self.conv2(F.pad(h, (self.pad, 0)))
        return F.gelu(x + h)


class TCN(nn.Module):
    def __init__(self, n_dynamic: int, n_static: int, hidden: int = 48, kernel: int = 3,
                 n_levels: int = 8, dropout: float = 0.1, head: str = "free"):
        super().__init__()
        self.head_kind = head
        self.inp = nn.Conv1d(n_dynamic + n_static, hidden, 1)
        self.blocks = nn.ModuleList(CausalBlock(hidden, kernel, 2 ** i, dropout)
                                    for i in range(n_levels))
        self.head = nn.Conv1d(hidden, 2, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(2 * b.pad for b in self.blocks)

    def forward(self, dyn, stat, phys=None):
        """dyn (B, C, T) with excess loss in channel 0, stat (B, S),
        phys (B, 3) = ITU a, ITU b, length km (physics head only) → logit, rate (B, T)."""
        s = stat[:, :, None].expand(-1, -1, dyn.shape[-1])
        h = self.inp(torch.cat([dyn, s], dim=1))
        for b in self.blocks:
            h = b(h)
        out = self.head(h)
        if self.head_kind == "physics":
            a, b, L = phys[:, 0:1], phys[:, 1:2], phys[:, 2:3]
            corr = -1.0 + 5.0 * torch.sigmoid(out[:, 1])   # dB, bounded to [-1, 4]
            A = F.relu(dyn[:, 0] - corr) + 1e-6
            return out[:, 0], torch.clamp((A / L / a) ** (1 / b), max=300.0)
        return out[:, 0], F.softplus(out[:, 1])
