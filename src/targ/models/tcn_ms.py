"""Causal TCN, MindSpore version (twin of tcn_torch.py, same layer names and maths).

Runs in .venv-ms (MindSpore 2.x). On Huawei Cloud the same file trains on
Ascend via ModelArts by setting framework: mindspore in configs/backend.yaml.
"""

from __future__ import annotations

import mindspore as ms
from mindspore import nn, ops


def _gelu(x):
    # exact (erf) GELU, same as torch.nn.functional.gelu default
    return 0.5 * x * (1.0 + ops.erf(x / 1.4142135623730951))


class CausalBlock(nn.Cell):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation, pad_mode="valid", has_bias=True)
        self.conv2 = nn.Conv1d(ch, ch, kernel, dilation=dilation, pad_mode="valid", has_bias=True)
        self.drop = nn.Dropout(p=dropout)

    def construct(self, x):
        h = _gelu(self.conv1(ops.pad(x, (self.pad, 0))))
        h = self.drop(h)
        h = self.conv2(ops.pad(h, (self.pad, 0)))
        return _gelu(x + h)


class TCN(nn.Cell):
    def __init__(self, n_dynamic: int, n_static: int, hidden: int = 48, kernel: int = 3,
                 n_levels: int = 8, dropout: float = 0.1, head: str = "free"):
        super().__init__()
        self.head_kind = head
        self.inp = nn.Conv1d(n_dynamic + n_static, hidden, 1, pad_mode="valid", has_bias=True)
        self.blocks = nn.CellList([CausalBlock(hidden, kernel, 2 ** i, dropout)
                                   for i in range(n_levels)])
        self.head = nn.Conv1d(hidden, 2, 1, pad_mode="valid", has_bias=True)

    def construct(self, dyn, stat, phys=None):
        s = ops.broadcast_to(stat[:, :, None], (stat.shape[0], stat.shape[1], dyn.shape[-1]))
        h = self.inp(ops.cat([dyn, s], axis=1))
        for b in self.blocks:
            h = b(h)
        out = self.head(h)
        if self.head_kind == "physics":
            a, b, L = phys[:, 0:1], phys[:, 1:2], phys[:, 2:3]
            corr = -1.0 + 5.0 * ops.sigmoid(out[:, 1])   # dB, bounded to [-1, 4]
            A = ops.relu(dyn[:, 0] - corr) + 1e-6
            return out[:, 0], ops.clamp(ops.pow(A / L / a, 1 / b), max=300.0)
        return out[:, 0], ops.softplus(out[:, 1])


def load_torch_state(net: TCN, state: dict) -> None:
    """Copy a PyTorch state_dict (numpy arrays) into the MindSpore net.

    PyTorch Conv1d weight is (out, in, k); MindSpore Conv1d stores (out, in, 1, k).
    """
    params = {p.name: p for p in net.get_parameters()}
    for name, arr in state.items():
        ms_name = name
        if ms_name not in params:
            raise KeyError(f"{name} has no MindSpore counterpart")
        p = params[ms_name]
        if tuple(p.shape) != tuple(arr.shape):
            arr = arr.reshape(tuple(p.shape))
        p.set_data(ms.Tensor(arr, dtype=p.dtype))
