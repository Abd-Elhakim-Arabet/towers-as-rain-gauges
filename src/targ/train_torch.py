"""Training and inference loops for the TCN (PyTorch). MindSpore twin: train_ms.py.

Data layout (numpy, framework-free):
  dyn     list of float32 arrays (n_links, C, T) — one per augmentation variant
  stat    float32 (n_links, S)
  phys    float32 (n_links, 3) ITU a, b, length km (used by the physics head)
  target  float32 (n_links, T) reference rain rate [mm/h], NaN where there is none
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .models.tcn_torch import TCN

WET_MM_H = 0.1


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(logit, rate, y, rate_weight: float):
    ok = torch.isfinite(y)
    if ok.sum() == 0:
        return logit.sum() * 0
    wet = (y[ok] >= WET_MM_H)
    bce = F.binary_cross_entropy_with_logits(logit[ok], wet.float())
    if wet.sum() == 0:
        return bce
    # the rate is only learned where the reference says it rains; rain/no-rain is the
    # logit's job. Learning rate on dry minutes too teaches the network to shrink.
    mse = F.mse_loss(rate[ok][wet], torch.clamp(y[ok][wet], max=100.0))
    return bce + rate_weight * mse


def train(dyn: list[np.ndarray], stat: np.ndarray, phys: np.ndarray, target: np.ndarray,
          t_range: tuple[int, int],
          *, val_fn=None, steps: int = 4000, batch: int = 64, crop: int = 2560, warm: int = 1024,
          lr: float = 2e-3, rate_weight: float = 0.05, seed: int = 0, model_kw: dict | None = None,
          log_every: int = 250, dyn_probs: list[float] | None = None):
    """Random crops inside t_range; loss only after `warm` minutes of context.

    val_fn(model) -> float (lower is better) picks the best checkpoint.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = device()
    model = TCN(dyn[0].shape[1], stat.shape[1], **(model_kw or {})).to(dev)
    assert model.receptive_field <= warm + 1, model.receptive_field
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    dyn_t = [torch.from_numpy(d).to(dev) for d in dyn]
    stat_t = torch.from_numpy(stat).to(dev)
    phys_t = torch.from_numpy(phys).to(dev)
    tgt_t = torch.from_numpy(target).to(dev)
    probs = dyn_probs or [1.0 / len(dyn)] * len(dyn)
    t0, t1 = t_range
    # only links that have at least one wet target in range are worth sampling more often
    n_links = stat.shape[0]
    best, best_state, hist = float("inf"), None, []
    tic = time.time()
    for step in range(1, steps + 1):
        model.train()
        li = torch.from_numpy(rng.integers(0, n_links, batch)).to(dev)
        st = rng.integers(t0, t1 - crop, batch)
        idx = torch.from_numpy(st[:, None] + np.arange(crop)[None]).to(dev)
        v = rng.choice(len(dyn), p=probs)
        x = dyn_t[v][li[:, None], :, idx].permute(0, 2, 1)       # (B, C, crop)
        y = tgt_t[li[:, None], idx][:, warm:]
        logit, rate = model(x, stat_t[li], phys_t[li])
        loss = loss_fn(logit[:, warm:], rate[:, warm:], y, rate_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            msg = f"step {step:5d} loss {loss.item():.4f} ({time.time() - tic:.0f}s)"
            if val_fn is not None:
                score = val_fn(model)
                hist.append({"step": step, "train_loss": float(loss.item()), "val": float(score)})
                msg += f" val {score:.4f}"
                if score < best:
                    best = score
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def predict(model, dyn: np.ndarray, stat: np.ndarray, phys: np.ndarray,
            t_range: tuple[int, int] | None = None,
            context: int = 1024, link_batch: int = 32):
    """Full-sequence inference. Returns (prob, rate) float32 (n_links, T or range)."""
    dev = next(model.parameters()).device
    model.eval()
    T = dyn.shape[-1]
    a, b = t_range or (0, T)
    s = max(0, a - context)
    P, R = [], []
    for i in range(0, dyn.shape[0], link_batch):
        x = torch.from_numpy(dyn[i:i + link_batch, :, s:b]).to(dev)
        logit, rate = model(x, torch.from_numpy(stat[i:i + link_batch]).to(dev),
                            torch.from_numpy(phys[i:i + link_batch]).to(dev))
        P.append(torch.sigmoid(logit)[:, a - s:].cpu().numpy())
        R.append(rate[:, a - s:].cpu().numpy())
    return np.concatenate(P), np.concatenate(R)


def state_numpy(model) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
