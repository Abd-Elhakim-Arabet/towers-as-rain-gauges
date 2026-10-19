"""Phase 2 on MindSpore: proves the Huawei framework path works with the same model.

Two checks, run from two environments (torch and MindSpore cannot share one venv here):

    .venv/bin/python    scripts/phase2_mindspore.py export   # PyTorch: dump weights + a test batch
    .venv-ms/bin/python scripts/phase2_mindspore.py check    # MindSpore: load, compare, train briefly

1. Port: the trained PyTorch weights are loaded into the MindSpore twin and both
   frameworks run the same 64 real Swedish sub-links × 3 days. Outputs must match
   to float32 round-off.
2. Training: the MindSpore twin trains from scratch for a few hundred steps on CPU
   (on Ascend via ModelArts once available) on the same data, to show the training
   loop runs and the loss falls. A full MindSpore training run is a config switch
   (framework: mindspore) once Huawei Cloud compute is available.

Writes results/phase2_mindspore.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "runs" / "phase2" / "ms_check"
OUT = ROOT / "results" / "phase2_mindspore.json"


def export():
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase2_openmrg as P2
    from targ import features as feat
    from targ.data import openmrg
    from targ.evaluation import OPENMRG_SPLITS
    from targ.models.tcn_torch import TCN

    p1 = json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())
    p2 = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())
    ds = openmrg.load_cml_1min().sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    ds = ds.sel(cml_id=np.setdiff1d(ds.cml_id.values, p1["qc_dropped"]))
    times = ds.time.values
    radar = openmrg.radar_along_cmls(ds).sel(cml_id=ds.cml_id, time=slice(times[0], times[-1])).load()
    dyn = P2.build_inputs(ds)
    stat = feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values)
    phys = feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)
    tgt = P2.minute_target(radar, times)
    a, _ = P2.idx(times, OPENMRG_SPLITS["test"])
    sel = np.random.default_rng(0).choice(dyn.shape[0], 64, replace=False)
    x = dyn[sel, :, a + 20 * 1440:a + 23 * 1440]          # 3 days in late August
    model = TCN(len(feat.CHANNELS), 3, **p2["variants"]["physics"]["model_kw"])
    model.load_state_dict(torch.load(ROOT / "runs" / "phase2" / "physics_selected.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        lg, rt = model(torch.from_numpy(x), torch.from_numpy(stat[sel]), torch.from_numpy(phys[sel]))
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DIR / "batch.npz", x=x, stat=stat[sel], phys=phys[sel], logit=lg.numpy(), rate=rt.numpy())
    np.savez(DIR / "weights.npz", **{k: v.numpy() for k, v in model.state_dict().items()})
    tr0, tr1 = P2.idx(times, OPENMRG_SPLITS["train"])
    np.savez(DIR / "train.npz", dyn=dyn[:, :, tr0:tr1], stat=stat, phys=phys, tgt=tgt[:, tr0:tr1])
    json.dump(p2["variants"]["physics"]["model_kw"], open(DIR / "model_kw.json", "w"))
    print("exported", x.shape)


def check():
    import mindspore as ms
    from mindspore import nn, ops

    from targ.models.tcn_ms import TCN, load_torch_state

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    kw = json.load(open(DIR / "model_kw.json"))
    b = np.load(DIR / "batch.npz")
    w = dict(np.load(DIR / "weights.npz"))
    net = TCN(b["x"].shape[1], 3, **kw)
    load_torch_state(net, w)
    net.set_train(False)
    lg, rt = net(ms.Tensor(b["x"]), ms.Tensor(b["stat"]), ms.Tensor(b["phys"]))
    lg, rt = lg.asnumpy(), rt.asnumpy()
    p_t, p_m = 1 / (1 + np.exp(-b["logit"])), 1 / (1 + np.exp(-lg))
    port = {
        "max_abs_diff_logit": float(np.abs(lg - b["logit"]).max()),
        "max_abs_diff_rain_mm_h": float(np.abs(rt - b["rate"]).max()),
        "max_abs_diff_prob": float(np.abs(p_m - p_t).max()),
        "rain_no_rain_agreement": float(np.mean((p_m > 0.2) == (p_t > 0.2))),
        "n_values": int(lg.size),
    }
    print("port check:", port, flush=True)

    # short training from scratch on CPU
    d = np.load(DIR / "train.npz")
    dyn, stat, phys, tgt = d["dyn"], d["stat"], d["phys"], d["tgt"]
    ms.set_seed(0)
    rng = np.random.default_rng(0)
    net2 = TCN(b["x"].shape[1], 3, **kw)
    opt = nn.AdamWeightDecay(net2.trainable_params(), learning_rate=2e-3, weight_decay=1e-4)
    crop, warm, batch, steps = 1536, 1024, 16, 300

    def loss_fn(x, s, p, y, m_wet, m_ok):
        logit, rate = net2(x, s, p)
        logit, rate = logit[:, warm:], rate[:, warm:]
        bce = ops.binary_cross_entropy_with_logits(logit, m_wet, ops.ones_like(logit) * m_ok,
                                                   ops.ones_like(logit), reduction="sum") / m_ok.sum()
        mse = ((rate - y) ** 2 * m_wet).sum() / (m_wet.sum() + 1e-6)
        return bce + 0.05 * mse

    grad_fn = ms.value_and_grad(loss_fn, None, opt.parameters)
    hist, tic = [], time.time()
    net2.set_train(True)
    for step in range(1, steps + 1):
        li = rng.integers(0, dyn.shape[0], batch)
        st = rng.integers(0, dyn.shape[-1] - crop, batch)
        x = np.stack([dyn[i, :, s:s + crop] for i, s in zip(li, st)])
        y = np.stack([tgt[i, s:s + crop] for i, s in zip(li, st)])[:, warm:]
        ok = np.isfinite(y)
        wet = (ok & (np.nan_to_num(y) >= 0.1)).astype(np.float32)
        yv = np.clip(np.nan_to_num(y), 0, 100).astype(np.float32)
        loss, grads = grad_fn(ms.Tensor(x), ms.Tensor(stat[li]), ms.Tensor(phys[li]), ms.Tensor(yv),
                              ms.Tensor(wet), ms.Tensor(ok.astype(np.float32)))
        opt(grads)
        hist.append(float(loss.asnumpy()))
        if step % 50 == 0:
            print(f"ms step {step} loss {np.mean(hist[-50:]):.4f} ({time.time() - tic:.0f}s)", flush=True)
    res = {"framework": f"MindSpore {ms.__version__} (CPU)", "port_check": port,
           "train_smoke": {"steps": steps, "batch": batch, "crop_min": crop,
                           "loss_first50": float(np.mean(hist[:50])),
                           "loss_last50": float(np.mean(hist[-50:])),
                           "seconds": round(time.time() - tic, 1)}}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"export": export, "check": check}[sys.argv[1]]()
