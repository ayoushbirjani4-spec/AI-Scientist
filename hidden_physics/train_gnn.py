"""Generate data and train the Stage B GNN ensemble.

python -m hidden_physics.train_gnn --n-traj 4000 --members 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .planner import SCRIPT
from .learned import (Dataset, LearnedPredictor, calibrate_error_corr, calibrate_floor,
                      calibrate_param_bias, PARAM_BIAS_SCALE,
                      generate_dataset, rollout_error, save_ensemble, train_member)


def split(ds: Dataset, seed: int = 0):
    """80 / 10 / 10 by trajectory = by (scene, g, mu) draw."""
    idx = np.random.default_rng(seed).permutation(len(ds.lengths))
    a, b = int(0.8 * len(idx)), int(0.9 * len(idx))
    return ds.subset(idx[:a]), ds.subset(idx[a:b]), ds.subset(idx[b:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=4000)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs-one", type=int, default=30)
    ap.add_argument("--epochs-roll", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-g-scaled", action="store_true", help="train the original g-input model")
    ap.add_argument("--recalibrate", action="store_true",
                    help="no training: recompute floor and error correlation for the model at --out")
    ap.add_argument("--data", default="data/gnn_dataset.npz")
    ap.add_argument("--out", default="models/gnn_gscaled.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()

    data_path = Path(a.data)
    if data_path.exists():
        ds = Dataset.load(data_path)
        print(f"loaded {len(ds.lengths)} trajectories from {data_path}")
    else:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        ds = generate_dataset(a.n_traj, seed=12345)
        ds.save(data_path)
        print(f"generated {len(ds.lengths)} trajectories in {time.time() - t0:.0f}s -> {data_path}")
    train, val, test = split(ds)
    if a.recalibrate:
        pred = LearnedPredictor(a.out)
        floor = calibrate_floor(pred.models, val)
        corr = calibrate_error_corr(pred.models, val, floor)
        pred.floor_var, pred.error_corr = floor, corr
        bias = calibrate_param_bias(pred, SCRIPT[:5])
        save_ensemble(a.out, pred.models, floor, pred.meta, corr, bias)
        print(f"shared bias sd: g {bias[0]:.2%} (relative), mu {bias[1]:.4f} (absolute), "
              f"after coverage scale {PARAM_BIAS_SCALE}")
        print(f"recalibrated {a.out}: lag-1 / lag-10 error correlation per column (x, y, vx, vy):")
        for c in range(4):
            print(f"  {np.diag(corr[c], 1)[5:20].mean():.2f} / {np.diag(corr[c], 10)[5:20].mean():.2f}")
        return
    print(f"split: {len(train.lengths)} train / {len(val.lengths)} val / {len(test.lengths)} test trajectories")

    models, hists = [], []
    for m in range(a.members):
        tm = time.time()
        model, hist = train_member(train, val, seed=m, epochs_one=a.epochs_one,
                                   epochs_roll=a.epochs_roll, hidden=a.hidden,
                                   g_scaled=not a.no_g_scaled)
        model.eval()
        models.append(model)
        hists.append(hist)
        print(f"member {m}: val pos RMSE {rollout_error(model, val):.4f} m ({time.time() - tm:.0f}s)", flush=True)

    floor = calibrate_floor(models, val)
    corr = calibrate_error_corr(models, val, floor)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"n_traj": len(ds.lengths), "members": a.members, "hidden": a.hidden,
            "epochs_one": a.epochs_one, "epochs_roll": a.epochs_roll, "split_seed": 0,
            "data": str(data_path), "g_scaled": not a.no_g_scaled, "train_seconds": time.time() - t0}
    save_ensemble(out, models, floor, meta, corr)
    bias = calibrate_param_bias(LearnedPredictor(out), SCRIPT[:5])
    save_ensemble(out, models, floor, meta, corr, bias)
    (out.parent / "train_log.json").write_text(json.dumps({"meta": meta, "history": hists}, indent=1))
    print(f"saved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
