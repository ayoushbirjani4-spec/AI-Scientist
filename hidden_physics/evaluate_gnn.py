"""Evaluate the Stage B GNN ensemble against the Stage A simulator predictor.

python -m hidden_physics.evaluate_gnn --seeds 10 --workers 2

1. Rollout accuracy on held-out trajectories (unseen scene + parameter draws):
   position error vs time per experiment kind, one-step velocity error for free
   flight / contact / impact transitions, and calibration of the predictive spread.
2. Speed: predicting 400 hypotheses, Stage A vs Stage B.
3. Inference: the full agent loop with Stage B as the world model (the world is
   still Pymunk), paired with the Stage A sweep runs (same seeds, truths, noise).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .adapter import LIKELIHOOD_COLS, NOISE_LEVELS
from .agent import run_episode
from .env import ExperimentSpec, World, simulate
from .evaluate import G_REL_TOL, MU_ABS_TOL, draw_truth, steps_to_stay, traj_error
from .learned import Dataset, LearnedPredictor, rollout_dataset, targets, transition_labels
from .planner import STRATEGIES
from .predictor import SimulatorPredictor
from .report import AXIS, COLORS, GRID, INK, MUTED, SURFACE  # noqa: F401  (shared style)
from .train_gnn import split

KIND_COLORS = {"drop": "#2a78d6", "flat_slide": "#eb6834", "ramp_slide": "#1baf7a", "impulse": "#eda100"}
MODEL = "models/gnn_ensemble.pt"
_PRED = None


# ------------------------------------------------------------ 1. rollout accuracy
def rollout_accuracy(pred: LearnedPredictor, test: Dataset, out: Path) -> list[str]:
    rolls = np.stack([rollout_dataset(m, test) for m in pred.models])        # [M, N, T, 6]
    mean = rolls.mean(0)
    T = test.states.shape[1]
    mask = np.arange(T)[None] < test.lengths[:, None]
    pos_err = np.sqrt(((mean[..., :2] - test.states[..., :2]) ** 2).sum(-1))  # [N, T]

    # error vs time per kind
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(T) / 30
    L = ["## 1. Rollout accuracy on held-out trajectories", "",
         f"{len(test.lengths)} test trajectories, each an unseen (scene, g, μ) draw. Open-loop rollout of the "
         "ensemble mean from the known setup, compared with the noiseless simulator.", "",
         "| experiment | n | position RMSE (m) | median final position error (m) |", "|---|---|---|---|"]
    for k in KIND_COLORS:
        sel = test.kinds == k
        if not sel.any():
            continue
        e, m = pos_err[sel], mask[sel]
        med = np.array([np.nanmedian(np.where(m[:, j], e[:, j], np.nan)) if m[:, j].any() else np.nan
                        for j in range(T)])
        ax.plot(t, med, color=KIND_COLORS[k], lw=2, label=k)
        final = [e[i, test.lengths[sel][i] - 1] for i in range(sel.sum())]
        L.append(f"| {k} | {sel.sum()} | {np.sqrt((e[m] ** 2).mean()):.4f} | {np.median(final):.4f} |")
    L.append(f"| all | {len(test.lengths)} | {np.sqrt((pos_err[mask] ** 2).mean()):.4f} | |")
    for name, nz in NOISE_LEVELS.items():
        ax.axhline(nz.pos, color=MUTED, lw=1, ls="--")
        ax.text(t[-1], nz.pos, f" {name} noise", va="center", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("time into experiment (s)")
    ax.set_ylabel("median position error (m)")
    ax.set_title("Stage B rollout error vs time (dashed: observation noise σ)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "rollout_error_vs_time.png", dpi=150)
    plt.close(fig)

    # one-step error by transition type (physical units)
    lab = transition_labels(test)
    S = torch.from_numpy(test.states)
    ctx = [torch.from_numpy(a) for a in (test.objf, test.surf, test.smask, test.glob)]
    dv_err = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for i in range(len(test.lengths)):
            n = test.lengths[i] - 1
            s0, s1 = S[i, :n], S[i, 1:n + 1]
            c = [x[i:i + 1].expand(n, *x.shape[1:]) for x in ctx]
            nxt = torch.stack([m.step(s0, *c) for m in pred.models]).mean(0)
            err = (nxt[:, 3:5] - s1[:, 3:5]).norm(dim=1).numpy()
            for cat in dv_err:
                dv_err[cat].extend(err[lab[i, :n] == cat])
    L += ["", "One-step velocity error (ensemble mean), by transition type:", "",
          "| transition | count | median |Δv error| (m/s) | 95th percentile |", "|---|---|---|---|"]
    for cat, nm in [(0, "free flight"), (1, "contact (sliding/resting)"), (2, "impact (lands this step)")]:
        e = np.array(dv_err[cat])
        if len(e):
            L.append(f"| {nm} | {len(e)} | {np.median(e):.4f} | {np.percentile(e, 95):.4f} |")

    # calibration of predictive spread (ensemble var + floor) on the likelihood columns
    var = rolls[..., LIKELIHOOD_COLS].var(0) + pred.floor_var[None, :T]
    z = np.abs(mean[..., LIKELIHOOD_COLS] - test.states[..., LIKELIHOOD_COLS]) / np.sqrt(var)
    later = mask & (np.arange(T)[None] > 0)     # frame 0 is the known setup: zero spread, zero error
    cov = (z[later] < 1.645).mean(0)
    L += ["", "Calibration: fraction of test predictions (after the first frame) inside the model's own "
          "90 % band (ensemble spread + calibrated floor; ideal 0.90):", "",
          "| x | y | vx | vy |", "|---|---|---|---|", "| " + " | ".join(f"{c:.2f}" for c in cov) + " |", ""]
    return L


# ------------------------------------------------------------ 1b. identifiability
def g_bias(pred: LearnedPredictor) -> list[str]:
    """Fit g to noise-free simulator trajectories through the learned model (mu known).
    A model can roll out accurately and still map g wrongly; this measures the latter."""
    from .planner import SCRIPT
    specs = [ExperimentSpec.make("drop", "ball", height=1.5), ExperimentSpec.make("drop", "block", height=1.0),
             SCRIPT[1], SCRIPT[2], SCRIPT[3]]
    gs = np.linspace(2, 20, 721)
    true_gs = [4, 8, 12, 16]
    L = ["## 1b. Can g be recovered through the model?", "",
         "Best-fit g (grid step 0.025) when noise-free simulator trajectories are fitted through the "
         "learned model with μ = 0.3 known. Stage A recovers g exactly.", "",
         "| experiment | " + " | ".join(f"g* = {g}" for g in true_gs) + " |", "|---|" + "---|" * len(true_gs)]
    rel = []
    for sp in specs:
        row = []
        for gt in true_gs:
            _, s = simulate(sp, gt, 0.3)
            m = pred.predict_many(sp, gs, np.full_like(gs, 0.3))
            sse = (((m - s[:, LIKELIHOOD_COLS]) / np.array([0.01, 0.01, 0.1, 0.1])) ** 2).sum((1, 2))
            b = gs[np.argmin(sse)]
            rel.append(abs(b - gt) / gt)
            row.append(f"{b:.2f}")
        L.append(f"| {sp} | " + " | ".join(row) + " |")
    L += ["", f"Median |bias| {np.median(rel):.1%}, worst {np.max(rel):.1%}.", ""]
    return L


# ------------------------------------------------------------ 2. speed
def speed(pred: LearnedPredictor) -> list[str]:
    spec = ExperimentSpec.make("ramp_slide", "block", angle_deg=35, start=2)
    gs, mus = np.linspace(2, 20, 400), np.linspace(0.05, 0.8, 400)
    a = SimulatorPredictor(cache_size=0)
    t0 = time.perf_counter(); a.predict_many(spec, gs, mus); ta = time.perf_counter() - t0
    t0 = time.perf_counter(); pred.predict_many_with_var(spec, gs, mus); tb = time.perf_counter() - t0
    return ["## 2. Speed", "", "Predicting one ramp-slide experiment for 400 hypotheses:", "",
            "| predictor | seconds |", "|---|---|",
            f"| Stage A (Pymunk) | {ta:.3f} |", f"| Stage B (GNN ensemble of {len(pred.models)}) | {tb:.3f} |", ""]


# ------------------------------------------------------------ 3. inference
def _init(path):
    global _PRED
    torch.set_num_threads(1)
    _PRED = LearnedPredictor(path)


def _job(args):
    strategy, noise_name, seed, budget = args
    noise = NOISE_LEVELS[noise_name]
    truth = draw_truth(seed)
    world = World(truth, noise.pos, noise.vel, seed=seed)
    t0 = time.perf_counter()
    recs = run_episode(world, STRATEGIES[strategy](), noise, budget, seed=seed, predictor=_PRED)
    wall = time.perf_counter() - t0
    n = steps_to_stay(recs, [truth] * budget)
    return {"strategy": strategy, "noise": noise_name, "seed": seed,
            "steps_to_threshold": n if n is not None else "",
            "g_rel_err_curve": json.dumps([abs(r["g_mean"] - truth.g) / truth.g for r in recs]),
            "mu_abs_err_curve": json.dumps([abs(r["mu_mean"] - truth.mu) for r in recs]),
            "cov_g": float(np.mean([r["g_ci"][0] <= truth.g <= r["g_ci"][1] for r in recs])),
            "cov_mu": float(np.mean([r["mu_ci"][0] <= truth.mu <= r["mu_ci"][1] for r in recs])),
            "final_traj_rmse": traj_error(recs[-1]["g_mean"], recs[-1]["mu_mean"], truth),
            "seconds": wall}


def _censored(curves_g, curves_mu, budget):
    out = []
    for cg, cm in zip(curves_g, curves_mu):
        ok = [(g <= G_REL_TOL and m <= MU_ABS_TOL) for g, m in zip(cg[:budget], cm[:budget])]
        n = next((i + 1 for i in range(len(ok)) if all(ok[i:])), budget + 1)
        out.append(n)
    return np.array(out)


def inference(path: str, seeds: int, workers: int, out: Path, noises=("medium", "high"),
              strategies=("active", "random")) -> list[str]:
    stage_a = list(csv.DictReader(open("results/metrics.csv")))
    budget = len(json.loads(stage_a[0]["g_rel_err_curve"]))
    jobs = [(s, n, seed, budget) for n in noises for s in strategies for seed in range(seeds)]
    with Pool(workers, initializer=_init, initargs=(path,)) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(r)
            print(f"  inference {i}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["noise"], r["strategy"], r["seed"]))
    with open(out / "stageB_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    L = ["## 3. Inference with Stage B as the world model", "",
         f"Same seeds (0–{seeds - 1}), hidden physics and observation noise as the Stage A sweep; the world is "
         f"still Pymunk. Budget {budget}; threshold |g error| ≤ 1 %, |μ error| ≤ 0.01, held to the end "
         f"(never reached counts as {budget + 1}).", "",
         "| noise | strategy | predictor | mean exps (never = 11) | reached | final g rel err (median) | "
         "final μ err (median) | 90% CI coverage g / μ | traj RMSE m (median) | s / run |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, len(noises), figsize=(5.5 * len(noises), 6.5), sharex=True, squeeze=False)
    x = np.arange(1, budget + 1)
    for j, n in enumerate(noises):
        for s in strategies:
            A = [r for r in stage_a if r["noise"] == n and r["strategy"] == s and int(r["seed"]) < seeds]
            B = [r for r in rows if r["noise"] == n and r["strategy"] == s]
            for tag, R, ls in [("A (simulator)", A, "-"), ("B (GNN)", B, "--")]:
                cg = [json.loads(r["g_rel_err_curve"]) for r in R]
                cm = [json.loads(r["mu_abs_err_curve"]) for r in R]
                y = _censored(cg, cm, budget)
                covg = np.mean([float(r.get("ci_coverage_g", r.get("cov_g"))) for r in R])
                covm = np.mean([float(r.get("ci_coverage_mu", r.get("cov_mu"))) for r in R])
                secs = np.mean([float(r.get("total_compute_s", r.get("seconds"))) for r in R])
                L.append(f"| {n} | {s} | {tag} | {y.mean():.2f} | {(y <= budget).sum()}/{len(y)} | "
                         f"{np.median([c[-1] for c in cg]):.4f} | {np.median([c[-1] for c in cm]):.4f} | "
                         f"{covg:.2f} / {covm:.2f} | {np.median([float(r['final_traj_rmse']) for r in R]):.4f} | "
                         f"{secs:.0f} |")
                axes[0, j].plot(x, np.median(cg, 0), color=COLORS[s], ls=ls, lw=2, label=f"{s} · {tag}")
                axes[1, j].plot(x, np.median(cm, 0), color=COLORS[s], ls=ls, lw=2, label=f"{s} · {tag}")
        for i, tol in enumerate([G_REL_TOL, MU_ABS_TOL]):
            axes[i, j].axhline(tol, color=MUTED, lw=1, ls=":")
            axes[i, j].set_yscale("log")
        axes[0, j].set_title(f"{n} noise", loc="left")
        axes[1, j].set_xlabel("experiment number")
    axes[0, 0].set_ylabel("median |g error| / g")
    axes[1, 0].set_ylabel("median |μ error|")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Estimation error with the simulator (solid) vs the learned GNN (dashed) as world model",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "stageA_vs_stageB.png", dpi=150)
    plt.close(fig)
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default="data/gnn_dataset.npz")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--out", default="results/gnn")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pred = LearnedPredictor(a.model)
    _, _, test = split(Dataset.load(a.data))
    L = ["# Stage B: GNN world model vs Stage A simulator", "",
         f"Model: {pred.version}, trained on {pred.meta['n_traj']} trajectories "
         f"(80/10/10 split by scene and parameter draw).", "",
         "![rollout error](rollout_error_vs_time.png)", ""]
    L += rollout_accuracy(pred, test, out)
    L += g_bias(pred)
    L += speed(pred)
    if not a.skip_inference:
        L += ["![stage A vs B](stageA_vs_stageB.png)", ""]
        L += inference(a.model, a.seeds, a.workers, out)
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
