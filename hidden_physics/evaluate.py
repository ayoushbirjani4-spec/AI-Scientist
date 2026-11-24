"""Evaluation harness. The only code that sees true hidden parameters.

python -m hidden_physics.evaluate --seeds 30 --budget 10 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .adapter import NOISE_LEVELS
from .agent import run_episode
from .change import ChangeDetector
from .env import ExperimentSpec, HiddenPhysics, World, simulate
from .adapter import LIKELIHOOD_COLS
from .planner import STRATEGIES
from .store import ExperimentStore

G_REL_TOL = 0.01      # |g - g*| / g* <= 1 %
MU_ABS_TOL = 0.01     # |mu - mu*| <= 0.01
TRUTH_G = (3.0, 18.0)
TRUTH_MU = (0.10, 0.70)

TEST_SPECS = [  # held-out experiments for trajectory prediction error
    ExperimentSpec.make("drop", "block", height=1.2),
    ExperimentSpec.make("flat_slide", "block", v0=3.0),
    ExperimentSpec.make("ramp_slide", "heavy_block", angle_deg=42, start=2.0),
    ExperimentSpec.make("impulse", "ball", impulse=3.0, direction_deg=60),
]


def draw_truth(seed: int) -> HiddenPhysics:
    r = np.random.default_rng(10_000 + seed)
    return HiddenPhysics(float(r.uniform(*TRUTH_G)), float(r.uniform(*TRUTH_MU)))


def within(rec: dict, truth: HiddenPhysics) -> bool:
    return (abs(rec["g_mean"] - truth.g) / truth.g <= G_REL_TOL
            and abs(rec["mu_mean"] - truth.mu) <= MU_ABS_TOL)


def steps_to_stay(recs: list[dict], truths: list[HiddenPhysics], start: int = 0):
    """First experiment count n (1-based, counted from `start`) after which the
    estimate stays within tolerance to the end of the budget. None if never."""
    ok = [within(r, t) for r, t in zip(recs, truths)][start:]
    for i in range(len(ok)):
        if all(ok[i:]):
            return i + 1
    return None


def traj_error(g: float, mu: float, truth: HiddenPhysics) -> float:
    errs = []
    for s in TEST_SPECS:
        _, a = simulate(s, g, mu)
        _, b = simulate(s, truth.g, truth.mu)
        errs.append(np.sqrt(np.mean((a[:, [0, 1]] - b[:, [0, 1]]) ** 2)))
    return float(np.mean(errs))


# ---------------------------------------------------------------- main sweep
def _job_sweep(args):
    strategy, noise_name, seed, budget = args
    noise = NOISE_LEVELS[noise_name]
    truth = draw_truth(seed)
    world = World(truth, noise.pos, noise.vel, seed=seed)
    t0 = time.perf_counter()
    recs = run_episode(world, STRATEGIES[strategy](), noise, budget, seed=seed)
    wall = time.perf_counter() - t0
    truths = [truth] * budget
    n = steps_to_stay(recs, truths)
    cum_cost = np.cumsum([r["cost"] for r in recs])
    last = recs[-1]
    return {
        "strategy": strategy, "noise": noise_name, "seed": seed,
        "g_true": truth.g, "mu_true": truth.mu,
        "steps_to_threshold": n if n is not None else "",
        "cost_to_threshold": float(cum_cost[n - 1]) if n is not None else "",
        "reached": n is not None,
        "final_g_abs_err": abs(last["g_mean"] - truth.g),
        "final_g_rel_err": abs(last["g_mean"] - truth.g) / truth.g,
        "final_mu_abs_err": abs(last["mu_mean"] - truth.mu),
        "final_traj_rmse": traj_error(last["g_mean"], last["mu_mean"], truth),
        "ci_coverage_g": float(np.mean([r["g_ci"][0] <= truth.g <= r["g_ci"][1] for r in recs])),
        "ci_coverage_mu": float(np.mean([r["mu_ci"][0] <= truth.mu <= r["mu_ci"][1] for r in recs])),
        "mean_plan_latency_s": float(np.mean([r["plan_time"] for r in recs])),
        "total_compute_s": wall,
        "g_rel_err_curve": json.dumps([abs(r["g_mean"] - truth.g) / truth.g for r in recs]),
        "mu_abs_err_curve": json.dumps([abs(r["mu_mean"] - truth.mu) for r in recs]),
        "kinds": json.dumps([r["spec"]["kind"] for r in recs]),
    }


# ---------------------------------------------------------- change detection
def _job_change(args):
    strategy, noise_name, seed, budget, change_at, with_change = args
    noise = NOISE_LEVELS[noise_name]
    before = draw_truth(seed)
    after = draw_truth(seed + 5000) if with_change else before
    world = World(before, noise.pos, noise.vel, seed=seed)

    def hook(step):  # evaluation-only: physics changes between experiments
        if step == change_at:
            world._set_physics(after)

    det = ChangeDetector()
    recs = run_episode(world, STRATEGIES[strategy](), noise, budget, seed=seed,
                       detector=det, before_step=hook)
    truths = [before] * change_at + [after] * (budget - change_at)
    flags = [r["step"] for r in recs if r["change_flag"]]
    pre = [f for f in flags if f < change_at] if with_change else flags
    post = [f for f in flags if f >= change_at] if with_change else []
    rec_n = steps_to_stay(recs, truths, start=change_at) if with_change else None
    return {
        "strategy": strategy, "noise": noise_name, "seed": seed, "with_change": with_change,
        "g_before": before.g, "mu_before": before.mu, "g_after": after.g, "mu_after": after.mu,
        "false_alarms": len(pre),
        "detected": bool(post),
        "detection_delay": (post[0] - change_at + 1) if post else "",
        "recovery_steps": rec_n if rec_n is not None else "",
        "z_curve": json.dumps([round(r["z"], 3) for r in recs]),
    }


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_demo_run(out_root: Path, strategy="active", noise_name="medium", seed=0, budget=10):
    """One fully logged run for the store / dashboard."""
    noise = NOISE_LEVELS[noise_name]
    truth = draw_truth(seed)
    run_id = f"{strategy}_{noise_name}_seed{seed}"
    store = ExperimentStore(out_root, run_id, {"strategy": strategy, "noise": noise_name,
                                               "noise_pos": noise.pos, "noise_vel": noise.vel,
                                               "seed": seed, "budget": budget})
    world = World(truth, noise.pos, noise.vel, seed=seed)
    run_episode(world, STRATEGIES[strategy](), noise, budget, seed=seed, store=store)
    store.write_truth([{"from_step": 0, "g": truth.g, "mu": truth.mu}])
    return store.dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--change-seeds", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-change", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    jobs = [(s, n, seed, a.budget) for n in NOISE_LEVELS for s in STRATEGIES for seed in range(a.seeds)]
    jobs.sort(key=lambda j: j[0] != "active")  # start the slow ones first
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_job_sweep, jobs), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"sweep {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
        rows.sort(key=lambda r: (r["noise"], r["strategy"], r["seed"]))
        _write_csv(out / "metrics.csv", rows)

        if not a.skip_change:
            cjobs = [(s, n, seed, 14, 6, wc) for n in NOISE_LEVELS for s in ("active", "random")
                     for seed in range(a.change_seeds) for wc in (True, False)]
            crow = []
            for i, r in enumerate(pool.imap_unordered(_job_change, cjobs), 1):
                crow.append(r)
                if i % 20 == 0 or i == len(cjobs):
                    print(f"change {i}/{len(cjobs)}  {time.time() - t0:.0f}s", flush=True)
            crow.sort(key=lambda r: (r["noise"], r["strategy"], r["with_change"], r["seed"]))
            _write_csv(out / "change_detection.csv", crow)

    save_demo_run(Path("runs"))
    from .report import make_report
    make_report(out)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
