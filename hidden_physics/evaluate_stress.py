"""Stress test: physics changes partway through an experiment.

python -m hidden_physics.evaluate_stress --seeds 6 --workers 2

Compares the between-experiment detector alone with between + within-experiment
CUSUM, at three noise levels, on runs with a mid-experiment change and on clean
control runs. Active strategy, Stage A predictor.
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .adapter import NOISE_LEVELS
from .agent import run_episode
from .change import ChangeDetector
from .env import World
from .evaluate import draw_truth, steps_to_stay
from .planner import STRATEGIES

BUDGET, CHANGE_STEP = 10, 4          # change inside the 5th experiment
T_CHANGE = (0.15, 0.65)              # seconds into it; inside every experiment kind
FRAME_DT = 1 / 30


def _job(args):
    noise_name, detector, with_change, seed = args
    noise = NOISE_LEVELS[noise_name]
    before = draw_truth(seed)
    after = draw_truth(seed + 5000)
    t_c = float(np.random.default_rng(20_000 + seed).uniform(*T_CHANGE))
    world = World(before, noise.pos, noise.vel, seed=seed)

    def hook(step):  # evaluation only: the agent is not told
        if with_change and step == CHANGE_STEP:
            world._schedule_change(t_c, after)

    det = ChangeDetector(within=(detector == "between+within"))
    recs = run_episode(world, STRATEGIES["active"](), noise, BUDGET, seed=seed, detector=det, before_step=hook)
    flags = [r for r in recs if r["change_flag"]]
    pre = [r for r in flags if r["step"] < CHANGE_STEP] if with_change else flags
    post = [r for r in flags if r["step"] >= CHANGE_STEP] if with_change else []
    first = post[0] if post else None
    loc = (abs(first["change_frame"] * FRAME_DT - t_c)
           if first and first["step"] == CHANGE_STEP and first["change_frame"] is not None else "")
    rec_n = (steps_to_stay(recs, [before] * (CHANGE_STEP + 1) + [after] * (BUDGET - CHANGE_STEP - 1),
                           start=CHANGE_STEP + 1) if with_change else None)
    return {"noise": noise_name, "detector": detector, "with_change": with_change, "seed": seed,
            "t_change": t_c if with_change else "", "false_alarm": bool(pre), "detected": bool(post),
            "delay": (first["step"] - CHANGE_STEP + 1) if first else "",
            "loc_error_s": loc, "recovery": rec_n if rec_n is not None else ""}


def _mean(xs):
    xs = [float(x) for x in xs if x != ""]
    return f"{np.mean(xs):.2f}" if xs else "–"


def _med_mean(xs):
    xs = [float(x) for x in xs if x != ""]
    return f"{np.median(xs):.2f} / {np.mean(xs):.2f}" if xs else "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path("results/stress")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, d, c, s) for n in NOISE_LEVELS for d in ("between", "between+within")
            for c in (True, False) for s in range(a.seeds)]
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(r)
            print(f"stress {i}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["noise"], r["detector"], not r["with_change"], r["seed"]))
    with open(out / "stress.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    L = ["# Stress test: physics changes inside an experiment", "",
         f"Active strategy, Stage A predictor, budget {BUDGET}. In change runs, g and μ are re-drawn "
         f"{T_CHANGE[0]}–{T_CHANGE[1]} s into experiment {CHANGE_STEP + 1}; control runs have no change. "
         f"{a.seeds} seeds per cell. Delay counts experiments from the contaminated one (1 = flagged on it). "
         "Recovery counts experiments after the contaminated one until the estimate stays within "
         "|g error| ≤ 1 % and |μ error| ≤ 0.01. Within-experiment CUSUM settings (drift 3, h 8.3) were "
         "chosen on separate calibration seeds (calibrate_cusum.py).", "",
         "| noise | detector | detected | mean delay | flagged on the contaminated experiment | "
         "change-time error, median / mean (s) | false-alarm runs | recovered | mean recovery |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in NOISE_LEVELS:
        for d in ("between", "between+within"):
            ch = [r for r in rows if r["noise"] == n and r["detector"] == d and r["with_change"]]
            ct = [r for r in rows if r["noise"] == n and r["detector"] == d and not r["with_change"]]
            det = [r for r in ch if r["detected"]]
            on_exp = sum(1 for r in det if r["delay"] == 1)
            fa = sum(r["false_alarm"] for r in ch + ct)
            rec = [r for r in ch if r["recovery"] != ""]
            L.append(f"| {n} | {d} | {len(det)}/{len(ch)} | {_mean(r['delay'] for r in det)} | {on_exp}/{len(ch)} | "
                     f"{_med_mean(r['loc_error_s'] for r in det)} | {fa}/{len(ch) + len(ct)} | {len(rec)}/{len(ch)} | "
                     f"{_mean(r['recovery'] for r in rec)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
