"""Choose the within-experiment CUSUM settings on calibration seeds (100+), kept
separate from the stress-test seeds (0+) that are reported.

python -m hidden_physics.calibrate_cusum --seeds 8 --workers 2

Records, after every update, the per-frame residuals of the latest experiment at
the MAP, in clean runs and in runs with a mid-experiment change (between-experiment
detector only, so traces are not altered by within-experiment resets). Then scores
CUSUM settings offline: h is set just above the largest statistic seen in any clean
experiment, and settings are compared on how often and how early they catch changes.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .adapter import NOISE_LEVELS
from .agent import run_episode
from .change import ChangeDetector
from .env import World
from .evaluate import draw_truth
from .evaluate_stress import BUDGET, CHANGE_STEP, T_CHANGE
from .planner import STRATEGIES

SEED0 = 100
CAP = 9.0


class Recorder(ChangeDetector):
    def __init__(self):
        super().__init__(within=False)
        self.traces = []

    def check(self, belief):
        spec, obs = belief.history[-1]
        g, mu = belief.map
        self.traces.append(belief.residual_frames(spec, obs, g, mu).tolist())
        return super().check(belief)


def _job(args):
    noise_name, with_change, seed = args
    noise = NOISE_LEVELS[noise_name]
    world = World(draw_truth(seed), noise.pos, noise.vel, seed=seed)
    t_c = float(np.random.default_rng(20_000 + seed).uniform(*T_CHANGE))

    def hook(step):
        if with_change and step == CHANGE_STEP:
            world._schedule_change(t_c, draw_truth(seed + 5000))

    det = Recorder()
    run_episode(world, STRATEGIES["active"](), noise, BUDGET, seed=seed, detector=det, before_step=hook)
    return {"noise": noise_name, "with_change": with_change, "seed": seed, "t_change": t_c, "traces": det.traces}


def cusum(z, drift):
    """Max CUSUM statistic and the excursion start frame at each frame."""
    s, start, out = 0.0, 0, []
    for t, zt in enumerate(np.minimum(z, CAP)):
        if s == 0.0:
            start = t
        s = max(0.0, s + zt - 1.0 - drift)
        out.append((s, start))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path("results/stress")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, c, SEED0 + s) for n in NOISE_LEVELS for c in (False, True) for s in range(a.seeds)]
    # Resumable: each finished run is appended, so a crash (e.g. out of memory) loses nothing.
    log = out / "cusum_calibration_traces.jsonl"
    runs = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    done = {(r["noise"], r["with_change"], r["seed"]) for r in runs}
    todo = [j for j in jobs if j not in done]
    with Pool(a.workers) as pool, open(log, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_job, todo), 1):
            runs.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"calibration {len(done) + i}/{len(jobs)}", flush=True)
    runs = [r for r in runs if (r["noise"], r["with_change"], r["seed"]) in set(jobs)]

    clean = [z for r in runs if not r["with_change"] for z in r["traces"]]
    pre = [z for r in runs if r["with_change"] for z in r["traces"][:CHANGE_STEP]]
    changed = [(r["traces"][CHANGE_STEP], r["t_change"], r["noise"]) for r in runs if r["with_change"]]
    L = ["# CUSUM calibration (seeds 100+, not used for reporting)", "",
         "h = 1.25 × the largest CUSUM value seen in any clean experiment (clean runs plus the "
         "pre-change part of change runs), so no clean calibration experiment would flag. A setting is only "
         "eligible if h exceeds the most one capped frame can add (cap − 1 − drift), so a single bad frame "
         "can never trigger it (the spec: flag persistent deviation, not a single error).", "",
         "| drift | h | needs ≥ 2 frames | changes caught on the contaminated experiment | mean change-time error (s) |",
         "|---|---|---|---|---|"]
    best = None
    for drift in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        worst = max(max(s for s, _ in cusum(z, drift)) for z in clean + pre)
        h = 1.25 * max(worst, 1.0)
        caught, errs = 0, []
        for z, t_c, _ in changed:
            hit = next(((s, st) for s, st in cusum(z, drift) if s > h), None)
            if hit:
                caught += 1
                errs.append(abs(hit[1] / 30 - t_c))
        persistent = h > CAP - 1.0 - drift
        L.append(f"| {drift} | {h:.1f} | {'yes' if persistent else 'no'} | {caught}/{len(changed)} | "
                 f"{np.mean(errs) if errs else float('nan'):.3f} |")
        if persistent and (best is None or caught > best[2]):
            best = (drift, h, caught)
    L += ["", f"Chosen: drift = {best[0]}, h = {best[1]:.1f} (most changes caught among eligible settings; ties go to the smaller drift).", ""]
    (out / "cusum_calibration.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
