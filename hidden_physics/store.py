"""Experiment store: everything needed to reproduce and inspect a run.

runs/<run_id>/config.json      run id, seed, scene/noise config, versions
runs/<run_id>/steps.jsonl      one line per experiment: spec, belief snapshot, timings
runs/<run_id>/traj_XXX.npz     observed and predicted trajectories
runs/<run_id>/eval_truth.json  hidden parameters (written by the evaluator only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymunk

from . import env

SIM_VERSION = f"pymunk-{pymunk.version}"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


class ExperimentStore:
    def __init__(self, root: str | Path, run_id: str, config: dict):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg = {"run_id": run_id, "simulator_version": SIM_VERSION,
               "dt": env.DT, "record_every": env.RECORD_EVERY, "damping": env.DAMPING,
               "surface_friction": env.SURFACE_FRICTION, "objects": env.OBJECTS, **config}
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, default=_jsonable))
        (self.dir / "steps.jsonl").write_text("")

    def log_step(self, step: int, record: dict, times, observed, predicted, **arrays) -> None:
        """arrays: optional belief snapshot (g_axis, mu_axis, weights)."""
        with open(self.dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(record, default=_jsonable) + "\n")
        np.savez_compressed(self.dir / f"traj_{step:03d}.npz",
                            times=times, observed=observed, predicted=predicted, **arrays)

    def write_truth(self, truth: list[dict]) -> None:
        (self.dir / "eval_truth.json").write_text(json.dumps(truth, indent=2))


def load_run(path: str | Path) -> dict:
    p = Path(path)
    steps = [json.loads(l) for l in (p / "steps.jsonl").read_text().splitlines() if l]
    trajs = [dict(np.load(f)) for f in sorted(p.glob("traj_*.npz"))]
    truth_f = p / "eval_truth.json"
    return {"config": json.loads((p / "config.json").read_text()), "steps": steps,
            "trajectories": trajs,
            "truth": json.loads(truth_f.read_text()) if truth_f.exists() else None}
