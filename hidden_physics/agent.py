"""The main loop: run experiment -> adapt -> update belief -> (detect change) -> plan."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .adapter import LIKELIHOOD_COLS, NoiseLevel, likelihood_sigma
from .belief import GridBelief
from .change import ChangeDetector
from .env import DURATION, World
from .predictor import SimulatorPredictor
from .store import ExperimentStore


def run_episode(world: World, planner, noise: NoiseLevel, budget: int, seed: int,
                detector: ChangeDetector | None = None, store: ExperimentStore | None = None,
                before_step: Callable[[int], None] | None = None,
                predictor: SimulatorPredictor | None = None,
                on_step: Callable[[dict], None] | None = None) -> list[dict]:
    """The agent sees only trajectories returned by world.execute_experiment.

    `before_step` is an evaluation hook (e.g. to change physics between
    experiments); the agent is not told when it fires.
    """
    rng = np.random.default_rng(seed)
    predictor = predictor or SimulatorPredictor()
    belief = GridBelief(predictor, likelihood_sigma(noise))
    records = []
    for step in range(budget):
        if before_step:
            before_step(step)
        t0 = time.perf_counter()
        spec = planner.choose(belief, rng, step)
        t_plan = time.perf_counter() - t0

        traj = world.execute_experiment(spec)

        t0 = time.perf_counter()
        belief.update(traj)
        flagged = detector.check(belief) if detector else False
        t_update = time.perf_counter() - t0

        cost = DURATION[spec.kind] if planner.name == "passive" else spec.cost
        rec = {"step": step, "spec": spec.to_dict(), "spec_str": str(spec), "cost": cost,
               "plan_time": t_plan, "update_time": t_update, "change_flag": flagged,
               "z": detector.z_history[-1] if detector else None,
               "change_frame": detector.last_change_frame if detector else None, **belief.snapshot()}
        if planner.name == "active" and planner.last_scores:
            rec["chosen_eig"] = planner.last_scores[0]["eig"]
            rec["alternatives"] = [{"spec": str(d["spec"]), "eig": d["eig"], "cost": d["cost"],
                                    "score": d["score"]} for d in planner.last_scores[:5]]
        records.append(rec)
        if store:
            g, mu = belief.mean
            store.log_step(step, rec, traj.times, traj.states[:, LIKELIHOOD_COLS],
                           predictor.predict(spec, g, mu),
                           g_axis=belief.g_axis, mu_axis=belief.mu_axis,
                           weights=belief.weights.reshape(belief.n, belief.n))
        if on_step:
            on_step(rec)
    return records
