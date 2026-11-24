"""Experiment selection strategies: active (information gain minus cost) and baselines."""
from __future__ import annotations

import numpy as np

from .belief import GridBelief
from .env import OBJECTS, SETTING_RANGES, ExperimentSpec

KINDS = tuple(SETTING_RANGES)
OBJS = tuple(OBJECTS)


def random_spec(rng: np.random.Generator, kinds=KINDS, objs=OBJS) -> ExperimentSpec:
    kind = kinds[rng.integers(len(kinds))]
    obj = objs[rng.integers(len(objs))]
    params = {k: rng.uniform(lo, hi) for k, (lo, hi) in SETTING_RANGES[kind].items()}
    return ExperimentSpec.make(kind, obj, **params)


def info_gain_lower_bound(preds: np.ndarray, w: np.ndarray, sigma: np.ndarray) -> float:
    """Pairwise-distance lower bound on mutual information I(theta; y) for a
    Gaussian-noise mixture (Kolchinsky & Tracey 2017, Bhattacharyya version).

    preds: [K, T, C] predicted outcomes for K hypotheses; w: [K] weights.
    It grows with how far apart outcomes are relative to noise, and is capped at
    the entropy of w.
    """
    y = (preds / sigma).reshape(len(preds), -1)
    sq = (y * y).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * y @ y.T, 0.0)
    D = d2 / 8.0
    return float(-(w * np.log(np.exp(-D) @ w)).sum())


class ActivePlanner:
    name = "active"

    def __init__(self, cost_weight: float = 0.15, n_hypotheses: int = 32,
                 n_candidates: int = 32, n_repeats: int = 4):
        self.cost_weight = cost_weight
        self.k = n_hypotheses
        self.n_candidates = n_candidates
        self.n_repeats = n_repeats
        self.last_scores: list[dict] = []

    def choose(self, belief: GridBelief, rng: np.random.Generator, step: int) -> ExperimentSpec:
        gs, mus, w = belief.sample(self.k, rng)
        cands = [random_spec(rng) for _ in range(self.n_candidates)]
        seen = []
        for spec, _ in reversed(belief.history):          # "repeat" options
            if spec not in seen:
                seen.append(spec)
            if len(seen) >= self.n_repeats:
                break
        cands += seen
        scored = []
        for spec in cands:
            preds = belief.predictor.predict_many(spec, gs, mus)
            eig = info_gain_lower_bound(preds, w, belief.sigma)
            scored.append({"spec": spec, "eig": eig, "cost": spec.cost,
                           "score": eig - self.cost_weight * spec.cost})
        scored.sort(key=lambda d: -d["score"])
        self.last_scores = scored
        return scored[0]["spec"]


class RandomPlanner:
    """Uniform over the same library and settings; sometimes repeats an earlier run."""
    name = "random"

    def __init__(self, p_repeat: float = 0.15):
        self.p_repeat = p_repeat

    def choose(self, belief, rng, step):
        if belief.history and rng.random() < self.p_repeat:
            return belief.history[rng.integers(len(belief.history))][0]
        return random_spec(rng)


SCRIPT = [
    ExperimentSpec.make("drop", "ball", height=1.5),
    ExperimentSpec.make("flat_slide", "block", v0=2.5),
    ExperimentSpec.make("ramp_slide", "block", angle_deg=30, start=2.0),
    ExperimentSpec.make("impulse", "heavy_block", impulse=4.0, direction_deg=20),
    ExperimentSpec.make("drop", "block", height=1.0),
    ExperimentSpec.make("flat_slide", "heavy_block", v0=3.5),
    ExperimentSpec.make("ramp_slide", "block", angle_deg=40, start=2.5),
    ExperimentSpec.make("impulse", "block", impulse=2.0, direction_deg=45),
]


class ScriptedPlanner:
    """A fixed sequence a sensible human might run, cycled."""
    name = "scripted"

    def choose(self, belief, rng, step):
        return SCRIPT[step % len(SCRIPT)]


class PassivePlanner:
    """No intervention: the agent watches events the world produces on its own
    (objects falling or sliding to rest). No ramps, no designed impulses, and the
    agent cannot pick objects or settings."""
    name = "passive"

    def choose(self, belief, rng, step):
        return random_spec(rng, kinds=("drop", "flat_slide"))


STRATEGIES = {
    "active": ActivePlanner,
    "random": RandomPlanner,
    "scripted": ScriptedPlanner,
    "passive": PassivePlanner,
}
