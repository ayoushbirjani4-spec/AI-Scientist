"""Stage A predictive world model: an isolated simulator copy.

Candidate parameters are passed in explicitly; it never sees the World's hidden values.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .adapter import LIKELIHOOD_COLS
from .env import ExperimentSpec, simulate


class SimulatorPredictor:
    version = "stageA-pymunk-1"

    def __init__(self, cache_size: int = 4_000):
        self._cached = lru_cache(maxsize=cache_size)(self._run)

    @staticmethod
    def _run(spec: ExperimentSpec, g: float, mu: float) -> np.ndarray:
        _, s = simulate(spec, g, mu)
        out = s[:, LIKELIHOOD_COLS]
        out.setflags(write=False)
        return out

    def predict(self, spec: ExperimentSpec, g: float, mu: float) -> np.ndarray:
        """Predicted [T, 4] (x, y, vx, vy) trajectory."""
        return self._cached(spec, float(g), float(mu))

    def predict_many(self, spec: ExperimentSpec, gs, mus) -> np.ndarray:
        return np.stack([self.predict(spec, g, m) for g, m in zip(gs, mus)])
