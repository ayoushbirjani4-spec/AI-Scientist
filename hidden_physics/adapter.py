"""Observation adapter: structured trajectories -> normalized tensors and transitions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import DT, OBJECTS, RECORD_EVERY, Trajectory

STATE_COLS = ("x", "y", "angle", "vx", "vy", "omega")
LIKELIHOOD_COLS = (0, 1, 3, 4)          # x, y, vx, vy are used for inference
# Fixed normalization scales (so features are O(1) regardless of scene).
STATE_SCALE = np.array([1.0, 1.0, np.pi, 2.0, 2.0, 10.0])
OBJ_ORDER = tuple(OBJECTS)              # consistent object ordering


@dataclass
class NoiseLevel:
    name: str
    pos: float   # m
    vel: float   # m/s


NOISE_LEVELS = {
    "low": NoiseLevel("low", 0.002, 0.02),
    "medium": NoiseLevel("medium", 0.01, 0.10),
    "high": NoiseLevel("high", 0.03, 0.30),
}

# Stage A predictor-error term (risk 2): added in quadrature to measurement noise.
MODEL_ERR_POS = 0.001
MODEL_ERR_VEL = 0.01


def likelihood_sigma(noise: NoiseLevel) -> np.ndarray:
    """Per-column sigma for the LIKELIHOOD_COLS, including predictor error."""
    sp = np.hypot(noise.pos, MODEL_ERR_POS)
    sv = np.hypot(noise.vel, MODEL_ERR_VEL)
    return np.array([sp, sp, sv, sv])


def to_tensor(traj: Trajectory) -> np.ndarray:
    """[T, 6] normalized state features."""
    return traj.states / STATE_SCALE


def object_features(obj: str) -> np.ndarray:
    """One-hot object ID plus known mass and size."""
    o = OBJECTS[obj]
    onehot = np.eye(len(OBJ_ORDER))[OBJ_ORDER.index(obj)]
    size = (2 * o["radius"], 2 * o["radius"]) if o["shape"] == "circle" else o["size"]
    return np.concatenate([onehot, [o["mass"], *size]])


def transitions(traj: Trajectory) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """(state_t, action_t, dt, state_t+1). The only action is the initial setup, so
    action_t is the experiment's settings at t=0 and zeros afterwards."""
    x = to_tensor(traj)
    a0 = np.array([v for _, v in traj.spec.params] + [0.0] * (2 - len(traj.spec.params)))
    zero = np.zeros_like(a0)
    dt = DT * RECORD_EVERY
    return [(x[i], a0 if i == 0 else zero, dt, x[i + 1]) for i in range(len(x) - 1)]
