"""2D Pymunk world with hidden gravity g and object friction mu.

Units are SI (m, s, kg). y points up; gravity acts along -y.

Friction convention (risk 3): Pymunk multiplies the friction of the two
touching shapes. Floor and ramp friction are fixed at SURFACE_FRICTION = 1.0,
so the object's own friction is the single hidden mu. Space damping is fixed
at DAMPING (1.0 = no damping) so it can never mimic friction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymunk

DT = 1.0 / 240.0          # fixed simulation step
RECORD_EVERY = 8          # observations at 30 Hz
DAMPING = 1.0             # Pymunk velocity multiplier per second; 1.0 = none
SURFACE_FRICTION = 1.0
SURFACE_ELASTICITY = 1.0
ITERATIONS = 20
COLLISION_SLOP = 1e-3     # Pymunk default (0.1) is far too large for metres

# Known geometry and mass. The agent may see all of this.
OBJECTS = {
    "ball": dict(shape="circle", radius=0.10, mass=1.0, elasticity=0.0),
    "block": dict(shape="box", size=(0.30, 0.20), mass=1.0, elasticity=0.0),
    "heavy_block": dict(shape="box", size=(0.40, 0.20), mass=3.0, elasticity=0.0),
}

# Observation window per experiment kind (s) and fixed interaction cost.
DURATION = {"drop": 0.8, "flat_slide": 1.5, "ramp_slide": 1.2, "impulse": 1.2}
SETUP_COST = {"drop": 1.0, "flat_slide": 1.0, "ramp_slide": 2.0, "impulse": 1.5}

# Continuous settings the planner may choose (risk 1).
SETTING_RANGES = {
    "drop": {"height": (0.3, 2.0)},
    "flat_slide": {"v0": (0.5, 4.0)},
    "ramp_slide": {"angle_deg": (10.0, 45.0), "start": (0.5, 2.5)},
    "impulse": {"impulse": (0.5, 4.0), "direction_deg": (0.0, 75.0)},
}
RAMP_LENGTH = 3.0


@dataclass(frozen=True)
class HiddenPhysics:
    g: float
    mu: float


@dataclass(frozen=True)
class ExperimentSpec:
    kind: str
    obj: str
    params: tuple  # sorted (name, value) pairs, hashable

    @staticmethod
    def make(kind: str, obj: str, **params) -> "ExperimentSpec":
        assert kind in DURATION, kind
        assert obj in OBJECTS, obj
        return ExperimentSpec(kind, obj, tuple(sorted((k, float(v)) for k, v in params.items())))

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def cost(self) -> float:
        return DURATION[self.kind] + SETUP_COST[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "obj": self.obj, **self.p}

    def __str__(self) -> str:
        s = ", ".join(f"{k}={v:.2f}" for k, v in self.params)
        return f"{self.kind}({self.obj}, {s})"


def _half_height(obj: str) -> float:
    o = OBJECTS[obj]
    return o["radius"] if o["shape"] == "circle" else o["size"][1] / 2


def _make_body(obj: str, mu: float) -> tuple[pymunk.Body, pymunk.Shape]:
    o = OBJECTS[obj]
    m = o["mass"]
    if o["shape"] == "circle":
        body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, o["radius"]))
        shape = pymunk.Circle(body, o["radius"])
    else:
        body = pymunk.Body(m, pymunk.moment_for_box(m, o["size"]))
        shape = pymunk.Poly.create_box(body, o["size"], radius=0.0)
    shape.friction = mu
    shape.elasticity = o["elasticity"]
    return body, shape


def _static_segment(space, a, b, offset_normal, r=0.01):
    """Segment whose top surface lies exactly on the line a-b."""
    nx, ny = offset_normal
    a2 = (a[0] - nx * r, a[1] - ny * r)
    b2 = (b[0] - nx * r, b[1] - ny * r)
    seg = pymunk.Segment(space.static_body, a2, b2, r)
    seg.friction = SURFACE_FRICTION
    seg.elasticity = SURFACE_ELASTICITY
    space.add(seg)


@dataclass(frozen=True)
class Surface:
    """A static line segment whose top face runs from a to b, with unit normal n."""
    a: tuple
    b: tuple
    n: tuple


@dataclass(frozen=True)
class SceneSetup:
    """Everything fixed by the experiment spec (the agent's action), before any
    dynamics: initial pose and velocity of the object, and the static surfaces.
    Shared by the simulator and the learned world model."""
    position: tuple
    angle: float
    velocity: tuple
    omega: float
    surfaces: tuple


FLOOR = Surface((-50.0, 0.0), (50.0, 0.0), (0.0, 1.0))


def scene_setup(spec: ExperimentSpec) -> SceneSetup:
    hh = _half_height(spec.obj)
    p = spec.p
    surfaces = [FLOOR]
    pos, angle, vel = (0.0, hh), 0.0, (0.0, 0.0)
    if spec.kind == "drop":
        pos = (0.0, p["height"] + hh)
    elif spec.kind == "flat_slide":
        vel = (p["v0"], 0.0)
    elif spec.kind == "ramp_slide":
        th = math.radians(p["angle_deg"])
        n = (math.sin(th), math.cos(th))            # ramp surface normal
        top = (-RAMP_LENGTH * math.cos(th), RAMP_LENGTH * math.sin(th))
        surfaces.append(Surface(top, (0.0, 0.0), n))
        s = p["start"]
        base = (-s * math.cos(th), s * math.sin(th))
        pos = (base[0] + n[0] * hh, base[1] + n[1] * hh)
        angle = -th
    elif spec.kind == "impulse":
        # impulse J applied at the centre of a resting object: v = J / m, no spin
        ph = math.radians(p["direction_deg"])
        m = OBJECTS[spec.obj]["mass"]
        vel = (p["impulse"] * math.cos(ph) / m, p["impulse"] * math.sin(ph) / m)
    else:
        raise ValueError(spec.kind)
    return SceneSetup(pos, angle, vel, 0.0, tuple(surfaces))


def simulate(spec: ExperimentSpec, g: float, mu: float,
             change: tuple[float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run one experiment under explicit (g, mu).

    change = (t, g2, mu2) switches the physics to (g2, mu2) at time t into the
    experiment (stress test only). Returns (times [T], states [T, 6]) with columns
    x, y, angle, vx, vy, omega. Deterministic: same inputs give the same trajectory.
    """
    space = pymunk.Space()
    space.gravity = (0.0, -g)
    space.damping = DAMPING
    space.iterations = ITERATIONS
    space.collision_slop = COLLISION_SLOP

    setup = scene_setup(spec)
    for srf in setup.surfaces:
        _static_segment(space, srf.a, srf.b, srf.n)
    body, shape = _make_body(spec.obj, mu)
    body.position = setup.position
    body.angle = setup.angle
    space.add(body, shape)
    if spec.kind == "impulse":
        p = spec.p
        ph = math.radians(p["direction_deg"])
        J = p["impulse"]
        body.apply_impulse_at_local_point((J * math.cos(ph), J * math.sin(ph)), (0, 0))
        # local frame == world frame here (angle 0)
    else:
        body.velocity = setup.velocity

    n_steps = int(round(DURATION[spec.kind] / DT))
    n_frames = n_steps // RECORD_EVERY + 1
    states = np.empty((n_frames, 6))
    k = 0
    for i in range(n_steps + 1):
        if change is not None and i == int(round(change[0] / DT)):
            space.gravity = (0.0, -change[1])
            shape.friction = change[2]
        if i % RECORD_EVERY == 0:
            states[k] = (body.position.x, body.position.y, body.angle,
                         body.velocity.x, body.velocity.y, body.angular_velocity)
            k += 1
        if i < n_steps:
            space.step(DT)
    times = np.arange(n_frames) * DT * RECORD_EVERY
    return times, states


@dataclass
class Trajectory:
    spec: ExperimentSpec
    times: np.ndarray
    states: np.ndarray      # noisy observed states [T, 6]
    run_index: int


class World:
    """The environment. Holds the hidden physics privately.

    Only the evaluation harness should call `_reveal` or `_set_physics`.
    The agent talks to it through `execute_experiment`.
    """

    def __init__(self, physics: HiddenPhysics, noise_pos: float, noise_vel: float,
                 noise_ang: float = 0.02, seed: int = 0):
        self.__physics = physics
        self._noise = np.array([noise_pos, noise_pos, noise_ang, noise_vel, noise_vel, noise_ang * 10])
        self._rng = np.random.default_rng(seed)
        self._count = 0
        self.__pending: tuple[float, HiddenPhysics] | None = None

    def execute_experiment(self, spec: ExperimentSpec) -> Trajectory:
        change = None
        if self.__pending is not None:
            t_c, new = self.__pending
            change = (t_c, new.g, new.mu)
        t, s = simulate(spec, self.__physics.g, self.__physics.mu, change)
        if change is not None:
            self.__physics, self.__pending = self.__pending[1], None
        noisy = s + self._rng.normal(size=s.shape) * self._noise
        traj = Trajectory(spec, t, noisy, self._count)
        self._count += 1
        return traj

    # --- evaluation-only hooks ---
    def _reveal(self) -> HiddenPhysics:
        return self.__physics

    def _schedule_change(self, t_change: float, physics: HiddenPhysics) -> None:
        """Change physics t_change seconds into the next experiment (stress test)."""
        self.__pending = (t_change, physics)

    def _set_physics(self, physics: HiddenPhysics) -> None:
        """Change physics between experiments (change-detection tests)."""
        self.__physics = physics
