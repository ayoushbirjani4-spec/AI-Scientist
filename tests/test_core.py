"""Run with `python -m pytest tests` or plain `python tests/test_core.py`."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hidden_physics.adapter import NOISE_LEVELS, likelihood_sigma, transitions
from hidden_physics.belief import GridBelief
from hidden_physics.change import ChangeDetector
from hidden_physics.env import ExperimentSpec as E, HiddenPhysics, World, simulate
from hidden_physics.planner import ActivePlanner, info_gain_lower_bound
from hidden_physics.predictor import SimulatorPredictor

G, MU = 9.81, 0.3


def test_reproducible():
    a = simulate(E.make("ramp_slide", "block", angle_deg=30, start=2), G, MU)[1]
    b = simulate(E.make("ramp_slide", "block", angle_deg=30, start=2), G, MU)[1]
    assert np.array_equal(a, b)
    w1 = World(HiddenPhysics(G, MU), 0.01, 0.1, seed=4).execute_experiment(E.make("drop", "ball", height=1))
    w2 = World(HiddenPhysics(G, MU), 0.01, 0.1, seed=4).execute_experiment(E.make("drop", "ball", height=1))
    assert np.array_equal(w1.states, w2.states)


def test_friction_definition():
    # flat slide: deceleration = mu * g (floor friction fixed at 1.0)
    t, s = simulate(E.make("flat_slide", "block", v0=2.0), G, MU)
    assert abs(s[10, 3] - (2.0 - MU * G * t[10])) < 1e-6
    # ramp: a = g (sin th - mu cos th)
    th = math.radians(30)
    t, s = simulate(E.make("ramp_slide", "block", angle_deg=30, start=2.0), G, MU)
    v = math.hypot(s[8, 3], s[8, 4])
    assert abs(v - G * (math.sin(th) - MU * math.cos(th)) * t[8]) < 1e-3
    # ramp shallower than friction angle: the block stays put
    _, s = simulate(E.make("ramp_slide", "block", angle_deg=10, start=2.0), G, MU)
    assert math.hypot(s[-1, 3], s[-1, 4]) < 1e-6


def test_hidden_params_not_exposed():
    w = World(HiddenPhysics(G, MU), 0.01, 0.1)
    public = [a for a in dir(w) if not a.startswith("_")]
    assert public == ["execute_experiment"]


def test_transitions_shape():
    tr = World(HiddenPhysics(G, MU), 0.0, 0.0).execute_experiment(E.make("flat_slide", "block", v0=1))
    ts = transitions(tr)
    assert len(ts) == len(tr.times) - 1 and ts[0][0].shape == (6,)


def test_posterior_converges():
    nz = NOISE_LEVELS["medium"]
    w = World(HiddenPhysics(12.3, 0.51), nz.pos, nz.vel, seed=2)
    b = GridBelief(SimulatorPredictor(), likelihood_sigma(nz))
    for sp in [E.make("drop", "ball", height=1.5), E.make("ramp_slide", "block", angle_deg=40, start=2),
               E.make("flat_slide", "block", v0=3)]:
        b.update(w.execute_experiment(sp))
    g, mu = b.mean
    assert abs(g - 12.3) / 12.3 < 0.01 and abs(mu - 0.51) < 0.01
    assert b.sd[0] > 0 and b.sd[1] > 0          # belief has not collapsed to a point


def test_drop_does_not_identify_mu():
    nz = NOISE_LEVELS["low"]
    w = World(HiddenPhysics(G, MU), nz.pos, nz.vel, seed=0)
    b = GridBelief(SimulatorPredictor(), likelihood_sigma(nz))
    b.update(w.execute_experiment(E.make("drop", "ball", height=1.5)))
    assert b.sd[0] < 0.05 and b.sd[1] > 0.15   # g known, mu still broad


def test_grid_expands_when_truth_is_cut_off():
    # A grid that shrank onto the wrong spot must not pin the estimate to its edge.
    nz = NOISE_LEVELS["medium"]
    w = World(HiddenPhysics(4.37, 0.45), nz.pos, nz.vel, seed=0)
    b = GridBelief(SimulatorPredictor(), likelihood_sigma(nz))
    b._set_grid((5.0, 6.0), (0.30, 0.50))                 # truth g = 4.37 is outside
    for _ in range(2):                                    # widening happens on the next update
        b.update(w.execute_experiment(E.make("drop", "block", height=1.3)))
    assert b.g_axis[0] <= 4.37 <= b.g_axis[-1] and b.n_expands > 0
    assert abs(b.mean[0] - 4.37) < 0.05


def test_info_gain_prefers_friction_experiment_when_g_known():
    nz = NOISE_LEVELS["medium"]
    pred = SimulatorPredictor()
    sigma = likelihood_sigma(nz)
    gs = np.full(8, G)
    mus = np.linspace(0.1, 0.7, 8)
    w = np.full(8, 1 / 8)
    drop = info_gain_lower_bound(pred.predict_many(E.make("drop", "ball", height=1), gs, mus), w, sigma)
    slide = info_gain_lower_bound(pred.predict_many(E.make("flat_slide", "block", v0=3), gs, mus), w, sigma)
    assert slide > drop + 1.0


def test_planner_picks_mu_experiment_after_drop():
    nz = NOISE_LEVELS["medium"]
    w = World(HiddenPhysics(G, MU), nz.pos, nz.vel, seed=1)
    b = GridBelief(SimulatorPredictor(), likelihood_sigma(nz))
    b.update(w.execute_experiment(E.make("drop", "ball", height=1.5)))
    spec = ActivePlanner().choose(b, np.random.default_rng(0), 1)
    assert spec.kind in ("flat_slide", "ramp_slide", "impulse") and spec.obj != "ball"


def test_change_detector():
    nz = NOISE_LEVELS["medium"]
    w = World(HiddenPhysics(G, MU), nz.pos, nz.vel, seed=3)
    b = GridBelief(SimulatorPredictor(), likelihood_sigma(nz))
    d = ChangeDetector()
    specs = [E.make("drop", "ball", height=1.5), E.make("flat_slide", "block", v0=3),
             E.make("ramp_slide", "block", angle_deg=40, start=2)]
    flags = []
    for sp in specs:
        b.update(w.execute_experiment(sp)); flags.append(d.check(b))
    assert not any(flags)
    w._set_physics(HiddenPhysics(6.0, 0.5))
    for sp in specs:
        b.update(w.execute_experiment(sp)); flags.append(d.check(b))
    assert any(flags[3:])
    g, mu = b.mean
    assert abs(g - 6.0) < 0.3


def test_change_inside_experiment():
    sp = E.make("flat_slide", "block", v0=3.0)
    _, base = simulate(sp, G, MU)
    _, ch = simulate(sp, G, MU, change=(0.6, 6.0, 0.6))
    k = int(round(0.6 * 30))                      # frame recorded at the change step
    assert np.array_equal(base[:k + 1], ch[:k + 1]) and not np.allclose(base[k + 2:], ch[k + 2:])
    w = World(HiddenPhysics(G, MU), 0.0, 0.0, noise_ang=0.0)
    w._schedule_change(0.6, HiddenPhysics(6.0, 0.6))
    assert np.allclose(w.execute_experiment(sp).states, ch)
    assert w._reveal() == HiddenPhysics(6.0, 0.6)  # new physics persists afterwards
    assert np.allclose(w.execute_experiment(sp).states, simulate(sp, 6.0, 0.6)[1])


def test_within_experiment_detector():
    nz = NOISE_LEVELS["medium"]
    specs = [E.make("drop", "ball", height=1.5), E.make("flat_slide", "block", v0=3),
             E.make("ramp_slide", "block", angle_deg=40, start=2)]
    w = World(HiddenPhysics(G, MU), nz.pos, nz.vel, seed=3)
    b = GridBelief(SimulatorPredictor(), likelihood_sigma(nz))
    d = ChangeDetector(within=True)
    flags = []
    for sp in specs:                              # clean: no false alarms
        b.update(w.execute_experiment(sp)); flags.append(d.check(b))
    assert not any(flags)
    w._schedule_change(0.6, HiddenPhysics(6.0, 0.6))
    b.update(w.execute_experiment(E.make("flat_slide", "block", v0=3)))
    assert d.check(b)                             # flagged on the contaminated experiment itself
    assert abs(d.last_change_frame / 30 - 0.6) <= 0.25   # residuals lag the change by a few frames
    assert len(b.history) == 0                    # contaminated + pre-change data dropped


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
