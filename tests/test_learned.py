"""Stage B tests. Run with `python -m pytest tests` or `python tests/test_learned.py`."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hidden_physics.adapter import NOISE_LEVELS, likelihood_sigma
from hidden_physics.belief import GridBelief
from hidden_physics.env import ExperimentSpec as E, HiddenPhysics, World, simulate
from hidden_physics.learned import (GraphStep, LearnedPredictor, generate_dataset, initial_state,
                                    integrate, n_frames, save_ensemble, targets, transition_labels)


def _tiny_ensemble(path, members=2, g_scaled=False):
    ms = []
    for k in range(members):
        torch.manual_seed(k)
        m = GraphStep(16, g_scaled)
        m.eval()
        ms.append(m)
    save_ensemble(path, ms, np.full((46, 4), 1e-4, np.float32), {"n_traj": 0})


def test_targets_invert_integrate():
    _, s = simulate(E.make("ramp_slide", "block", angle_deg=35, start=2), 9.81, 0.3)
    s0, s1 = torch.tensor(s[:-1], dtype=torch.float64), torch.tensor(s[1:], dtype=torch.float64)
    assert torch.allclose(integrate(s0, targets(s0, s1)), s1, atol=1e-9)


def test_initial_state_matches_simulator():
    for sp in [E.make("drop", "ball", height=1.0), E.make("impulse", "heavy_block", impulse=3, direction_deg=30),
               E.make("ramp_slide", "block", angle_deg=25, start=1.5)]:
        _, s = simulate(sp, 9.81, 0.3)
        assert np.allclose(initial_state(sp), s[0], atol=1e-5), sp


def test_dataset_and_labels():
    ds = generate_dataset(20, seed=0)
    assert ds.states.shape[0] == 20 and (ds.lengths > 20).all()
    lab = transition_labels(ds)
    assert set(np.unique(lab)) <= {-1, 0, 1, 2}
    assert (lab[ds.kinds == "flat_slide"] != 0).any()      # slides are in contact


def test_predictor_shapes_and_variance():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ens.pt"
        _tiny_ensemble(p)
        pred = LearnedPredictor(p)
        sp = E.make("flat_slide", "block", v0=2)
        m, v = pred.predict_many_with_var(sp, [9.8, 5.0, 12.0], [0.3, 0.5, 0.2])
        assert m.shape == v.shape == (3, n_frames(sp), 4)
        assert (v >= 1e-4 - 1e-9).all()                   # never below the calibrated floor
        assert np.allclose(m[:, 0, :2], initial_state(sp)[:2], atol=1e-6)


def test_g_scaled_structure():
    """In g-scaled mode g is not a network input (globals carry mu only) and the
    physical output scale is exactly proportional to g."""
    m = GraphStep(16, g_scaled=True)
    glob = torch.tensor([[0.5, 0.3], [1.0, 0.3]])          # g = 5 and 10
    assert torch.equal(m.glob_features(glob), glob[:, 1:2])
    sc = m.out_scale(glob)
    assert torch.allclose(sc[1], 2 * sc[0])
    # free fall is representable exactly: F = (0, -1) gives dv_y = -g dt
    from hidden_physics.learned import FRAME_DT
    assert torch.allclose(sc[:, 1], glob[:, 0] * 10 * FRAME_DT)


def test_old_checkpoints_still_load():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ens.pt"
        _tiny_ensemble(p, g_scaled=False)
        ck = torch.load(p, weights_only=False)
        del ck["g_scaled"]                      # checkpoints saved before the flag existed
        torch.save(ck, p)
        assert not LearnedPredictor(p).models[0].g_scaled


def test_correlated_likelihood():
    """Identity correlation must reproduce the independent-frames Gaussian exactly;
    strong correlation must make the same residuals less decisive between hypotheses."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ens.pt"
        _tiny_ensemble(p, g_scaled=True)
        pred = LearnedPredictor(p)
        nz = NOISE_LEVELS["medium"]
        sp = E.make("drop", "ball", height=1)
        obs = World(HiddenPhysics(9.8, 0.3), nz.pos, nz.vel).execute_experiment(sp).states[:, [0, 1, 3, 4]]
        b = GridBelief(pred, likelihood_sigma(nz), n=6)
        pred.error_corr = np.stack([np.eye(46)] * 4)
        m, v = pred.predict_many_with_var(sp, b.G, b.M)
        s2 = b.sigma ** 2 + v
        ref = -0.5 * ((m - obs[None]) ** 2 / s2 + np.log(s2)).sum((1, 2))
        assert np.allclose(b._loglik(sp, obs), ref)
        spread_indep = np.ptp(b._loglik(sp, obs))
        pred.error_corr = np.stack([np.full((46, 46), 0.95) + 0.05 * np.eye(46)] * 4)
        assert np.ptp(b._loglik(sp, obs)) < spread_indep


def test_shared_bias_widens_intervals():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ens.pt"
        _tiny_ensemble(p)
        pred = LearnedPredictor(p)
        assert pred.param_bias_sd is None                        # older checkpoints: no bias term
        nz = NOISE_LEVELS["medium"]
        traj = World(HiddenPhysics(9.8, 0.3), nz.pos, nz.vel).execute_experiment(E.make("flat_slide", "block", v0=2.5))
        b0 = GridBelief(pred, likelihood_sigma(nz), n=8)
        b0.update(traj)
        pred.param_bias_sd = np.array([0.02, 0.01])
        b1 = GridBelief(pred, likelihood_sigma(nz), n=8)
        b1.update(traj)
        assert np.allclose(b0.weights, b1.weights) and b0.map == b1.map   # bias only widens the report
        c0, c1 = b0.credible_interval(), b1.credible_interval()
        for k in ("g", "mu"):                                    # never narrower; g clearly wider
            assert c1[k][0] <= c0[k][0] + 1e-12 and c1[k][1] >= c0[k][1] - 1e-12
        assert c1["g"][1] - c1["g"][0] > c0["g"][1] - c0["g"][0]
        assert b1.sd[0] > b0.sd[0] and b1.sd[1] > b0.sd[1]


def test_belief_runs_with_learned_predictor():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ens.pt"
        _tiny_ensemble(p)
        nz = NOISE_LEVELS["high"]
        b = GridBelief(LearnedPredictor(p), likelihood_sigma(nz), n=8)
        b.update(World(HiddenPhysics(9.8, 0.3), nz.pos, nz.vel).execute_experiment(E.make("drop", "ball", height=1)))
        assert np.isfinite(b.loglik).all() and abs(b.weights.sum() - 1) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
