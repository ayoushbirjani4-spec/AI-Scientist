"""Stage B: a parameter-conditioned graph network world model, trained as an ensemble.

Graph per time step
  nodes   the dynamic object (state + known shape/mass) and each static surface
          (floor, and the ramp when present)
  edges   surface -> object, with the object's pose and velocity expressed in the
          surface frame (normal gap, along-surface position, relative angle)
  globals candidate g and mu
  output  change in velocity and spin over one observation step (1/30 s), plus a
          small position correction; integrated (trapezoid) to the next state

The model never calls the simulator: it gets the experiment setup (initial pose,
initial velocity, surfaces, from env.scene_setup) and rolls forward on its own.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .adapter import LIKELIHOOD_COLS
from .env import (DT, DURATION, OBJECTS, RECORD_EVERY, ExperimentSpec, scene_setup,
                  simulate)

FRAME_DT = DT * RECORD_EVERY
OBJ_NAMES = tuple(OBJECTS)
MAX_SURF = 2
VEL_SCALE, OMEGA_SCALE = 3.0, 10.0
G_SCALE = 10.0
# Per-dimension scale for the rollout loss: x, y, angle, vx, vy, omega
STATE_LOSS_SCALE = torch.tensor([0.02, 0.02, 0.05, 0.2, 0.2, 1.0])


def n_frames(spec: ExperimentSpec) -> int:
    return int(round(DURATION[spec.kind] / DT)) // RECORD_EVERY + 1


# ------------------------------------------------------------------ static context
def object_features(obj: str) -> np.ndarray:
    o = OBJECTS[obj]
    onehot = np.eye(len(OBJ_NAMES))[OBJ_NAMES.index(obj)]
    if o["shape"] == "circle":
        w = h = 2 * o["radius"]
        circ = 1.0
    else:
        w, h = o["size"]
        circ = 0.0
    return np.concatenate([onehot, [o["mass"] / 3.0, w, h, circ]]).astype(np.float32)


def surface_features(spec: ExperimentSpec) -> tuple[np.ndarray, np.ndarray]:
    """[MAX_SURF, 7] = a(2), unit tangent(2), unit normal(2), length; and mask."""
    out = np.zeros((MAX_SURF, 7), np.float32)
    mask = np.zeros(MAX_SURF, np.float32)
    for k, s in enumerate(scene_setup(spec).surfaces):
        a, b = np.array(s.a), np.array(s.b)
        L = np.linalg.norm(b - a)
        out[k] = [*a, *((b - a) / L), *s.n, L]
        mask[k] = 1.0
    return out, mask


def initial_state(spec: ExperimentSpec) -> np.ndarray:
    s = scene_setup(spec)
    return np.array([*s.position, s.angle, *s.velocity, s.omega], np.float32)


# ------------------------------------------------------------------ model
def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class GraphStep(nn.Module):
    """One message-passing step: surfaces -> object, then an object update.

    g_scaled=False  inputs include g; outputs are raw velocity changes (first model).
    g_scaled=True   dimensional analysis: the network never sees g. Velocities enter
                    as v / (g dt) and gaps also as gap / (g dt^2), and its outputs are
                    multiplied by g dt (velocity) and g dt^2 (position correction):
                        dv = g dt * F(dimensionless state, geometry, mu)
                    Free fall is F = (0, -1), sliding F = -mu, a ramp sin - mu cos,
                    and an impact dv = -(1+e) v is linear in the scaled velocity, so
                    the dependence on g is exact by construction instead of learned.
    """
    NODE_IN = 4 + 1 + 7          # sin, cos, vx, vy | omega | object features

    def __init__(self, hidden: int = 128, g_scaled: bool = False):
        super().__init__()
        self.hidden = hidden
        self.g_scaled = g_scaled
        self.edge_in = 13 + (1 if g_scaled else 0)
        self.glob_in = 1 if g_scaled else 3
        self.edge_mlp = _mlp(self.edge_in + self.NODE_IN + self.glob_in, hidden, hidden)
        self.node_mlp = _mlp(self.NODE_IN + hidden + self.glob_in, hidden, 6)
        self.register_buffer("tgt_mean", torch.zeros(6))
        self.register_buffer("tgt_std", torch.ones(6))

    # ---- scaling helpers
    @staticmethod
    def vel_unit(glob):
        """g * dt per sample: the velocity a body gains in one step of free fall."""
        return glob[:, 0] * G_SCALE * FRAME_DT

    def out_scale(self, glob):
        """Factor turning network outputs into physical (dv, dw, position residual)."""
        if not self.g_scaled:
            return torch.ones(glob.shape[0], 6, dtype=glob.dtype)
        u = self.vel_unit(glob)[:, None]
        return torch.cat([u.expand(-1, 3), (u * FRAME_DT).expand(-1, 3)], 1)

    def glob_features(self, glob):
        if self.g_scaled:
            return glob[:, 1:2]                                     # mu only
        return torch.cat([glob, (glob[:, 0] * glob[:, 1])[:, None]], 1)

    def graph_inputs(self, state, objf, surf, smask, glob):
        pos, th, v, w = state[:, :2], state[:, 2], state[:, 3:5], state[:, 5]
        if self.g_scaled:
            u = self.vel_unit(glob)
            vel = lambda x: torch.asinh(x / u.view(-1, *[1] * (x.dim() - 1))) / 3
            vfeat, wfeat = vel(v), vel(w)
        else:
            vel = lambda x: x / VEL_SCALE
            vfeat, wfeat = v / VEL_SCALE, w / OMEGA_SCALE
        node = torch.cat([torch.sin(th)[:, None], torch.cos(th)[:, None], vfeat, wfeat[:, None], objf], 1)
        a, t, n, L = surf[..., 0:2], surf[..., 2:4], surf[..., 4:6], surf[..., 6]
        rel = pos[:, None, :] - a
        st = (rel * t).sum(-1)
        d = (rel * n).sum(-1)
        dth = th[:, None] - torch.atan2(t[..., 1], t[..., 0])
        # support distance of the known shape along the surface normal (pure geometry)
        wdt, hgt, circ = objf[:, None, 4], objf[:, None, 5], objf[:, None, 6]
        box_sup = (wdt / 2 * torch.sin(dth)).abs() + (hgt / 2 * torch.cos(dth)).abs()
        sup = circ * hgt / 2 + (1 - circ) * box_sup
        gap = d - sup
        vt = (v[:, None, :] * t).sum(-1)
        vn = (v[:, None, :] * n).sum(-1)
        feats = [gap.clamp(-0.1, 1.0), d.clamp(-0.5, 3.0), st.clamp(-1, 1), (L - st).clamp(-1, 1),
                 n[..., 0], n[..., 1], vel(vt), vel(vn),
                 torch.sin(dth), torch.cos(dth), torch.sin(2 * dth), torch.cos(2 * dth),
                 (gap < 0.02).float()]
        if self.g_scaled:   # gap in units of one free-fall step of displacement
            feats.append(torch.asinh(gap / (self.vel_unit(glob)[:, None] * FRAME_DT)) / 3)
        return node, torch.stack(feats, -1)

    def forward(self, state, objf, surf, smask, glob, as_step: bool = False):
        """Normalized targets, or (as_step=True) the integrated next state."""
        gf = self.glob_features(glob)
        node, edge = self.graph_inputs(state, objf, surf, smask, glob)
        S = edge.shape[1]
        e_in = torch.cat([edge, node[:, None].expand(-1, S, -1), gf[:, None].expand(-1, S, -1)], -1)
        msg = self.edge_mlp(e_in) * smask[..., None]
        agg = msg.sum(1)
        out = self.node_mlp(torch.cat([node, agg, gf], 1))
        if as_step:
            return integrate(state, (out * self.tgt_std + self.tgt_mean) * self.out_scale(glob))
        return out

    def step(self, state, objf, surf, smask, glob):
        return self.forward(state, objf, surf, smask, glob, as_step=True)


def integrate(state, out):
    """out = (dvx, dvy, domega, res_x, res_y, res_angle)."""
    v_old, w_old = state[:, 3:5], state[:, 5]
    v_new = v_old + out[:, 0:2]
    w_new = w_old + out[:, 2]
    pos = state[:, :2] + FRAME_DT * (v_old + v_new) / 2 + out[:, 3:5]
    th = state[:, 2] + FRAME_DT * (w_old + w_new) / 2 + out[:, 5]
    return torch.cat([pos, th[:, None], v_new, w_new[:, None]], 1)


def targets(s0, s1):
    """Inverse of `integrate`: what the network should output to map s0 -> s1."""
    dv = s1[:, 3:6] - s0[:, 3:6]
    res_pos = s1[:, 0:2] - s0[:, 0:2] - FRAME_DT * (s0[:, 3:5] + s1[:, 3:5]) / 2
    res_th = s1[:, 2] - s0[:, 2] - FRAME_DT * (s0[:, 5] + s1[:, 5]) / 2
    return torch.cat([dv, res_pos, res_th[:, None]], 1)


# ------------------------------------------------------------------ data
@dataclass
class Dataset:
    states: np.ndarray    # [N, Tmax, 6]
    lengths: np.ndarray   # [N]
    objf: np.ndarray      # [N, 7]
    surf: np.ndarray      # [N, MAX_SURF, 7]
    smask: np.ndarray     # [N, MAX_SURF]
    glob: np.ndarray      # [N, 2] (g / G_SCALE, mu)
    kinds: np.ndarray     # [N] experiment kind (str)
    specs: list

    def subset(self, idx):
        return Dataset(self.states[idx], self.lengths[idx], self.objf[idx], self.surf[idx],
                       self.smask[idx], self.glob[idx], self.kinds[idx], [self.specs[i] for i in idx])

    def save(self, path):
        np.savez_compressed(path, states=self.states, lengths=self.lengths, objf=self.objf, surf=self.surf,
                            smask=self.smask, glob=self.glob, kinds=self.kinds,
                            specs=np.array([json.dumps(s.to_dict()) for s in self.specs]))

    @staticmethod
    def load(path):
        z = np.load(path)
        specs = []
        for js in z["specs"]:
            d = json.loads(str(js))
            specs.append(ExperimentSpec.make(d.pop("kind"), d.pop("obj"), **d))
        return Dataset(z["states"], z["lengths"], z["objf"], z["surf"], z["smask"], z["glob"], z["kinds"], specs)


def context_arrays(spec: ExperimentSpec, g: float, mu: float):
    surf, mask = surface_features(spec)
    return object_features(spec.obj), surf, mask, np.array([g / G_SCALE, mu], np.float32)


def generate_dataset(n: int, seed: int, g_range=(2.0, 20.0), mu_range=(0.05, 0.80)) -> Dataset:
    """Each trajectory is a fresh (scene, g, mu) draw, so a split by trajectory is a
    split by scene and parameter setting, never by adjacent frames."""
    from .planner import random_spec
    rng = np.random.default_rng(seed)
    tmax = max(int(round(d / DT)) // RECORD_EVERY + 1 for d in DURATION.values())
    states = np.zeros((n, tmax, 6), np.float32)
    lengths = np.zeros(n, np.int64)
    objf = np.zeros((n, 7), np.float32)
    surf = np.zeros((n, MAX_SURF, 7), np.float32)
    smask = np.zeros((n, MAX_SURF), np.float32)
    glob = np.zeros((n, 2), np.float32)
    kinds, specs = [], []
    for i in range(n):
        g, mu = rng.uniform(*g_range), rng.uniform(*mu_range)
        spec = random_spec(rng)
        _, s = simulate(spec, g, mu)
        states[i, :len(s)] = s
        lengths[i] = len(s)
        objf[i], surf[i], smask[i], glob[i] = context_arrays(spec, g, mu)
        kinds.append(spec.kind)
        specs.append(spec)
    return Dataset(states, lengths, objf, surf, smask, glob, np.array(kinds), specs)


def transition_labels(ds: Dataset) -> np.ndarray:
    """[N, Tmax-1] category per transition: 0 free flight, 1 contact (sliding or
    resting), 2 impact (lands on a surface during the step). -1 = padding."""
    lab = -np.ones((len(ds.lengths), ds.states.shape[1] - 1), np.int64)
    with torch.no_grad():
        for i in range(len(ds.lengths)):
            T = ds.lengths[i]
            st = torch.from_numpy(ds.states[i, :T])
            ctx = [torch.from_numpy(np.repeat(a[i][None], T, 0)) for a in (ds.objf, ds.surf, ds.smask, ds.glob)]
            _, edge = GraphStep(8).graph_inputs(st, *ctx)
            gap = torch.where(ctx[2] > 0, edge[..., 0], torch.full_like(edge[..., 0], 9.0)).min(1).values
            touch = (gap < 0.02).numpy()
            lab[i, :T - 1] = np.where(touch[:-1], 1, np.where(touch[1:], 2, 0))
    return lab


# ------------------------------------------------------------------ training
def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def train_member(train: Dataset, val: Dataset, seed: int, epochs_one: int = 25, epochs_roll: int = 12,
                 rollout_k: int = 10, batch: int = 512, hidden: int = 128, input_noise=(0.0, 0.0),
                 g_scaled: bool = False, log=print) -> tuple[GraphStep, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GraphStep(hidden, g_scaled)
    S = _t(train.states)
    ctx = [_t(a) for a in (train.objf, train.surf, train.smask, train.glob)]

    # transitions (traj, frame)
    pairs = np.array([(i, k) for i in range(len(train.lengths)) for k in range(train.lengths[i] - 1)])
    s0, s1 = S[pairs[:, 0], pairs[:, 1]], S[pairs[:, 0], pairs[:, 1] + 1]
    G = ctx[3][pairs[:, 0]]
    scale = model.out_scale(G)
    tg = targets(s0, s1) / scale
    model.tgt_mean.copy_(tg.mean(0))
    model.tgt_std.copy_(tg.std(0) + 1e-6)
    noise = torch.tensor([input_noise[0], input_noise[0], input_noise[0] * 5,
                          input_noise[1], input_noise[1], input_noise[1] * 5])

    hist = {"one_step": [], "rollout": [], "val_rollout": []}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs_one)
    for ep in range(epochs_one):
        perm = rng.permutation(len(pairs))
        tot = 0.0
        for b in range(0, len(perm), batch):
            j = perm[b:b + batch]
            ii = pairs[j, 0]
            x0 = s0[j] + torch.randn(len(j), 6) * noise           # input corruption
            y = (targets(x0, s1[j]) / scale[j] - model.tgt_mean) / model.tgt_std
            pred = model(x0, *(c[ii] for c in ctx))
            # Huber: impact steps are heavy-tailed (F ~ 50 on a landing vs ~1 otherwise).
            # Errors under 0.5 std stay quadratic (the precision identifying g needs);
            # large misses on impacts count linearly so they don't dominate.
            loss = nn.functional.smooth_l1_loss(pred, y, beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        sched.step()
        hist["one_step"].append(tot / len(perm))
        if ep % 5 == 4 or ep == epochs_one - 1:
            log(f"  member {seed} one-step epoch {ep + 1}/{epochs_one}: loss {hist['one_step'][-1]:.4f}")

    # multi-step rollout loss on windows of K steps
    wins = np.array([(i, k) for i in range(len(train.lengths)) for k in range(0, train.lengths[i] - rollout_k, 3)])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs_roll, 1))
    for ep in range(epochs_roll):
        perm = rng.permutation(len(wins))
        tot = 0.0
        for b in range(0, len(perm), batch // 2):
            j = perm[b:b + batch // 2]
            ii, kk = wins[j, 0], wins[j, 1]
            c = [a[ii] for a in ctx]
            x = S[ii, kk] + torch.randn(len(j), 6) * noise * 0.5
            loss = 0.0
            for step in range(1, rollout_k + 1):
                x = model.step(x, *c)
                err = (x - S[ii, kk + step]) / STATE_LOSS_SCALE
                loss = loss + nn.functional.smooth_l1_loss(err, torch.zeros_like(err), beta=1.0)
            loss = loss / rollout_k
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(j)
        sched.step()
        hist["rollout"].append(tot / len(perm))
        vr = rollout_error(model, val)
        hist["val_rollout"].append(vr)
        log(f"  member {seed} rollout epoch {ep + 1}/{epochs_roll}: loss {hist['rollout'][-1]:.4f}  "
            f"val pos RMSE {vr:.4f} m")
    return model, hist


@torch.no_grad()
def rollout_dataset(model: GraphStep, ds: Dataset) -> np.ndarray:
    """Full-length open-loop rollouts from each trajectory's first frame. [N, Tmax, 6]"""
    S = _t(ds.states)
    ctx = [_t(a) for a in (ds.objf, ds.surf, ds.smask, ds.glob)]
    x = S[:, 0]
    out = [x]
    for _ in range(S.shape[1] - 1):
        x = model.step(x, *ctx)
        out.append(x)
    return torch.stack(out, 1).numpy()


def rollout_error(model: GraphStep, ds: Dataset) -> float:
    r = rollout_dataset(model, ds)
    mask = np.arange(ds.states.shape[1])[None] < ds.lengths[:, None]
    e = ((r[..., :2] - ds.states[..., :2]) ** 2).sum(-1)
    return float(np.sqrt(e[mask].mean()))


def calibrate_floor(models: list[GraphStep], val: Dataset) -> np.ndarray:
    """Per-frame, per-column extra variance so that the ensemble's predictive spread
    plus this floor matches the observed rollout error on validation data. [Tmax, 4]"""
    rolls = np.stack([rollout_dataset(m, val) for m in models])[..., LIKELIHOOD_COLS]   # [M,N,T,4]
    truth = val.states[..., LIKELIHOOD_COLS]
    mask = (np.arange(val.states.shape[1])[None] < val.lengths[:, None])[..., None]
    err2 = ((rolls.mean(0) - truth) ** 2 * mask).sum(0) / np.maximum(mask.sum(0), 1)
    ens = (rolls.var(0) * mask).sum(0) / np.maximum(mask.sum(0), 1)
    floor = np.maximum(err2 - ens, 0.0)
    return np.maximum.accumulate(floor, axis=0).astype(np.float32)   # never shrinks with time


def calibrate_error_corr(models: list[GraphStep], val: Dataset, floor: np.ndarray,
                         shrink: float = 0.05) -> np.ndarray:
    """Frame-to-frame correlation of the ensemble-mean rollout error, per column,
    estimated on validation data. [4, Tmax, Tmax]

    Rollout errors are correlated: a slightly biased model is off the same way for a
    whole trajectory. Treating frames as independent then makes the posterior far too
    confident. This matrix lets the belief use a full-covariance likelihood instead."""
    rolls = np.stack([rollout_dataset(m, val) for m in models])[..., LIKELIHOOD_COLS]   # [M,N,T,4]
    T = val.states.shape[1]
    var = rolls.var(0) + floor[None, :T]
    u = (rolls.mean(0) - val.states[..., LIKELIHOOD_COLS]) / np.sqrt(np.maximum(var, 1e-12))
    mask = (np.arange(T)[None] < val.lengths[:, None]).astype(float)                    # [N,T]
    u = u * mask[..., None]
    count = np.maximum(mask.T @ mask, 1.0)                                              # [T,T]
    corr = np.empty((u.shape[-1], T, T))
    for c in range(u.shape[-1]):
        cov = (u[..., c].T @ u[..., c]) / count
        d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        r = cov / np.outer(d, d)
        r = (1 - shrink) * r + shrink * np.eye(T)              # shrink toward independence
        w, V = np.linalg.eigh((r + r.T) / 2)                  # make it a valid correlation matrix
        r = (V * np.maximum(w, 1e-6)) @ V.T
        d = np.sqrt(np.diag(r))
        corr[c] = r / np.outer(d, d)
    return corr.astype(np.float32)


def save_ensemble(path, models, floor, meta, error_corr=None):
    torch.save({"state_dicts": [m.state_dict() for m in models], "hidden": models[0].hidden,
                "g_scaled": models[0].g_scaled, "floor_var": floor, "error_corr": error_corr,
                "meta": meta}, path)


# ------------------------------------------------------------------ predictor
class LearnedPredictor:
    """Drop-in replacement for SimulatorPredictor, backed by the GNN ensemble.

    predict_many_with_var also returns predictive variance: ensemble spread plus a
    calibrated floor, which the belief adds to its noise model.
    """

    def __init__(self, path: str | Path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.models = []
        g_scaled = ck.get("g_scaled", False)
        for sd in ck["state_dicts"]:
            m = GraphStep(ck["hidden"], g_scaled)
            m.load_state_dict(sd)
            m.eval()
            self.models.append(m)
        self.floor_var = ck["floor_var"]
        self.error_corr = ck.get("error_corr")      # None: belief treats frames as independent
        self.meta = ck["meta"]
        self.version = f"stageB-gnn{'-gscaled' if g_scaled else ''}-ensemble{len(self.models)}"
        # Run all members at once: stack their weights and vmap one functional step.
        from torch.func import functional_call, stack_module_state
        self._params, self._buffers = stack_module_state(self.models)
        base = GraphStep(ck["hidden"], g_scaled).to("meta")

        def one(p, b, x, objf, surf, mask, glob):
            return functional_call(base, (p, b), (x, objf, surf, mask, glob), {"as_step": True})
        self._vstep = torch.vmap(one, in_dims=(0, 0, 0, None, None, None, None))

    @torch.no_grad()
    def rollout(self, spec: ExperimentSpec, gs, mus) -> np.ndarray:
        """[M, N, T, 6] rollouts for N hypotheses under each ensemble member."""
        gs, mus = np.asarray(gs, np.float32), np.asarray(mus, np.float32)
        N, T = len(gs), n_frames(spec)
        objf, surf, mask, _ = context_arrays(spec, 0.0, 0.0)
        ctx = [torch.from_numpy(np.repeat(a[None], N, 0)) for a in (objf, surf, mask)]
        glob = torch.from_numpy(np.stack([gs / G_SCALE, mus], 1))
        x0 = torch.from_numpy(np.repeat(initial_state(spec)[None], N, 0))
        x = x0.expand(len(self.models), -1, -1)
        traj = [x]
        for _ in range(T - 1):
            x = self._vstep(self._params, self._buffers, x, *ctx, glob)
            traj.append(x)
        return torch.stack(traj, 2).numpy().astype(np.float64)       # [M, N, T, 6]

    def predict_many_with_var(self, spec, gs, mus, chunk: int = 100):
        gs, mus = np.asarray(gs), np.asarray(mus)
        means, vars_ = [], []
        for i in range(0, len(gs), chunk):        # chunked: keeps peak memory low
            r = self.rollout(spec, gs[i:i + chunk], mus[i:i + chunk])[..., LIKELIHOOD_COLS]
            means.append(r.mean(0))
            vars_.append(r.var(0))
        T = means[0].shape[1]
        return np.concatenate(means), np.concatenate(vars_) + self.floor_var[:T][None]

    def predict_many(self, spec, gs, mus):
        return self.predict_many_with_var(spec, gs, mus)[0]

    def predict(self, spec, g, mu):
        return self.predict_many(spec, [g], [mu])[0]

    def predict_with_var(self, spec, g, mu):
        m, v = self.predict_many_with_var(spec, [g], [mu])
        return m[0], v[0]
