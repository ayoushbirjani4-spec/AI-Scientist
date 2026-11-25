"""Belief over (g, mu): an adaptive grid with tempering and refinement.

Each grid point is a hypothesis. The likelihood is Student-t (heavy-tailed, so a
single contact frame cannot dominate) with sigma = measurement noise and predictor
error added in quadrature.

Two safeguards stop the belief collapsing too early:
* Tempering: if the likelihood is so sharp that fewer than `min_ess` grid cells
  carry weight, it is flattened (log-likelihood / T) until they do. This treats
  "the grid is too coarse to represent the posterior" as extra model error. A thin
  ridge such as mu*g = const then stays visibly uncertain instead of snapping to
  whichever cell happens to lie closest to it.
* Refinement: the grid zooms to the tempered posterior mean +/- 3 sd and every
  logged experiment is re-scored on the new grid, repeating while the box shrinks.
  As resolution improves, T falls back toward 1.
"""
from __future__ import annotations

import numpy as np

from .adapter import LIKELIHOOD_COLS
from .env import ExperimentSpec, Trajectory
from .predictor import SimulatorPredictor

PRIOR_G = (2.0, 20.0)
PRIOR_MU = (0.05, 0.80)


def _weighted_quantile(axis: np.ndarray, w: np.ndarray, q: float) -> float:
    c = np.cumsum(w) - w / 2          # CDF at cell centres
    return float(np.interp(q, c, axis))


def _mixture_quantile(centres: np.ndarray, w: np.ndarray, sd: np.ndarray, q: float) -> float:
    """q-quantile of sum_i w_i N(centres_i, sd_i^2), by bisection on the CDF."""
    from math import erf, sqrt
    keep = w > 1e-12
    c, w, sd = centres[keep], w[keep] / w[keep].sum(), np.maximum(sd[keep], 1e-12)
    cdf = lambda x: float(w @ (0.5 * (1 + np.vectorize(erf)((x - c) / (sd * sqrt(2))))))
    lo, hi = float((c - 6 * sd).min()), float((c + 6 * sd).max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cdf(mid) < q else (lo, mid)
    return 0.5 * (lo + hi)


class GridBelief:
    def __init__(self, predictor: SimulatorPredictor, sigma: np.ndarray, n: int = 20,
                 g_range=PRIOR_G, mu_range=PRIOR_MU, max_refines: int = 6,
                 nu: float | None = 4.0, min_ess: float = 4.0, shrink: float = 0.85):
        """nu: Student-t degrees of freedom for a heavy-tailed likelihood; None = Gaussian."""
        self.nu = nu
        self.min_ess = min_ess
        self.shrink = shrink
        self.T = 1.0
        self.predictor = predictor
        # Shared model bias: one offset for the whole run, so it never averages away
        # over experiments. (relative sd of g, absolute sd of mu); None = no bias term.
        bsd = getattr(predictor, "param_bias_sd", None)
        self.bias_sd = None if bsd is None else (float(bsd[0]), float(bsd[1]))
        self.sigma = np.asarray(sigma)
        self.n = n
        self.prior_g, self.prior_mu = tuple(g_range), tuple(mu_range)
        self.max_refines = max_refines
        self.history: list[tuple[ExperimentSpec, np.ndarray]] = []
        self.n_refines = 0
        self._set_grid(self.prior_g, self.prior_mu)

    # ---------------- grid management ----------------
    def _set_grid(self, g_range, mu_range):
        self.g_axis = np.linspace(*g_range, self.n)
        self.mu_axis = np.linspace(*mu_range, self.n)
        G, M = np.meshgrid(self.g_axis, self.mu_axis, indexing="ij")
        self.G, self.M = G.ravel(), M.ravel()
        self.loglik = np.zeros(self.n * self.n)
        for spec, obs in self.history:
            self.loglik += self._loglik(spec, obs)
        self._fit_temperature()

    def _fit_temperature(self):
        ll = self.loglik - self.loglik.max()

        def ess(T):
            w = np.exp(ll / T)
            w /= w.sum()
            return 1.0 / (w @ w)

        if ess(1.0) >= self.min_ess:
            self.T = 1.0
            return
        lo, hi = 1.0, 4.0
        while ess(hi) < self.min_ess and hi < 1e12:
            lo, hi = hi, hi * 4
        for _ in range(25):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if ess(mid) < self.min_ess else (lo, mid)
        self.T = float(hi)

    def _loglik(self, spec: ExperimentSpec, obs: np.ndarray) -> np.ndarray:
        if getattr(self.predictor, "error_corr", None) is not None:
            return self._loglik_correlated(spec, obs)
        if hasattr(self.predictor, "predict_many_with_var"):
            # Learned model: per-hypothesis predictive variance joins the noise model,
            # so the scale differs across hypotheses and its log term matters.
            preds, var = self.predictor.predict_many_with_var(spec, self.G, self.M)
            scale2 = self.sigma ** 2 + var
            r2 = (preds - obs[None]) ** 2 / scale2
            norm = -0.5 * np.log(scale2).sum((1, 2))
        else:
            preds = self.predictor.predict_many(spec, self.G, self.M)    # [N, T, 4]
            r = (preds - obs[None]) / self.sigma
            r2 = r * r
            norm = 0.0
        if self.nu is None:
            return -0.5 * r2.sum((1, 2)) + norm
        return -0.5 * (self.nu + 1) * np.log1p(r2 / self.nu).sum((1, 2)) + norm

    def _loglik_correlated(self, spec: ExperimentSpec, obs: np.ndarray) -> np.ndarray:
        """Gaussian likelihood with model error correlated across frames.

        Per column: Sigma = diag(sigma^2) + D C D, with D the model's per-frame error
        spread (ensemble + calibrated floor, per hypothesis) and C the frame-to-frame
        error correlation measured on validation data. Correlated frames then count as
        less independent evidence, which keeps a slightly biased model from producing
        an overconfident posterior."""
        preds, var = self.predictor.predict_many_with_var(spec, self.G, self.M)    # [N, T, C]
        T = preds.shape[1]
        r = preds - obs[None]
        ll = np.zeros(len(preds))
        for c in range(preds.shape[2]):
            d = np.sqrt(var[:, :, c])
            S = d[:, :, None] * self.predictor.error_corr[c, :T, :T][None] * d[:, None, :]
            S[:, np.arange(T), np.arange(T)] += self.sigma[c] ** 2
            L = np.linalg.cholesky(S)
            z = np.linalg.solve(L, r[:, :, c][..., None])[..., 0]
            ll += -0.5 * (z * z).sum(1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
        return ll

    def residual_frames(self, spec: ExperimentSpec, obs: np.ndarray, g: float, mu: float) -> np.ndarray:
        """Per-frame mean squared normalized residual [T]; about 1 when the model fits."""
        if hasattr(self.predictor, "predict_with_var"):
            pred, var = self.predictor.predict_with_var(spec, g, mu)
            return np.mean((pred - obs) ** 2 / (self.sigma ** 2 + var), axis=1)
        r = (self.predictor.predict(spec, g, mu) - obs) / self.sigma
        return np.mean(r * r, axis=1)

    def residual_z(self, spec: ExperimentSpec, obs: np.ndarray, g: float, mu: float) -> float:
        """Mean squared normalized residual over the whole experiment."""
        return float(np.mean(self.residual_frames(spec, obs, g, mu)))

    def _maybe_refine(self):
        for _ in range(self.max_refines):
            (mg, mm), (sg, sm) = self.mean, self._raw_sd()
            cg, cm = self.g_axis[1] - self.g_axis[0], self.mu_axis[1] - self.mu_axis[0]
            hg, hm = max(3 * sg, 1.5 * cg), max(3 * sm, 1.5 * cm)
            gr = (max(self.prior_g[0], mg - hg), min(self.prior_g[1], mg + hg))
            mr = (max(self.prior_mu[0], mm - hm), min(self.prior_mu[1], mm + hm))
            old_area = (self.g_axis[-1] - self.g_axis[0]) * (self.mu_axis[-1] - self.mu_axis[0])
            if (gr[1] - gr[0]) * (mr[1] - mr[0]) > self.shrink * old_area:
                return
            self._set_grid(gr, mr)
            self.n_refines += 1

    # ---------------- public API ----------------
    def update(self, traj: Trajectory) -> None:
        obs = traj.states[:, LIKELIHOOD_COLS]
        self.history.append((traj.spec, obs))
        self.loglik += self._loglik(traj.spec, obs)
        self._fit_temperature()
        self._maybe_refine()

    def reset(self, keep_last: int = 0) -> None:
        """Broaden back to the prior, keeping only the most recent experiments."""
        self.history = self.history[-keep_last:] if keep_last else []
        self._set_grid(self.prior_g, self.prior_mu)
        self._maybe_refine()

    @property
    def weights(self) -> np.ndarray:
        w = np.exp((self.loglik - self.loglik.max()) / self.T)
        return w / w.sum()

    @property
    def mean(self) -> tuple[float, float]:
        w = self.weights
        return float(w @ self.G), float(w @ self.M)

    def _raw_sd(self) -> tuple[float, float]:
        """Spread of the grid posterior alone (no shared-bias term); drives grid refinement."""
        w = self.weights
        mg, mm = w @ self.G, w @ self.M
        return float(np.sqrt(w @ (self.G - mg) ** 2)), float(np.sqrt(w @ (self.M - mm) ** 2))

    @property
    def sd(self) -> tuple[float, float]:
        sg, sm = self._raw_sd()
        if self.bias_sd is None:
            return sg, sm
        w = self.weights
        bg, bm = self.bias_sd
        return float(np.sqrt(sg ** 2 + w @ (self.G * bg) ** 2)), float(np.sqrt(sm ** 2 + bm ** 2))

    @property
    def map(self) -> tuple[float, float]:
        i = int(np.argmax(self.loglik))
        return float(self.G[i]), float(self.M[i])

    def credible_interval(self, level: float = 0.9) -> dict:
        w = self.weights.reshape(self.n, self.n)
        lo, hi = (1 - level) / 2, (1 + level) / 2
        wg, wm = w.sum(1), w.sum(0)
        if self.bias_sd is not None:
            # Grid posterior is over the parameters as the model sees them; the true value
            # is off by the shared bias, so each cell becomes a normal of the bias width.
            # A cell's own width (uniform: var = width^2 / 12) is added so the interval
            # stays smooth on a coarse grid; the result never falls inside the plain grid
            # interval, and is clipped to the prior.
            bg, bm = self.bias_sd
            cg, cm = self.g_axis[1] - self.g_axis[0], self.mu_axis[1] - self.mu_axis[0]
            sg = np.sqrt((self.g_axis * bg) ** 2 + cg ** 2 / 12)
            sm = np.full_like(self.mu_axis, np.sqrt(bm ** 2 + cm ** 2 / 12))
            def ci(ax, w, sd, pr):
                a = min(_mixture_quantile(ax, w, sd, lo), _weighted_quantile(ax, w, lo))
                b = max(_mixture_quantile(ax, w, sd, hi), _weighted_quantile(ax, w, hi))
                return float(max(a, pr[0])), float(min(b, pr[1]))
            return {"g": ci(self.g_axis, wg, sg, self.prior_g), "mu": ci(self.mu_axis, wm, sm, self.prior_mu)}
        return {"g": (_weighted_quantile(self.g_axis, wg, lo), _weighted_quantile(self.g_axis, wg, hi)),
                "mu": (_weighted_quantile(self.mu_axis, wm, lo), _weighted_quantile(self.mu_axis, wm, hi))}

    @property
    def entropy(self) -> float:
        """Differential entropy estimate (nats) of the (g, mu) posterior."""
        w = self.weights
        cell = (self.g_axis[1] - self.g_axis[0]) * (self.mu_axis[1] - self.mu_axis[0])
        nz = w[w > 0]
        return float(-(nz * np.log(nz)).sum() + np.log(cell))

    def sample(self, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """k distinct hypotheses drawn by weight, with renormalized weights."""
        w = self.weights
        k = min(k, int((w > 0).sum()))
        idx = rng.choice(len(w), size=k, replace=False, p=w)
        ww = w[idx] / w[idx].sum()
        return self.G[idx], self.M[idx], ww

    def snapshot(self) -> dict:
        (mg, mm), (sg, sm) = self.mean, self.sd
        ci = self.credible_interval()
        return {"g_mean": mg, "mu_mean": mm, "g_sd": sg, "mu_sd": sm,
                "g_ci": ci["g"], "mu_ci": ci["mu"], "entropy": self.entropy,
                "grid_g": (float(self.g_axis[0]), float(self.g_axis[-1])),
                "grid_mu": (float(self.mu_axis[0]), float(self.mu_axis[-1])),
                "n_history": len(self.history), "temperature": self.T}
