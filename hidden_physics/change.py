"""Change detector: flags persistent prediction residuals, then broadens the belief.

Between experiments: after each update we take the posterior MAP hypothesis and
compute, for every logged experiment, the mean squared normalized residual z (about
1 when one hypothesis explains everything). If the physics changed, no single
(g, mu) fits both old and new data, so the worst z grows. A change is flagged only
after `persistence` consecutive experiments exceed the threshold.

Within an experiment (within=True): a CUSUM over the latest experiment's per-frame
residuals at the MAP. Each frame is capped at `frame_cap`, so one bad frame (an
impact) cannot trigger it; only a sustained run of misfit can. The run's first
frame estimates when the change happened. If that is at the very start, the change
came before the experiment and the experiment is kept; otherwise it is contaminated
and discarded, because the predictors can only simulate from the setup at t = 0.
drift and h were chosen on calibration seeds (calibrate_cusum.py), separate from
the seeds the stress test reports.
"""
from __future__ import annotations

import numpy as np

from .belief import GridBelief


class ChangeDetector:
    def __init__(self, threshold: float = 1.6, persistence: int = 2, within: bool = False,
                 drift: float = 3.0, h: float = 8.3, frame_cap: float = 9.0, start_frames: int = 2):
        self.threshold = threshold
        self.persistence = persistence
        self.within = within
        self.drift, self.h, self.frame_cap, self.start_frames = drift, h, frame_cap, start_frames
        self.streak = 0
        self.z_history: list[float] = []
        self.last_change_frame: int | None = None

    def statistic(self, belief: GridBelief) -> float:
        g, mu = belief.map
        return max(belief.residual_z(spec, obs, g, mu) for spec, obs in belief.history)

    def change_point(self, belief: GridBelief) -> int | None:
        """Frame where the residual level steps up in the latest experiment, or None.

        The calibrated CUSUM decides whether there is a sustained misfit; the frame
        is then the standard mean-shift estimate (where the cumulative sum of
        z - mean(z) bottoms out), which a high CUSUM drift would place too late."""
        spec, obs = belief.history[-1]
        g, mu = belief.map
        z = np.minimum(belief.residual_frames(spec, obs, g, mu), self.frame_cap)
        s = 0.0
        for zt in z:
            s = max(0.0, s + zt - 1.0 - self.drift)
            if s > self.h:
                return int(np.argmin(np.cumsum(z - z.mean()))) + 1
        return None

    def check(self, belief: GridBelief) -> bool:
        """Call after belief.update. Returns True (and resets the belief) on a flag."""
        z = self.statistic(belief)
        self.z_history.append(z)
        self.last_change_frame = None
        if self.within:
            cp = self.change_point(belief)
            if cp is not None:
                self.last_change_frame = cp
                belief.reset(keep_last=1 if cp <= self.start_frames else 0)
                self.streak = 0
                return True
        self.streak = self.streak + 1 if z > self.threshold else 0
        if self.streak >= self.persistence:
            belief.reset(keep_last=self.streak)
            self.streak = 0
            return True
        return False
