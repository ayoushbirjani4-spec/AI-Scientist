# Hidden Physics: active experiment design for g and μ

An agent in a 2D Pymunk world infers hidden gravity `g` and object friction `μ`,
choosing experiments that reduce its uncertainty, and detects when physics changes
between experiments. It estimates parameters inside a known model; it does not
discover laws.

**Research question:** does active experiment selection need fewer experiments to
identify `(g, μ)` than random selection, a fixed script, or passive observation?
Tested at low, medium and high observation noise. Results: `results/summary.md`.

Built: all 7 milestones. That covers the Must tier, the change detector, the
Streamlit dashboard, and a GNN world model ensemble (Stage B, two versions)
evaluated against the simulator predictor. Also built: a stress test for physics
changes inside an experiment. Results: `results/gnn_gscaled/summary.md`,
`results/stress/summary.md`.

Requires about 1 GB of free RAM for 3 workers. The full sweep takes about an hour
on a laptop.

## Run

```
pip install pymunk numpy matplotlib streamlit plotly torch
python tests/test_core.py                       # or: python -m pytest tests
streamlit run dashboard.py                      # dashboard at http://localhost:8501
python -m hidden_physics.train_gnn              # Stage B v2 (g-scaled): data + 5-member ensemble (~30 min on CPU)
python -m hidden_physics.evaluate_gnn --model models/gnn_gscaled.pt --out results/gnn_gscaled --workers 1
python -m hidden_physics.calibrate_cusum --workers 1   # choose within-experiment detector settings (seeds 100+)
python -m hidden_physics.evaluate_stress --workers 1   # stress test: physics changes inside an experiment
python -m hidden_physics.evaluate --seeds 20 --budget 10 --change-seeds 8 --workers 3
```

Outputs: `results/metrics.csv`, `results/change_detection.csv`,
`results/experiments_to_threshold.png`, `results/error_curves.png`,
`results/summary.md`, and one fully logged run in `runs/`.

## Dashboard

`streamlit run dashboard.py` has three views:

- **Run viewer.** Pick a logged run and step through it. For each experiment it
  shows the simulated world (setup, observed noisy path, predicted path under the
  posterior mean), predicted vs observed x, y, vx, vy, the g and μ estimates with
  90 % intervals over the run, the posterior over the current grid, and why the
  experiment was chosen (top-5 candidates with info gain, cost and score for the
  active planner). Every explanation comes from logged quantities.
- **New run.** Choose strategy, noise, budget and seed, optionally change physics
  before a given experiment, and watch the run stream in. It is saved to `runs/`.
- **Results.** The strategy comparison from `results/`.

Ground truth (true paths, true values, errors, and the Results view) appears only
with **Evaluation view** switched on in the sidebar.

## Stage B: GNN world model

`learned.py` is a parameter-conditioned graph network:
- **Nodes:** the object, plus each static surface (floor, and the ramp if present).
- **Edges:** surface → object, with pose and velocity expressed in the surface
  frame, plus the shape's support distance from known geometry.
- **Globals:** g, μ and μ·g.
- **Output:** the change in velocity and spin over one 1/30 s step, plus a small
  position correction, integrated with the trapezoid rule.

It never calls the simulator. It starts from the experiment setup
(`env.scene_setup`) and rolls forward on its own.

Training data is 4,000 trajectories, split 80/10/10 by trajectory, so by scene and
parameter draw, never by adjacent frames. Training uses a one-step loss, then a
10-step rollout loss. The ensemble is 5 independently seeded members. A per-frame
variance floor is calibrated on validation data so that ensemble spread plus floor
matches the observed error. `GridBelief` adds this predictive variance to its
noise model automatically when the predictor provides it.

Two versions were trained, on the same data split and with the same evaluation.

**v1, g as an input** (`models/gnn_ensemble.pt`, `results/gnn/summary.md`). Rollouts
looked good: median end-of-rollout error 0.8–1.9 cm, 9 cm on ramps. Inference
through it was poor: only 1–3 of 10 runs reached the threshold. The diagnosis was
that accuracy is not identifiability. Fitting g to noise-free trajectories through
the model was biased by 3.4 % median and up to 44 % on ramps. Inference needs the
prediction's dependence on g to be right, not only the prediction itself.

**v2, g-scaled** (`models/gnn_gscaled.pt`, `results/gnn_gscaled/summary.md`,
`python -m hidden_physics.train_gnn`). This applies dimensional analysis: the
network never sees g, and it predicts velocity change in units of one gravity step.

  Δv = g·Δt · F(v / (g·Δt), gap / (g·Δt²), geometry, μ)

Free fall is F = (0, −1), sliding is F = −μ, a ramp is sin θ − μ cos θ, and an impact
is linear in the scaled velocity. The dependence on g is exact by construction
instead of learned. One more change was needed. Impact steps reach F ≈ 50 against
about 1 otherwise, which swamped the precision needed elsewhere, so training uses a
Huber loss (β = 0.5 std). A median/MAD normalisation was tried first and made things
worse.

| | v1 (g input) | v2 (g-scaled) | Stage A |
|---|---|---|---|
| median final position error, drop / slide / impulse / ramp (cm) | 0.8 / 1.6 / 1.9 / 9.3 | **0.3 / 0.4 / 0.7 / 2.0** | 0 |
| one-step Δv error, free flight / contact (m/s) | 0.0095 / 0.0085 | **0.0017 / 0.0043** | 0 |
| g bias when fitted through the model, median / worst | 3.4 % / 44 % | **0.6 % / 2.5 %** | 0 |
| runs reaching threshold (medium/high noise × active/random) | 3, 2, 1, 2 of 10 | **10, 8, 10, 9 of 10** | 10 of 10 |
| mean experiments to threshold, active, medium noise | 8.8 | **2.3** | 1.7 |
| 90 % credible-interval coverage g / μ | 0.08–0.43 | 0.19–0.72 | 0.88–0.97 |

What this shows:
- With v2 as the world model, the agent identifies g and μ in most runs, and active
  selection still beats random (2.3 vs 5.1 experiments at medium noise). The
  simulator is still better.
- **Open issue: overconfidence.** Credible intervals are still too narrow. The
  remaining ~0.5 % g bias is systematic, shared across frames and experiments, and
  the likelihood treats model error as independent per frame. Calibrating the
  predictive variance on whole-trajectory, parameter-fit error would address this.
- The model's own 90 % bands cover 91–99 % of held-out points, so per-prediction
  calibration is fine. The problem is error that is correlated over a trajectory.
- **Speed.** The 5 members run at once with `torch.func.vmap`, at about the same
  speed as Pymunk for 400 hypotheses.

## Stress test: physics changes inside an experiment

The world can switch g and μ partway through an experiment
(`World._schedule_change`, evaluation only). The detector gained a within-experiment
mode (`ChangeDetector(within=True)`):
- **Detection:** a CUSUM over the latest experiment's per-frame residuals at the MAP.
  Frames are capped so that one bad frame can never trigger it.
- **Change time:** where the residual level steps up, i.e. the standard step-change
  estimate.
- **Contaminated experiment:** discarded, because both predictors simulate from the
  setup at t = 0 and can't restart from a mid-trajectory state.

The CUSUM settings (drift 3, h 8.3) were chosen on calibration seeds 100+
(`results/stress/cusum_calibration.md`). The stress test reports seeds 0–5. In each
change run, g and μ are re-drawn 0.15–0.65 s into experiment 5. It uses the active
strategy and the Stage A predictor (`results/stress/summary.md`):

| noise | detector | mean delay (exps) | flagged on the contaminated exp | change-time error median | false-alarm runs | mean recovery (exps) |
|---|---|---|---|---|---|---|
| low | between only | 2.17 | 0/6 | – | 0/12 | 2.67 |
| low | + within | **1.00** | **6/6** | 0.07 s | **4/12** | 2.00 |
| medium | between only | 2.17 | 0/6 | – | 0/12 | 2.83 |
| medium | + within | **1.17** | 5/6 | 0.16 s | 0/12 | 2.00 |
| high | between only | 2.83 | 0/6 | – | 0/12 | 2.67 |
| high | + within | **1.83** | 1/6 | 0.15 s | 0/12 | 2.83 |

The within detector finds changes about one experiment sooner at every noise level.
At high noise it rarely catches the change inside the experiment itself.

**Open issue: low-noise false alarms (4/12).** Calibration cut them from 6/12 but
did not remove them, and I did not retune on the test seeds. The cause is Stage A,
not the detector. Pymunk stops a sliding block at a whole substep, so the rest
position jumps slightly with μ·g. At 2 mm noise that leaves a lasting offset of a
few millimetres, which looks like a change. The principled fix is a larger
predictor-error term for stopping events at low noise.

## Layout

| File | Role |
|---|---|
| `env.py` | Pymunk world, experiment specs, `World` (holds hidden physics privately) |
| `adapter.py` | normalization, object features, transition tuples, noise levels, likelihood σ |
| `predictor.py` | Stage A: isolated simulator copy, parameters passed explicitly |
| `belief.py` | adaptive grid posterior over (g, μ) with refine/re-score and reset |
| `planner.py` | active planner (info-gain lower bound − cost) and the three baselines |
| `change.py` | residual-based change detector with persistence |
| `agent.py` | main loop |
| `store.py` | run config, per-step JSONL, trajectories (npz), eval-only truth file |
| `evaluate.py`, `report.py` | sweep, metrics, plots, summary (the only code that sees truth) |
| `dashboard.py` | Streamlit dashboard (truth shown only in its evaluation view) |
| `learned.py` | Stage B GNN, dataset generation, training, `LearnedPredictor` |
| `train_gnn.py`, `evaluate_gnn.py` | train the ensemble; evaluate it against Stage A |
| `calibrate_cusum.py`, `evaluate_stress.py` | choose within-experiment detector settings; mid-experiment change stress test |

## Design decisions worth knowing

- **Friction (risk 3).** Floor and ramp friction fixed at 1.0, so the object's
  friction is the single hidden μ (Pymunk multiplies the pair). Damping fixed at
  1.0 (none). `collision_slop` lowered to 1 mm, since Pymunk's default (0.1) assumes
  pixel units. Tests check flat-slide deceleration = μg and ramp acceleration
  = g(sin θ − μ cos θ) against the analytic values.
- **Ball is a 2D disk**, so it rolls at 2/3·v0 and its slide barely reveals μ. The
  planner has to learn not to use it for friction.
- **Ball is inelastic (elasticity 0).** With a bouncing ball, Pymunk's contact
  timing snaps to the 1/240 s step, so the simulated trajectory jumps as g changes
  and the likelihood in g was rough and lopsided (χ² 73 vs 1094 at ±0.05). That gave
  a g error floor of about 0.2 % at every noise level and 90 % intervals that missed the
  truth. Without the bounce the roughness drops about 15×.
- **Continuous settings and cost (risk 1).** Each experiment has continuous settings
  and a cost = observation duration + setup cost (ramps cost the most). Active score
  = information-gain estimate − 0.15 × cost.
- **Information gain** uses the pairwise Bhattacharyya lower bound on I(θ; y)
  (Kolchinsky & Tracey 2017) over 32 hypotheses drawn from the posterior: cheap,
  deterministic, and capped at log K, so very broad beliefs saturate it.
- **Belief.** 20×20 grid, Student-t likelihood (ν = 4) so a single contact frame
  can't dominate, σ = measurement noise ⊕ a predictor-error term (risk 2).
  *Tempering:* if fewer than 4 effective cells carry weight, the log-likelihood is
  divided by T until they do. Without this, a single slide at low noise (a thin
  μ·g ridge) snapped onto whichever coarse cell sat nearest the ridge, and the zoom
  then cut the truth out of the grid. *Refinement:* zoom to the tempered mean ± 3 sd,
  re-score all logged experiments, repeat while the box shrinks by ≥ 15 %. T falls
  back to 1 as resolution improves.
- **Passive** = the world's own events (things falling or sliding to rest): drops
  and flat slides with random objects and settings, no ramps, no designed impulses,
  no setup cost.
- **Threshold** for "identified": |g−g*|/g* ≤ 1 % and |μ−μ*| ≤ 0.01, and it must
  hold for every remaining experiment in the budget.
- **Change detector.** After each update, the worst normalized residual (mean r²)
  over all logged experiments at the MAP. Above 1.6 for 2 experiments in a row →
  reset the belief to the prior, keeping the suspect experiments.
