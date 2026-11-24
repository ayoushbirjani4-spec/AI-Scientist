"""Streamlit dashboard for the hidden-physics agent.

    streamlit run dashboard.py

Views
  Run viewer   - pick a logged run; step through the world, predicted vs observed
                 trajectories, the (g, mu) posterior and why each experiment was chosen.
  New run      - start a run (optionally with a physics change) and watch it live.
  Results      - the strategy comparison from results/ (evaluation only).
Ground truth is hidden unless "Evaluation view" is switched on in the sidebar.
Explanations are generated only from logged quantities.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from hidden_physics.adapter import NOISE_LEVELS
from hidden_physics.agent import run_episode
from hidden_physics.change import ChangeDetector
from hidden_physics.env import OBJECTS, RAMP_LENGTH, ExperimentSpec, HiddenPhysics, World, simulate
from hidden_physics.evaluate import draw_truth
from hidden_physics.planner import STRATEGIES
from hidden_physics.store import ExperimentStore, load_run

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
# Prefer the g-scaled ensemble; fall back to the first (g-input) one.
GNN_PATH = next((p for p in (ROOT / "models" / "gnn_gscaled.pt", ROOT / "models" / "gnn_ensemble.pt")
                 if p.exists()), ROOT / "models" / "gnn_gscaled.pt")

PRED = "#2a78d6"     # predicted (model) - categorical slot 1
OBS = "#eb6834"      # observed (noisy)  - categorical slot 2
TRUTH = "#8a8980"    # ground truth, evaluation only - neutral ink

st.set_page_config(page_title="Hidden Physics", layout="wide")


# ------------------------------------------------------------------ helpers
def list_runs() -> list[str]:
    if not RUNS.exists():
        return []
    runs = [p for p in RUNS.iterdir() if (p / "steps.jsonl").exists()]
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]


@st.cache_resource(show_spinner="Loading GNN ensemble…")
def load_gnn():
    from hidden_physics.learned import LearnedPredictor
    return LearnedPredictor(GNN_PATH)


@st.cache_data(show_spinner=False)
def cached_run(name: str, mtime: float) -> dict:
    return load_run(RUNS / name)


def spec_from_record(rec: dict) -> ExperimentSpec:
    d = dict(rec["spec"])
    return ExperimentSpec.make(d.pop("kind"), d.pop("obj"), **d)


def truth_at(truth: list[dict] | None, step: int) -> dict | None:
    if not truth:
        return None
    cur = None
    for t in truth:
        if t["from_step"] <= step:
            cur = t
    return cur


def object_outline(obj: str, x: float, y: float, angle: float) -> tuple[list, list]:
    o = OBJECTS[obj]
    if o["shape"] == "circle":
        a = np.linspace(0, 2 * np.pi, 40)
        return list(x + o["radius"] * np.cos(a)), list(y + o["radius"] * np.sin(a))
    w, h = o["size"]
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]])
    c, s = math.cos(angle), math.sin(angle)
    rot = pts @ np.array([[c, s], [-s, c]])
    return list(x + rot[:, 0]), list(y + rot[:, 1])


# ------------------------------------------------------------------ figures
def world_figure(spec: ExperimentSpec, traj: dict, truth: dict | None) -> go.Figure:
    obs, pred = traj["observed"], traj["predicted"]
    fig = go.Figure()
    xs = np.concatenate([obs[:, 0], pred[:, 0]])
    lo, hi = min(xs.min(), -0.5) - 0.3, max(xs.max(), 0.5) + 0.3
    if spec.kind == "ramp_slide":
        th = math.radians(spec.p["angle_deg"])
        top = (-RAMP_LENGTH * math.cos(th), RAMP_LENGTH * math.sin(th))
        fig.add_trace(go.Scatter(x=[top[0], 0, top[0], top[0]], y=[top[1], 0, 0, top[1]], fill="toself",
                                 fillcolor="rgba(140,140,130,0.18)", line=dict(color=TRUTH, width=1),
                                 hoverinfo="skip", showlegend=False))
        lo = min(lo, top[0] - 0.2)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines", line=dict(color=TRUTH, width=2),
                             hoverinfo="skip", showlegend=False))
    # start pose (known setup) and predicted end pose
    _, s0 = simulate(spec, 9.81, 0.3)  # only for the initial pose, which does not depend on g, mu
    ox, oy = object_outline(spec.obj, s0[0, 0], s0[0, 1], s0[0, 2])
    fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color=TRUTH, width=1.5, dash="dot"),
                             name="start pose", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pred[:, 0], y=pred[:, 1], mode="lines", line=dict(color=PRED, width=2),
                             name="predicted path (posterior mean)",
                             hovertemplate="x %{x:.3f} m<br>y %{y:.3f} m<extra>predicted</extra>"))
    fig.add_trace(go.Scatter(x=obs[:, 0], y=obs[:, 1], mode="markers",
                             marker=dict(color=OBS, size=8, line=dict(color="white", width=1)),
                             name="observed (noisy)", customdata=traj["times"],
                             hovertemplate="t %{customdata:.3f} s<br>x %{x:.3f} m<br>y %{y:.3f} m<extra>observed</extra>"))
    if truth:
        _, st_ = simulate(spec, truth["g"], truth["mu"])
        fig.add_trace(go.Scatter(x=st_[:, 0], y=st_[:, 1], mode="lines", line=dict(color=TRUTH, width=2, dash="dash"),
                                 name="true path (eval only)"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                      legend=dict(orientation="h", y=-0.15), xaxis_title="x (m)", yaxis_title="y (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def trajectory_figure(spec: ExperimentSpec, traj: dict, truth: dict | None) -> go.Figure:
    t, obs, pred = traj["times"], traj["observed"], traj["predicted"]
    names = ["x (m)", "y (m)", "vx (m/s)", "vy (m/s)"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=names, vertical_spacing=0.16, horizontal_spacing=0.08)
    st_true = simulate(spec, truth["g"], truth["mu"])[1][:, [0, 1, 3, 4]] if truth else None
    for i in range(4):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(x=t, y=pred[:, i], mode="lines", line=dict(color=PRED, width=2),
                                 name="predicted", legendgroup="p", showlegend=i == 0,
                                 hovertemplate="t %{x:.3f} s: %{y:.3f}<extra>predicted</extra>"), r, c)
        fig.add_trace(go.Scatter(x=t, y=obs[:, i], mode="markers", marker=dict(color=OBS, size=6),
                                 name="observed", legendgroup="o", showlegend=i == 0,
                                 hovertemplate="t %{x:.3f} s: %{y:.3f}<extra>observed</extra>"), r, c)
        if st_true is not None:
            fig.add_trace(go.Scatter(x=t, y=st_true[:, i], mode="lines", line=dict(color=TRUTH, width=1.5, dash="dash"),
                                     name="true (eval only)", legendgroup="t", showlegend=i == 0), r, c)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.12))
    fig.update_xaxes(title_text="t (s)", row=2)
    return fig


def estimate_figure(steps: list[dict], key: str, label: str, truth: list[dict] | None,
                    current: int) -> go.Figure:
    x = [s["step"] + 1 for s in steps]
    m = [s[f"{key}_mean"] for s in steps]
    lo = [s[f"{key}_ci"][0] for s in steps]
    hi = [s[f"{key}_ci"][1] for s in steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=hi + lo[::-1], mode="lines", fill="toself",
                             fillcolor="rgba(42,120,214,0.15)",
                             line=dict(width=0), hoverinfo="skip", name="90% credible interval"))
    fig.add_trace(go.Scatter(x=x, y=m, mode="lines+markers", line=dict(color=PRED, width=2), marker=dict(size=8),
                             name="posterior mean", customdata=np.c_[lo, hi],
                             hovertemplate="after exp %{x}: %{y:.4f}<br>90% CI %{customdata[0]:.4f} – %{customdata[1]:.4f}<extra></extra>"))
    flags = [s["step"] + 1 for s in steps if s.get("change_flag")]
    for f in flags:
        fig.add_vline(x=f, line=dict(color=OBS, width=1.5, dash="dot"))
    if truth:
        ty = [truth_at(truth, s["step"])[key] for s in steps]
        fig.add_trace(go.Scatter(x=x, y=ty, mode="lines", line=dict(color=TRUTH, width=1.5, dash="dash", shape="hv"),
                                 name="true value (eval only)"))
    fig.add_vline(x=current + 1, line=dict(color=TRUTH, width=1))
    # The first experiment's interval is often prior-wide; zoom to the rest so they stay readable.
    k = 1 if len(steps) > 2 else 0
    vals = lo[k:] + hi[k:] + m[k:] + ([truth_at(truth, s["step"])[key] for s in steps[k:]] if truth else [])
    pad = 0.15 * (max(vals) - min(vals) or abs(max(vals)) * 0.01 or 1e-3)
    fig.update_yaxes(range=[min(vals) - pad, max(vals) + pad])
    if k:
        fig.add_annotation(x=1, y=1, xref="x", yref="paper", text="exp 1 clipped", showarrow=False,
                           font=dict(size=10, color=TRUTH), yanchor="bottom")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title=dict(text=label, x=0, font=dict(size=14)),
                      xaxis_title="experiment", hovermode="closest", legend=dict(orientation="h", y=-0.3),
                      xaxis=dict(dtick=1))
    return fig


def posterior_figure(traj: dict, truth: dict | None) -> go.Figure | None:
    if "weights" not in traj:
        return None
    fig = go.Figure(go.Heatmap(x=traj["mu_axis"], y=traj["g_axis"], z=traj["weights"],
                               colorscale=[[0, "rgba(42,120,214,0.02)"], [1, PRED]], showscale=False,
                               hovertemplate="μ %{x:.4f}<br>g %{y:.3f}<br>weight %{z:.3f}<extra></extra>"))
    if truth and traj["mu_axis"][0] <= truth["mu"] <= traj["mu_axis"][-1] and \
            traj["g_axis"][0] <= truth["g"] <= traj["g_axis"][-1]:
        fig.add_trace(go.Scatter(x=[truth["mu"]], y=[truth["g"]], mode="markers",
                                 marker=dict(symbol="x", size=12, color=OBS), name="truth (eval only)"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="μ", yaxis_title="g (m/s²)",
                      title=dict(text="Posterior over the current grid", x=0, font=dict(size=14)), showlegend=False)
    return fig


def explain(steps: list[dict], i: int, strategy: str) -> str:
    rec = steps[i]
    lines = [f"**Experiment {i + 1}: `{rec['spec_str']}`** (cost {rec['cost']:.1f})"]
    if strategy == "active" and "chosen_eig" in rec:
        lines.append(f"Chosen because its expected information gain was {rec['chosen_eig']:.2f} nats, "
                     f"the best score after subtracting 0.15 × cost.")
    elif strategy == "random":
        lines.append("Picked at random from the experiment library.")
    elif strategy == "scripted":
        lines.append("Next item in the fixed script.")
    else:
        lines.append("Passive: the agent only watched an event the world produced.")
    if i > 0:
        p = steps[i - 1]
        lines.append(f"g uncertainty (sd) {p['g_sd']:.4f} → {rec['g_sd']:.4f}; "
                     f"μ uncertainty {p['mu_sd']:.4f} → {rec['mu_sd']:.4f}; "
                     f"entropy fell by {p['entropy'] - rec['entropy']:.2f} nats.")
    else:
        lines.append(f"After it: g sd {rec['g_sd']:.4f}, μ sd {rec['mu_sd']:.4f}.")
    if rec.get("temperature", 1) > 1.01:
        lines.append(f"The likelihood was too sharp for the grid, so it was tempered (T = {rec['temperature']:.3g}); "
                     "intervals are deliberately wider than the data alone imply.")
    if rec.get("z") is not None:
        lines.append(f"Worst prediction residual z = {rec['z']:.2f} (about 1 means the model fits; flag threshold 1.6).")
    if rec.get("change_flag"):
        lines.append("**Physics change suspected:** residuals stayed high for 2 experiments, "
                     "so the belief was reset to the prior, keeping only the recent experiments.")
    return "  \n".join(lines)


# ------------------------------------------------------------------ views
def run_viewer(eval_view: bool):
    runs = list_runs()
    if not runs:
        st.info("No logged runs yet. Start one under **New run**.")
        return
    default = runs.index(st.session_state["select_run"]) if st.session_state.get("select_run") in runs else 0
    name = st.sidebar.selectbox("Run", runs, index=default)
    run = cached_run(name, (RUNS / name / "steps.jsonl").stat().st_mtime)
    cfg, steps, trajs = run["config"], run["steps"], run["trajectories"]
    truth = run["truth"] if eval_view else None
    if not steps:
        st.warning("This run has no steps logged.")
        return

    st.subheader(f"{cfg.get('strategy', '?')} strategy · {cfg.get('noise', '?')} noise · seed {cfg.get('seed', '?')}")
    st.caption(f"World model: {cfg.get('world_model', 'Stage A · simulator')}")
    i = st.slider("Experiment", 1, len(steps), len(steps)) - 1
    rec, traj = steps[i], trajs[i]
    spec = spec_from_record(rec)
    tr = truth_at(truth, i)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("g estimate (m/s²)", f"{rec['g_mean']:.3f}", help="Posterior mean after this experiment")
        st.caption(f"90% interval {rec['g_ci'][0]:.3f} – {rec['g_ci'][1]:.3f}")
    with c2:
        st.metric("μ estimate", f"{rec['mu_mean']:.4f}")
        st.caption(f"90% interval {rec['mu_ci'][0]:.4f} – {rec['mu_ci'][1]:.4f}")
    with c3:
        st.metric("Total cost so far", f"{sum(s['cost'] for s in steps[:i + 1]):.1f}")
        st.caption(f"experiment {i + 1} of {len(steps)}")
    if tr:
        st.caption(f"Evaluation view · true g = {tr['g']:.3f}, μ = {tr['mu']:.4f} · "
                   f"error {abs(rec['g_mean'] - tr['g']) / tr['g']:.3%} (g), {abs(rec['mu_mean'] - tr['mu']):.4f} (μ)")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Simulated world")
        st.plotly_chart(world_figure(spec, traj, tr), width="stretch")
    with right:
        st.markdown("##### Why this experiment")
        st.markdown(explain(steps, i, cfg.get("strategy", "")))
        if rec.get("alternatives"):
            st.dataframe(pd.DataFrame(rec["alternatives"]).rename(columns={"spec": "candidate", "eig": "info gain"})
                         .style.format({"info gain": "{:.2f}", "cost": "{:.1f}", "score": "{:.2f}"}),
                         hide_index=True, width="stretch")

    st.markdown("##### Predicted vs observed")
    st.plotly_chart(trajectory_figure(spec, traj, tr), width="stretch")

    a, b, c = st.columns(3)
    a.plotly_chart(estimate_figure(steps, "g", "g (m/s²)", truth, i), width="stretch")
    b.plotly_chart(estimate_figure(steps, "mu", "μ", truth, i), width="stretch")
    pf = posterior_figure(traj, tr)
    if pf:
        c.plotly_chart(pf, width="stretch")
    else:
        c.info("This run was logged before posterior grids were stored.")

    with st.expander("All experiments in this run"):
        st.dataframe(pd.DataFrame([{
            "#": s["step"] + 1, "experiment": s["spec_str"], "cost": s["cost"],
            "g": s["g_mean"], "g sd": s["g_sd"], "μ": s["mu_mean"], "μ sd": s["mu_sd"],
            "entropy": s["entropy"], "change flag": s["change_flag"], "plan s": s["plan_time"]}
            for s in steps]), hide_index=True, width="stretch")


def new_run():
    st.subheader("Start a new run")
    with st.form("new_run"):
        c1, c2, c3, c4 = st.columns(4)
        strategy = c1.selectbox("Strategy", list(STRATEGIES))
        noise_name = c2.selectbox("Noise", list(NOISE_LEVELS), index=1)
        budget = c3.number_input("Experiments", 2, 20, 8)
        seed = c4.number_input("Seed (sets the hidden physics)", 0, 10_000, 0)
        c5, c6, c7 = st.columns(3)
        change_at = c5.number_input("Change physics before experiment (0 = never)", 0, 20, 0)
        models = ["Stage A · simulator"] + (["Stage B · GNN ensemble"] if GNN_PATH.exists() else [])
        world_model = c6.selectbox("Agent's world model", models,
                                   help="Stage B appears once models/gnn_ensemble.pt is trained")
        use_detector = c7.checkbox("Use change detector", value=True)
        go_ = st.form_submit_button("Run", type="primary")
    st.caption("The agent never sees the hidden physics; it is drawn from the seed and saved "
               "to an evaluation-only file. Each experiment takes about 1–3 s.")
    if not go_:
        return

    noise = NOISE_LEVELS[noise_name]
    truth = draw_truth(int(seed))
    after = draw_truth(int(seed) + 5000)
    run_id = f"{strategy}_{noise_name}_seed{seed}_{time.strftime('%H%M%S')}"
    store = ExperimentStore(RUNS, run_id, {"strategy": strategy, "noise": noise_name, "noise_pos": noise.pos,
                                           "noise_vel": noise.vel, "seed": int(seed), "budget": int(budget),
                                           "change_at": int(change_at), "world_model": world_model})
    predictor = load_gnn() if world_model.startswith("Stage B") else None
    world = World(truth, noise.pos, noise.vel, seed=int(seed))
    truth_log = [{"from_step": 0, "g": truth.g, "mu": truth.mu}]

    def hook(step):  # evaluation-only: the agent is not told
        if change_at and step == change_at - 1:
            world._set_physics(after)
            truth_log.append({"from_step": step, "g": after.g, "mu": after.mu})

    status = st.status("Running…", expanded=True)
    bar = st.progress(0.0)

    def on_step(rec):
        status.write(f"{rec['step'] + 1}. `{rec['spec_str']}` → g = {rec['g_mean']:.3f} ± {rec['g_sd']:.3f}, "
                     f"μ = {rec['mu_mean']:.4f} ± {rec['mu_sd']:.4f}"
                     + ("  **· change flagged**" if rec["change_flag"] else ""))
        bar.progress((rec["step"] + 1) / budget)

    run_episode(world, STRATEGIES[strategy](), noise, int(budget), seed=int(seed),
                detector=ChangeDetector() if use_detector else None, store=store,
                before_step=hook, on_step=on_step, predictor=predictor)
    store.write_truth(truth_log)
    status.update(label="Done", state="complete")
    st.session_state["select_run"] = run_id
    st.success("Saved. Open **Run viewer** to step through it.")


def results_view():
    st.subheader("Strategy comparison (evaluation)")
    if not (RESULTS / "metrics.csv").exists():
        st.info("No results yet. Run `python -m hidden_physics.evaluate`.")
        return
    df = pd.read_csv(RESULTS / "metrics.csv")
    budget = len(json.loads(df["g_rel_err_curve"].iloc[0]))
    df["exps"] = df["steps_to_threshold"].fillna(budget + 1)
    st.caption(f"{df['seed'].nunique()} seeds per cell · threshold |g error| ≤ 1 % and |μ error| ≤ 0.01, "
               f"held to the end of the {budget}-experiment budget · never reached counts as {budget + 1}")
    order = ["active", "random", "scripted", "passive"]
    colors = {"active": "#2a78d6", "random": "#eb6834", "scripted": "#1baf7a", "passive": "#eda100"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=["low noise", "medium noise", "high noise"], shared_yaxes=True)
    for j, n in enumerate(["low", "medium", "high"]):
        for s in order:
            d = df[(df.noise == n) & (df.strategy == s)]
            fig.add_trace(go.Box(y=d["exps"], name=s, marker_color=colors[s], boxmean=True, boxpoints="all",
                                 jitter=0.4, pointpos=0, showlegend=False, customdata=d["seed"],
                                 hovertemplate="seed %{customdata}: %{y} experiments<extra>" + s + "</extra>"),
                          1, j + 1)
    fig.update_yaxes(title_text="experiments needed", row=1, col=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    tab = (df.groupby(["noise", "strategy"])
           .agg(mean_exps=("exps", "mean"), reached=("reached", "mean"), g_err=("final_g_rel_err", "median"),
                mu_err=("final_mu_abs_err", "median"), cov_g=("ci_coverage_g", "mean"),
                cov_mu=("ci_coverage_mu", "mean"), plan_s=("mean_plan_latency_s", "mean"))
           .reindex(pd.MultiIndex.from_product([["low", "medium", "high"], order])).reset_index()
           .rename(columns={"level_0": "noise", "level_1": "strategy"}))
    st.dataframe(tab.style.format({"mean_exps": "{:.2f}", "reached": "{:.0%}", "g_err": "{:.4%}",
                                   "mu_err": "{:.4f}", "cov_g": "{:.0%}", "cov_mu": "{:.0%}", "plan_s": "{:.2f}"}),
                 hide_index=True, width="stretch")
    if (RESULTS / "error_curves.png").exists():
        st.image(str(RESULTS / "error_curves.png"), width="stretch")
    cd = RESULTS / "change_detection.csv"
    if cd.exists():
        st.markdown("##### Change detection")
        c = pd.read_csv(cd)
        ch = c[c.with_change]
        summ = ch.groupby(["noise", "strategy"]).agg(detected=("detected", "mean"),
                                                     delay=("detection_delay", "mean"),
                                                     recovery=("recovery_steps", "mean")).reset_index()
        fa = c.groupby(["noise", "strategy"])["false_alarms"].apply(lambda s: (s > 0).mean()).reset_index(name="false_alarm_runs")
        st.dataframe(summ.merge(fa).style.format({"detected": "{:.0%}", "delay": "{:.2f}", "recovery": "{:.2f}",
                                                  "false_alarm_runs": "{:.0%}"}),
                     hide_index=True, width="stretch")


# ------------------------------------------------------------------ layout
st.sidebar.title("Hidden Physics")
eval_view = st.sidebar.toggle("Evaluation view (show ground truth)", value=False)
pages = ["Run viewer", "New run"] + (["Results"] if eval_view else [])
page = st.sidebar.radio("View", pages)
if not eval_view:
    st.sidebar.caption("Results are evaluation-only; switch on the evaluation view to see them.")

if page == "Run viewer":
    run_viewer(eval_view)
elif page == "New run":
    new_run()
else:
    results_view()
