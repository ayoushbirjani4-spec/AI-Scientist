"""Turn results/*.csv into plots and summary.md (every noise level is reported)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STRATS = ["active", "random", "scripted", "passive"]
NOISES = ["low", "medium", "high"]
COLORS = {"active": "#2a78d6", "random": "#eb6834", "scripted": "#1baf7a", "passive": "#eda100"}
INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6a63", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 11,
    "text.color": INK,
})


def _read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _censored_steps(rows, budget):
    """Steps to threshold; runs that never reached it count as budget + 1."""
    return np.array([float(r["steps_to_threshold"]) if r["steps_to_threshold"] else budget + 1
                     for r in rows])


def _boot_ci(x, rng, n=4000):
    m = rng.choice(x, size=(n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def make_report(out: Path):
    out = Path(out)
    rows = _read(out / "metrics.csv")
    budget = len(json.loads(rows[0]["g_rel_err_curve"]))
    rng = np.random.default_rng(0)
    by = {(n, s): sorted([r for r in rows if r["noise"] == n and r["strategy"] == s],
                         key=lambda r: int(r["seed"])) for n in NOISES for s in STRATS}
    n_seeds = len(by[("low", "active")])

    # ---------- Figure 1: experiments needed, per strategy, per noise level
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, n in zip(axes, NOISES):
        for i, s in enumerate(STRATS):
            y = _censored_steps(by[(n, s)], budget)
            jit = rng.uniform(-0.18, 0.18, len(y))
            ax.scatter(i + jit, y + rng.uniform(-0.15, 0.15, len(y)), s=14, color=COLORS[s],
                       alpha=0.45, linewidths=0)
            lo, hi = _boot_ci(y, rng)
            ax.errorbar(i + 0.32, y.mean(), yerr=[[y.mean() - lo], [hi - y.mean()]], fmt="o",
                        color=INK, ms=6, lw=1.6, capsize=3)
        ax.axhline(budget + 0.5, color=AXIS, lw=1, ls="--")
        ax.set_xticks(range(4), STRATS)
        ax.set_title(f"{n} noise", loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("experiments until estimate stays within tolerance")
    axes[0].set_yticks(list(range(1, budget + 1)) + [budget + 1],
                       [str(i) for i in range(1, budget + 1)] + ["not\nreached"])
    fig.suptitle(f"Experiments needed to identify g (±1%) and μ (±0.01) — {n_seeds} seeds per cell; "
                 "dots = runs, black = mean with 95% bootstrap CI", x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "experiments_to_threshold.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 2: error curves (median across seeds)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    x = np.arange(1, budget + 1)
    for j, n in enumerate(NOISES):
        for s in STRATS:
            for i, (key, tol) in enumerate([("g_rel_err_curve", 0.01), ("mu_abs_err_curve", 0.01)]):
                c = np.array([json.loads(r[key]) for r in by[(n, s)]])
                axes[i, j].plot(x, np.median(c, 0), color=COLORS[s], lw=2, marker="o", ms=4, label=s)
        for i, tol in enumerate([0.01, 0.01]):
            axes[i, j].axhline(tol, color=MUTED, lw=1, ls="--")
            axes[i, j].set_yscale("log")
        axes[0, j].set_title(f"{n} noise", loc="left")
        axes[1, j].set_xlabel("experiment number")
    axes[0, 0].set_ylabel("median |g error| / g")
    axes[1, 0].set_ylabel("median |μ error|")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Median estimation error vs experiments run (dashed = tolerance)",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "error_curves.png", dpi=150)
    plt.close(fig)

    # ---------- Summary tables
    L = ["# Results", "",
         f"Budget {budget} experiments, {n_seeds} seeds per (strategy, noise). True g ~ U(3, 18), "
         "μ ~ U(0.1, 0.7), same truths and noise seeds across strategies (paired design). "
         "Threshold: |g−g*|/g* ≤ 1 % and |μ−μ*| ≤ 0.01, and it must hold through the end of the budget. "
         f"Runs that never reach it count as {budget + 1} in means.", "",
         "![experiments to threshold](experiments_to_threshold.png)", "",
         "![error curves](error_curves.png)", ""]
    for n in NOISES:
        L += [f"## {n} noise", "",
              "| strategy | mean exps (95% CI) | median | reached | mean cost to threshold | "
              "final g rel err (median) | final μ err (median) | traj RMSE m (median) | 90% CI coverage g / μ | plan latency s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in STRATS:
            rs = by[(n, s)]
            y = _censored_steps(rs, budget)
            lo, hi = _boot_ci(y, rng)
            reached = [r for r in rs if r["reached"] == "True"]
            cost = np.mean([float(r["cost_to_threshold"]) for r in reached]) if reached else float("nan")
            f = lambda k: np.median([float(r[k]) for r in rs])
            m = lambda k: np.mean([float(r[k]) for r in rs])
            L.append(f"| {s} | {y.mean():.2f} ({lo:.2f}–{hi:.2f}) | {np.median(y):.0f} | "
                     f"{len(reached)}/{len(rs)} | {cost:.1f} | {f('final_g_rel_err'):.4f} | "
                     f"{f('final_mu_abs_err'):.4f} | {f('final_traj_rmse'):.4f} | "
                     f"{m('ci_coverage_g'):.2f} / {m('ci_coverage_mu'):.2f} | {m('mean_plan_latency_s'):.2f} |")
        L += ["", "Paired difference in experiments needed (baseline − active; positive = active needs fewer):", "",
              "| baseline | mean diff (95% paired bootstrap CI) | active fewer / tie / more |", "|---|---|---|"]
        ya = _censored_steps(by[(n, "active")], budget)
        for s in STRATS[1:]:
            d = _censored_steps(by[(n, s)], budget) - ya
            lo, hi = _boot_ci(d, rng)
            L.append(f"| {s} | {d.mean():+.2f} ({lo:+.2f} to {hi:+.2f}) | "
                     f"{(d > 0).sum()} / {(d == 0).sum()} / {(d < 0).sum()} |")
        L.append("")

    # experiment mix chosen by each strategy
    L += ["## Which experiments each strategy ran (all noise levels)", "",
          "| strategy | drop | flat_slide | ramp_slide | impulse |", "|---|---|---|---|---|"]
    for s in STRATS:
        ks = [k for r in rows if r["strategy"] == s for k in json.loads(r["kinds"])]
        L.append(f"| {s} | " + " | ".join(f"{ks.count(k) / len(ks):.0%}"
                                          for k in ["drop", "flat_slide", "ramp_slide", "impulse"]) + " |")
    L.append("")

    cpath = out / "change_detection.csv"
    if cpath.exists():
        cr = _read(cpath)
        L += ["## Change detection", "",
              "Physics re-drawn between experiment 6 and 7 (14-experiment budget); control runs have no change. "
              "Detector: max normalized residual at the MAP over all logged experiments > 1.6 for 2 consecutive "
              "experiments, then the belief resets to the prior keeping the suspect experiments.", "",
              "| noise | strategy | detected | mean delay (exps) | false-alarm runs (change runs pre-change + control runs) | "
              "recovered to tolerance | mean recovery (exps after change) |", "|---|---|---|---|---|---|---|"]
        for n in NOISES:
            for s in ("active", "random"):
                ch = [r for r in cr if r["noise"] == n and r["strategy"] == s and r["with_change"] == "True"]
                ct = [r for r in cr if r["noise"] == n and r["strategy"] == s and r["with_change"] == "False"]
                if not ch:
                    continue
                det = [r for r in ch if r["detected"] == "True"]
                delay = np.mean([float(r["detection_delay"]) for r in det]) if det else float("nan")
                fa = sum(int(r["false_alarms"]) > 0 for r in ch + ct)
                rec = [r for r in ch if r["recovery_steps"]]
                recm = np.mean([float(r["recovery_steps"]) for r in rec]) if rec else float("nan")
                L.append(f"| {n} | {s} | {len(det)}/{len(ch)} | {delay:.2f} | {fa}/{len(ch) + len(ct)} | "
                         f"{len(rec)}/{len(ch)} | {recm:.2f} |")
        L.append("")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import sys
    make_report(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
