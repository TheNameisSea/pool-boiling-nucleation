#!/usr/bin/env python3
"""Fig. 5 and Table 3.

(a) A real pool-boiling birth (Twall 90, 20 sites, three frames after its site freed): the distance
    field along the first cell row above the wall, with and without the new seed, and the blends
    p*phi_seed + (1-p)*phi_noseed for p = 0.5, 0.7, 0.9.
(b) One step from true inputs: matched recall of real births against site count, for the baselines,
    wall-band upweighting and the phase-indicator term on both architectures, with the squared-error
    ceiling (l2opt_ceiling.csv) and the single-bubble control.
(c) The same models in autoregressive rollout: nucleation ratio (predicted / reference events).
Table 3: every training change, one step ahead (pooled matched recall / precision) and in rollout.

Inputs: the CSVs in results/diagnostics, results/runs/*/metrics.csv, and for (a) two saved frames
(fig5a_frames.npz) plus the Twall 90 JSON metadata. Seconds on a laptop.
Writes figures/fig5_cause_fix.{png,pdf} and tables/table3_interventions.md.
"""
import os, sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "scripts")
from birth_predictability import meta
from diagnostics import DX
from l2opt_ceiling import ALIGN, WIN, sdf, seed_component

D = "results/diagnostics"
from paths import POOL  # noqa: E402
BLUE, AQUA = "#2a78d6", "#1baf7a"                  # indicator term, wall-band upweighting
INK, INK2, GREY, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUES = ("#9ec5f4", "#5598e7", "#256abf")          # sequential steps 200 / 350 / 500 for p = 0.5 / 0.7 / 0.9
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "font.family": "DejaVu Sans"})
SITES = [12, 20, 26, 32, 37]
MATCH_FILES = ("tf_birth_matching.csv", "tf_birth_matching_baselines.csv", "tf_birth_matching_1070_1077_1089.csv",
               "tf_birth_matching_1184.csv", "tf_birth_matching_1181_1194.csv", "tf_birth_matching_1200.csv")
TW, F_IDX, SITE = "90", 41, 10                     # the birth shown in (a); in l2opt_ceiling_births.csv p* = 0.90, delay 3


def matched():
    m = pd.concat([pd.read_csv(f"{D}/{f}") for f in
                   MATCH_FILES])
    m["job"] = m.job.astype(str)
    return m[m.n_sites >= 12].drop_duplicates(["job", "n_sites"]).set_index(["job", "n_sites"]).recall_tol0


def rollout(job):
    d = pd.read_csv(f"{D}/model_diagnostics_{job}_minarea16.csv").set_index("n_sites").sort_index()
    return d.nuc_ratio.loc[SITES].values


def panel_a(ax):
    m = meta(os.path.join(POOL, f"Twall_{TW}.json"))
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    # The two frames (raw frames ALIGN - WIN + F_IDX - 1 and ALIGN - WIN + F_IDX of Twall 90) are stored
    # in results/diagnostics/fig5a_frames.npz, so the figure does not need the HDF5 data.
    prev, cur = np.load(f"{D}/fig5a_frames.npz")["dfun"]
    seed = seed_component(cur > 0, prev > 0, cols, SITE)
    phi_s = cur
    phi_n = cur + (sdf((cur > 0) & ~seed) - sdf(cur > 0))
    c, half, row = cols[SITE], 9, 1
    x = (np.arange(c - half, c + half + 1) - c) * DX * 32          # cells from the site
    s, n = phi_s[row, c - half:c + half + 1], phi_n[row, c - half:c + half + 1]
    ax.axhspan(0, 0.5, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.plot(x, n, color=GREY, lw=1.6, ls="--", label="no seed")
    for p, col in zip((0.5, 0.7, 0.9), BLUES):
        ax.plot(x, p * s + (1 - p) * n, color=col, lw=1.6, label=f"optimal blend, p={p}")
    ax.plot(x, s, color=INK, lw=1.6, label="seed (truth)")
    ax.text(x[-1], 0.02, "vapour", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.text(x[-1], -0.15, "liquid", ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel("cells along the wall from the site")
    ax.set_ylabel(r"$\phi$ one cell above the wall")
    ax.set_ylim(-0.6, 0.13)
    ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1])
    ax.set_title("(a) The blend hides the seed", loc="left")
    ax.legend(loc="lower center", ncol=1, fontsize=6.0, handlelength=1.6, borderaxespad=0.2,
              labelspacing=0.25, frameon=False)


def lines(ax, series, ylabel, title, ceiling=None, control=None):
    if ceiling is not None:
        ax.plot(SITES, ceiling, color=INK, lw=1.4, ls=(0, (4, 2)), zorder=2)
        ax.text(SITES[-1] + 1.2, ceiling[-1], "squared-\nerror\nceiling", fontsize=6.3, color=INK, va="center")
    if control is not None:
        ax.axhline(control, color=INK2, lw=1, ls=":", zorder=1)
        ax.text(SITES[-1] + 1.2, control, "control", fontsize=6.3, color=INK2, va="center")
    for vals, color, ls, mk in series:
        if isinstance(vals, tuple):                                # (low, mid, high): a range over runs
            lo, mid, hi = vals
            ax.fill_between(SITES, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
            vals = mid
        ax.plot(SITES, vals, color=color, ls=ls, lw=1.8, marker=mk, ms=4.5, mec="white", mew=0.8, zorder=3)
    ax.set_xticks(SITES)
    ax.set_xlabel("active nucleation sites $n_s$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(axis="y", color=GRID, lw=0.5)


def main():
    r = matched()
    rec = lambda j: r.loc[j].reindex(SITES).values
    ceil = pd.read_csv(f"{D}/l2opt_ceiling.csv").query("study == 'pool'").set_index("n_sites").count_cnn.loc[SITES].values
    ctl = pd.read_csv(f"{D}/tf_birth_matching_control.csv")
    ctl_recall = (ctl.recall_tol0 * ctl.ref_births).sum() / ctl.ref_births.sum()
    rng = lambda f, jobs: (lambda a: (a.min(0), a.mean(0), a.max(0)))(np.vstack([f(j) for j in jobs]))
    BASE_A, BASE_U, UPW_A = ("735", "766", "773"), ("739", "855"), ("952", "980")
    IND_A, IND_U = ("1055", "1077", "1089", "1184"), ("1070", "1181")
    ind = np.vstack([rec(j) for j in IND_A])

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.0, 1.05, 1.05]))
    panel_a(a)
    one = [(rng(rec, BASE_A), GREY, "-", "o"), (rng(rec, BASE_U), GREY, "--", "s"),
           (rng(rec, UPW_A), AQUA, "-", "o"), (rec("1034"), AQUA, "--", "s"),
           (rng(rec, IND_A), BLUE, "-", "o"), (rng(rec, IND_U), BLUE, "--", "s")]
    lines(b, one, "matched recall of real births", "(b) One step from true inputs",
          ceiling=ceil, control=ctl_recall)
    roll = [(rng(rollout, BASE_A), GREY, "-", "o"), (rng(rollout, BASE_U), GREY, "--", "s"),
            (rng(rollout, UPW_A), AQUA, "-", "o"), (rollout("1034"), AQUA, "--", "s"),
            (rng(rollout, IND_A), BLUE, "-", "o"), (rng(rollout, IND_U), BLUE, "--", "s")]
    lines(c, roll, "nucleation ratio (pred / ref)", "(c) Autoregressive rollout", control=0.615)
    h = [Line2D([], [], color=GREY, lw=1.8, label="baseline (AViT 3 runs, U-Net 2)"),
         Line2D([], [], color=AQUA, lw=1.8, label="wall-band L2 upweighting (AViT 2 runs, U-Net 1)"),
         Line2D([], [], color=BLUE, lw=1.8, label="phase-indicator term, with or without upweighting (AViT 4 runs, U-Net 2)"),
         Line2D([], [], color=INK2, lw=1.8, ls="-", marker="o", ms=4.5, label="axial transformer"),
         Line2D([], [], color=INK2, lw=1.8, ls="--", marker="s", ms=4.5, label="U-Net")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, -0.02),
               title="lines show the mean, shading the range over runs", title_fontsize=6.3)
    fig.tight_layout(rect=(0, 0.13, 0.95, 1), w_pad=2.0)
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig5_cause_fix.{ext}", dpi=300, bbox_inches="tight")
    print("one-step recall, 12+ sites:")
    for name, v in (("baseline AViT mean", rng(rec, BASE_A)[1]), ("upweighting AViT mean", rng(rec, UPW_A)[1]),
                    ("indicator AViT mean", ind.mean(0)),
                    ("739", rec("739")), ("1070", rec("1070")), ("1181", rec("1181")), ("ceiling", ceil)):
        print(f"  {name:>20}: {np.round(v, 2)}")
    print(f"  control recall {ctl_recall:.2f}")
    print("wrote figures/fig5_cause_fix.{png,pdf}")


ROWS = (  # (label, architecture, runs) -- every pool model with matched births; seeds pooled per condition
    ("none (baseline)", "axial transformer", ("735", "766", "773")),
    ("none (baseline)", "U-Net", ("739", "855")),
    ("thermophysical conditioning", "axial transformer (FiLM)", ("716", "813", "814")),
    ("timing-tolerant loss", "axial transformer", ("942",)),
    ("first output frame only†", "axial transformer", ("974",)),
    ("wall-band L2 upweighting ×10", "axial transformer", ("952", "980")),
    ("wall-band L2 upweighting ×30", "axial transformer", ("1011",)),
    ("wall-band L2 upweighting ×10", "U-Net", ("1034",)),
    ("local wall-feature path + upweighting", "axial transformer", ("1052",)),
    ("**phase-indicator term**", "axial transformer", ("1089", "1184")),
    ("phase-indicator term + upweighting", "axial transformer", ("1055", "1077")),
    ("**phase-indicator term**", "U-Net", ("1181",)),
    ("phase-indicator term + upweighting", "U-Net", ("1070",)),
    ("phase-indicator term over the whole domain", "axial transformer", ("1200",)),
    ("phase-indicator term + pushforward training", "axial transformer", ("1194",)),
)


def table3():
    m = pd.concat([pd.read_csv(f"{D}/{f}") for f in
                   MATCH_FILES])
    m["job"] = m.job.astype(str)
    m = m[m.n_sites >= 12].drop_duplicates(["job", "n_sites"])
    m["hit"] = m.recall_tol0 * m.ref_births
    m["phit"] = m.precision_tol0 * m.pred_births
    out = ["| training change | architecture | runs | recall | precision | rollout ratio | validation loss |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for lab, arch, jobs in ROWS:
        x = m[m.job.isin(jobs)]
        g = x.groupby("job")[["hit", "ref_births", "phit", "pred_births"]].sum()
        rec = g.hit / g.ref_births
        prec = g.phit / g.pred_births.where(g.pred_births > 0)
        roll = [rollout(j).mean() for j in jobs if os.path.exists(f"{D}/model_diagnostics_{j}_minarea16.csv")]
        val = [pd.read_csv(f"results/runs/{j}/metrics.csv").dropna(subset=["val_loss_epoch"]).val_loss_epoch.iloc[-1]
               for j in jobs]
        span = lambda v, f: f.format(min(v)) if len(v) == 1 or f.format(min(v)) == f.format(max(v)) \
            else f"{f.format(min(v))}–{f.format(max(v))}"
        out.append(f"| {lab} | {arch} | {len(jobs)} | {span(list(rec), '{:.2f}')} | "
                   f"{span([v for v in prec if v == v], '{:.2f}') if prec.notna().any() else '—'} | "
                   f"{span(roll, '{:.2f}') if roll else '—'} | {span(val, '{:.3f}')} |")
    ctl = pd.read_csv(f"{D}/tf_birth_matching_control.csv")
    out.append(f"| none (single-bubble control) | axial transformer | 1 | "
               f"{(ctl.recall_tol0 * ctl.ref_births).sum() / ctl.ref_births.sum():.2f} | 1.00 | 0.62 | — |")
    note = ("\n*One step from true inputs, the six evaluation simulations with 12–37 active sites pooled. Recall: "
            "the share of real births the model reproduces at the right site and frame. Precision: the share "
            "of its births that are real. Rollout ratio: predicted over reference nucleation events in "
            "300-frame autoregressive rollouts, mean over site counts. Validation loss: the standard "
            "relative-$L^2$ loss over all five output frames, final epoch. Ranges are over training seeds. "
            "† Trained on the first output frame only, so it has no meaningful rollout and its five-frame "
            "validation loss is not comparable.*\n")
    os.makedirs("tables", exist_ok=True)
    with open("tables/table3_interventions.md", "w") as f:
        f.write("<!-- generated by scripts/fig5_cause_fix.py from results/diagnostics/tf_birth_matching*.csv, "
                "model_diagnostics_*_minarea16.csv and results/runs/*/metrics.csv; do not edit by hand -->\n\n"
                "**Table 3.** Every training change tested, one step ahead and in rollout.\n\n"
                + "\n".join(out) + "\n" + note)
    print("\n".join(out))


if __name__ == "__main__":
    main()
    table3()
