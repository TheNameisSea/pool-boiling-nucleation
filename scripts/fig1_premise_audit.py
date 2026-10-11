#!/usr/bin/env python3
"""Fig. 1: the premise audit.

  (a) Active nucleation sites of all 20 simulations, against the site count (= Twall - 70).
  (b) Frames between a site freeing and the next seed on it, pool boiling (12-37 sites) vs the
      single-bubble control, with the 5-frame input window shaded.
  (c) ROC of "seed next frame" on held-out simulations: the local CNN vs the delay since the site freed.

Inputs: the simulation JSON metadata ($BUBBLEML_ROOT, see scripts/pull_study.py --meta-only);
results/diagnostics/reoccupation_by_delay.csv; results/diagnostics/timing_readability_preds_pool.npz.
Writes figures/fig1_premise_audit.{png,pdf}. Seconds on a laptop.
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from timing_readability import auc
from paths import POOL
OUT = "figures/fig1_premise_audit"
BLUE, ORANGE = "#2a78d6", "#eb6834"                 # pool boiling, control
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WINDOW = 5

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "font.family": "DejaVu Sans"})


def panel_a(ax):
    sims = []
    for f in glob.glob(f"{POOL}/Twall_*.json"):
        m = json.load(open(f))
        sims.append((m["heater"]["wallTemp"], np.array(m["nuc_sites_x"], float)))
    sims.sort()
    prev = 0
    for tw, xs in sims:
        n = len(xs)
        assert prev == 0 or np.allclose(xs[:prev], last[:prev]), "site sets are not nested"
        ax.scatter(xs[:prev], np.full(prev, n), s=7, color=INK2, lw=0, zorder=2)
        ax.scatter(xs[prev:], np.full(n - prev, n), s=16, color=BLUE, lw=0.6, edgecolor="white", zorder=3)
        prev, last = n, xs
    ax.set_xlabel("site position on the heated wall, $x$")
    ax.set_ylabel("active sites $n_s$ (= $T_w$ − 70)")
    ax.set_title("(a) Where: one fixed site sequence", loc="left")
    ax.scatter([], [], s=16, color=BLUE, label="sites added at this $T_w$")
    ax.scatter([], [], s=7, color=INK2, label="sites already active below it")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=7, handletextpad=0.2)
    ax.set_ylim(0, 40)
    ax.grid(axis="x", color=GRID, lw=0.5)


def panel_b(ax):
    df = pd.read_csv("results/diagnostics/reoccupation_by_delay.csv")
    df["seeds"] = df.counted + df.merged
    groups = (("pool boiling, 12–37 sites", df[(df.study == "pool") & (df.twall >= 82)], BLUE),
              ("single-bubble control", df[df.study == "single"], ORANGE))
    ds = np.arange(1, 8)
    w = 0.38
    ax.axvspan(0.5, WINDOW + 0.5, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.text(WINDOW + 0.35, 0.98, "freeing visible\nin the 5-frame\ninput window", ha="right", va="top",
            fontsize=6.5, color=INK2, transform=ax.get_xaxis_transform())
    for i, (lab, g, c) in enumerate(groups):
        s = g.groupby("delay").seeds.sum().reindex(ds, fill_value=0)
        share = s / s.sum()
        inside = share[ds <= WINDOW].sum()
        ax.bar(ds + (i - 0.5) * w, share, width=w - 0.04, color=c, label=f"{lab} ({inside:.0%} inside)",
               zorder=2)
    ax.set_xticks(ds, [str(d) for d in ds[:-1]] + ["≥7"])
    ax.set_xlabel("frames since the site freed, $d$")
    ax.set_ylabel("share of seeds")
    ax.set_title("(b) When: trigger in the input", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=7, handletextpad=0.4)
    ax.set_xlim(0.5, 7.5)
    ax.grid(axis="y", color=GRID, lw=0.5)


def roc(y, p):
    o = np.argsort(-p, kind="stable")
    y = y[o].astype(bool)
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(~y) / (~y).sum()])
    return fpr, tpr


def panel_c(ax):
    z = np.load("results/diagnostics/timing_readability_preds_pool.npz")
    y, d = z["y"].astype(bool), z["d"]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    for p, lab, c, ls in ((z["p_cnn"], "local CNN on the input window", BLUE, "-"),
                          (z["p_hazard"], "delay since the site freed", INK2, "--")):
        f, t = roc(y, p)
        a3, a4 = auc(y[d == 3], p[d == 3]), auc(y[d == 4], p[d == 4])
        ax.plot(f, t, color=c, lw=2, ls=ls, zorder=3,
                label=f"{lab}: AUC {auc(y, p):.2f}\n(within $d$ = 3 / 4: {a3:.2f} / {a4:.2f})")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("(c) Which frame: readable", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), fontsize=7, handlelength=2.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(color=GRID, lw=0.5)
    ax.text(0.97, 0.05, f"held-out simulations\n{len(y):,} site-frames\n{y.mean():.0%} seeds",
            ha="right", va="bottom", fontsize=6.5, color=INK2)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    panel_a(axs[0]); panel_b(axs[1]); panel_c(axs[2])
    fig.tight_layout(w_pad=1.6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
