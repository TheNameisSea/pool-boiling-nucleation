#!/usr/bin/env python3
"""Fig. 6: standard metrics against nucleation, over every pool-boiling model with matched births.

22 models (run 974 is left out: trained on the first output frame only, its five-frame validation
loss is not comparable). Each panel puts a standard metric on x and nucleation on y:
(a) validation loss (upstream relative L2 over five output frames, final epoch) vs one-step matched recall;
(b) one-step relative L2 of the distance field from true inputs vs one-step matched recall;
(c) rollout eikonal error |<|grad phi|> - 1| at frame 250 vs rollout nucleation ratio.
Recall and rollout ratio are pooled / averaged over the five sims with 12-37 active sites.

Also writes results/diagnostics/d4_metrics_vs_nucleation.csv (one row per model) and prints Spearman
rank correlations. Inputs are the CSVs in results/diagnostics and results/runs; seconds on a laptop.
Writes figures/fig6_metrics_blind.{png,pdf}.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "scripts")
from fig5_cause_fix import AQUA, BLUE, D, GREY, GRID, INK, INK2, MATCH_FILES  # noqa: E402  (also sets rcParams)

ORANGE = "#eb6834"
GROUPS = (  # (label, colour, {job: architecture})
    ("baseline (incl. thermophysical conditioning)", GREY,
     {"735": "A", "766": "A", "773": "A", "716": "A", "813": "A", "814": "A", "739": "U", "855": "U"}),
    ("other loss / architecture changes", ORANGE, {"942": "A", "1052": "A"}),
    ("wall-band L2 upweighting", AQUA, {"952": "A", "980": "A", "1011": "A", "1034": "U"}),
    ("phase-indicator term (all variants)", BLUE,
     {"1055": "A", "1077": "A", "1089": "A", "1184": "A", "1200": "A", "1194": "A", "1070": "U", "1181": "U"}),
)
OUT_CSV = f"{D}/d4_metrics_vs_nucleation.csv"


def collect():
    m = pd.concat([pd.read_csv(f"{D}/{f}") for f in MATCH_FILES])
    m["job"] = m.job.astype(str)
    m = m[m.n_sites >= 12].drop_duplicates(["job", "n_sites"])
    m["hit"], m["phit"] = m.recall_tol0 * m.ref_births, m.precision_tol0 * m.pred_births
    g = m.groupby("job")[["hit", "ref_births", "phit", "pred_births"]].sum()
    rows = []
    for label, colour, jobs in GROUPS:
        for j, arch in jobs.items():
            val = pd.read_csv(f"results/runs/{j}/metrics.csv").dropna(subset=["val_loss_epoch"]).val_loss_epoch
            tf = pd.read_csv(f"{D}/model_diagnostics_{j}_minarea16_teacherforced.csv").query("n_sites >= 12")
            ro = pd.read_csv(f"{D}/model_diagnostics_{j}_minarea16.csv").query("n_sites >= 12")
            rows.append(dict(job=j, group=label, arch={"A": "axial transformer", "U": "U-Net"}[arch],
                             val_loss=val.iloc[-1], val_loss_last5=val.iloc[-5:].mean(),
                             recall=g.hit[j] / g.ref_births[j], precision=g.phit[j] / g.pred_births[j],
                             onestep_rel_l2_dfun=tf.tf_onestep_rel_l2_dfun.mean(),
                             rollout_ratio=ro.nuc_ratio.mean(), rollout_eikonal_err=(ro.grad_mean - 1).abs().mean(),
                             rollout_rel_l2_dfun=ro.rel_l2_dfun.mean(), interface_width=ro.interface_width.mean()))
    return pd.DataFrame(rows)


def panel(ax, df, x, y, xlabel, ylabel, title):
    for label, colour, _ in GROUPS:
        for arch, mk in (("axial transformer", "o"), ("U-Net", "s")):
            s = df[(df.group == label) & (df.arch == arch)]
            ax.scatter(s[x], s[y], s=30, marker=mk, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_ylim(-0.04, 1.0)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def main():
    df = collect()
    df.round(4).to_csv(OUT_CSV, index=False)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.7))

    avit = df[(df.arch == "axial transformer") & (df.group != "baseline (incl. thermophysical conditioning)")]
    lo, hi = df[df.job.isin(["735", "766", "773"])].val_loss.agg(["min", "max"])
    a.axvspan(lo, hi, color=GRID, alpha=0.7, lw=0, zorder=0)
    a.text(hi + 0.002, 0.52, "unconditioned\nbaselines'\nrange", fontsize=6.3, color=INK2, va="center")
    panel(a, df, "val_loss", "recall", "validation loss (relative $L^2$)", "matched recall of real births",
          "(a) Validation loss")
    panel(b, df, "onestep_rel_l2_dfun", "recall", r"one-step relative $L^2$ of $\phi$",
          "matched recall of real births", "(b) One-step field error")
    panel(c, df, "rollout_eikonal_err", "rollout_ratio", r"eikonal error $|\langle|\nabla\phi|\rangle - 1|$",
          "rollout nucleation ratio", "(c) Rollout admissibility")
    c.set_ylim(-0.02, 0.5)
    h = [Line2D([], [], ls="", marker="o", ms=5.5, color=col, mec="white", label=lab) for lab, col, _ in GROUPS]
    h += [Line2D([], [], ls="", marker="o", ms=5.5, color=INK2, label="axial transformer"),
          Line2D([], [], ls="", marker="s", ms=5.5, color=INK2, label="U-Net")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.2)
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig6_metrics_blind.{ext}", dpi=300, bbox_inches="tight")

    print(df.set_index("job").round(3).to_string())
    print("\nSpearman rank correlation with nucleation (all 22 models):")
    for x, y in (("val_loss", "recall"), ("onestep_rel_l2_dfun", "recall"), ("rollout_eikonal_err", "rollout_ratio"),
                 ("rollout_rel_l2_dfun", "rollout_ratio")):
        print(f"  {x:>22} vs {y:<14} rho = {df[x].corr(df[y], method='spearman'):+.2f}")
    print(f"\nAViT models with a changed loss: validation loss {avit.val_loss.min():.3f}–{avit.val_loss.max():.3f}, "
          f"recall {avit.recall.min():.2f}–{avit.recall.max():.2f}")
    near = df[(df.val_loss >= lo - 0.005) & (df.val_loss <= hi + 0.005)]
    print(f"Models within 0.005 of the unconditioned baselines' validation loss ({lo:.3f}–{hi:.3f}): "
          f"{len(near)}, recall {near.recall.min():.3f}–{near.recall.max():.2f}")
    print(f"wrote figures/fig6_metrics_blind.{{png,pdf}} and {OUT_CSV}")


if __name__ == "__main__":
    main()
