#!/usr/bin/env python3
"""Fig. 4 and Table 2.

Fig. 4: per-step level-set reinitialisation (`rollout_diagnostics.py --project`). Rollout nucleation
ratio and mid-rollout mean |∇φ| against site count, with and without reinitialisation, for the AViT
(run 735) and the U-Net (run 739).
Table 2: penalty the training loss charges for omitting young bubbles, pool boiling vs the
single-bubble control (`loss_weighting.csv`), next to the models' one-step error.

Inputs are the CSVs in results/diagnostics; seconds on a laptop.
Writes figures/fig4_projection.{png,pdf} and tables/table2_loss_penalty.md.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = "results/diagnostics"
BLUE, ORANGE = "#2a78d6", "#eb6834"                 # 735 (AViT), 739 (U-Net)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "font.family": "DejaVu Sans"})
MODELS = (("735", "axial transformer (28.9 M)", BLUE), ("739", "U-Net (566 M)", ORANGE))


def load(job, projected):
    f = f"{D}/model_diagnostics_{job}_minarea16{'_projected' if projected else ''}.csv"
    return pd.read_csv(f).sort_values("n_sites")


def fig4():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for job, lab, c in MODELS:
        for proj, ls, mk, fill in ((False, "-", "o", c), (True, "--", "o", "white")):
            d = load(job, proj)
            kw = dict(color=c, ls=ls, lw=1.8, marker=mk, ms=5, mfc=fill, mec=c, mew=1.4)
            a.plot(d.n_sites, d.nuc_ratio, **kw)
            b.plot(d.n_sites, d.grad_mean, **kw)
    ref = load("735", False)
    b.fill_between(ref.n_sites, ref.ref_grad_mean.min(), ref.ref_grad_mean.max(), color=GRID, lw=0,
                   zorder=0)
    b.text(36.5, ref.ref_grad_mean.max() + 0.01, "reference", ha="right", va="bottom", fontsize=6.5,
           color=INK2)
    for ax in (a, b):                                   # shade the 5-site simulation (few events)
        ax.axvspan(3.2, 8.2, color=GRID, alpha=0.45, lw=0, zorder=0)
    a.text(4.9, 0.03, "sparse,\nnoisy", ha="center", va="bottom", fontsize=6.3, color=INK2)
    a.set_ylabel("rollout nucleation ratio (pred / ref)")
    b.set_ylabel(r"mean $|\nabla\phi|$ at mid-rollout")
    for ax in (a, b):
        ax.set_xlabel("active nucleation sites $n_s$")
        ax.set_xticks([5, 12, 20, 26, 32, 37])
        ax.grid(axis="y", color=GRID, lw=0.5)
    a.set_title("(a) Nucleation does not recover", loc="left")
    b.set_title("(b) The level set is repaired", loc="left")
    a.set_ylim(-0.01, 0.215)
    # legend: model by colour, projection by line style / fill
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=1.8, marker="o", ms=5, label=lab) for _, lab, c in MODELS]
    h += [Line2D([], [], color=INK2, lw=1.8, ls="-", marker="o", ms=5, mfc=INK2, label="as predicted"),
          Line2D([], [], color=INK2, lw=1.8, ls="--", marker="o", ms=5, mfc="white", mec=INK2,
                 label="reinitialised every step")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig4_projection.{ext}", dpi=300, bbox_inches="tight")
    for job, _, _ in MODELS:
        u, p = load(job, False), load(job, True)
        print(f"{job}: nuc ratio mean {u.nuc_ratio.mean():.3f} -> {p.nuc_ratio.mean():.3f}; "
              f"|grad| mean {u.grad_mean.mean():.3f} -> {p.grad_mean.mean():.3f}")


def table2():
    lw = pd.read_csv(f"{D}/loss_weighting.csv")
    rows = ["| study | sites | frames with a young bubble | penalty when present | penalty, all frames "
            "| model one-step error |", "|---|---:|---:|---:|---:|---:|"]
    pool = lw[lw.study == "pool"].sort_values("twall")
    for _, r in pool.iterrows():
        ns = int(round(r.twall - 70))
        err = r["onestep_735 uncond"]
        rows.append(f"| pool boiling | {ns} | {r.frac_frames_with_young:.2f} | {r.pen_young_when_present:.4f} "
                    f"| {r.pen_young_mean:.4f} | {'—' if pd.isna(err) else f'{err:.3f}'} |")
    ctl = lw[lw.study != "pool"]
    rng = lambda c, f: f"{f.format(ctl[c].min())}–{f.format(ctl[c].max())}"
    rows.append(f"| single-bubble control | 1 | {rng('frac_frames_with_young', '{:.2f}')} | "
                f"{rng('pen_young_when_present', '{:.3f}')} | {rng('pen_young_mean', '{:.4f}')} | "
                f"{rng('onestep_778 control', '{:.3f}')} |")
    note = ("\n*Young bubble: a wall-attached bubble born within the preceding five frames. Penalty: the "
            "distance-field term of the training loss (relative $L^2$) for a frame that is exact except "
            "that its young bubbles are removed. Model one-step error: relative $L^2$ of the axial "
            "transformer (pool) or the control model's distance field, one step from true inputs.*\n")
    os.makedirs("tables", exist_ok=True)
    out = "tables/table2_loss_penalty.md"
    with open(out, "w") as f:
        f.write("<!-- generated by scripts/fig4_projection_table2.py from results/diagnostics/"
                "loss_weighting.csv; do not edit by hand -->\n\n"
                "**Table 2.** Loss penalty for omitting young bubbles.\n\n" + "\n".join(rows) + "\n" + note)
    print("\n".join(rows))


if __name__ == "__main__":
    fig4()
    table2()
    print("wrote figures/fig4_projection.{png,pdf} and tables/table2_loss_penalty.md")
