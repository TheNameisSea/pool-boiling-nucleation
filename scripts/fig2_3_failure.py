#!/usr/bin/env python3
"""Figs. 2 and 3.

Fig. 2. The eight baselines (three AViT, three FiLM-AViT, two U-Net runs) against the single-bubble
control, from 12 active sites up:
  (a) one step from true inputs: matched recall of real births (right site, right frame);
  (b) 300-frame autoregressive rollout: nucleation ratio (predicted / reference events).
Fig. 3. Site freeing in each source's own rollout:
  (a) debounced frees per site per 100 frames;
  (b) probability that a freed site is reseeded within 5 frames.
  Reference vs the AViT (735) and the conditioned AViT (716); the control model and its reference as
  pooled points.

Inputs: the CSVs in results/diagnostics (tf_birth_matching*.csv, model_diagnostics_*_minarea16.csv,
site_freeing.csv). Seconds on a laptop.
Writes figures/fig2_failure.{png,pdf} and fig3_reseeding.{png,pdf}.
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
from fig5_cause_fix import D, GREY, GRID, INK, INK2, MATCH_FILES, SITES  # noqa: E402  (also sets rcParams)

ARCH = (  # (label, jobs, marker, line style)
    ("axial transformer (AViT)", ("735", "766", "773"), "o", "-"),
    ("conditioned AViT (FiLM)", ("716", "813", "814"), "^", "-."),
    ("U-Net", ("739", "855"), "s", "--"),
)
BLUE = "#2a78d6"


def fig2():
    m = pd.concat([pd.read_csv(f"{D}/{f}") for f in MATCH_FILES])
    m["job"] = m.job.astype(str)
    m = m.drop_duplicates(["job", "n_sites"]).set_index(["job", "n_sites"])
    ctl = pd.read_csv(f"{D}/tf_birth_matching_control.csv")
    ctl_rec = (ctl.recall_tol0 * ctl.ref_births).sum() / ctl.ref_births.sum()
    c = pd.read_csv(f"{D}/model_diagnostics_778_minarea16.csv")
    ctl_roll = c.nuc_total.sum() / c.ref_nuc_total.sum()

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, get, ctl_v, title, ylab in (
            (a, lambda j: m.loc[j].recall_tol0.reindex(SITES).values, ctl_rec,
             "(a) One step from true inputs", "matched recall of real births"),
            (b, lambda j: pd.read_csv(f"{D}/model_diagnostics_{j}_minarea16.csv").set_index("n_sites")
             .nuc_ratio.reindex(SITES).values, ctl_roll,
             "(b) 300-frame rollout", "nucleation ratio (pred / ref)")):
        ax.axhline(ctl_v, color=BLUE, lw=1.6, ls=":", zorder=2)
        ax.text(SITES[-1], ctl_v + 0.03, f"single-bubble control ({ctl_v:.2f})", fontsize=6.5, color=INK2,
                ha="right", va="bottom")
        for _, jobs, mk, ls in ARCH:                       # mean over seeds, shading = range over seeds
            v = np.vstack([get(j) for j in jobs])
            ax.fill_between(SITES, v.min(0), v.max(0), color=GREY, alpha=0.18, lw=0, zorder=2)
            ax.plot(SITES, v.mean(0), color=GREY, ls=ls, lw=1.5, marker=mk, ms=4.5, mec="white", mew=0.7,
                    zorder=3)
        ax.set_xticks(SITES)
        ax.set_xlabel("active nucleation sites $n_s$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)   # symlog keeps values near zero distinguishable
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.01, 0.03, 0.1, 0.3, 1.0])
        ax.set_yticklabels(["0", "0.01", "0.03", "0.1", "0.3", "1"])
        ax.grid(axis="y", color=GRID, lw=0.5)
    b.yaxis.set_tick_params(labelleft=True)
    h = [Line2D([], [], color=GREY, ls=ls, marker=mk, ms=4.5, mec="white", label=f"{lab}, {len(j)} seeds (mean, range)")
         for lab, j, mk, ls in ARCH]
    h.append(Line2D([], [], color=BLUE, ls=":", lw=1.6, label="single-bubble control (AViT)"))
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig2_failure.{ext}", dpi=300, bbox_inches="tight")
    print(f"control: one-step recall {ctl_rec:.2f}, rollout ratio {ctl_roll:.2f}")
    for lab, jobs, _, _ in ARCH:
        r = np.vstack([m.loc[j].recall_tol0.reindex(SITES).values for j in jobs])
        print(f"  {lab:<28} one-step recall {r.min():.3f}-{r.max():.3f}")


def fig3():
    s = pd.read_csv(f"{D}/site_freeing.csv")
    pool = s[(s.study == "pool") & (s.n_sites >= 12)]
    sites = sorted(pool.n_sites.unique())
    single = s[s.study == "single"]
    # pooled control: frees and responding frees summed over the six sims
    ctl = {src: (g.respond * g.n_frees_deb).sum() / g.n_frees_deb.sum() for src, g in single.groupby("source")}
    ctl_frees = {src: g.frees_deb.mean() for src, g in single.groupby("source")}
    style = {"reference": (INK, "-", "D"), "735 uncond": (GREY, "-", "o"), "716 cond": (GREY, "-.", "^")}
    label = {"reference": "reference simulation", "735 uncond": "AViT", "716 cond": "conditioned AViT"}

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, col, title, ylab in ((a, "frees_deb", "(a) Sites free", "site freeings per site per 100 frames"),
                                 (b, "respond", "(b) Freed sites are not reseeded",
                                  "P(seed within 5 frames | site freed)")):
        for src, (colr, ls, mk) in style.items():
            g = pool[pool.source == src].set_index("n_sites").reindex(sites)
            ax.plot(sites, g[col].values, color=colr, ls=ls, lw=1.5, marker=mk, ms=4.5, mec="white", mew=0.7,
                    zorder=3)
        ax.set_xticks(sites)
        ax.set_xlabel("active nucleation sites $n_s$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
    b.set_ylim(-0.03, 1.05)
    a.set_ylim(0, 9)
    b.axhline(ctl["778 control"], color=BLUE, ls=":", lw=1.6)
    b.text(sites[0], ctl["778 control"] + 0.03, f"single-bubble control model ({ctl['778 control']:.2f}; "
           f"its reference {ctl['reference']:.2f})", fontsize=6.3, color=INK2, va="bottom")
    h = [Line2D([], [], color=c, ls=ls, marker=mk, ms=4.5, mec="white", label=label[k])
         for k, (c, ls, mk) in style.items()]
    h.append(Line2D([], [], color=BLUE, ls=":", lw=1.6, label="single-bubble control model"))
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/fig3_reseeding.{ext}", dpi=300, bbox_inches="tight")
    print(pool[["n_sites", "source", "duty", "frees_deb", "respond", "n_births"]].round(3).to_string(index=False))
    print("control pooled P(reseed | freed):", {k: round(v, 2) for k, v in ctl.items()},
          "frees/site/100f:", {k: round(v, 2) for k, v in ctl_frees.items()})


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    fig2()
    fig3()
    print("wrote figures/fig2_failure.{png,pdf} and fig3_reseeding.{png,pdf}")
