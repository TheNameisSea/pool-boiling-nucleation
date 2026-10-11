# Reproducing the results: a guide for reviewers

This guide takes you from raw data to every number, figure and table in the paper. There are three
levels of effort:

| Level | What it checks | Needs | Time |
|---|---|---|---|
| A | Figures and tables follow from the committed measurements | CPU, JSON metadata only | minutes |
| B | The measurements follow from the data and saved model predictions | CPU, the BubbleML data, our trained checkpoints or saved predictions | hours |
| C | The models themselves (retraining) | GPUs, the BubbleML data | about 24 × 16 GPU-hours |

Run every command from the repository root, after the setup in [README.md](README.md). Two environment
variables locate the large files:
- `BUBBLEML_ROOT` holds the BubbleML 2.0 studies;
- `WORK_ROOT` holds the checkpoints (`$WORK_ROOT/checkpoints`) and the saved predictions
  (`$WORK_ROOT/rollout_fields_<run>`, `$WORK_ROOT/tf_fields_<run>`).

## What is and is not in the repository

- **Included:**
  - all code;
  - the data configs and the run registry (`configs/runs.csv`);
  - every per-run measurement (`results/diagnostics/*.csv`);
  - every training curve (`results/runs/<run>/metrics.csv`).
- **Not included:**
  - **The BubbleML data**, about 336 GB for pool boiling and 78 GB for the two single-bubble studies.
    Download it with `scripts/pull_study.py`.
  - **The 24 trained checkpoints.** They are on Zenodo ([doi:10.5281/zenodo.23299635](https://doi.org/10.5281/zenodo.23299635), about 14 GB, weights and training
    configuration only, with SHA-256 sums in `manifest.csv`). Retraining them is level C.
  - **The saved predictions**, about 1.8 GB per run and mode, which the CPU analyses read. They are
    regenerated from the checkpoints by `bin/evaluate_run.sh`.
  - **Upstream Bubbleformer**, which is fetched by `setup/get_bubbleformer.sh`.

  With the Zenodo checkpoints, level B needs no training: download them to `$WORK_ROOT/checkpoints/`
  and run `bin/evaluate_run.sh <run> $WORK_ROOT/checkpoints/run<run>.ckpt` for each run.

## Level A: figures and tables from the committed measurements

```bash
python scripts/pull_study.py PoolBoiling-Saturated-FC72-2D --meta-only --yes
python scripts/fig1_premise_audit.py          # Fig. 1
python scripts/fig2_3_failure.py              # Figs. 2, 3
python scripts/fig4_projection_table2.py      # Fig. 4, Table 2
python scripts/fig5_cause_fix.py              # Fig. 5, Table 3
python scripts/fig6_metrics_blind.py          # Fig. 6; prints the Spearman correlations of Section 7
```

We checked that the figures come out pixel-identical to the published ones, and the tables
identical.

## Data

| Study | Size | Needed for |
|---|---|---|
| `PoolBoiling-Saturated-FC72-2D`, all 20 simulations | ~336 GB | Training. Also the premise-audit classifier, the ceiling and the probe, which train on the 14 simulations outside the evaluation set |
| `PoolBoiling-Saturated-FC72-2D`, Twall 75 82 90 96 102 107 only | ~101 GB | All evaluations and every other analysis (`--sims 75 82 90 96 102 107`) |
| `SingleBubble-Saturated-FC72-2D` and `SingleBubble-Saturated-R515B-2D` | ~39 GB each | The single-bubble control model and its reference |

## Level C: training (`configs/runs.csv`)

Every model is trained for 150 epochs at batch size 2 on one GPU, through `bin/train.sh`:

```bash
# a baseline: plain upstream training
bin/train.sh model_cfg=avit_small data_cfg=poolboiling_fc72_nocond seed=42                         # run 735
# the paper's fix: the phase-indicator term
TRAIN_ENTRY=src/train_indicator.py BAND_WEIGHT=1 IND_WEIGHT=1 \
    bin/train.sh model_cfg=avit_small data_cfg=poolboiling_fc72_nocond seed=42                     # run 1089
# the U-Net, with AdamW at learning rate 1e-4 (it diverges under the released Lion recipe)
TRAIN_ENTRY=src/train_indicator.py BAND_WEIGHT=1 IND_WEIGHT=1 \
    bin/train.sh model_cfg=unet_modern data_cfg=poolboiling_fc72_nocond optim_cfg=adamw optim_cfg.params.lr=1e-4   # run 1181
```

For each row of `configs/runs.csv`, set the `loss_env` variables and `TRAIN_ENTRY=<train_entry>` (when
one is given), then pass `model_cfg`, `data_cfg`, `seed=<seed>` and `optimiser_override`.

**Resources:**
- About 16 h per run on an uncontended 24 GB A5000.
- The AViT runs need about 11.4 GB of GPU memory.
- The 566 M-parameter U-Net needs about 18.9 GB, so give it a nearly empty 24 GB card.
- Host memory is about 4 GB per run.

**Determinism:** GPU training is not bit-reproducible. Retrained models should land within the
seed-to-seed ranges the paper reports (for example, matched recall 0.84 for both seeds of the
phase-indicator AViT, and 0.00–0.02 for the baselines), not on identical numbers. Validation loss is
always upstream's relative L2 over five output frames, so it is comparable across all runs.

## Level B: measurements

### 1. Reference curves (no model)

```bash
python scripts/validate_diagnostics.py                         # must print PASS; writes reference_diagnostics.csv
python scripts/validate_diagnostics.py --min-area 4 --name reference_diagnostics_minarea4.csv
python scripts/validate_diagnostics.py --data-dir $BUBBLEML_ROOT/SingleBubble-Saturated-FC72-2D \
    --sims 90 92 94 96 98 100 --name singlebubble_reference_diagnostics.csv           # (and --min-area 4)
```

### 2. Evaluate each trained run (one GPU, about 10 min per mode)

```bash
bin/evaluate_run.sh <run> $WORK_ROOT/checkpoints/run<run>.ckpt      # Zenodo checkpoint
bin/evaluate_run.sh <run> $WORK_ROOT/checkpoints/<run dir>/.../epoch=149-step=150000.ckpt   # or your own retrained one
```

This writes:
- `model_diagnostics_<run>_minarea16.csv`: the 300-frame rollout from true frames 95–99;
- `..._teacherforced.csv`: one step from true inputs at every frame 100–399;
- the saved predictions used below;
- for some runs, the reinitialised (`--project`) and 4-cell-threshold variants.

Every rollout re-measures its own target frames and prints `SELF-CHECK ... PASS` when they reproduce
the reference curve. A FAIL means the alignment, units or aggregation are wrong, and that run's
numbers should not be used.

### 3. Analyses (CPU)

| Command | Output (`results/diagnostics/`) | Memory |
|---|---|---|
| `python scripts/tf_birth_matching.py 735 766 773 716 813 814 739 855 [more runs]` | `tf_birth_matching*.csv` | 8 GB |
| `MATCH_STUDY=single MATCH_OUT=results/diagnostics/tf_birth_matching_control.csv python scripts/tf_birth_matching.py 778` | control | 8 GB |
| `python scripts/birth_predictability.py` | `birth_predictability.csv` | 48 GB |
| `python scripts/reoccupation.py` | `reoccupation_by_delay.csv` | 32 GB |
| `python scripts/timing_readability.py` | `timing_readability*.csv/.npz` | 32 GB |
| `python scripts/horizon_readability.py` | `horizon_readability.csv`, `timing_readability_groupcv.csv` | 32 GB |
| `python scripts/site_freeing.py` | `site_freeing.csv` | 32 GB |
| `python scripts/site_freeing.py 952 1055 1077 1089 1184 1070 1181 1200 1194` | `site_freeing_rollouts.csv` | 32 GB |
| `python scripts/tf_reseeding.py 735 766 773 739 855 952 980 1055 1077 1089 1184 1070 1181 1200 1194` | `tf_reseeding.csv` | 8 GB |
| `python scripts/loss_weighting.py` | `loss_weighting.csv` | 32 GB |
| `python scripts/l2opt_ceiling.py` | `l2opt_ceiling*.csv` | 32 GB |
| `python scripts/seed_response.py` | `seed_response*.csv` | 48 GB |
| `python scripts/representation_probe.py --extract --job 735 --checkpoint <ckpt>` (GPU), then `--probe --job 735` | `representation_probe_735.csv` | 32 GB |
| `python scripts/cond_anatomy.py` | printed (Section 5.4) | 16 GB |
| `python scripts/birth_analysis.py` | printed (Section 3.3, 4-cell threshold) | 16 GB |

## Where each result comes from

Section numbers follow the paper.

| Paper | Numbers | Script | File |
|---|---|---|---|
| 3.3 event threshold | 99.1 % / 100 % of births kept; 204 vs 26 events at 4 cells | `validate_diagnostics.py`, `rollout_diagnostics.py --min-area 4`, `birth_analysis.py` | `*_minarea4.csv`, printed |
| 3.3 reference curve, self-check | 0.100 … 1.130 events/frame; mean \|∇φ\| 0.993–1.004 | `validate_diagnostics.py` | `reference_diagnostics.csv` |
| 3.3 tolerance | ±1 frame changes pooled recall by ≤ 0.007 | `tf_birth_matching.py` (`recall_tol1`) | `tf_birth_matching*.csv` |
| 3.4, Table 1, Fig. 1 | nested site sets; delays; 1467 on-site births; hazard 0.61 / 0.75; merge share | `fig1_premise_audit.py`, `birth_predictability.py`, `reoccupation.py` | `birth_predictability.csv`, `reoccupation_by_delay.csv` |
| 3.4 | AUC 0.96 vs 0.79; 0.93 / 0.91 at fixed delay; merge AUC 0.94; φ-only / T-only 0.95; log loss 0.22; CV 0.955–0.964 | `timing_readability.py`, `horizon_readability.py` | `timing_readability*.csv` |
| 4.1, Fig. 2a | matched recall 0.2–2.0 % pooled, ≤ 3.3 % per simulation; control 20/26 | `tf_birth_matching.py` | `tf_birth_matching_baselines.csv`, `_control.csv` |
| 4.2, Fig. 2b | rollout ratio 0.00–0.08; U-Net one event in 1,432 | `rollout_diagnostics.py` | `model_diagnostics_*_minarea16.csv` (`nuc_total`, `ref_nuc_total`) |
| 4.3, Fig. 3 | frees 2.0–5.6 vs 7.2–7.9; reseeding 1–2 % vs 34–56 %; control 71 % | `site_freeing.py` | `site_freeing.csv` |
| 5.2, Table 2 | penalties; about ten times; 0.6–1 vs about 3 times the one-step error | `loss_weighting.py`, `fig4_projection_table2.py` | `loss_weighting.csv` |
| 5.3, Fig. 4 | \|∇φ\| 1.137 → 0.988; nucleation 0.071 → 0.040 | `rollout_diagnostics.py --project` | `*_projected.csv` |
| 5.4 | void ratio > 1 at 17 of 18; bubble counts and sizes | `rollout_diagnostics.py`, `cond_anatomy.py` | `model_diagnostics_716/813/814*.csv`, printed |
| 6.1–6.2, Fig. 5a | p\* median 0.90; ceiling 0.61 … 0.47 | `l2opt_ceiling.py`, `fig5_cause_fix.py` | `l2opt_ceiling*.csv` |
| 6.3 | probe AUC 0.869 / 0.877–0.878 vs output 0.789 | `representation_probe.py` | `representation_probe_735.csv` |
| 6.4–6.5, Table 3, Fig. 5b–c | recall / precision / rollout / validation loss of every training change; intervals; interface width of the whole-domain run | `tf_birth_matching.py`, `rollout_diagnostics.py`, `fig5_cause_fix.py`, `seed_response.py` | `tf_birth_matching*.csv`, `model_diagnostics_*.csv`, `results/runs/*/metrics.csv` |
| 6.6 | rollout reseeding 13–30 % / 6–17 %; one-step reseeding 87–98 % / 66–84 % of the reference | `site_freeing.py <runs>`, `tf_reseeding.py` | `site_freeing_rollouts.csv`, `tf_reseeding.csv` |
| 7, Fig. 6 | ρ = −0.10, +0.15, +0.15 (+0.02 baselines), −0.47; 12 models within 0.005 | `fig6_metrics_blind.py` | `d4_metrics_vs_nucleation.csv` |

## Conventions that matter when checking numbers

- **Frames:**
  - Evaluation uses raw frames 100–399 of the six evaluation simulations (Twall 75, 82, 90, 96, 102,
    107, i.e. 5–37 active sites). These are training simulations, which is deliberate (Section 3.1).
  - Arrays are `(t, y, x)` with row 0 at the heated wall, and dx = 1/32.
- **5-site simulation excluded from pooled one-step numbers:** pooled one-step numbers use the five
  simulations with 12–37 sites (Section 3.3).
- **One-step vs rollout:** "one step" means the model is given the true 5-frame window at every start
  frame. "Rollout" means it is fed its own predictions for 300 frames from true frames 95–99.
- **Two reference birth counts:** the one-step count (1,437 from 12 sites up) also scores the
  transition into frame 100, so it is 5 births higher than the rollout count (1,432).
- **Duplicate CSVs:** files named `model_diagnostics_<run>rf_*` are repeat rollouts that also saved
  the predicted fields. Their numbers are identical to `model_diagnostics_<run>_*`.
