# Learned pool-boiling forecasters and nucleation

Code and results for the paper *"Why learned pool-boiling forecasters do not nucleate, and a loss term
that makes them"* (Hai Do; journal and DOI to be added on publication).

The repository contains:
- the evaluation protocol (nucleation events matched to the reference simulation, one step ahead and in
  rollout);
- the premise audit of the BubbleML pool-boiling data;
- the training-loss wrappers, including the wall-band phase-indicator term;
- every per-run result file behind the paper's figures and tables.

The forecasters themselves are the published Bubbleformer models (upstream code, unmodified).

## Layout

| Path | Contents |
|---|---|
| `src/` | Training-loss wrappers around upstream Bubbleformer: the phase-indicator loss (`indicator_loss.py`, `train_indicator.py`), wall-band upweighting, the timing-tolerant loss, first-frame-only training, the local wall path and pushforward training |
| `scripts/` | Diagnostics (`diagnostics.py`), evaluation (`rollout_diagnostics.py`, `tf_birth_matching.py`), the premise audit, the analyses, and one script per figure (`fig*.py`) |
| `configs/data_cfg/` | Data configs, copied into the upstream config tree by `setup/get_bubbleformer.sh` |
| `configs/runs.csv` | Every trained model in the paper: architecture, config, seed, loss and its settings |
| `results/diagnostics/` | Per-run and reference measurements (CSV), the inputs of every figure and table |
| `results/runs/<run>/` | Training curves (`metrics.csv`, including validation loss) and the resolved Hydra config (`hparams.yaml`) |
| `figures/`, `tables/` | Outputs of the `fig*.py` scripts |
| `bin/` | `train.sh` (one training run), `evaluate_run.sh` (all evaluations of one run), `run_python.sh` |
| `setup/` | Conda environment and upstream checkout |

Run ids (735, 1089, ...) are the labels of the 24 trained models. `configs/runs.csv` gives the
paper's name for each.

## Setup

```bash
setup/environment.sh nucleation && conda activate nucleation   # python 3.10, torch 2.5.0 (CUDA 12.4)
setup/get_bubbleformer.sh                                      # upstream Bubbleformer at commit 8d1dade, + our data configs
export BUBBLEML_ROOT=/path/to/bubbleml2                        # data (default: ./data/bubbleml2)
export WORK_ROOT=/path/to/work                                 # checkpoints and saved predictions (default: ./work)
```

Data: BubbleML 2.0 on Hugging Face (`hpcforge/BubbleML_2`, DOI 10.57967/hf/5594, CC BY-NC-SA 4.0).
Download it with `scripts/pull_study.py`, for example `python scripts/pull_study.py
PoolBoiling-Saturated-FC72-2D`.

## Regenerate the figures and tables (minutes, no GPU)

The figure scripts read the committed CSVs. Only the simulations' JSON metadata needs downloading
(a few MB):

```bash
python scripts/pull_study.py PoolBoiling-Saturated-FC72-2D --meta-only --yes
for f in fig1_premise_audit fig2_3_failure fig4_projection_table2 fig5_cause_fix fig6_metrics_blind; do
    python scripts/$f.py; done
```

The outputs go to `figures/` and `tables/`. They are pixel-identical to the published figures.

## Reproduce the results from scratch

See [REPRODUCE.md](REPRODUCE.md), which covers:
- data;
- training the 24 models (about 16 GPU-hours each);
- evaluation;
- the analyses;
- which script produces which number in the paper.

## Licence

- Code: MIT (see `LICENSE`).
- Trained checkpoints (on Zenodo, doi:10.5281/zenodo.23299635): CC BY-NC-SA 4.0, matching the licence of the training data.
- Upstream Bubbleformer: MIT. It is not included here and is fetched by `setup/get_bubbleformer.sh`.
- BubbleML data: CC BY-NC-SA 4.0. The repository includes derived measurements only, plus two frames
  used in Fig. 5a (`results/diagnostics/fig5a_frames.npz`).

## Trained models

The 24 trained models are on Zenodo under CC BY-NC-SA 4.0: [doi:10.5281/zenodo.23299635](https://doi.org/10.5281/zenodo.23299635). They contain weights and training configuration
only (`scripts/export_checkpoints.py`), and `manifest.csv` lists each file with its SHA-256. Place
them under `$WORK_ROOT/checkpoints/` and evaluate any of them with
`bin/evaluate_run.sh <run> $WORK_ROOT/checkpoints/run<run>.ckpt`.

## Citation

[To be added on publication.]
