# Project provenance and preparation notes

## Original work

The source is a University of Bristol physics undergraduate coursework notebook for Advanced Computational Physics & Machine Learning. The final report was generated on 23 April 2026. The original notebook and report remain unchanged on the author's computer.

The archived notebook records completed training runs for ViT-B/16, Swin-T and EfficientNet-V2-S, each with 15 epochs, followed by weighted five-view TTA evaluation. Metrics in this repository are transcribed from saved original outputs, not results of a fresh reproduction run.

## Public-copy changes

- Replaced workstation-specific data and checkpoint paths with repository-relative paths and `GALAXY_DATA_DIR`.
- Cleared execution outputs and private notebook metadata. Dataset files, checkpoints, download credentials, assignment material and the original report are not uploaded.
- Made unreadable images fail explicitly instead of silently dropping them, which could otherwise invalidate denominators and failure-example indexing.
- Removed a synthetic ensemble convergence curve; there was no recorded ensemble evaluation for every epoch. Recorded single-model training losses remain available as data.
- Removed direct comparisons between this local validation split and a historical Kaggle private leaderboard.
- Calculated the previously missing per-task single-model RMSE arrays from the collected predictions before plotting them.
- Removed unsupported qualitative error thresholds and assertions that aggregate output sums prove probability consistency.
- Corrected the empirical CDF to run from 1/n to 1.

The original model choices, optimizer settings, losses, split seed, 15-epoch setting, TTA transformations and ensemble weights are retained. No claim is made that these settings are optimal.

## Interpretation and limitations

`GZ2DecisionTreeModel` is the legacy class name. Its implementation is a backbone followed by 11 independent linear heads whose outputs are concatenated. It does not implement a hard probabilistic decision tree or guarantee bounded, normalized output probabilities. Aggregate diagnostics are descriptive, not a proof of hierarchical consistency.

The reported 88.28% accuracy concerns only the argmax of the first three outputs against the argmax of their volunteer-label proportions. The primary 37-output task is regression, so this number is not a general 37-class accuracy.

The split is a local random 80/20 split. Model selection and evaluation use that validation split; there is no independent held-out test evaluation in this archive. Results cannot establish an official competition ranking or an unbiased estimate after hyperparameter selection.

The original report discloses generative-AI assistance for parts of visualization code, TTA tensor operations and writing refinement/translation. The public copy does not present AI-assisted portions as unaided work.

## Checks during preparation

The public copy is intended to receive JSON/notebook-format and Python syntax checks, relative-path checks, and a scan for workstation paths or credentials before upload. Full GPU training is not rerun as part of packaging.
