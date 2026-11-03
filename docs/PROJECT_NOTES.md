# Project provenance and repository history

## Coursework experiment

The source is an undergraduate physics coursework project for Advanced Computational Physics & Machine Learning at the University of Bristol. The final report was generated on 23 April 2026.

The saved notebook contains completed 15-epoch runs for ViT-B/16, Swin-T and EfficientNet-V2-S, followed by weighted five-view TTA evaluation. `results/training_history.json` was parsed from the saved training output; `results/validation_metrics.json` records the saved local validation metrics. Full GPU training has not been rerun as part of preparing this repository.

## Public reference notebook

`notebooks/galaxy_morphology.ipynb` retains the coursework workflow as a cleaned reference. Local paths and execution output were removed. Unreadable images now fail explicitly, missing task-comparison arrays were filled from collected predictions, and the empirical CDF ends at one.

A synthetic ensemble convergence curve was removed because the archive contained only a final ensemble evaluation, not an ensemble score at every epoch. Direct comparisons with a historical private competition leaderboard were also removed because the evaluation datasets differ.

## November 2026 refactor

The `src/galaxy_morphology` package adds a configurable CLI, strict input checks, self-describing checkpoints, same-split ensemble validation, legacy-checkpoint import, and offline tests. It also normalizes gradient accumulation by the actual sample count in the current accumulation group, including a short final group.

The CLI is the maintained entry point. Its new runs write to `runs/`, while `results/` remains an archive of the April experiment. Model architecture and the main baseline settings are retained, but the refactored implementation is not claimed to reproduce archived numbers bit for bit.

## Authorship and assistance

The original report discloses generative-AI assistance for parts of visualization code, TTA tensor operations and writing refinement/translation. The repository packaging, CLI refactor, tests and documentation were also prepared with AI assistance in November 2026. These additions are distinct from the original coursework submission.

The desktop coursework files remain the source archive. Dataset images, model checkpoints, credentials and the assignment handout are not included in this repository.

## Evaluation interpretation

The historical 0.07418 RMSE belongs to a local random 80/20 split used for checkpoint selection and model comparison. It is not an independently held-out test result or an official Kaggle score. The 88.28% root accuracy only compares argmax values of the first three output dimensions.

The legacy class name `GZ2DecisionTreeModel` in the reference notebook describes independent linear task heads, not a hard probabilistic tree. The refactored class is named `MultiTaskRegressor`. Metric definitions and method details are documented in [METHOD.md](METHOD.md).
