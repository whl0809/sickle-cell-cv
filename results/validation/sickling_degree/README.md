# Sickling-degree classifier validation

> **Status: unfinished.** These are the metrics an exploratory line of work happened to
> produce, not a validation suite. Nothing here is compared against
> `reference/kinetics-seven_Jianlu.xlsx`, and unlike the parent `results/validation/`
> there are no ROC, confusion, or accuracy figures — CSV and JSON only.

Metrics for the `semi_sickled` vs `final_sickled` classifier, copied out of the untracked
`runs/` training scratch. Held-out test split: 1,686 crops (706 `final_sickled`,
980 `semi_sickled`), grouped by source acquisition so no video straddles train and test.

`model_comparison.csv` is generated from the `test_metrics.json` and `classes.json` files
below; regenerate it if runs are added.

## Per run

| File | Contents |
|------|----------|
| `classes.json` | Class order and the training config: model, image size, augmentation, loss |
| `metrics.csv` | Per-epoch train/val loss, accuracy, macro F1, and learning rates |
| `test_metrics.json` | Test accuracy, macro F1, per-class precision/recall/F1, confusion matrix |
| `misclassified_test_predictions.csv` | Per-crop prediction and probability for the whole test split |

## `convnext_tiny_..._163854` extras

This run is the analysis baseline — the ensemble, threshold, and morphology experiments
all score against its `best_model.pt`.

| Path | Contents |
|------|----------|
| `threshold_tuning/*.json` | Chosen `final_sickled` threshold per objective. Accuracy and macro F1 both land at 0.491, so argmax is fine; optimising `positive_recall` collapses it to 0.071. |
| `ensemble/test_ensemble_metrics.json` | convnext_tiny + vit_b_16 + efficientnet_b3, probability-averaged: the best result on this dataset |
| `ensemble_convnext_vit/test_ensemble_metrics.json` | convnext_tiny + vit_b_16 only |
| `morphology_features/` | 21 hand-computed shape and intensity features, no CNN |
| `morphology_hybrid_convnext/` | Those features concatenated with the CNN probabilities |

## Not kept

`best_model.pt` / `last_model.pt` (~110–660 MB per run), `splits.csv` (regenerated
deterministically from `--seed`), the exported misclassified PNGs, the threshold and
ensemble per-crop CSVs, and the morphology `*_cache.npz` files all stay in `runs/`.
Regenerate any of them with the CLIs in `code/training/`.
