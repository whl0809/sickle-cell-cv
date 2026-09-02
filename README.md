# Sickle Cell Video Analysis

Deep-learning pipeline for quantifying **sickling kinetics** of red blood cells (RBCs)
from time-lapse microscopy video. Each frame is segmented with
[Cellpose](https://github.com/MouseLand/cellpose), every detected cell is classified
into an RBC **morphological subtype** with a Vision Transformer (ViT), and a second
ViT / Siamese-ViT head decides whether that cell has **sickled (changed)** or is still
**unchanged**. The output is a per-video *sickled-fraction vs. time* curve that can be
compared against wet-lab kinetics measurements.

---

## Cell subtype labels

| ID | Label | Subtype |
|----|-------|---------|
| 0 | A | Discocyte |
| 1 | B | Cup-shape |
| 2 | C | Stomatocyte |
| 3 | D | Reticulocyte |
| 4 | E | Echinocyte |
| 5 | F | Granular |
| 6 | G | ISC (irreversibly sickled cell) |

Binary state per cell: `1 = changed/sickled`, `0 = unchanged`.
Current pipelines use the 7-class scheme (**A–G**); the archived ones in
[code/legacy/](code/legacy/) use the older 6-class scheme (**A–F**).

### Sickling degree

A second, independent axis. The subtype scheme above says *what shape* a cell is and the
binary state says *whether* it sickled; sickling degree asks **how far** a sickled cell
went, and has its own two-class scheme:

| Label | Folder in the dataset | Meaning |
|-------|----------------------|---------|
| `final_sickled` | `Sickled1-FinalSickled` | Fully sickled — the classic elongated crescent |
| `semi_sickled`  | `Sickled2-SemiSickled`  | Partially deformed; sickling started but did not complete |

This axis is produced by its own classifier and its own scripts. That work is
**unfinished** — see [Sickling degree](#sickling-degree--work-in-progress) below.

---

## Repository layout

```
<repo-root>/
├── code/
│   ├── pipelines/        current inference pipelines (all weights present)
│   ├── training/         sickling-degree classifier CLIs        [unfinished]
│   ├── legacy/           superseded pipelines (see note below)
│   └── notebooks/        training, analysis, and debugging notebooks
├── models/               ViT / Siamese / Cellpose weights        [not in git]
├── data/
│   ├── videos/           source .mp4 microscopy video            [not in git]
│   └── <datasets>/       cropped single-cell PNG datasets        [not in git]
├── runs/                 sickling-degree training scratch        [not in git]
├── results/              one folder per pipeline run
│   └── validation/       model validation metrics (ROC, accuracy, confusion)
├── reference/            measured kinetics workbook, ground-truth slides
├── docs/worklog.md       running log of findings and debugging notes
└── requirements.txt
```

---

## Pipelines

Three pipelines are current. Each takes videos in and writes a run folder out. A
separate, unfinished line of work classifies *sickling degree* instead of subtype — see
[Sickling degree](#sickling-degree--work-in-progress).

| Script | Classes | Purpose |
|--------|---------|---------|
| [pipeline_one_general_model.py](code/pipelines/pipeline_one_general_model.py) | 7 (A–G) | Single general subtype model + all-subtype Siamese change detector. Most recent. |
| [pipeline_isc_reti.py](code/pipelines/pipeline_isc_reti.py) | 7 (A–G) | Specialised heads for ISC (G) and reticulocytes (D). |
| [pipeline_seven_class.py](code/pipelines/pipeline_seven_class.py) | 7 (A–G) | 7-class classifier with per-subtype binary heads, no Siamese stage. |

### Usage

```bash
pip install -r requirements.txt

python code/pipelines/pipeline_one_general_model.py \
    -i data/videos/V1.mp4,data/videos/V2.mp4,data/videos/V3.mp4,data/videos/V4.mp4 \
    -o results/my_run_name \
    --frame_skip 2 \
    --max_frame 480
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-i, --inputs` | *required* | Comma-separated video paths |
| `-o, --output_dir` | *required* | Output directory (one subfolder per video) |
| `--frame_skip` | `2` | Process every Nth frame |
| `--max_frame` | `480` | Maximum frames to process |

Weights are resolved from `models/` relative to each script's own location, so the
pipelines can be launched from any working directory.

Acquisition is assumed to be **4 fps**, so plotted time = `frame_index * frame_skip / 4` s.

### Archived pipelines

[code/legacy/](code/legacy/) holds six earlier pipelines. They are kept for reference
but **cannot run as-is**: each one loads weight files that are not present in `models/`
— `direct_vit_C.pt`, `direct_vit_F.pt`,
`best_model_vit_torch_macos_raw_vit_large_binary_F.pth`, `siamese_vit_c_Haolin.pt`,
and `OG1_siamese_vit_c_Haolin.pt`. That missing-weight boundary is exactly what
separates `legacy/` from `pipelines/`.

| Script | Classes | Note |
|--------|---------|------|
| `pipeline_one_general_model_haolin.py` | 7 | Pre-refactor one-general-model version |
| `pipeline_isc_reti_haolin.py` | 7 | Pre-refactor ISC/reticulocyte version |
| `six_class_0811.py` | 6 | Main 6-class reference pipeline |
| `six_class_0811_macos.py` | 6 | macOS / MPS variant |
| `six_class_0811_cached_cellpose.py` | 6 | Reuses a cached Cellpose segmentation |
| `six_class_baseline.py` | 6 | Oldest baseline, CUDA only |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [train_subtype_vit.ipynb](code/notebooks/train_subtype_vit.ipynb) | Trains the ViT subtype classifier from one-folder-per-class crops. |
| [train_siamese_change.ipynb](code/notebooks/train_siamese_change.ipynb) | Trains the `SiameseViTChange` sickling detector; writes the metrics in `results/validation/`. |
| [train_siamese_change_backup.ipynb](code/notebooks/train_siamese_change_backup.ipynb) | Earlier snapshot of the Siamese trainer (subtype G). |
| [analyze_polymerization_mae.ipynb](code/notebooks/analyze_polymerization_mae.ipynb) | Compares model curves against `reference/kinetics-seven_Jianlu.xlsx`; matching-timepoint MAE, no interpolation. |
| [debug_cellpose.ipynb](code/notebooks/debug_cellpose.ipynb) | Segmentation debugging — downscaling, cell diameter, `remove_edge_masks`. |
| [prepare_dataset.ipynb](code/notebooks/prepare_dataset.ipynb) | Reorganises the paired image dataset into per-class folders. |

Each notebook opens with a small anchor cell that walks up from the working directory
until it finds `requirements.txt`, then defines `REPO_ROOT`, `DATA_DIR`, `MODELS_DIR`,
`RESULTS_DIR`, `REFERENCE_DIR`, `VALIDATION_DIR`, and `CKPT_DIR`. Run that cell first;
every path in the notebook is built from those, so it does not matter whether Jupyter
was started in `code/notebooks/` or at the repository root. Validation metrics and
figures are written to `results/validation/`.

### Where training output goes

Training **never writes into `models/` directly.** Every `torch.save` in the training
notebooks resolves through `CKPT_DIR`, which is `models/rbc_ckpts/`:

```
models/                  released weights -- what the pipelines load. Read-only to training.
└── rbc_ckpts/           staging area -- everything a training run produces lands here.
```

This exists to stop a training run from clobbering a checkpoint a pipeline depends on.
The hazard was real: the Siamese trainer's best-checkpoint path was
`siamese_vit_ISC_Haolin.pt`, the exact file
[pipeline_isc_reti.py](code/pipelines/pipeline_isc_reti.py) loads, so re-running training
would have silently replaced it.

Promoting a staged checkpoint is therefore a deliberate step — inspect it, then move it
up into `models/` under a name no released weight already uses:

```bash
mv models/rbc_ckpts/siamese_vit_ISC_epoch10.pt models/siamese_vit_ISC_v2.pt
```

Checkpoint *reads* (the `CKPT_PATH` config lines in the evaluation cells) still point at
`models/`, since those cells evaluate released weights. Point one at `CKPT_DIR` when you
want to evaluate something you just trained.

---

---

## Sickling degree — work in progress

> **Status: unfinished, exploratory.** This is an active line of work folded into the
> repo so it stops living in a loose directory, not a finished pipeline. It has not been
> validated against the wet-lab kinetics workbook the way the subtype pipelines have, the
> three scripts are successive attempts rather than three supported options, and the
> unfinished parts are listed at the end of this section. Read the numbers below as
> where things stand, not as results.

`models/semi_final_classifier.pt` is a ConvNeXt-Tiny that splits already-sickled cells
into `semi_sickled` and `final_sickled`. Three scripts in `code/pipelines/` wrap it. They
share the same front half as the subtype pipelines — Cellpose segmentation, tracking, and
the all-subtype Siamese head that decides when a cell has sickled.

They are a **lineage, not a menu**: each was written to fix the previous one, and
`pipeline_semi_final_sickling_time.py` contains every function in
`pipeline_semi_final_endpoint.py` plus ten more. Start from the last one.

| Script | Order | What it added | Degree label |
|--------|-------|---------------|--------------|
| `pipeline_semi_final_detection.py` | first | Degree classification at all, on top of the A–G subtype outputs | Re-decided every frame, so it flickers near the boundary |
| `pipeline_semi_final_endpoint.py` | second | Collapsed the flicker into one label per cell | Averaged over the last `--endpoint_subtype_frames` sickled frames |
| `pipeline_semi_final_sickling_time.py` | current | Per-cell sickling onset and completion time, plus stability scoring | Same endpoint label as above |

```bash
python code/pipelines/pipeline_semi_final_sickling_time.py \
    -i data/videos/V1.mp4 -o results/<run-name> --frame_skip 2 --max_frame 480
```

Comma-separate `-i` for several videos, as with the subtype pipelines.

### Flags beyond `--frame_skip` / `--max_frame`

Endpoint and sickling-time only. **The defaults are not tuned** — `--sickling_ema 1.0`
disables smoothing and `--sickling_min_persist 1` disables persistence, i.e. the two
flicker-suppression mechanisms the subtype runs in `results/` lean on heavily are
present but switched off.

| Flag | Default | Meaning |
|------|---------|---------|
| `--endpoint_subtype_frames` | `5` | Average degree probabilities over the last K sickled observations |
| `--sickling_threshold` | `0.73` | Siamese score above which a cell counts as sickled |
| `--sickling_min_persist` | `1` | Consecutive above-threshold frames required |
| `--sickling_ema` | `1.0` | EMA on the Siamese score; `1.0` disables smoothing |
| `--completion_stable_frames` | `6` | Consecutive low-change frames that mark sickling complete |
| `--completion_change_threshold` | `0.05` | Maximum change score still considered stable |
| `--completion_score_mode` | `combined` | `combined`, `probability`, or `all` |
| `--completion_min_duration_sec` | `0` | Minimum time after onset before completion may be reported |

On top of the standard run-folder contents these write
`endpoint_semi_final_sickled_fraction.{csv,png}` (per-frame semi/final/total sickled %),
and the sickling-time script adds `sickling_time_report.csv` (per-cell onset and
completion), `sickling_time_summary.csv`, and `stability_score_*.csv`. The first script
writes `semi_final_sickled_fraction.*` instead, having no endpoint stage.

### The classifier

[code/training/](code/training/) is the CLI toolkit that produced the checkpoint:
`train_vit.py` (fine-tunes `vit_b_16`, `convnext_tiny`, `efficientnet_b0/b3`, `resnet50`,
or `swin_t`, and doubles as the shared library the pipelines import),
`evaluate_vit.py`, `ensemble_evaluate.py`, `tune_threshold.py`,
`export_misclassifications.py`, `train_morphology_features.py`, `predict_vit.py`, and
`move_bad_images.py`.

```bash
python code/training/train_vit.py --model convnext_tiny --image-size 320 \
    --loss focal --augmentation conservative --epochs 20
python code/training/evaluate_vit.py --run-dir runs/<run-name> --split test
```

Splits are grouped by source acquisition (`--split-by group`), so crops from one video
never straddle train and test. `splits.csv` regenerates deterministically from `--seed`
and is not tracked. Training writes to `runs/<model>_sickling_degree_<timestamp>/`, which
is untracked — the same read-only-`models/` policy as `models/rbc_ckpts/`. The current
checkpoint was promoted by hand and is byte-identical to
`runs/convnext_tiny_sickling_degree_20260505_215821/best_model.pt`.

### Where things stand

Held-out test split, 1,686 crops. Full per-run metrics in
[results/validation/sickling_degree/](results/validation/sickling_degree/).

| Approach | Accuracy | Macro F1 |
|----------|---------:|---------:|
| 3-model ensemble — convnext_tiny + vit_b_16 + efficientnet_b3 | 0.9100 | 0.9074 |
| 2-model ensemble — convnext_tiny + vit_b_16 | 0.9077 | 0.9051 |
| `convnext_tiny` @320, focal (run `..._163854`) | 0.9071 | 0.9040 |
| Morphology + CNN-probability hybrid | 0.9047 | 0.9023 |
| `convnext_tiny` @320, focal (run `..._215821`) — the promoted checkpoint | 0.9051 | 0.9012 |
| `vit_b_16` @224, cross-entropy | 0.8983 | 0.8956 |
| `efficientnet_b3` @300, cross-entropy | 0.8878 | 0.8841 |
| 21 morphology features alone, no CNN | 0.7972 | 0.7938 |

Two findings that should save someone a repeat: **morphology features are a dead end
here** — 21 hand-computed shape and intensity descriptors land 11 points behind the CNN,
and adding them on top of the CNN probabilities does not beat the CNN alone, so the CNN
has already captured what they encode. And **threshold tuning is a no-op** — the
validation optimum for both accuracy and macro F1 is 0.491, so argmax is fine.

Also note the two `convnext_tiny` runs are easy to confuse: `..._163854` is what every
ensemble, threshold, and morphology experiment scores against, but `..._215821` is the
one promoted to `semi_final_classifier.pt`. The 0.0028 macro F1 gap is inside noise; the
checkpoints are different files.

### What is unfinished

- **No validation against ground truth.** Nothing here is compared against
  `reference/kinetics-seven_Jianlu.xlsx`, which is how the subtype pipelines are
  judged. Until that happens the sickling *times* are unverified output, not a result.
- **Two of four videos.** V1 has been run through everything and V2 once; V3 and V4
  never. No pooled multi-video summary exists.
- **The temporal parameters are untuned**, per the flag table above. Whether the
  endpoint labels are stable under sensible EMA and persistence settings is untested.
- **The best result is not wired in.** The 3-model ensemble leads, but no pipeline uses
  it — it would mean three backbones per frame for +0.003 macro F1. Unresolved, not
  decided.
- **`pipeline_semi_final_endpoint.py` is probably redundant** now that the sickling-time
  script is a strict superset of it. It has not been retired because nothing has
  confirmed the two agree.
- **No validation figures.** `results/validation/` has ROC curves, confusion matrices,
  and per-cell-type accuracy plots for the subtype models; the sickling-degree subfolder
  has CSV and JSON only.
- Three training runs (`vit_sickling_degree_20260503_19{2107,3546,4817}`) died after
  writing `splits.csv` and were left in `runs/`.

## Run configurations

Each folder in `results/` is one experiment; the name encodes the key settings.
Temporal smoothing (EMA on the pair score) plus a *K-in-a-row* confirmation rule
suppress per-frame classification flicker.

| Folder | Configuration |
|--------|---------------|
| `AllSubtypes_OneModel` | Single general 7-class model |
| `RET&ISC_2inrow_EMA0.5` | Reticulocyte + ISC pipeline, 2-in-a-row, EMA 0.5 |
| `With3inarow_tr0.8` | 3-consecutive-frame confirmation, threshold 0.8 (per-video outputs only, no pooled summary) |
| `NMS_3inRow_tr0.7_Reupdate_EMA0.2` | 3-in-a-row, thr 0.7, EMA 0.2, re-update |
| `MS_2(3)inRow_tr0.7_EMA0.2_OGmodel` | 2/3-in-a-row, thr 0.7, EMA 0.2, original model (subfolders `V*_MS`) |
| `newly_trained_model_withD` | Retrained model including subtype D |
| `newly_trained_EMA0.5_withD` | Retrained, EMA 0.5, with D |
| `newlytrained_2inRow_EMA0.5_withD` | Retrained, 2-in-a-row, EMA 0.5, with D |

The `SemiFinal_*` folders are from the unfinished sickling-degree work rather than
tuned experiments — single-video smoke runs at default settings, kept because they are
the only recorded output of those scripts. Four of the five are V1.

| Folder | Pipeline | Note |
|--------|----------|------|
| `SemiFinal_perframe_V1` | `pipeline_semi_final_detection` | Earliest script; per-frame label |
| `SemiFinal_endpoint_V1` | `pipeline_semi_final_endpoint` | Second script |
| `SemiFinal_endpoint_time_V1` | `pipeline_semi_final_sickling_time` | Current script |
| `SemiFinal_endpoint_time_V2` | `pipeline_semi_final_sickling_time` | Only non-V1 run |
| `SemiFinal_endpoint_time_probonly_V1` | `pipeline_semi_final_sickling_time` | `--completion_score_mode probability` |

### What a run folder contains

```
results/<run-name>/
├── V1../V4/                            per-video outputs (V1_MS../V4_MS in the MS run)
│   ├── state_ratio_report.csv          per-frame, per-class sickled counts
│   ├── state_ratio_plot_binary.csv     sickled fraction (%) vs. time
│   ├── state_ratio_plot*.png           per-class / overall curves
│   ├── frame0_class_pie.png            subtype composition at t=0
│   ├── masks_{BEFORE,AFTER}_remove_edge_cells.png
│   └── df.pkl                          tidy per-frame dataframe
├── state_ratio_report.csv              pooled across V1–V4
├── state_ratio_plot_binary.{csv,png}
└── combined_{state_ratio_plot,frame0_class_pie}.png
```

`results/validation/` holds classifier validation metrics that are not tied to one
video run: ROC curves and AUCs, per-cell-type accuracy, confusion matrices, raw
validation predictions, and the MGH2133 polymerized-fraction comparison.

---

## Data not in this repository

`.gitignore` excludes raw input media, model weights, and anything the pipeline
regenerates. To reproduce a run you need these locally:

- **`models/`** — ~4.3 GB of weights. Required to run anything:
  - `cyto3_train0327` — fine-tuned Cellpose segmentation model
  - `best_model_vit_torch_macos_seven.pth` — 7-class subtype ViT
  - `best_model_vit_torch_macos_raw_vit_large_binary.pth` — binary change ViT
  - `best_model_vit_torch_macos_raw_vit_large_binary_G.pth`, `direct_vit_{D,E,G}.pt` — per-subtype heads
  - `siamese_vit_{All,ISC,Reti}_Haolin.pt`, `OG_siamese_vit_c_Haolin.pt` — Siamese change detectors
  - `semi_final_classifier.pt` — ConvNeXt-Tiny sickling-degree classifier (semi vs. final)
  - `rbc_ckpts/best_vit.pth` — ViT baseline from `train_subtype_vit.ipynb`
- **`data/videos/`** — `V1.mp4` … `V4.mp4` (~380 MB each).
- **`data/`** image datasets, needed only for *training*:
  `Alldataset_for_subtypeclassification/`, `Paired dataset_10122025/`,
  `C dataset/`, `D_Reticulocyte/` (~80,000 cropped single-cell PNGs), and
  `semi_final_degree/` — 11,154 crops for the sickling-degree classifier,
  4,663 `Sickled1-FinalSickled` and 6,491 `Sickled2-SemiSickled` (~565 MB).
  `semi_final_degree_bad_images/` holds the 7 crops `move_bad_images.py` quarantined
  as truncated.
- **Per-run intermediates**, recreated automatically by the pipeline:
  `V*/V*.avi` (annotated video, up to 1.9 GB), `cell_info.pkl` (~145–190 MB),
  `first_frame.png` (~36 MB), `masks_remove_edge_cells.npy` (~6 MB).
- **`runs/`** — sickling-degree training scratch (~2.1 GB): per-run `best_model.pt` /
  `last_model.pt`, `splits.csv`, morphology feature caches, and exported misclassified
  crops. The metrics worth keeping were copied to
  `results/validation/sickling_degree/`.

Tracked content is code, notebooks, every `state_ratio_*.csv`, the summary plots,
`df.pkl`, and the reference workbooks — about 25 MB in total.

Since the weights are not in git, a fresh clone cannot run until `models/` is populated
out of band (shared drive, GitHub Release assets, or Hugging Face Hub).

---

## Reorganisation notes

The repository was restructured after the initial import (commit `810c9f8`), which
holds the original flat layout if anything needs recovering. Changes:

- Pipelines split into `code/pipelines/` (weights present) and `code/legacy/`
  (weights missing), replacing names like
  `cell_video_classify_multiple_add_comp_filt_windows_latest_0917_seven_colors_by_herui.py`.
- Model weight paths in the scripts now resolve via `MODELS_DIR`, anchored with
  `Path(__file__).resolve().parents[2]`, instead of bare filenames that required running
  from inside `code/`.
- Notebook paths were retargeted from stale absolute paths pointing at an earlier local
  working directory to a `REPO_ROOT` anchor, so datasets, weights, and outputs resolve from any working
  directory. Outputs that previously landed in whatever directory Jupyter happened to be
  started in (checkpoints, ROC/accuracy/confusion figures and CSVs) now go to
  `models/rbc_ckpts/` and `results/validation/`.
- Neither the scripts nor the notebooks depend on the name of the top-level folder, so
  the repository can be cloned or renamed freely.
- `cell_video_classify_multiple_add_comp_filt_windows_latest_0917_seven_colors_by_herui.py`
  was deleted as a byte-identical duplicate of `pipeline_OneGeneralModel_Haolin.py`.
- A 5 MB stored file-copy log was cleared from `prepare_dataset.ipynb`.
- `code/README.md` became `docs/worklog.md`; `code/.gitignore` was folded into the root one.

Code inside the scripts was not otherwise refactored — the pipeline logic is unchanged.

### Sickling-degree branch

The sickling-degree work arrived as a second flat working directory and was folded in on
the same conventions:

- The three `pipeline_semi_final_*` scripts moved to `code/pipelines/`, and the training
  and evaluation CLIs to a new `code/training/`. Weight lookups went from bare filenames
  resolved against the script's own directory (`script_path()` / `bundle_path()`) to
  `MODELS_DIR`, anchored the same way as the subtype pipelines. `train_vit.py` is
  imported by the pipelines, so each one puts `code/training/` on `sys.path` from the
  same repo-root anchor.
- `pipeline_semi_final_detection.py` loaded its classifier straight out of
  `runs/convnext_tiny_sickling_degree_20260505_215821/best_model.pt`, i.e. a depended-on
  weight sitting in untracked training scratch. It now loads
  `models/semi_final_classifier.pt`, which is byte-identical to that checkpoint.
- `train_vit.py` and `move_bad_images.py` defaulted `--data-dir` to `7. SemiSickled`
  relative to the working directory. That dataset is now `data/semi_final_degree/`,
  resolved from the repo root; `--bad-dir` likewise.
- `pipeline_OneGeneralModel_kaiyu_Haolin.py` was dropped as a duplicate of
  `code/pipelines/pipeline_one_general_model.py` — the only differences were blank lines,
  commented-out weight paths, and the older `bundle_path()` scheme. Its one real
  improvement, a `ViTFeatureExtractor` → `ViTImageProcessor` fallback for
  transformers ≥ 4.41, was ported onto the three current subtype pipelines, which had
  been importing a name recent releases no longer export.
- `pipeline_OneGeneralModel_kaiyu_Haolin_pack/` — a sidecar folder holding five weights
  the pipeline loaded from its own directory — was dissolved into `models/`. Its
  `requirements.txt` was a subset of the root one and its `PACK_MANIFEST.md` described a
  layout that no longer exists; both were removed.
- `README_pipeline_semi_final_endpoint.md` was folded into this file.
- `pipeline_semi_final_detection_package.zip` (452 MB) — a hand-built bundle for handing
  the endpoint pipeline to a collaborator — was deleted. Its three weights were
  byte-identical to the copies now under `models/`, its two scripts differed only in the
  weight-path scheme replaced above, and its README is folded into this file. `.gitignore`
  still excludes `/dist/` and `*.zip` so a rebuilt bundle stays out of git.
- 11,177 `*Zone.Identifier` files — one per file, written by Windows/WSL when the tree
  was downloaded — were deleted, and the pattern added to `.gitignore`.
- The five output folders became `results/SemiFinal_*`. Under the existing ignore rules
  this tracks the CSV reports, the sickled-fraction plots, and `df.pkl`, and drops the
  annotated `.avi`, `cell_info.pkl`, `first_frame.png`, and the mask arrays — about
  3 MB tracked out of 7.7 GB on disk.
- `runs/` stays untracked. Per-run `classes.json`, `metrics.csv`, `test_metrics.json`,
  and the threshold / ensemble / morphology results were copied to
  `results/validation/sickling_degree/`, with `model_comparison.csv` generated across
  them. Three aborted runs that wrote only a `splits.csv` before dying were left in the
  scratch directory.
