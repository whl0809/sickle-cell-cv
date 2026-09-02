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

---

## Repository layout

```
<repo-root>/
├── code/
│   ├── pipelines/        current inference pipelines (all weights present)
│   ├── legacy/           superseded pipelines (see note below)
│   └── notebooks/        training, analysis, and debugging notebooks
├── models/               ViT / Siamese / Cellpose weights        [not in git]
├── data/
│   ├── videos/           source .mp4 microscopy video            [not in git]
│   └── <datasets>/       cropped single-cell PNG datasets        [not in git]
├── results/              one folder per pipeline run
│   └── validation/       model validation metrics (ROC, accuracy, confusion)
├── reference/            measured kinetics workbook, ground-truth slides
├── docs/worklog.md       running log of findings and debugging notes
└── requirements.txt
```

---

## Pipelines

Three pipelines are current. Each takes videos in and writes a run folder out.

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
  - `rbc_ckpts/best_vit.pth` — ViT baseline from `train_subtype_vit.ipynb`
- **`data/videos/`** — `V1.mp4` … `V4.mp4` (~380 MB each).
- **`data/`** image datasets, needed only for *training*:
  `Alldataset_for_subtypeclassification/`, `Paired dataset_10122025/`,
  `C dataset/`, `D_Reticulocyte/` (~80,000 cropped single-cell PNGs).
- **Per-run intermediates**, recreated automatically by the pipeline:
  `V*/V*.avi` (annotated video, up to 1.6 GB), `cell_info.pkl` (~145 MB),
  `first_frame.png` (~36 MB), `masks_remove_edge_cells.npy` (~6 MB).

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
