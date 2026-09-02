# DaoLab — Sickle Cell Video Analysis

Deep-learning pipeline for quantifying **sickling kinetics** of red blood cells (RBCs)
from time-lapse microscopy video. Each frame is segmented with
[Cellpose](https://github.com/MouseLand/cellpose), every detected cell is classified
into an RBC **morphological subtype** with a Vision Transformer (ViT), and a second
ViT / Siamese-ViT head decides whether that cell has **sickled (changed)** or is still
**unchanged**. The result is a per-video *sickled-fraction vs. time* curve that can be
compared against wet-lab kinetics measurements.

> This is the **initial import** of the working directory as-is. Scripts still contain
> hard-coded paths and duplicated variants from different contributors; cleanup is
> intentionally deferred to a later commit so this snapshot stays faithful.

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

Earlier pipelines use the 6-class scheme (**A–F**); the newer
`pipeline_OneGeneralModel_*` scripts use the full 7-class scheme (**A–G**).

---

## Repository layout

```
DaoLab/
├── code/                       # all pipelines, training notebooks, analysis
└── <run-name>/                 # one folder per pipeline run / model config
    ├── V1../V4/                # per-video outputs (V1_MS../V4_MS in the MS run)
    │   ├── state_ratio_report.csv        # per-frame, per-class sickled counts
    │   ├── state_ratio_plot_binary.csv   # sickled fraction (%) vs. time
    │   ├── state_ratio_plot*.png         # per-class / overall curves
    │   ├── frame0_class_pie.png          # subtype composition at t=0
    │   ├── masks_{BEFORE,AFTER}_remove_edge_cells.png
    │   └── df.pkl                        # tidy per-frame dataframe
    ├── state_ratio_report.csv            # pooled across V1–V4
    ├── state_ratio_plot_binary.{csv,png}
    └── combined_{state_ratio_plot,frame0_class_pie}.png
```

### `code/` — pipelines

| File | Purpose |
|------|---------|
| `cell_video_classify_multiple_add_comp_filt_windows_latest_0811.py` | Main 6-class (A–F) pipeline, Windows/CUDA. Current reference version. |
| `..._latest_0811_WORKONMAC.py` | Same pipeline adapted for macOS / MPS. |
| `..._latest_0811_loadcellpose.py` | Variant that reuses a cached Cellpose segmentation. |
| `..._latest_0917_seven_colors_by_herui.py` | 7-class variant with subtype colour overlay (Herui). |
| `cell_video_classify_multiple_add_filt_windows_latest_0920_seven_by_kaiyu.py` | 7-class variant, refactored by Kaiyu. |
| `pipeline_OneGeneralModel_Haolin.py` | Single general 7-class model for all subtypes. |
| `pipeline_OneGeneralModel_kaiyu_Haolin.py` | Latest one-general-model pipeline (Kaiyu + Haolin). |
| `pipeline_ISC_Reti_Haolin.py` | Pipeline specialised for ISC (G) and reticulocytes (D). |
| `pipeline_ISC_Reti_kaiyu_Haolin.py` | Refactored ISC/Reticulocyte pipeline. |
| `cell_video_classify_multiple_add_comp_filt_windows.py` | Older baseline, kept for reference. |

### `code/` — notebooks

| Notebook | Purpose |
|----------|---------|
| `rbc_vit_classifier.ipynb` | Trains the ViT subtype classifier from one-folder-per-class crops. |
| `vit_train_F_Haolin.ipynb` | Trains the **Siamese ViT** change detector (`SiameseViTChange`). |
| `vit_train_Haolin_backup.ipynb` | Backup of the Siamese training notebook. |
| `RBC_polymerization_MAE_matching_only.ipynb` | Compares model curves against `kinetics-seven_Jianlu.xlsx`; matching-timepoint MAE, no interpolation. |
| `Cellpose_debug.ipynb` | Segmentation debugging — downscaling, diameter, `remove_edge_masks`. |
| `copyDataset.ipynb` | Reorganises the paired image dataset into per-class folders. |

`code/README.md` holds the running **work log** (findings, debugging notes, ffmpeg
recipes) and is worth reading alongside this file.

---

## Run configurations

Each top-level output folder is one experiment. The name encodes the key settings:

| Folder | Configuration |
|--------|---------------|
| `With3inarow_tr0.8` | 3-consecutive-frame confirmation, threshold 0.8 (per-video outputs only, no pooled summary) |
| `NMS_3inRow_tr0.7_Reupdate_EMA0.2` | 3-in-a-row, thr 0.7, EMA smoothing 0.2, re-update |
| `MS_2(3)inRow_tr0.7_EMA0.2_OGmodel` | 2/3-in-a-row, thr 0.7, EMA 0.2, original model (subfolders `V*_MS`) |
| `RET&ISC_2inrow_EMA0.5` | Reticulocyte + ISC pipeline, 2-in-a-row, EMA 0.5 |
| `AllSubtypes_OneModel` | Single general 7-class model |
| `newly_trained_model_withD` | Retrained model including subtype D |
| `newly_trained_EMA0.5_withD` | Retrained, EMA 0.5, with D |
| `newlytrained_2inRow_EMA0.5_withD` | Retrained, 2-in-a-row, EMA 0.5, with D |

Temporal smoothing (EMA on the pair score) plus a *K-in-a-row* confirmation rule are
used to suppress per-frame classification flicker.

---

## Usage

```bash
cd code
python cell_video_classify_multiple_add_comp_filt_windows_latest_0811.py \
    -i V1.mp4,V2.mp4,V3.mp4,V4.mp4 \
    -o ../my_run_name \
    --frame_skip 2 \
    --max_frame 480
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-i, --inputs` | *required* | Comma-separated video paths |
| `-o, --output_dir` | *required* | Output directory (one subfolder per video) |
| `--frame_skip` | `2` | Process every Nth frame |
| `--max_frame` | `480` | Maximum frames to process |

Acquisition is assumed to be **4 fps**, so plotted time = `frame_index * frame_skip / 4` s.

### Dependencies

Python 3.10+, plus:

```
torch torchvision transformers cellpose opencv-python
numpy pandas matplotlib scikit-image pillow tqdm openpyxl
```

---

## Data not in this repository

The `.gitignore` excludes everything too large for GitHub. To reproduce a run you need
these locally, in the layout the scripts expect:

- **Source videos** — `V1.mp4` … `V4.mp4` in the repo root.
- **Model weights** in `code/`:
  - `cyto3_train0327` — fine-tuned Cellpose segmentation model
  - `best_model_vit_torch_macos_seven.pth` — 7-class subtype ViT
  - `best_model_vit_torch_macos_raw_vit_large_binary.pth` — binary change ViT
  - `best_model_vit_torch_macos_raw_vit_large_binary_G.pth`, `direct_vit_{D,E,G}.pt` — per-subtype heads
  - `siamese_vit_{All,ISC,Reti}_Haolin.pt`, `OG_siamese_vit_c_Haolin.pt` — Siamese change detectors
  - `rbc_ckpts/best_vit.pth`
- **Image datasets** (cropped single-cell PNGs):
  `Alldataset_for_subtypeclassification/`, `Paired dataset_10122025/`,
  `C dataset/`, `D_Reticulocyte/`
- **Per-run intermediates** regenerated by the pipeline:
  `V*/V*.avi` (annotated video), `cell_info.pkl`, `first_frame.png`,
  `masks_remove_edge_cells.npy`

What *is* tracked: all code and notebooks, every `state_ratio_*.csv`, the summary
plots, `df.pkl`, and the reference kinetics workbook `kinetics-seven_Jianlu.xlsx`.
