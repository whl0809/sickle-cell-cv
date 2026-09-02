# --- repo-root-anchored paths (added during 2026 reorganisation) ---
# Weights live in <repo>/models/ and are resolved from this file's location, so the
# pipeline can be launched from any working directory. The shared classifier helpers
# (build_model / build_transforms / load_checkpoint) live in <repo>/code/training/.
import sys as _sys
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
_sys.path.insert(0, str(REPO_ROOT / "code" / "training"))

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
import os
from cellpose import models
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import pickle
import torch.nn.functional as F
from train_vit import build_model, build_transforms, load_checkpoint, normalize_model_name

#------------- Constants ----------------
def model_path(filename):
    """Resolve a weight file under <repo>/models/, failing loudly if it is absent."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model weight: {path}\n"
            "models/ is not tracked in git; see 'Data not in this repository' in README.md."
        )
    return str(path)


# Note (kaiyu): in cv2, color is BGR
BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 180, 0)
LABEL_CHANGED = 0
LABEL_UNCHANGED = 1
SICKLE_SUBTYPE_COLORS = {
    "unsickled": BLUE,
    "semi_sickled": (0, 165, 255),
    "final_sickled": RED,
}

# This semi/final pipeline does not classify cell types A-G.

# ============ Device ==============
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Siamese ViT model for unsickled/sickled change detection.
class SiameseViTChange(nn.Module):
    def __init__(self, backbone="google/vit-base-patch16-224-in21k", proj_dim=512, dropout=0.1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(backbone)
        h = self.vit.config.hidden_size          # 768 for ViT-Base
        self.proj = nn.Sequential(
            nn.Linear(h, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(proj_dim*2, proj_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, 1)            # BCEWithLogitsLoss
        )

    def encode(self, x):
        x = x.contiguous()
        out = self.vit(pixel_values=x)
        cls = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        cls = cls.contiguous()
        proj = self.proj(cls).contiguous()
        return proj

    def forward(self, x0, x1):
        f0, f1 = self.encode(x0), self.encode(x1)
        f0 = F.normalize(f0, dim=1)
        f1 = F.normalize(f1, dim=1)

        z = torch.cat([(f0 - f1).abs(), f0 * f1], dim=1)
        logit = self.head(z)
        return logit.reshape(-1)
    
    
pair_model_path_All = model_path("siamese_vit_All_Haolin.pt")
pair_model_All = SiameseViTChange()
pair_model_All.load_state_dict(torch.load(pair_model_path_All, map_location=device))
pair_model_All.to(device)
pair_model_All.eval()

semi_final_checkpoint_path = model_path("semi_final_classifier.pt")
semi_final_checkpoint = load_checkpoint(semi_final_checkpoint_path, map_location="cpu")
semi_final_classes = semi_final_checkpoint["classes"]
semi_final_model_name = normalize_model_name(semi_final_checkpoint.get("model_name", "convnext_tiny"))
semi_final_image_size = semi_final_checkpoint.get("image_size", 320)
semi_final_augmentation = semi_final_checkpoint.get("augmentation", "conservative")
semi_final_model = build_model(
    num_classes=len(semi_final_classes),
    pretrained=False,
    model_name=semi_final_model_name,
    image_size=semi_final_image_size,
)
semi_final_model.load_state_dict(semi_final_checkpoint["model_state"])
semi_final_model.to(device)
semi_final_model.eval()
_, semi_final_transform = build_transforms(
    semi_final_image_size,
    augmentation=semi_final_augmentation,
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict_sickled_subtype_probs(cell_pil):
    tensor = semi_final_transform(cell_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(semi_final_model(tensor), dim=1).squeeze(0).detach().cpu().numpy()
    return probabilities


def assign_endpoint_subtypes(cell_info, endpoint_frames=5):
    endpoint_frames = max(1, int(endpoint_frames))
    for cid, info in cell_info.items():
        sickled_frames = [
            frame_index
            for frame_index, label in sorted(info.get('state_history', {}).items())
            if label == LABEL_CHANGED and frame_index in info.get('subtype_prob_vectors', {})
        ]
        if not sickled_frames:
            continue

        selected_frames = sickled_frames[-endpoint_frames:]
        probabilities = np.stack([info['subtype_prob_vectors'][frame_index] for frame_index in selected_frames], axis=0)
        mean_probabilities = probabilities.mean(axis=0)
        class_index = int(np.argmax(mean_probabilities))
        subtype = semi_final_classes[class_index]
        subtype_prob = float(mean_probabilities[class_index])

        info['endpoint_subtype'] = subtype
        info['endpoint_subtype_prob'] = subtype_prob
        info['endpoint_subtype_frames'] = selected_frames
        for frame_index in sickled_frames:
            info.setdefault('subtype_history', {})[frame_index] = subtype
            info.setdefault('subtype_prob_history', {})[frame_index] = subtype_prob


DEBUG_MODE = False


def DEBUG_PRINT(msg, *args):
    if DEBUG_MODE:
        print(f"DEBUG: {msg}", *args)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Endpoint semi/final sickling pipeline with filtering and visualization.'
    )
    parser.add_argument(
        '-i', '--inputs', type=str, required=True,
        help='Comma-separated list of input video files, e.g., v1.mov,v2.mov,v3.mov'
    )
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--frame_skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--max_frame', type=int, default=480, help='Max number of frames to process')
    parser.add_argument(
        '--endpoint_subtype_frames', type=int, default=5,
        help='Average semi/final probabilities over the last K sickled observations for each cell'
    )
    parser.add_argument(
        '--sickling_threshold', type=float, default=0.73,
        help='Pairwise model threshold for marking a cell as sickled'
    )
    parser.add_argument(
        '--sickling_min_persist', type=int, default=1,
        help='Number of consecutive above-threshold observations required to mark sickling'
    )
    parser.add_argument(
        '--sickling_ema', type=float, default=1,
        help='EMA coefficient for pairwise sickling scores; 1.0 disables smoothing'
    )
    parser.add_argument(
        '--completion_stable_frames', type=int, default=6,
        help='Number of consecutive low-change sickled observations required to mark sickling complete'
    )
    parser.add_argument(
        '--completion_change_threshold', type=float, default=0.05,
        help='Maximum stability change score considered complete'
    )
    parser.add_argument(
        '--completion_score_mode',
        choices=['combined', 'probability', 'all'],
        default='combined',
        help='Criteria used to decide completion: weighted combined score, probability only, or all criteria stable'
    )
    parser.add_argument(
        '--completion_min_duration_sec', type=float, default=0,
        help='Minimum time after sickling onset before completion can be reported'
    )
    return parser.parse_args()

# Removes edge cells that fall below a specified threshold
def remove_edge_cells(masks, threshold=0.3):
    unique_ids = range(1, np.max(masks)+1)

    # average cell takes up ~1500 slots in an (729, 1094) ndarray
    avg_cell_area = sum([np.sum(masks == cid) for cid in unique_ids]) / len(unique_ids)

    # generates list of edges to be removed
    top    = masks[0, :]
    bottom = masks[-1, :]
    left   = masks[:, 0]
    right  = masks[:, -1]
    border_pixels = np.concatenate([top, bottom, left, right])
    edge_ids = np.unique(border_pixels[border_pixels > 0])
    remove_edge_list = []
    for edge_id in edge_ids:
        if np.sum(masks == edge_id) < threshold*avg_cell_area:
            remove_edge_list.append(edge_id)

    # generates new set of masks without targeted edge cells
    filtered_masks = np.zeros(masks.shape)
    cellidnumber = 1
    for cid in unique_ids:
        if cid not in remove_edge_list:
            filtered_masks += ((masks == cid)*cellidnumber)
            cellidnumber += 1
    
    return filtered_masks
    
def plot_sickled_subtype_ratio(df, out_path, frame_skip, fps, title='Semi/final sickled fraction'):
    time_sec = df['FrameIndex'] * frame_skip / fps
    semi_percent = df['SemiFraction'] * 100
    final_percent = df['FinalFraction'] * 100
    total_percent = df['SickledFraction'] * 100

    plt.figure(figsize=(8, 5))
    plt.plot(time_sec, semi_percent, label='Semi-sickled fraction', color='#f5a623')
    plt.plot(time_sec, final_percent, label='Final-sickled fraction', color='#d62728')
    plt.plot(time_sec, total_percent, label='Total sickled fraction', color='black', linestyle='--', alpha=0.55)
    plt.xlabel('Time (s)')
    plt.ylabel('Fraction of tracked cells (%)')
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    csv_out_path = out_path.replace(".png", ".csv")
    df_out = pd.DataFrame({
        "Time_sec": time_sec,
        "Semi_sickled_fraction_percent": semi_percent,
        "Final_sickled_fraction_percent": final_percent,
        "Total_sickled_fraction_percent": total_percent,
    })
    df_out.to_csv(csv_out_path, index=False)
    print(f"Saved curve data to {csv_out_path}")


def get_mask_shape_features(mask):
    if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
        return np.full(6, np.nan, dtype=np.float32)

    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(6, np.nan, dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
    aspect_ratio = max(w, h) / max(1.0, float(min(w, h)))
    solidity = area / (hull_area + 1e-6)
    extent = area / (float(w * h) + 1e-6)

    return np.array(
        [np.log1p(area), np.log1p(perimeter), circularity, aspect_ratio, solidity, extent],
        dtype=np.float32,
    )


def get_crop_texture_features(cell_crop, hist_bins=16):
    if cell_crop is None or cell_crop.size == 0:
        return {
            "texture": np.full(3, np.nan, dtype=np.float32),
            "hist": np.full(hist_bins, np.nan, dtype=np.float32),
        }

    gray = cv2.cvtColor(cell_crop, cv2.COLOR_RGB2GRAY) if cell_crop.ndim == 3 else cell_crop
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    mean_intensity = float(np.mean(gray)) / 255.0
    std_intensity = float(np.std(gray)) / 255.0
    edges = cv2.Canny(gray, 50, 150)
    edge_fraction = float(np.mean(edges > 0))
    hist = cv2.calcHist([gray], [0], None, [hist_bins], [0, 256]).astype(np.float32).reshape(-1)
    hist = hist / (hist.sum() + 1e-6)
    return {
        "texture": np.array([mean_intensity, std_intensity, edge_fraction], dtype=np.float32),
        "hist": hist,
    }


def make_stability_features(cell_crop, mask=None, subtype_probs=None):
    texture_features = get_crop_texture_features(cell_crop)
    return {
        "shape": get_mask_shape_features(mask),
        "texture": texture_features["texture"],
        "hist": texture_features["hist"],
        "subtype_probs": None if subtype_probs is None else np.asarray(subtype_probs, dtype=np.float32),
    }


def mean_valid_abs_delta(a, b, relative=False):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return None
    diff = np.abs(a[valid] - b[valid])
    if relative:
        diff = diff / (np.abs(a[valid]) + 1e-6)
    return float(np.mean(np.clip(diff, 0.0, 1.0)))


def stability_change_score(prev_features, cur_features):
    components = []
    component_values = {}

    subtype_prev = prev_features.get("subtype_probs")
    subtype_cur = cur_features.get("subtype_probs")
    if subtype_prev is not None and subtype_cur is not None:
        subtype_delta = mean_valid_abs_delta(subtype_prev, subtype_cur)
        if subtype_delta is not None:
            component_values["subtype_delta"] = subtype_delta
            components.append((0.40, subtype_delta))

    shape_delta = mean_valid_abs_delta(prev_features.get("shape"), cur_features.get("shape"), relative=True)
    if shape_delta is not None:
        component_values["shape_delta"] = shape_delta
        components.append((0.35, shape_delta))

    texture_delta = mean_valid_abs_delta(prev_features.get("texture"), cur_features.get("texture"))
    hist_delta = mean_valid_abs_delta(prev_features.get("hist"), cur_features.get("hist"))
    if texture_delta is not None:
        component_values["texture_delta"] = texture_delta
    if hist_delta is not None:
        component_values["hist_delta"] = hist_delta
    if texture_delta is not None and hist_delta is not None:
        texture_intensity_delta = 0.5 * texture_delta + 0.5 * hist_delta
        component_values["texture_intensity_delta"] = texture_intensity_delta
        components.append((0.25, texture_intensity_delta))
    elif texture_delta is not None:
        component_values["texture_intensity_delta"] = texture_delta
        components.append((0.25, texture_delta))
    elif hist_delta is not None:
        component_values["texture_intensity_delta"] = hist_delta
        components.append((0.25, hist_delta))

    if not components:
        return None, component_values

    weight_sum = sum(weight for weight, _ in components)
    score = sum(weight * value for weight, value in components) / weight_sum
    component_values["combined_score"] = score
    return score, component_values


def estimate_sickling_completion(
    cell_info,
    fps,
    stable_frames=6,
    change_threshold=0.05,
    score_mode="combined",
    min_duration_sec=1.0,
):
    stable_frames = max(1, int(stable_frames))
    min_duration_frames = int(np.ceil(max(0.0, float(min_duration_sec)) * fps)) if fps else 0
    columns = [
        "CellID",
        "Subtype",
        "StartFrame",
        "EndFrame",
        "StartTimeSec",
        "EndTimeSec",
        "DurationSec",
        "CompletionStatus",
        "MeanChangeScore",
        "MaxChangeScore",
        "MeanSubtypeDelta",
        "MeanShapeDelta",
        "MeanTextureDelta",
        "MeanHistDelta",
        "StableFramesRequired",
        "ChangeThreshold",
        "ScoreMode",
        "MinDurationSec",
        "EndpointSubtypeProb",
    ]
    records = []

    for cid, info in cell_info.items():
        sickled_frames = [
            frame_index
            for frame_index, label in sorted(info.get("state_history", {}).items())
            if label == LABEL_CHANGED
        ]
        if not sickled_frames:
            continue

        start_frame = sickled_frames[0]
        info["sickling_start_frame"] = start_frame
        info["sickling_start_time_sec"] = start_frame / fps if fps else None

        stable_run = 0
        stable_candidate = None
        prev_frame = None
        end_frame = None
        scores = []
        component_history = []

        for frame_index in sickled_frames:
            features = info.get("stability_features", {}).get(frame_index)
            if features is None:
                continue

            if prev_frame is not None:
                prev_features = info.get("stability_features", {}).get(prev_frame)
                score, component_values = stability_change_score(prev_features, features) if prev_features is not None else (None, {})
                if score is not None:
                    component_values["cell_id"] = cid
                    component_values["prev_frame"] = prev_frame
                    component_values["frame"] = frame_index
                    component_values["time_sec"] = frame_index / fps if fps else None
                    info.setdefault("stability_scores", {})[frame_index] = score
                    info.setdefault("stability_score_components", {})[frame_index] = component_values
                    scores.append(score)
                    component_history.append(component_values)
                    probability_score = component_values.get("subtype_delta")
                    shape_score = component_values.get("shape_delta")
                    texture_score = component_values.get("texture_intensity_delta")
                    if score_mode == "probability":
                        is_stable = probability_score is not None and probability_score <= change_threshold
                    elif score_mode == "all":
                        is_stable = (
                            probability_score is not None
                            and shape_score is not None
                            and texture_score is not None
                            and probability_score <= change_threshold
                            and shape_score <= change_threshold
                            and texture_score <= change_threshold
                        )
                    else:
                        is_stable = score <= change_threshold

                    if is_stable:
                        stable_run += 1
                        if stable_candidate is None:
                            stable_candidate = frame_index
                        far_enough_from_start = (frame_index - start_frame) >= min_duration_frames
                        if stable_run >= stable_frames and far_enough_from_start:
                            end_frame = stable_candidate
                            break
                    else:
                        stable_run = 0
                        stable_candidate = None

            prev_frame = frame_index

        subtype = info.get("endpoint_subtype", "unknown")
        end_time = end_frame / fps if end_frame is not None and fps else None
        start_time = info.get("sickling_start_time_sec")
        duration = end_time - start_time if end_time is not None and start_time is not None else None

        info["sickling_end_frame"] = end_frame
        info["sickling_end_time_sec"] = end_time
        info["sickling_duration_sec"] = duration
        info["sickling_completion_status"] = "completed" if end_frame is not None else "not_completed"

        def mean_component(name):
            values = [component[name] for component in component_history if name in component]
            return float(np.mean(values)) if values else None

        records.append({
            "CellID": cid,
            "Subtype": subtype,
            "StartFrame": start_frame,
            "EndFrame": end_frame,
            "StartTimeSec": start_time,
            "EndTimeSec": end_time,
            "DurationSec": duration,
            "CompletionStatus": info["sickling_completion_status"],
            "MeanChangeScore": float(np.mean(scores)) if scores else None,
            "MaxChangeScore": float(np.max(scores)) if scores else None,
            "MeanSubtypeDelta": mean_component("subtype_delta"),
            "MeanShapeDelta": mean_component("shape_delta"),
            "MeanTextureDelta": mean_component("texture_delta"),
            "MeanHistDelta": mean_component("hist_delta"),
            "StableFramesRequired": stable_frames,
            "ChangeThreshold": change_threshold,
            "ScoreMode": score_mode,
            "MinDurationSec": min_duration_sec,
            "EndpointSubtypeProb": info.get("endpoint_subtype_prob"),
        })

    return pd.DataFrame(records, columns=columns)


def summarize_sickling_times(time_df):
    if time_df.empty:
        return pd.DataFrame()

    completed = time_df[time_df["CompletionStatus"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=[
            "Subtype", "CompletedCells", "MeanDurationSec", "MedianDurationSec", "StdDurationSec"
        ])

    summary = completed.groupby("Subtype")["DurationSec"].agg(["count", "mean", "median", "std"]).reset_index()
    summary = summary.rename(columns={
        "count": "CompletedCells",
        "mean": "MeanDurationSec",
        "median": "MedianDurationSec",
        "std": "StdDurationSec",
    })
    return summary


def collect_stability_score_report(cell_info, fps, change_threshold):
    rows = []
    for cid, info in cell_info.items():
        subtype = info.get("endpoint_subtype", "unknown")
        for frame_index, components in sorted(info.get("stability_score_components", {}).items()):
            probability_score = components.get("subtype_delta")
            shape_score = components.get("shape_delta")
            texture_score = components.get("texture_intensity_delta")
            combined_score = components.get("combined_score")
            row = {
                "CellID": cid,
                "Subtype": subtype,
                "PrevFrame": components.get("prev_frame"),
                "Frame": frame_index,
                "TimeSec": components.get("time_sec", frame_index / fps if fps else None),
                "CombinedScore": combined_score,
                "ProbabilityScore": probability_score,
                "ShapeScore": shape_score,
                "TextureIntensityScore": texture_score,
                "RawTextureDelta": components.get("texture_delta"),
                "HistogramDelta": components.get("hist_delta"),
                "CombinedStable": combined_score <= change_threshold if combined_score is not None else None,
                "ProbabilityStable": probability_score <= change_threshold if probability_score is not None else None,
                "ShapeStable": shape_score <= change_threshold if shape_score is not None else None,
                "TextureIntensityStable": texture_score <= change_threshold if texture_score is not None else None,
                "ChangeThreshold": change_threshold,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_stability_criteria(score_df, change_threshold):
    rows = []
    criteria = [
        ("Probability", "ProbabilityScore"),
        ("Shape", "ShapeScore"),
        ("TextureIntensity", "TextureIntensityScore"),
        ("Combined", "CombinedScore"),
    ]
    for name, column in criteria:
        if score_df.empty or column not in score_df:
            values = pd.Series(dtype=float)
        else:
            values = pd.to_numeric(score_df[column], errors="coerce").dropna()
        rows.append({
            "Criterion": name,
            "Count": int(values.count()),
            "MeanScore": float(values.mean()) if not values.empty else None,
            "MedianScore": float(values.median()) if not values.empty else None,
            "StdScore": float(values.std()) if len(values) > 1 else None,
            "MinScore": float(values.min()) if not values.empty else None,
            "MaxScore": float(values.max()) if not values.empty else None,
            "FractionStable": float((values <= change_threshold).mean()) if not values.empty else None,
            "FractionHighChange": float((values > change_threshold).mean()) if not values.empty else None,
            "ChangeThreshold": change_threshold,
        })
    return pd.DataFrame(rows)


def summarize_stability_consistency(score_df, change_threshold):
    if score_df.empty:
        return pd.DataFrame(columns=["Comparison", "PairsCompared", "Correlation", "StableAgreementRate"])

    rows = []
    pairs = [
        ("Probability_vs_Shape", "ProbabilityScore", "ShapeScore"),
        ("Probability_vs_TextureIntensity", "ProbabilityScore", "TextureIntensityScore"),
        ("Shape_vs_TextureIntensity", "ShapeScore", "TextureIntensityScore"),
    ]
    for name, left, right in pairs:
        pair_df = score_df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair_df.empty:
            rows.append({
                "Comparison": name,
                "PairsCompared": 0,
                "Correlation": None,
                "StableAgreementRate": None,
            })
            continue
        left_stable = pair_df[left] <= change_threshold
        right_stable = pair_df[right] <= change_threshold
        rows.append({
            "Comparison": name,
            "PairsCompared": int(len(pair_df)),
            "Correlation": float(pair_df[left].corr(pair_df[right])) if len(pair_df) > 1 else None,
            "StableAgreementRate": float((left_stable == right_stable).mean()),
        })

    all_df = score_df[["ProbabilityScore", "ShapeScore", "TextureIntensityScore"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if all_df.empty:
        rows.append({
            "Comparison": "All_three",
            "PairsCompared": 0,
            "Correlation": None,
            "StableAgreementRate": None,
        })
    else:
        stable_matrix = all_df <= change_threshold
        all_agree = stable_matrix.all(axis=1) | (~stable_matrix).all(axis=1)
        rows.append({
            "Comparison": "All_three",
            "PairsCompared": int(len(all_df)),
            "Correlation": None,
            "StableAgreementRate": float(all_agree.mean()),
        })

    return pd.DataFrame(rows)


# ===  For cell tracing - Optical Flow function ===
def estimate_next_bboxes(prev_frame_gray, curr_frame_gray, prev_bboxes):
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray,
                                        None, 0.5, 3, 21, 3, 5, 1.2, 0)
    updated_bboxes = {}
    for cid, bbox in prev_bboxes.items():
        x, y, w, h = bbox
        region_flow = flow[y:y+h, x:x+w]
        if region_flow.size == 0:
            updated_bboxes[cid] = bbox
            continue
        dx = np.mean(region_flow[..., 0])
        dy = np.mean(region_flow[..., 1])
        nx, ny = int(x + dx), int(y + dy)
        updated_bboxes[cid] = (nx, ny, w, h)
    return updated_bboxes

def resize_frame(frame, ratio):
    h, w = frame.shape[:2]
    new_size = (int(w * ratio), int(h * ratio))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def upscale_bbox(bbox, ratio):
    x, y, w, h = bbox
    return tuple(int(coord / ratio) for coord in (x, y, w, h))


def segment_frame_downscaled_ds(original_frame, model_path, out_path, ratio=0.2, diameter=30,is_frame_0=False):
    orig_h, orig_w = original_frame.shape[:2]

    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized_frame = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks, flows, styles = cellpose_model.eval(resized_frame, diameter=diameter, channels=[0, 0])

    # Debug only for frame 0
    if is_frame_0 :

        plt.imshow(masks, cmap='gray')
        plt.title('Cellpose masks (BEFORE remove edge cells)')
        plt.axis('off')
        plt.imsave(out_path+"/masks_BEFORE_remove_edge_cells.png", masks, cmap="gray")

    masks = remove_edge_cells(masks)

    # Debug only for frame 0    
    if is_frame_0:
        DEBUG_PRINT("Saving masks.npy before remove_edge_cells")
        savepath = out_path+"/masks_remove_edge_cells.npy"
        np.save(savepath, masks)
        plt.imshow(masks, cmap='gray')
        plt.title('Cellpose masks (AFTER remove edge cells)')
        plt.axis('off')
        plt.imsave(out_path+"/masks_AFTER_remove_edge_cells.png", masks, cmap="gray")
        DEBUG_PRINT("masks.npy AFTER remove_edge_cells saved at", savepath)
        DEBUG_PRINT("    number of zero elements;", np.count_nonzero(masks == 0))

    unique_ids = np.unique(masks)[1:]
    bboxes = {}
    for cid in unique_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        x, y, w, h = cv2.boundingRect(contours[0])

        x_orig = int(x / ratio)
        y_orig = int(y / ratio)
        w_orig = int(w / ratio)
        h_orig = int(h / ratio)

        bboxes[cid] = (x_orig, y_orig, w_orig, h_orig)

    return bboxes, masks, unique_ids, resized_frame

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def check_size_outline(box1,box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    #tolerance = 1e-6
    fold=5
    c1=0
    c2=0
    if w2==0 or h2==0:
        return True
    if w1>w2:
        if w1-fold*w2>0:
            c1=1
    else:
        if w2-fold*w1>0:
            c1=1
    if h1>h2:
        if h1-fold*h2>0:
            c2=1
    else:
        if h2-fold*h1>0:
            c2=1
    #print(w1,w2)
    #print(h1,h2)
    #print(c1,c2)
    if c1==1 and c2==1:
        return True
    else:
        return False
    '''
    if (abs(w1 / w2 - fold) < tolerance or abs(w2 / w1 - fold) < tolerance) and (abs(h1 / h2 - fold) < tolerance or abs(h2 / h1 - fold) < tolerance):
        return True
    else:
        return False
    '''

def check_pos_outline_iter(box1,box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if abs(x1-x2)>=200 and abs(y1-y2)>=200:
        return True
    # elif  abs(x1-x2)>=200  or abs(y1 - y2) >= 200:
    #     return True
    # elif abs(x1-x2)>=200 and abs(y1-y2)>=50:
    #     return True
    # elif abs(y1 - y2) >= 200 and abs(x1 - x2) >= 50:
    #     return True
    else:
        return False


def check_pos_outline(box1,box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if abs(x1-x2)>=100 or abs(y1-y2)>=100:
        return True
    # elif  abs(x1-x2)>=200  or abs(y1 - y2) >= 200:
    #     return True
    # elif abs(x1-x2)>=200 and abs(y1-y2)>=50:
    #     return True
    # elif abs(y1 - y2) >= 200 and abs(x1 - x2) >= 50:
    #     return True
    else:
        return False


def center_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    #cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    #cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    cx1, cy1 = x1, y1
    cx2, cy2 = x2, y2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def match_cells_tracking(prev_cells, curr_masks, bboxes):
    matches = {}
    unmatched = list(np.unique(curr_masks))
    if 0 in unmatched:
        unmatched.remove(0)  # remove background
    used_curr = set()

    for track_id, prev_info in prev_cells.items():
        prev_frame_index = prev_info['latest_frame_index']
        prev_box = prev_info['bbox'][prev_frame_index]
        min_dist = float('inf')
        best_iou = 0
        best_cid = None
        best_box = None
        #print(track_id,prev_box)
        #exit()

        for cid in unmatched:
            x, y, w, h = bboxes[cid][0], bboxes[cid][1], bboxes[cid][2], bboxes[cid][3]
            curr_box = (x, y, w, h)
            dist = center_distance(prev_box, curr_box)
            iou = compute_iou(prev_box, curr_box)
            # if check_pos_outline_iter(prev_box, curr_box):continue

            #if not prev_cells[int(cid)]['class']==prev_cells[int(track_id)]['class']:continue
            if dist < min_dist:
                min_dist = dist
                best_iou = iou
                best_cid = cid
                best_box = curr_box
            elif dist < 50 and iou > best_iou:
                min_dist = dist
                best_iou = iou
                best_cid = cid
                best_box = curr_box


        # check_id=105
        # if int(track_id) == check_id:
        #     print(best_box)

        # Note (kaiyu): other fields are removed since not used
        if best_cid is not None:
            if check_size_outline(prev_box, best_box) or check_pos_outline(prev_box, best_box):
                matches[track_id] = {
                    'bbox': prev_box,
                    'mask_id': None,
                }
                # if int(track_id) == check_id:
                #     #print(cid, 'doesn\'t pass check_outline')
                #     print('best_cid is None - ', prev_box)
            else:

                matches[track_id] = {
                    'bbox': best_box,
                    'mask_id': best_cid,
                }
                # if int(track_id)==check_id:
                #     print('best_cid is not None - ',best_box)
                # used_curr.add(best_cid)
        else:
            matches[track_id]={
                'bbox': prev_box,
                'mask_id': None,
            }
            # if int(track_id)==check_id:
            #     print('best_cid is None - ', prev_box)

    return matches, used_curr

#------------------ Remove false positives ---------------
def remove_bin_label_false_positives(cell_info):
    for cid in cell_info:
        frame_indices = list(sorted(cell_info[cid]["state_history"].keys()))
        print(f"Removing false positives for {cid}")        
        # Imagine     B B B R R B B R R B R R R R
        # Make a box  B B B|R R B B R R B|R R R R
        #                   Left          Right
        # Count the number of R in the box (4)
        # Calculate R_count / (Right - Left)
        # Determine box_left
        box_left = 0   # list index to frame_indices
        for frame_index in frame_indices:
            bin_label = cell_info[cid]["state_history"][frame_index]
            if bin_label == LABEL_CHANGED:
                break
            box_left += 1

        # Determine box_right
        box_right = len(frame_indices) # len(cell_info[cid]["state_history"])  # list index to frame_indices
        for frame_index in reversed(frame_indices):
            bin_label = cell_info[cid]["state_history"][frame_index]
            if bin_label == LABEL_UNCHANGED:
                break
            box_right -= 1
            
        if box_right <= box_left:
            DEBUG_PRINT("[remove_false_positive] box_right <= box_left -- no false positive detected")
            continue

        # Count labels in the box
        changed_count = 0
        for list_index in range(box_left, box_right):
            frame_index = frame_indices[list_index]
            bin_label = cell_info[cid]["state_history"][frame_index]
            if bin_label == LABEL_CHANGED:
                changed_count += 1

        # Calculate ratio
        if changed_count / (box_right - box_left) > 0.5:
            for list_index in range(box_left, box_right):
                frame_index = frame_indices[list_index]
                cell_info[cid]["state_history"][frame_index] = LABEL_CHANGED
        else:
            for list_index in range(box_left, box_right):
                frame_index = frame_indices[list_index]
                cell_info[cid]["state_history"][frame_index] = LABEL_UNCHANGED            

# -------------------- Save intermediate results (in case we couldn't finish running for all videos) ---------------------
def save_intermediate_results(cell_info, df, out_path):
    """
    Saves cell_info, df after processing a video to out_path/cell_info.pkl and out_path/df.pkl
    
    cell_info: cell info collected after processing a video
    df: data frame after processing a video
    out_path: the output directory corresponding to an input video
    """
    os.makedirs(out_path, exist_ok=True)

    # Save cell_info dict
    cell_info_path = os.path.join(out_path, "cell_info.pkl")
    with open(cell_info_path, "wb") as f:
        pickle.dump(cell_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save pandas DataFrame
    df_path = os.path.join(out_path, "df.pkl")
    df.to_pickle(df_path, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved cell_info to {cell_info_path} and DataFrame to {df_path}")


# -------------------- Main processing function ---------------------
def process_video(
    video_path,
    out_path,
    output_video_path,
    transform,
    cellpose_model_path=None,
    frame_skip=2,
    max_frame=480,
    fps=4,
    endpoint_subtype_frames=5,
    sickling_threshold=0.73,
    sickling_min_persist=1,
    sickling_ema=1,
    completion_stable_frames=6,
    completion_change_threshold=0.05,
    completion_score_mode="combined",
    completion_min_duration_sec=0,
):
    if cellpose_model_path is None:
        cellpose_model_path = model_path('cyto3_train0327')

    print('- Initialization......')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print('- Initialization......Done~')
    output_fps = fps / frame_skip

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Can not load the 0 time point from the video.")
    DEBUG_PRINT("Save first frame png")
    cv2.imwrite(out_path+"/first_frame.png", cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    # Note (kaiyu):
    # We first run through the predictions, and only after all
    # predictions are made, we generate the video, so that we 
    # could remove false positives from the binary model's predictions.
    
    # ======= Process Cells =========
    DEBUG_PRINT("Run segment_frame_downscaled_ds w/ cellpose_model")
    bboxes, masks_seg, unique_ids, resized_frame = segment_frame_downscaled_ds( first_frame, cellpose_model_path, out_path,is_frame_0=True)

    # Note (kaiyu): cell_info is a SUPER IMPORTANT data structure; it holds all relevant
    # information output by the models to produce results
    cell_info = {}

    print('- Initialize cells in frame 0 as unsickled......')
    for cid in tqdm(unique_ids, desc='Process cells in frame 0 - progress:'):
        mask = (masks_seg == cid).astype(np.uint8)
        x, y, w, h=bboxes[cid][0],bboxes[cid][1],bboxes[cid][2],bboxes[cid][3]
        cell_crop = first_frame[y:y+h, x:x+w]
        cell_pil = Image.fromarray(cell_crop)
        cell_mask = (masks_seg == cid).astype(np.uint8)
        # Note (kaiyu): saves all bbox and state detections, per frame where detection was made. 
        cell_info[cid] = {
            'bbox': {0: (x, y, w, h)},     # cell bounding box; maps from frame id to bbox
            'state_history': {0: LABEL_UNCHANGED},  # frame 0 is the unsickled baseline
            'state_prob_history': {0: 1.0}, # maps from frame id to past state prediction prob
            'subtype_history': {0: "unsickled"},
            'subtype_prob_history': {0: 1.0},
            'subtype_prob_vectors': {},
            'stability_features': {0: make_stability_features(cell_crop, cell_mask)},
            'stability_scores': {},
            'latest_frame_index': 0       # the most recent frame index that was updated
        }
        # --- NEW: init pairwise model state ---
        cell_info[cid]['ref_tensor']      = transform(cell_pil)   # keep on CPU; .to(device) when using
        cell_info[cid]['pair_score_ema']  = None                  # EMA for anti-flicker
        cell_info[cid]['above_streak']    = 0                     # streak counter for K-in-a-row

    print('- Initialize cells in frame 0 as unsickled......Done~')

    frame_stats = []
    # frame_idx = 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frame > total_frames:
        max_frame = total_frames

    for frame_idx in tqdm(range(1, max_frame), desc="Processing frames"): # Note (Haolin): start from frame 1 to avoid overwriting frame 0 info
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            continue

        DEBUG_PRINT(f"[frame {frame_idx}] Run segment_frame_downscaled_ds w/ cellpose_model")
        #if frame_idx<95: continue

        bboxes, masks, unique_ids, resized_frame = segment_frame_downscaled_ds(frame, cellpose_model_path, out_path,ratio=0.1, diameter=30)
        #continue
        #exit()
        #print(masks)

        matched, used_cids = match_cells_tracking(cell_info, masks, bboxes)

        for cid, info in matched.items():
            x, y, w, h = info['bbox']
            cell_crop = frame[y:y+h, x:x+w]
            cell_pil = Image.fromarray(cell_crop)
            mask_id = info.get('mask_id')
            cell_mask = (masks == mask_id).astype(np.uint8) if mask_id is not None else None

            with torch.no_grad():
                # All subtypes use the same pairwise model
                # inputs
                ref_t = cell_info[cid]['ref_tensor'].to(device)   # (3,224,224)
                cur_t = transform(cell_pil).to(device)            # (3,224,224)

                # model
                logit = pair_model_All(ref_t.unsqueeze(0), cur_t.unsqueeze(0))  # (1,)
                p_changed = torch.sigmoid(logit[0]).item()

                # smooth with EMA to reduce flicker
                prev = cell_info[cid]['pair_score_ema']
                s_ema = p_changed if prev is None else (sickling_ema * p_changed + (1 - sickling_ema) * prev)
                cell_info[cid]['pair_score_ema'] = s_ema

                # update streak
                is_above = (s_ema >= sickling_threshold)
                streak = cell_info[cid].get('above_streak')
                streak = streak + 1 if is_above else 0
                cell_info[cid]['above_streak'] = streak

                # when we *first* reach exactly MIN_PERSIST, retro-label previous MIN_PERSIST-1 frames
                if streak == sickling_min_persist:
                    start_idx = frame_idx - (sickling_min_persist - 1)
                    for idx in range(start_idx, frame_idx):
                        cell_info[cid]['state_history'][idx] = LABEL_CHANGED

                # current frame label: changed iff we currently have >= MIN_PERSIST in a row
                bin_label = LABEL_CHANGED if (streak >= sickling_min_persist) else LABEL_UNCHANGED
                bin_prob  = float(s_ema)

                # write current frame's label into history
                cell_info[cid]['state_history'][frame_idx] = bin_label

            cell_info[cid]["bbox"][frame_idx] = (x, y, w, h)
            cell_info[cid]["state_history"][frame_idx] = bin_label
            cell_info[cid]["state_prob_history"][frame_idx] = bin_prob
            cell_info[cid]["latest_frame_index"] = frame_idx
            subtype_probs = None
            if bin_label == LABEL_CHANGED:
                subtype_probs = predict_sickled_subtype_probs(cell_pil)
                cell_info[cid].setdefault("subtype_prob_vectors", {})[frame_idx] = subtype_probs
            cell_info[cid].setdefault("stability_features", {})[frame_idx] = make_stability_features(
                cell_crop,
                cell_mask,
                subtype_probs,
            )

    #---------------------------------------------


    # ================ Remove False Positives in bin_labels =========
    remove_bin_label_false_positives(cell_info)
    assign_endpoint_subtypes(cell_info, endpoint_subtype_frames)
    sickling_time_df = estimate_sickling_completion(
        cell_info,
        fps,
        stable_frames=completion_stable_frames,
        change_threshold=completion_change_threshold,
        score_mode=completion_score_mode,
        min_duration_sec=completion_min_duration_sec,
    )
    sickling_time_df.to_csv(os.path.join(out_path, 'sickling_time_report.csv'), index=False)
    sickling_time_summary = summarize_sickling_times(sickling_time_df)
    sickling_time_summary.to_csv(os.path.join(out_path, 'sickling_time_summary.csv'), index=False)
    stability_score_df = collect_stability_score_report(cell_info, fps, completion_change_threshold)
    stability_score_df.to_csv(os.path.join(out_path, 'stability_score_report.csv'), index=False)
    stability_score_summary = summarize_stability_criteria(stability_score_df, completion_change_threshold)
    stability_score_summary.to_csv(os.path.join(out_path, 'stability_score_summary.csv'), index=False)
    stability_score_consistency = summarize_stability_consistency(stability_score_df, completion_change_threshold)
    stability_score_consistency.to_csv(os.path.join(out_path, 'stability_score_consistency.csv'), index=False)

    # ================ Collect Results =================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset frame back to zero    
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (W, H))

    for frame_index in range(max_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_skip != 0:
            continue        

        print(f"Collecting Results for frame {frame_index}......")        
        if frame_index == 0:
            out.write(frame)
            
        frame_record = {'Frame': frame_index}
        total_count = 0
        sickled_count = 0
        semi_count = 0
        final_count = 0
        
        annotated_frame = frame.copy()
        for cid, info in cell_info.items():
            if frame_index in info['bbox']:
                x, y, w, h = info['bbox'][frame_index]
                bin_label = info['state_history'][frame_index]
                total_count += 1

                subtype = "unsickled"
                subtype_prob = 1.0
                if bin_label == LABEL_CHANGED:
                    subtype = info.get('subtype_history', {}).get(frame_index, info.get('endpoint_subtype', 'semi_sickled'))
                    subtype_prob = info.get('subtype_prob_history', {}).get(frame_index, info.get('endpoint_subtype_prob', 1.0))
                    sickled_count += 1
                    if subtype == "semi_sickled":
                        semi_count += 1
                    elif subtype == "final_sickled":
                        final_count += 1

                info.setdefault('subtype_history', {})[frame_index] = subtype
                info.setdefault('subtype_prob_history', {})[frame_index] = subtype_prob
                color = SICKLE_SUBTYPE_COLORS["unsickled"]
                state_text = "unsickled"
                if bin_label == LABEL_CHANGED:
                    completion_frame = info.get("sickling_end_frame")
                    if completion_frame is not None and frame_index >= completion_frame:
                        color = GREEN
                        state_text = "completed"
                    else:
                        color = RED
                        state_text = "sickling"
                text = f"[{cid}] {state_text} | {subtype} ({subtype_prob:.2f})"
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                text_y = max(18, y - 24)
                cv2.putText(annotated_frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                if bin_label == LABEL_CHANGED:
                    start_time = info.get("sickling_start_time_sec")
                    end_time = info.get("sickling_end_time_sec")
                    duration = info.get("sickling_duration_sec")
                    start_text = f"{start_time:.1f}s" if start_time is not None else "..."
                    end_text = f"{end_time:.1f}s" if end_time is not None else "..."
                    duration_text = f"{duration:.1f}s" if duration is not None else "..."
                    time_text = f"start {start_text} end {end_text} dur {duration_text}"
                    if frame_index == info.get("sickling_start_frame"):
                        time_text += " START"
                    if frame_index == info.get("sickling_end_frame"):
                        time_text += " END"
                    cv2.putText(annotated_frame, time_text, (x, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Save every 10 frames
        # if frame_index % 10 == 0:
        #     DEBUG_PRINT(f"Save annotated frame png for frame index {frame_index}")
        #     cv2.imwrite(out_path+f"/frame_{frame_index}_annotated.png", annotated_frame)
                
        # Write annotated frame in output video
        out.write(annotated_frame)

        frame_record['TotalCells'] = total_count
        frame_record['SickledCells'] = sickled_count
        frame_record['SemiSickledCells'] = semi_count
        frame_record['FinalSickledCells'] = final_count
        frame_record['SickledFraction'] = sickled_count / total_count if total_count > 0 else 0
        frame_record['SemiFraction'] = semi_count / total_count if total_count > 0 else 0
        frame_record['FinalFraction'] = final_count / total_count if total_count > 0 else 0
        frame_stats.append(frame_record)
        #out.write(annotated_frame)
        
    cap.release()
    out.release()
    df = pd.DataFrame(frame_stats)
    df['FrameIndex'] = range(len(df))
    df.to_csv(out_path+'/state_ratio_report.csv', index=False)
    plot_sickled_subtype_ratio(df, out_path+'/endpoint_semi_final_sickled_fraction.png', frame_skip, fps)
    print("Finished - generate endpoint semi/final sickled csv and figure report.")
    if not sickling_time_summary.empty:
        print("Average sickling duration by endpoint subtype:")
        print(sickling_time_summary.to_string(index=False))
    if not stability_score_summary.empty:
        print("Stability criterion score summary:")
        print(stability_score_summary.to_string(index=False))
    subtype_counts = Counter()
    for info in cell_info.values():
        for subtype in info.get('subtype_history', {}).values():
            if subtype in {"semi_sickled", "final_sickled"}:
                subtype_counts[subtype] += 1
    return cell_info, df, subtype_counts, sickling_time_df, stability_score_df

def main():
    print("-------------------- Parameterization --------------------")
    args = parse_args()

    video_paths = [path.strip() for path in args.inputs.split(',') if path.strip()]
    all_out = args.output_dir
    frame_skip = args.frame_skip
    max_frame = args.max_frame
    endpoint_subtype_frames = args.endpoint_subtype_frames
    sickling_threshold = args.sickling_threshold
    sickling_min_persist = args.sickling_min_persist
    sickling_ema = args.sickling_ema
    completion_stable_frames = args.completion_stable_frames
    completion_change_threshold = args.completion_change_threshold
    completion_score_mode = args.completion_score_mode
    completion_min_duration_sec = args.completion_min_duration_sec
    fps = 4

    output = [os.path.splitext(os.path.basename(v))[0] + '.avi' for v in video_paths]
    out_path = [os.path.join(all_out, os.path.splitext(os.path.basename(v))[0]) for v in video_paths]

    os.makedirs(all_out, exist_ok=True)
    for op in out_path:
        os.makedirs(op, exist_ok=True)

    semi_final_stats = []
    sickling_time_stats = []
    stability_score_stats = []
    for idx, video_path in enumerate(video_paths):
        cell_info, df, subtype_counts, sickling_time_df, stability_score_df = process_video(
            video_path=video_path,
            out_path=out_path[idx],
            output_video_path=os.path.join(out_path[idx], output[idx]),
            transform=transform,
            frame_skip=frame_skip,
            max_frame=max_frame,
            fps=fps,
            endpoint_subtype_frames=endpoint_subtype_frames,
            sickling_threshold=sickling_threshold,
            sickling_min_persist=sickling_min_persist,
            sickling_ema=sickling_ema,
            completion_stable_frames=completion_stable_frames,
            completion_change_threshold=completion_change_threshold,
            completion_score_mode=completion_score_mode,
            completion_min_duration_sec=completion_min_duration_sec,
        )
        semi_final_stats.append(df)
        sickling_time_df.insert(0, "Video", os.path.splitext(os.path.basename(video_path))[0])
        sickling_time_stats.append(sickling_time_df)
        stability_score_df.insert(0, "Video", os.path.splitext(os.path.basename(video_path))[0])
        stability_score_stats.append(stability_score_df)
        save_intermediate_results(cell_info, df, out_path[idx])

    combined_df = pd.concat(semi_final_stats, ignore_index=True)
    count_columns = ['TotalCells', 'SickledCells', 'SemiSickledCells', 'FinalSickledCells']
    summed = combined_df.groupby('FrameIndex')[count_columns].sum().reset_index()
    frame_map = combined_df.groupby('FrameIndex')['Frame'].first().reset_index()
    final_df = pd.merge(summed, frame_map, on='FrameIndex')
    final_df['SickledFraction'] = np.where(final_df['TotalCells'] > 0, final_df['SickledCells'] / final_df['TotalCells'], 0)
    final_df['SemiFraction'] = np.where(final_df['TotalCells'] > 0, final_df['SemiSickledCells'] / final_df['TotalCells'], 0)
    final_df['FinalFraction'] = np.where(final_df['TotalCells'] > 0, final_df['FinalSickledCells'] / final_df['TotalCells'], 0)
    cols = ['FrameIndex', 'Frame'] + [col for col in final_df.columns if col not in ['FrameIndex', 'Frame']]
    final_df = final_df[cols]
    final_df.to_csv(os.path.join(all_out, 'state_ratio_report.csv'), index=False)
    plot_sickled_subtype_ratio(
        final_df,
        os.path.join(all_out, 'combined_endpoint_semi_final_sickled_fraction.png'),
        frame_skip,
        fps,
    )

    combined_time_df = pd.concat(sickling_time_stats, ignore_index=True)
    combined_time_df.to_csv(os.path.join(all_out, 'sickling_time_report.csv'), index=False)
    combined_time_summary = summarize_sickling_times(combined_time_df)
    combined_time_summary.to_csv(os.path.join(all_out, 'sickling_time_summary.csv'), index=False)
    if not combined_time_summary.empty:
        print("Combined average sickling duration by endpoint subtype:")
        print(combined_time_summary.to_string(index=False))

    combined_score_df = pd.concat(stability_score_stats, ignore_index=True)
    combined_score_df.to_csv(os.path.join(all_out, 'stability_score_report.csv'), index=False)
    combined_score_summary = summarize_stability_criteria(combined_score_df, completion_change_threshold)
    combined_score_summary.to_csv(os.path.join(all_out, 'stability_score_summary.csv'), index=False)
    combined_score_consistency = summarize_stability_consistency(combined_score_df, completion_change_threshold)
    combined_score_consistency.to_csv(os.path.join(all_out, 'stability_score_consistency.csv'), index=False)


if __name__ == "__main__":
    main()
