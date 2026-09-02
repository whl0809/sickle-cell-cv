import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from train_vit import (
    build_model,
    build_transforms,
    class_weights,
    load_checkpoint,
    metrics_from_predictions,
    normalize_model_name,
    seed_everything,
)


FEATURE_NAMES = [
    "area_fraction",
    "perimeter_sqrt_area",
    "circularity",
    "bbox_aspect_ratio",
    "bbox_extent",
    "eccentricity",
    "radial_std_over_mean",
    "radial_p90_p10_over_mean",
    "mean_intensity",
    "std_intensity",
    "p10_intensity",
    "p50_intensity",
    "p90_intensity",
    "inside_background_contrast",
    "dark_fraction",
    "bright_fraction",
    "gradient_mean",
    "gradient_std",
    "boundary_gradient_mean",
    "laplacian_std",
    "entropy_16bins",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier from explicit cell morphology and texture features."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run folder containing splits.csv.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional CNN checkpoint to append probabilities.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA for CNN probability extraction.")
    return parser.parse_args()


def read_splits(path):
    splits = {"train": [], "val": [], "test": []}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row["split"]
            if split not in splits:
                continue
            splits[split].append(
                {
                    "path": Path(row["path"]),
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "group": row["group"],
                }
            )
    return splits


def otsu_threshold(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    if float(values.max()) <= float(values.min()):
        return float(values.mean())

    hist, bin_edges = np.histogram(values, bins=128, range=(float(values.min()), float(values.max())))
    hist = hist.astype(np.float64)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1e-12)
    mean_fg = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1e-12))[::-1]
    variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return float(centers[int(np.argmax(variance_between))])


def largest_component(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            component = []
            queue = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            if len(component) > len(best):
                best = component

    output = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best)
        output[np.array(ys), np.array(xs)] = True
    return output


def erode(mask):
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    output = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            output &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return output


def make_cell_mask(gray):
    height, width = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    diff = np.abs(gray - background)
    threshold = max(otsu_threshold(diff), float(diff.mean() + 0.25 * diff.std()))
    mask = diff > threshold

    if mask.mean() < 0.01 or mask.mean() > 0.90:
        intensity_threshold = otsu_threshold(gray)
        darker = gray < intensity_threshold
        brighter = gray >= intensity_threshold
        mask = darker if darker.mean() < brighter.mean() else brighter

    mask = largest_component(mask)
    if mask.sum() == 0:
        mask = diff > max(float(diff.mean()), 1e-6)
    return mask, background


def entropy_16bins(values):
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=16, range=(0.0, 1.0))
    probs = hist.astype(np.float64)
    probs = probs / max(probs.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def extract_features(path):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    mask, background = make_cell_mask(gray)
    area = float(mask.sum())
    total_area = float(mask.size)
    if area <= 1:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    boundary = mask & ~erode(mask)
    perimeter = float(boundary.sum())
    ys, xs = np.nonzero(mask)
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    bbox_height = max_y - min_y + 1
    bbox_width = max_x - min_x + 1
    bbox_area = float(bbox_height * bbox_width)

    centered = np.column_stack([ys - ys.mean(), xs - xs.mean()])
    covariance = np.cov(centered, rowvar=False) if centered.shape[0] > 2 else np.eye(2)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-12))
    eccentricity = math.sqrt(max(0.0, 1.0 - float(eigenvalues[0] / eigenvalues[1])))

    by, bx = np.nonzero(boundary)
    radial = np.sqrt((by - ys.mean()) ** 2 + (bx - xs.mean()) ** 2)
    radial_mean = float(radial.mean()) if radial.size else 0.0
    radial_std = float(radial.std()) if radial.size else 0.0
    radial_p10, radial_p90 = np.percentile(radial, [10, 90]) if radial.size else (0.0, 0.0)

    gy, gx = np.gradient(gray)
    gradient = np.sqrt(gx * gx + gy * gy)
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )

    inside = gray[mask]
    inside_gradient = gradient[mask]
    boundary_gradient = gradient[boundary] if boundary.any() else inside_gradient
    inside_laplacian = laplacian[mask]

    features = [
        area / total_area,
        perimeter / math.sqrt(max(area, 1.0)),
        (4.0 * math.pi * area) / max(perimeter * perimeter, 1.0),
        max(bbox_width, bbox_height) / max(1.0, min(bbox_width, bbox_height)),
        area / max(bbox_area, 1.0),
        eccentricity,
        radial_std / max(radial_mean, 1e-6),
        float((radial_p90 - radial_p10) / max(radial_mean, 1e-6)),
        float(inside.mean()),
        float(inside.std()),
        float(np.percentile(inside, 10)),
        float(np.percentile(inside, 50)),
        float(np.percentile(inside, 90)),
        float(inside.mean() - background),
        float((inside < 0.35).mean()),
        float((inside > 0.75).mean()),
        float(inside_gradient.mean()),
        float(inside_gradient.std()),
        float(boundary_gradient.mean()),
        float(inside_laplacian.std()),
        entropy_16bins(inside),
    ]
    return np.asarray(features, dtype=np.float32)


def load_or_extract_features(samples, cache_path):
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], data["paths"].tolist()

    features = []
    labels = []
    paths = []
    for index, sample in enumerate(samples, start=1):
        if index % 1000 == 0:
            print(f"Extracted {index}/{len(samples)} features...")
        features.append(extract_features(sample["path"]))
        labels.append(sample["label"])
        paths.append(str(sample["path"]))

    features = np.vstack(features).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(cache_path, features=features, labels=labels, paths=np.asarray(paths))
    return features, labels, paths


def checkpoint_probabilities(samples, checkpoint_path, cache_path, batch_size, use_amp):
    if cache_path.exists():
        data = np.load(cache_path)
        return data["probabilities"].astype(np.float32), data["classes"].tolist()

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    model_name = normalize_model_name(checkpoint.get("model_name", "vit_b_16"))
    augmentation = checkpoint.get("augmentation", "standard")
    _, transform = build_transforms(image_size, augmentation=augmentation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(use_amp and device.type == "cuda")
    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        model_name=model_name,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    probabilities = []
    batch_tensors = []

    def flush_batch():
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device)
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            batch_probs = model(images).softmax(dim=1).cpu().numpy()
        probabilities.append(batch_probs)
        batch_tensors.clear()

    for index, sample in enumerate(samples, start=1):
        if index % 1000 == 0:
            print(f"Extracted {index}/{len(samples)} CNN probabilities...")
        with Image.open(sample["path"]) as image:
            batch_tensors.append(transform(image.convert("RGB")))
        if len(batch_tensors) >= batch_size:
            flush_batch()
    flush_batch()

    probabilities = np.vstack(probabilities).astype(np.float32)
    np.savez_compressed(cache_path, probabilities=probabilities, classes=np.asarray(classes))
    return probabilities, classes


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, inputs):
        return self.net(inputs)


def standardize(train_features, *other_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = [(features - mean) / std for features in (train_features, *other_features)]
    return standardized, mean.squeeze(0), std.squeeze(0)


def evaluate(model, features, labels):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).float())
        predictions = logits.argmax(dim=1).cpu().numpy().tolist()
    return metrics_from_predictions(labels.tolist(), predictions, num_classes=2)


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = args.output_dir or args.run_dir / "morphology_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = read_splits(args.run_dir / "splits.csv")

    all_samples = splits["train"] + splits["val"] + splits["test"]
    all_features, all_labels, all_paths = load_or_extract_features(all_samples, output_dir / "features_cache.npz")
    feature_names = list(FEATURE_NAMES)

    if args.checkpoint is not None:
        probabilities, cnn_classes = checkpoint_probabilities(
            all_samples,
            args.checkpoint,
            output_dir / "cnn_probabilities_cache.npz",
            args.batch_size,
            args.amp,
        )
        all_features = np.concatenate([all_features, probabilities], axis=1)
        feature_names.extend([f"cnn_prob_{class_name}" for class_name in cnn_classes])

    n_train = len(splits["train"])
    n_val = len(splits["val"])
    train_x = all_features[:n_train]
    val_x = all_features[n_train : n_train + n_val]
    test_x = all_features[n_train + n_val :]
    train_y = all_labels[:n_train]
    val_y = all_labels[n_train : n_train + n_val]
    test_y = all_labels[n_train + n_val :]

    (train_x, val_x, test_x), mean, std = standardize(train_x, val_x, test_x)
    model = FeatureMLP(input_dim=train_x.shape[1], hidden_dim=args.hidden_dim, num_classes=2)

    train_samples_for_weights = [{"label": int(label)} for label in train_y]
    weights = class_weights(train_samples_for_weights, 2)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_tensor = torch.from_numpy(train_x).float()
    train_labels = torch.from_numpy(train_y).long()
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    stale = 0
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tensor)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        train_metrics = evaluate(model, train_x, train_y)
        val_metrics = evaluate(model, val_x, val_y)
        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        rows.append(row)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"epoch {epoch}: train f1 {train_metrics['macro_f1']:.4f}, "
                f"val f1 {val_metrics['macro_f1']:.4f}"
            )
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_x, val_y)
    test_metrics = evaluate(model, test_x, test_y)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "feature_names": feature_names,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": feature_names,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "hidden_dim": args.hidden_dim,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        },
        output_dir / "morphology_model.pt",
    )

    print(f"Best epoch: {best_epoch}")
    print("Validation metrics")
    print(json.dumps(val_metrics, indent=2))
    print("Test metrics")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved result: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
