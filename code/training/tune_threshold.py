import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from train_vit import (
    build_model,
    build_transforms,
    load_checkpoint,
    metrics_from_predictions,
    normalize_model_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune a binary decision threshold on validation predictions."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--positive-class", default="final_sickled")
    parser.add_argument(
        "--metric",
        choices=["macro_f1", "accuracy", "positive_recall"],
        default="macro_f1",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    return parser.parse_args()


def read_split(path, split_name):
    samples = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] != split_name:
                continue
            samples.append(
                {
                    "path": Path(row["path"]),
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "group": row["group"],
                }
            )
    return samples


def predict_positive_probabilities(model, samples, transform, positive_index, device, batch_size, use_amp):
    y_true = []
    positive_probs = []
    batch_tensors = []
    batch_samples = []

    def flush_batch():
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device)
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            probabilities = model(images).softmax(dim=1).cpu()
        for sample, probs in zip(batch_samples, probabilities):
            y_true.append(sample["label"])
            positive_probs.append(float(probs[positive_index]))
        batch_tensors.clear()
        batch_samples.clear()

    for sample in samples:
        with Image.open(sample["path"]) as image:
            batch_tensors.append(transform(image.convert("RGB")))
        batch_samples.append(sample)
        if len(batch_tensors) >= batch_size:
            flush_batch()
    flush_batch()

    return y_true, positive_probs


def predictions_at_threshold(positive_probs, threshold, positive_index, negative_index):
    return [
        positive_index if probability >= threshold else negative_index
        for probability in positive_probs
    ]


def score_threshold(y_true, positive_probs, threshold, positive_index, negative_index, metric):
    y_pred = predictions_at_threshold(positive_probs, threshold, positive_index, negative_index)
    metrics = metrics_from_predictions(y_true, y_pred, num_classes=2)
    if metric == "positive_recall":
        score = metrics["per_class"][positive_index]["recall"]
    else:
        score = metrics[metric]
    return score, metrics


def threshold_candidates(positive_probs):
    candidates = {0.0, 1.0}
    candidates.update(round(index / 1000, 3) for index in range(1, 1000))
    for probability in positive_probs:
        candidates.add(probability)
    return sorted(candidates)


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or args.run_dir / "best_model.pt"
    output_dir = args.run_dir / "threshold_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    if len(classes) != 2:
        raise ValueError(f"Threshold tuning expects exactly two classes. Found: {classes}")
    if args.positive_class not in classes:
        raise ValueError(f"Unknown positive class {args.positive_class!r}. Classes: {classes}")

    positive_index = classes.index(args.positive_class)
    negative_index = 1 - positive_index
    image_size = checkpoint.get("image_size", 224)
    model_name = normalize_model_name(checkpoint.get("model_name", "vit_b_16"))
    augmentation = checkpoint.get("augmentation", "standard")
    _, eval_transform = build_transforms(image_size, augmentation=augmentation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Positive class: {args.positive_class}")

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        model_name=model_name,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    val_samples = read_split(args.run_dir / "splits.csv", "val")
    test_samples = read_split(args.run_dir / "splits.csv", "test")

    val_true, val_probs = predict_positive_probabilities(
        model,
        val_samples,
        eval_transform,
        positive_index,
        device,
        args.batch_size,
        use_amp,
    )
    test_true, test_probs = predict_positive_probabilities(
        model,
        test_samples,
        eval_transform,
        positive_index,
        device,
        args.batch_size,
        use_amp,
    )

    best = None
    rows = []
    for threshold in threshold_candidates(val_probs):
        score, metrics = score_threshold(
            val_true,
            val_probs,
            threshold,
            positive_index,
            negative_index,
            args.metric,
        )
        row = {
            "threshold": threshold,
            "score": score,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "positive_precision": metrics["per_class"][positive_index]["precision"],
            "positive_recall": metrics["per_class"][positive_index]["recall"],
            "negative_precision": metrics["per_class"][negative_index]["precision"],
            "negative_recall": metrics["per_class"][negative_index]["recall"],
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        }
        rows.append(row)

        tie_breaker = (metrics["macro_f1"], metrics["accuracy"])
        candidate = (score, tie_breaker, threshold, metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    _, _, best_threshold, best_val_metrics = best
    test_pred = predictions_at_threshold(test_probs, best_threshold, positive_index, negative_index)
    test_metrics = metrics_from_predictions(test_true, test_pred, num_classes=2)

    curve_path = output_dir / f"{args.positive_class}_{args.metric}_val_threshold_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "positive_class": args.positive_class,
        "negative_class": classes[negative_index],
        "optimized_metric": args.metric,
        "threshold": best_threshold,
        "classes": classes,
        "checkpoint": str(checkpoint_path),
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    result_path = output_dir / f"{args.positive_class}_{args.metric}_threshold_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Best validation threshold: {best_threshold:.6f}")
    print("Validation metrics at threshold:")
    print(json.dumps(best_val_metrics, indent=2))
    print("Test metrics at threshold:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved curve: {curve_path}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
