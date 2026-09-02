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
    parser = argparse.ArgumentParser(description="Average probabilities from multiple saved classifiers.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run folder containing the reference splits.csv.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
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


def load_member(checkpoint_path, expected_classes, device):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    if classes != expected_classes:
        raise ValueError(
            f"Class mismatch for {checkpoint_path}: expected {expected_classes}, found {classes}"
        )

    image_size = checkpoint.get("image_size", 224)
    model_name = normalize_model_name(checkpoint.get("model_name", "vit_b_16"))
    augmentation = checkpoint.get("augmentation", "standard")
    _, transform = build_transforms(image_size, augmentation=augmentation)

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        model_name=model_name,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return {
        "path": checkpoint_path,
        "model": model,
        "transform": transform,
        "model_name": model_name,
        "image_size": image_size,
        "augmentation": augmentation,
    }


def predict_member(member, samples, device, batch_size, use_amp):
    probabilities = []
    labels = []
    batch_tensors = []
    batch_labels = []

    def flush_batch():
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device)
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            batch_probs = member["model"](images).softmax(dim=1).cpu()
        probabilities.append(batch_probs)
        labels.extend(batch_labels)
        batch_tensors.clear()
        batch_labels.clear()

    for sample in samples:
        with Image.open(sample["path"]) as image:
            batch_tensors.append(member["transform"](image.convert("RGB")))
        batch_labels.append(sample["label"])
        if len(batch_tensors) >= batch_size:
            flush_batch()
    flush_batch()

    return torch.cat(probabilities, dim=0), labels


def main():
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_split(args.run_dir / "splits.csv", args.split)
    if not samples:
        raise RuntimeError(f"No samples found for split {args.split}")

    first_checkpoint = load_checkpoint(args.checkpoint[0], map_location="cpu")
    classes = first_checkpoint["classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Members: {len(args.checkpoint)}")

    members = [load_member(path, classes, device) for path in args.checkpoint]
    for member in members:
        print(
            f"- {member['model_name']} size={member['image_size']} "
            f"aug={member['augmentation']} checkpoint={member['path']}"
        )

    member_probabilities = []
    y_true = None
    member_metrics = []
    for member in members:
        probs, labels = predict_member(member, samples, device, args.batch_size, use_amp)
        if y_true is None:
            y_true = labels
        elif y_true != labels:
            raise RuntimeError("Label order mismatch while predicting ensemble members.")
        member_probabilities.append(probs)
        member_pred = probs.argmax(dim=1).tolist()
        member_metrics.append(
            {
                "checkpoint": str(member["path"]),
                "model_name": member["model_name"],
                "metrics": metrics_from_predictions(y_true, member_pred, num_classes=len(classes)),
            }
        )

    ensemble_probs = torch.stack(member_probabilities, dim=0).mean(dim=0)
    y_pred = ensemble_probs.argmax(dim=1).tolist()
    metrics = metrics_from_predictions(y_true, y_pred, num_classes=len(classes))

    predictions_path = output_dir / f"{args.split}_ensemble_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "true_class",
            "predicted_class",
            *[f"prob_{class_name}" for class_name in classes],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample, probs, pred in zip(samples, ensemble_probs.tolist(), y_pred):
            row = {
                "path": str(sample["path"]),
                "true_class": sample["class_name"],
                "predicted_class": classes[pred],
            }
            for class_name, probability in zip(classes, probs):
                row[f"prob_{class_name}"] = probability
            writer.writerow(row)

    result = {
        "split": args.split,
        "classes": classes,
        "checkpoints": [str(path) for path in args.checkpoint],
        "member_metrics": member_metrics,
        "ensemble_metrics": metrics,
    }
    result_path = output_dir / f"{args.split}_ensemble_metrics.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("Ensemble metrics")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {result_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
