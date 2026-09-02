import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image

from train_vit import build_model, build_transforms, load_checkpoint, normalize_model_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export false positives and false negatives from a saved ViT run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--positive-class", default="final_sickled")
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


def safe_filename(index, sample, predicted_class, probability):
    source = sample["path"]
    probability_text = f"{probability:.4f}".replace(".", "p")
    return f"{index:05d}_pred-{predicted_class}_p-{probability_text}_{source.name}"


def error_type(true_class, predicted_class, positive_class):
    if true_class == predicted_class:
        return "correct"
    if predicted_class == positive_class:
        return f"false_positive_{positive_class}"
    if true_class == positive_class:
        return f"false_negative_{positive_class}"
    return f"true_{true_class}__pred_{predicted_class}"


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or args.run_dir / "best_model.pt"
    output_dir = args.output_dir or args.run_dir / "misclassified" / args.split

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    if args.positive_class not in classes:
        raise ValueError(f"Unknown positive class {args.positive_class!r}. Classes: {classes}")

    image_size = checkpoint.get("image_size", 224)
    model_name = normalize_model_name(checkpoint.get("model_name", "vit_b_16"))
    augmentation = checkpoint.get("augmentation", "standard")
    _, eval_transform = build_transforms(image_size, augmentation=augmentation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        model_name=model_name,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    samples = read_split(args.run_dir / "splits.csv", args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"

    rows = []
    copied_counts = {}
    batch_tensors = []
    batch_samples = []

    def flush_batch():
        if not batch_tensors:
            return

        images = torch.stack(batch_tensors).to(device)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            probabilities = model(images).softmax(dim=1).cpu()

        for sample, probs in zip(batch_samples, probabilities):
            true_class = sample["class_name"]
            predicted_index = int(probs.argmax())
            predicted_class = classes[predicted_index]
            predicted_probability = float(probs[predicted_index])
            kind = error_type(true_class, predicted_class, args.positive_class)

            copied_to = ""
            if kind != "correct":
                copied_counts[kind] = copied_counts.get(kind, 0) + 1
                target_dir = output_dir / kind
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / safe_filename(
                    len(rows),
                    sample,
                    predicted_class,
                    predicted_probability,
                )
                shutil.copy2(sample["path"], target_path)
                copied_to = str(target_path)

            row = {
                "path": str(sample["path"]),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "error_type": kind,
                "predicted_probability": predicted_probability,
                "copied_to": copied_to,
            }
            for class_name, probability in zip(classes, probs.tolist()):
                row[f"prob_{class_name}"] = probability
            rows.append(row)

        batch_tensors.clear()
        batch_samples.clear()

    for sample in samples:
        with Image.open(sample["path"]) as image:
            batch_tensors.append(eval_transform(image.convert("RGB")))
        batch_samples.append(sample)
        if len(batch_tensors) >= args.batch_size:
            flush_batch()
    flush_batch()

    fieldnames = [
        "path",
        "true_class",
        "predicted_class",
        "error_type",
        "predicted_probability",
        *[f"prob_{class_name}" for class_name in classes],
        "copied_to",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_errors = sum(copied_counts.values())
    print(f"Scanned {len(samples)} {args.split} images.")
    print(f"Copied {total_errors} misclassified images to {output_dir}.")
    for kind, count in sorted(copied_counts.items()):
        print(f"{kind}: {count}")
    print(f"Saved predictions: {csv_path}")


if __name__ == "__main__":
    main()
