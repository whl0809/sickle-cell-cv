import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_vit import (
    MicroscopyDataset,
    build_criterion,
    build_model,
    build_transforms,
    class_weights,
    load_checkpoint,
    normalize_model_name,
    run_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved ViT run without retraining.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
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


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or args.run_dir / "best_model.pt"
    splits_path = args.run_dir / "splits.csv"

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    model_name = normalize_model_name(checkpoint.get("model_name", "vit_b_16"))
    augmentation = checkpoint.get("augmentation", "standard")
    loss_name = checkpoint.get("loss", "cross_entropy")
    focal_gamma = checkpoint.get("args", {}).get("focal_gamma", 2.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    splits = read_splits(splits_path)
    _, eval_transform = build_transforms(image_size, augmentation=augmentation)
    dataset = MicroscopyDataset(splits[args.split], eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        model_name=model_name,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    weights = class_weights(splits["train"], len(classes)).to(device)
    criterion = build_criterion(loss_name, weights, focal_gamma)
    metrics = run_epoch(
        model,
        loader,
        criterion,
        optimizer=None,
        device=device,
        training=False,
        use_amp=use_amp,
    )

    output_path = args.run_dir / f"{args.split}_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
