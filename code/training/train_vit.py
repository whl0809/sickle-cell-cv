# --- repo-root-anchored paths (added during 2026 reorganisation) ---
# This file sits at <repo>/code/training/, so the repo root is two levels up. Default
# dataset and output locations are resolved from it rather than the working directory.
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    efficientnet_b0,
    efficientnet_b3,
    resnet50,
    swin_t,
    vit_b_16,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MODEL_CHOICES = ["vit_b_16", "convnext_tiny", "efficientnet_b0", "efficientnet_b3", "resnet50", "swin_t"]
LOSS_CHOICES = ["cross_entropy", "focal"]
AUGMENTATION_CHOICES = ["standard", "conservative", "strong"]


CLASS_FOLDERS = {
    "final_sickled": "Sickled1-FinalSickled",
    "semi_sickled": "Sickled2-SemiSickled",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an image classifier for final-sickled vs semi-sickled microscopy images."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "semi_final_degree")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="vit_b_16")
    parser.add_argument("--augmentation", choices=AUGMENTATION_CHOICES, default="standard")
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--split-by", choices=["group", "image"], default="group")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_samples(data_dir):
    samples = []
    for class_index, (class_name, folder_name) in enumerate(CLASS_FOLDERS.items()):
        class_dir = data_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(
                    {
                        "path": path,
                        "label": class_index,
                        "class_name": class_name,
                        "group": make_group_key(path),
                    }
                )

    if not samples:
        raise RuntimeError(f"No image files found under {data_dir}")
    return samples


def make_group_key(path):
    """Keep crops from the same source acquisition in the same split."""
    stem = path.stem
    patterns = [
        r"^(.*?)_unknown_cell_\d+$",
        r"^(.*?)_cell_\d+$",
        r"^(.*?)_id\d+(?:\.\d+)?_frame\d+$",
        r"^(.*?)_id\d+(?:\.\d+)?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match and match.group(1):
            return f"{path.parent.name}:{match.group(1)}"
    return f"{path.parent.name}:{stem}"


def split_samples(samples, val_fraction, test_fraction, seed, split_by):
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if val_fraction + test_fraction >= 0.8:
        raise ValueError("Validation plus test fractions are too large.")

    random_state = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for label in sorted({sample["label"] for sample in samples}):
        class_samples = [sample for sample in samples if sample["label"] == label]

        if split_by == "image":
            units = [[sample] for sample in class_samples]
        else:
            grouped = defaultdict(list)
            for sample in class_samples:
                grouped[sample["group"]].append(sample)
            units = list(grouped.values())

        random_state.shuffle(units)
        total_images = sum(len(unit) for unit in units)
        target_test = round(total_images * test_fraction)
        target_val = round(total_images * val_fraction)

        current = "test"
        test_count = 0
        val_count = 0
        for unit in units:
            if current == "test" and test_count >= target_test:
                current = "val"
            if current == "val" and val_count >= target_val:
                current = "train"

            splits[current].extend(unit)
            if current == "test":
                test_count += len(unit)
            elif current == "val":
                val_count += len(unit)

    for split in splits.values():
        random_state.shuffle(split)
    return splits


class MicroscopyDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            with Image.open(sample["path"]) as image:
                image = image.convert("RGB")
                image = self.transform(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Could not read image: {sample['path']}") from exc
        return image, sample["label"]


def build_transforms(image_size, augmentation="standard"):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augmentation == "conservative":
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=20,
                            translate=(0.03, 0.03),
                            scale=(0.95, 1.05),
                            shear=5,
                        )
                    ],
                    p=0.85,
                ),
                transforms.RandomApply([transforms.RandomAutocontrast()], p=0.25),
                transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.25),
                transforms.ColorJitter(brightness=0.08, contrast=0.12, saturation=0.05, hue=0.01),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif augmentation == "strong":
        train_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.2)),
                transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.80, 1.20)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.20),
                transforms.RandomApply([transforms.RandomAutocontrast()], p=0.30),
                transforms.ColorJitter(brightness=0.18, contrast=0.20, saturation=0.12, hue=0.03),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.15, scale=(0.01, 0.04), ratio=(0.3, 3.3)),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return train_transform, eval_transform


def normalize_model_name(model_name):
    aliases = {
        "torchvision.vit_b_16": "vit_b_16",
        "vit": "vit_b_16",
        "convnext": "convnext_tiny",
        "efficientnet": "efficientnet_b0",
        "resnet": "resnet50",
        "swin": "swin_t",
    }
    return aliases.get(model_name, model_name)


def head_prefixes(model_name):
    model_name = normalize_model_name(model_name)
    if model_name == "vit_b_16":
        return ("heads.",)
    if model_name in {"convnext_tiny", "efficientnet_b0", "efficientnet_b3"}:
        return ("classifier.",)
    if model_name == "resnet50":
        return ("fc.",)
    if model_name == "swin_t":
        return ("head.",)
    raise ValueError(f"Unsupported model: {model_name}")


def is_head_parameter(name, model_name):
    return name.startswith(head_prefixes(model_name))


def build_model(num_classes, pretrained=True, model_name="vit_b_16", image_size=224):
    model_name = normalize_model_name(model_name)
    weights = None
    if pretrained and model_name == "vit_b_16" and image_size != 224:
        raise ValueError("Pretrained vit_b_16 in torchvision expects --image-size 224.")

    try:
        if model_name == "vit_b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            if pretrained:
                model = vit_b_16(weights=weights)
            else:
                model = vit_b_16(weights=None, image_size=image_size)
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        elif model_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = convnext_tiny(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b3(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            model = swin_t(weights=weights)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Could not load pretrained {model_name} weights ({exc}). Falling back to random initialization.")
        model = build_model(
            num_classes=num_classes,
            pretrained=False,
            model_name=model_name,
            image_size=image_size,
        )

    return model


def set_backbone_trainable(model, trainable, model_name="vit_b_16"):
    for name, parameter in model.named_parameters():
        if not is_head_parameter(name, model_name):
            parameter.requires_grad = trainable


def parameter_groups(model, lr, head_lr, weight_decay, model_name="vit_b_16"):
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_head_parameter(name, model_name):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def class_weights(samples, num_classes):
    counts = Counter(sample["label"] for sample in samples)
    total = sum(counts.values())
    weights = []
    for label in range(num_classes):
        weights.append(total / (num_classes * max(1, counts[label])))
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        return (((1.0 - pt) ** self.gamma) * loss).mean()


def build_criterion(loss_name, weights, focal_gamma):
    if loss_name == "focal":
        return FocalLoss(weight=weights, gamma=focal_gamma)
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"Unsupported loss: {loss_name}")


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix


def metrics_from_predictions(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[index][index] for index in range(num_classes))
    accuracy = correct / total if total else 0.0

    per_class = []
    f1_values = []
    for label in range(num_classes):
        tp = cm[label][label]
        fp = sum(cm[row][label] for row in range(num_classes) if row != label)
        fn = sum(cm[label][col] for col in range(num_classes) if col != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": sum(cm[label])})

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, training=False, use_amp=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_items = 0
    y_true = []
    y_pred = []
    iterator = tqdm(loader, leave=False) if tqdm else loader

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        predictions = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    metrics = metrics_from_predictions(y_true, y_pred, num_classes=2)
    metrics["loss"] = total_loss / total_items if total_items else math.inf
    return metrics


def write_splits_csv(path, splits):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "class_name", "group", "path"])
        for split_name, samples in splits.items():
            for sample in samples:
                writer.writerow(
                    [
                        split_name,
                        sample["label"],
                        sample["class_name"],
                        sample["group"],
                        str(sample["path"]),
                    ]
                )


def write_metrics_csv(path, rows):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "lr_backbone",
        "lr_head",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plain_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(path, model, args, classes, epoch, val_metrics):
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "class_folders": CLASS_FOLDERS,
            "model_name": normalize_model_name(args.model),
            "image_size": args.image_size,
            "augmentation": args.augmentation,
            "loss": args.loss,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "args": plain_value(vars(args)),
        },
        path,
    )


def summarize_split(name, samples):
    counts = Counter(sample["class_name"] for sample in samples)
    return f"{name}: {len(samples)} images ({dict(counts)})"


def main():
    args = parse_args()
    args.model = normalize_model_name(args.model)
    seed_everything(args.seed)

    classes = list(CLASS_FOLDERS.keys())
    run_name = time.strftime(f"{args.model}_sickling_degree_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args.data_dir)
    splits = split_samples(
        samples=samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_by=args.split_by,
    )
    write_splits_csv(run_dir / "splits.csv", splits)

    print(f"Run directory: {run_dir}")
    print(summarize_split("train", splits["train"]))
    print(summarize_split("val", splits["val"]))
    print(summarize_split("test", splits["test"]))

    train_transform, eval_transform = build_transforms(args.image_size, augmentation=args.augmentation)
    train_dataset = MicroscopyDataset(splits["train"], train_transform)
    val_dataset = MicroscopyDataset(splits["val"], eval_transform)
    test_dataset = MicroscopyDataset(splits["test"], eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")

    print(f"Model: {args.model}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Loss: {args.loss}")

    model = build_model(
        num_classes=len(classes),
        pretrained=not args.no_pretrained,
        model_name=args.model,
        image_size=args.image_size,
    )
    model.to(device)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False, model_name=args.model)

    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.lr, args.head_lr, args.weight_decay, model_name=args.model),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    weights = class_weights(splits["train"], len(classes)).to(device)
    criterion = build_criterion(args.loss, weights, args.focal_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    with (run_dir / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "classes": classes,
                "class_folders": CLASS_FOLDERS,
                "model_name": args.model,
                "image_size": args.image_size,
                "augmentation": args.augmentation,
                "loss": args.loss,
            },
            handle,
            indent=2,
        )

    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    metric_rows = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            print("Unfreezing backbone.")
            set_backbone_trainable(model, True, model_name=args.model)
            optimizer = torch.optim.AdamW(
                parameter_groups(model, args.lr, args.head_lr, args.weight_decay, model_name=args.model),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - epoch + 1),
            )

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            training=True,
            use_amp=use_amp,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            training=False,
            use_amp=use_amp,
        )
        scheduler.step()

        lrs = [group["lr"] for group in optimizer.param_groups]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr_backbone": lrs[0] if len(lrs) > 1 else "",
            "lr_head": lrs[-1],
        }
        metric_rows.append(row)
        write_metrics_csv(run_dir / "metrics.csv", metric_rows)

        print(
            "train loss {train_loss:.4f}, acc {train_accuracy:.4f}, f1 {train_macro_f1:.4f} | "
            "val loss {val_loss:.4f}, acc {val_accuracy:.4f}, f1 {val_macro_f1:.4f}".format(**row)
        )
        print(f"val confusion matrix: {val_metrics['confusion_matrix']}")

        save_checkpoint(run_dir / "last_model.pt", model, args, classes, epoch, val_metrics)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(run_dir / "best_model.pt", model, args, classes, epoch, val_metrics)
            print("Saved new best_model.pt")
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation F1 improvement.")
            break

    print(f"\nLoading best checkpoint from epoch {best_epoch}.")
    checkpoint = load_checkpoint(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        training=False,
        use_amp=use_amp,
    )

    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print("\nTest metrics")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nBest model: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
