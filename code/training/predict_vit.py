import argparse
from pathlib import Path

import torch
from PIL import Image

from train_vit import build_model, build_transforms, load_checkpoint, normalize_model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sickling degree for one image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--positive-class", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    model_name = normalize_model_name(checkpoint.get("model_name", "vit_b_16"))
    augmentation = checkpoint.get("augmentation", "standard")

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        model_name=model_name,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, transform = build_transforms(image_size, augmentation=augmentation)
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1).squeeze(0)

    for class_name, probability in zip(classes, probabilities.tolist()):
        print(f"{class_name}: {probability:.4f}")

    if args.threshold is not None:
        positive_class = args.positive_class or classes[0]
        if positive_class not in classes:
            raise ValueError(f"Unknown positive class {positive_class!r}. Classes: {classes}")
        if len(classes) != 2:
            raise ValueError("Threshold prediction expects exactly two classes.")
        positive_index = classes.index(positive_class)
        negative_index = 1 - positive_index
        predicted_index = positive_index if probabilities[positive_index] >= args.threshold else negative_index
    else:
        predicted_index = int(probabilities.argmax())

    print(f"prediction: {classes[predicted_index]}")


if __name__ == "__main__":
    main()
