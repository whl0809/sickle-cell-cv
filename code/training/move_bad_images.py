# --- repo-root-anchored paths (added during 2026 reorganisation) ---
# This file sits at <repo>/code/training/, so the repo root is two levels up. Default
# dataset and output locations are resolved from it rather than the working directory.
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

import argparse
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move unreadable or truncated image files out of a training dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "semi_final_degree")
    parser.add_argument("--bad-dir", type=Path, default=DATA_DIR / "semi_final_degree_bad_images")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report bad images without moving them.",
    )
    return parser.parse_args()


def image_is_readable(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
    except (OSError, UnidentifiedImageError, ValueError):
        return False
    return True


def unique_destination(path):
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    image_paths = sorted(
        path
        for path in args.data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    moved = []
    for path in image_paths:
        if image_is_readable(path):
            continue

        relative_path = path.relative_to(args.data_dir)
        destination = unique_destination(args.bad_dir / relative_path)
        moved.append((path, destination))

        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))

    action = "Would move" if args.dry_run else "Moved"
    print(f"Scanned {len(image_paths)} images.")
    print(f"{action} {len(moved)} bad images.")
    for source, destination in moved:
        print(f"{source} -> {destination}")


if __name__ == "__main__":
    main()
