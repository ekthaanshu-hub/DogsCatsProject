from pathlib import Path
import random
from PIL import Image, ImageOps

root = Path(__file__).resolve().parent
source = root / "data" / "PetImages"
output = root / "data" / "prepared"
limit = 2000
rng = random.Random(42)

if output.exists():
    raise SystemExit(
        "The prepared folder already exists. "
        "Rename it before preparing another dataset."
    )

for label in ["Cat", "Dog"]:
    if not (source / label).is_dir():
        raise SystemExit(f"Missing folder: {source / label}")

for label in ["Cat", "Dog"]:
    files = sorted((source / label).glob("*.jpg"))
    rng.shuffle(files)
    valid = []

    print(f"Checking {label} images...", flush=True)

    for path in files:
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
            valid.append(path)
        except (OSError, ValueError):
            continue

        if len(valid) == limit:
            break

    if len(valid) < 10:
        raise SystemExit(f"Too few valid images in {label}")

    train_end = int(len(valid) * 0.70)
    val_end = train_end + int(len(valid) * 0.15)

    splits = {
        "train": valid[:train_end],
        "validation": valid[train_end:val_end],
        "test": valid[val_end:],
    }

    for split, paths in splits.items():
        folder = output / split / label
        folder.mkdir(parents=True, exist_ok=True)

        for path in paths:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize(
                    (128, 128), Image.Resampling.LANCZOS
                )
                image.save(folder / path.name, "JPEG")

        print(f"{split}/{label}: {len(paths)} images", flush=True)

print("Dataset preparation complete!")