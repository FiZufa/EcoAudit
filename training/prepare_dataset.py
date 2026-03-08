import os
import shutil
import random
import yaml
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DIR  = Path("dataset")
OUTPUT_DIR  = Path("dataset_yolo")
SPLIT_RATIO = {"train": 0.80, "val": 0.15, "test": 0.05}
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# Collect class names from folder names (sorted for consistent IDs)
classes = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
class_to_id = {name: idx for idx, name in enumerate(classes)}

print(f"Found {len(classes)} classes: {classes}\n")

# Create output folders
for split in SPLIT_RATIO:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

counts = {"train": 0, "val": 0, "test": 0}

for class_name, class_id in class_to_id.items():
    class_dir = SOURCE_DIR / class_name
    images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val   = int(n * SPLIT_RATIO["val"])
    # remaining goes to test
    splits = (
        ("train", images[:n_train]),
        ("val",   images[n_train:n_train + n_val]),
        ("test",  images[n_train + n_val:]),
    )

    for split_name, split_images in splits:
        for img_path in split_images:
            # Copy image
            dest_img = OUTPUT_DIR / "images" / split_name / img_path.name
            # Avoid name collision across classes
            if dest_img.exists():
                dest_img = OUTPUT_DIR / "images" / split_name / f"{class_name}_{img_path.name}"
            shutil.copy2(img_path, dest_img)

            # Write label file (full-image bounding box)
            label_name = dest_img.stem + ".txt"
            dest_lbl = OUTPUT_DIR / "labels" / split_name / label_name
            dest_lbl.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")

            counts[split_name] += 1

    print(f"  {class_name:<20} (id={class_id})  total={n}  "
          f"train={n_train}  val={n_val}  test={n - n_train - n_val}")

# Write data.yaml
data_yaml = {
    "path": str(OUTPUT_DIR.resolve()),
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc":    len(classes),
    "names": classes,
}

with open(OUTPUT_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"\nDone!")
print(f"  Train : {counts['train']} images")
print(f"  Val   : {counts['val']} images")
print(f"  Test  : {counts['test']} images")
print(f"\nOutput  : {OUTPUT_DIR.resolve()}")
print(f"Config  : {(OUTPUT_DIR / 'data.yaml').resolve()}")
