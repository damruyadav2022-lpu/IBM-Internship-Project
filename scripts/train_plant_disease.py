"""
Train YOLOv8 on the PlantVillage dataset for plant disease detection.
38 disease classes, ~87,000 images.

USAGE (after dataset finishes downloading):
    python scripts/train_plant_disease.py

The Kaggle dataset downloads to:  data/plantvillage/new-plant-diseases-dataset.zip
and unzips automatically with --unzip flag.

TIME:
    ~30–60 min on GPU  |  3–5 hours on CPU (not recommended)
    Tip: Use Google Colab free GPU if no local GPU.
"""
import os
import sys
import shutil
from pathlib import Path

ROOT       = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "yolo_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# ── Find the dataset ──────────────────────────────────────────────────────────

def find_train_valid(search_root: Path):
    """
    Recursively find a train/ directory that has subdirectories (classes).
    Returns (train_dir, valid_dir) or (None, None).
    """
    for train_dir in sorted(search_root.rglob("train")):
        if not train_dir.is_dir():
            continue
        subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if len(subdirs) < 2:          # must have at least 2 class folders
            continue
        # look for valid/ or validation/ sibling
        for vname in ("valid", "validation", "val", "test"):
            valid_dir = train_dir.parent / vname
            if valid_dir.exists():
                return train_dir, valid_dir
        # no separate valid dir — use train itself (ultralytics will split)
        return train_dir, None
    return None, None


def check_dataset():
    """Return the dataset root if it exists, else None."""
    search_roots = [
        ROOT / "data" / "plantvillage",
        ROOT / "data",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        train_dir, valid_dir = find_train_valid(root)
        if train_dir:
            return train_dir, valid_dir
    return None, None


def update_env(rel_path: str):
    """Write YOLO_MODEL_PATH into .env."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        print(f"  No .env found — add manually:  YOLO_MODEL_PATH={rel_path}")
        return

    # Handle UTF-16 BOM (Windows PowerShell sometimes writes it)
    raw = env_file.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        content = raw.decode('utf-16')
    else:
        content = raw.decode('utf-8', errors='replace')

    new_line = f"YOLO_MODEL_PATH={rel_path}"
    if "YOLO_MODEL_PATH=" in content:
        lines = [new_line if l.startswith("YOLO_MODEL_PATH=") else l
                 for l in content.splitlines()]
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        with env_file.open("a", encoding="utf-8") as f:
            f.write(f"\n{new_line}\n")
    print(f"  .env updated: {new_line}")


def train():
    from ultralytics import YOLO
    import torch

    # ── Locate dataset ────────────────────────────────────────────────────────
    train_dir, valid_dir = check_dataset()

    if train_dir is None:
        print("\nERROR: PlantVillage dataset not found.")
        print("Make sure the Kaggle download completed and extracted to data/plantvillage/")
        print("\nExpected structure:")
        print("  data/plantvillage/")
        print("  └── New Plant Diseases Dataset(Augmented)/")
        print("      ├── train/")
        print("      │   ├── Apple___Apple_scab/")
        print("      │   ├── Tomato___Late_blight/")
        print("      │   └── ... (38 classes total)")
        print("      └── valid/")
        sys.exit(1)

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    data_root = train_dir.parent   # dir that contains train/ (and valid/)

    print("=" * 62)
    print("  KrishiRakshak - YOLOv8 Plant Disease Training")
    print("=" * 62)
    print(f"  Dataset root : {data_root}")
    print(f"  Train dir    : {train_dir}")
    print(f"  Valid dir    : {valid_dir or 'auto-split from train'}")
    print(f"  Classes      : {len(classes)}")
    print(f"  Sample crops : {', '.join(classes[:5])} ...")

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device       : GPU - {gpu_name}")
        batch = 64
        epochs = 30
    else:
        device = "cpu"
        print(f"  Device       : CPU (slow — consider Google Colab for GPU)")
        batch = 16
        epochs = 10    # fewer epochs on CPU to keep it reasonable

    print(f"  Epochs       : {epochs}  |  Batch: {batch}")
    print("=" * 62)
    print("\nStarting training... (check runs/classify/plant_disease/ for progress)\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = YOLO("yolov8n-cls.pt")    # start from pretrained nano classifier

    results = model.train(
        data=str(data_root),
        epochs=epochs,
        imgsz=224,
        batch=batch,
        device=device,
        project="runs/classify",
        name="plant_disease",
        exist_ok=True,
        patience=7,
        save=True,
        plots=True,
        verbose=True,
        workers=4 if device != "cpu" else 0,
    )

    # ── Save best weights ─────────────────────────────────────────────────────
    # Ultralytics saves to runs/classify/plant_disease/weights/best.pt
    best_candidates = [
        Path("runs/classify/plant_disease/weights/best.pt"),
        Path("runs/classify/plant_disease2/weights/best.pt"),
    ]
    # Also search dynamically
    for p in sorted(Path("runs").rglob("best.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
        best_candidates.insert(0, p)

    best_pt = None
    for candidate in best_candidates:
        if candidate.exists():
            best_pt = candidate
            break

    if best_pt:
        dest = WEIGHTS_DIR / "plant_disease_best.pt"
        shutil.copy(best_pt, dest)
        size_mb = dest.stat().st_size / 1024 / 1024

        print("\n" + "=" * 62)
        print(f"  Training complete!")
        print(f"  Best model  : {dest}  ({size_mb:.1f} MB)")

        rel = str(dest.relative_to(ROOT)).replace("\\", "/")
        update_env(rel)

        print(f"""
  NEXT STEPS
  ----------
  1. Restart your backend:
       cd backend
       uvicorn main:app --reload

  2. Open the frontend at http://localhost:3000
     -> Click "Disease Detection" tab
     -> Upload a leaf photo to test

  3. Your model detects {len(classes)} plant disease classes:
{chr(10).join(f"     - {c}" for c in classes[:10])}
     ... and {max(0, len(classes)-10)} more
        """)
    else:
        print("\nWARNING: Could not find best.pt after training.")
        print("Check runs/classify/plant_disease/weights/")


if __name__ == "__main__":
    train()
