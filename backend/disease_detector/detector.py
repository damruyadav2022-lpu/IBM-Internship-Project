"""
YOLOv8 Plant Disease Detector.

This module wraps Ultralytics YOLOv8 for plant disease detection from images.

Model priority (auto-selected, no dataset download needed):
  1. YOLO_MODEL_PATH env var  — explicit path set by user
  2. yolo_weights/plant_disease_best.pt    — user-trained on Colab/local
  3. yolo_weights/plant_disease_38cls.pt   — 31-class pre-trained (auto-downloaded)
  4. yolo_weights/plant_disease_peachfawn_better.pt — 15-class pre-trained
  5. yolo_weights/yolov8n-cls.pt           — base classifier fallback

Pre-trained models are auto-downloaded from HuggingFace on first run.
No dataset or training required.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DISEASE_CLASSES_FILE = DATA_DIR / "disease_classes.json"

# Weights folder — one level above backend/
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "yolo_weights"

# Explicit override — takes top priority
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "")

# Confidence threshold — detections below this are ignored
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))

# Max image dimension to avoid OOM on small servers
MAX_IMAGE_SIZE = 640

# Pre-trained plant disease models (tried in priority order, auto-downloaded)
PRETRAINED_MODELS = [
    {
        "path": "plant_disease_best.pt",         # user-trained (best)
        "hf_repo": None,
        "hf_file": None,
        "classes": "user-trained",
    },
    {
        "path": "plant_disease_38cls.pt",         # 31-class HuggingFace model
        "hf_repo": "Lucario-K17/YOLOv8-5_Plant-Diseases",
        "hf_file": "yolov8_plant_disease.pt",
        "classes": "31 classes (Tomato, Potato, Rice, Cassava, Mango, Sugarcane)",
    },
    {
        "path": "plant_disease_peachfawn_better.pt",  # 15-class detection model
        "hf_repo": "peachfawn/yolov8-plant-disease-Better",
        "hf_file": "best.pt",
        "classes": "15 classes (Tomato, Potato, Pepper)",
    },
    {
        "path": "yolov8n-cls.pt",                # base YOLOv8 fallback
        "hf_repo": None,
        "hf_file": None,
        "classes": "base ImageNet classifier",
    },
]


# ──────────────────────────────────────────────
# Disease knowledge base loader
# ──────────────────────────────────────────────

def _load_disease_knowledge() -> Dict[str, Any]:
    """Load disease treatment knowledge from disease_classes.json."""
    try:
        with open(DISEASE_CLASSES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("diseases", {}), data.get("general_advice", {})
    except FileNotFoundError:
        logger.warning(f"Disease classes file not found: {DISEASE_CLASSES_FILE}")
        return {}, {}
    except Exception as e:
        logger.error(f"Failed to load disease knowledge: {e}")
        return {}, {}


# ──────────────────────────────────────────────
# Main Detector class
# ──────────────────────────────────────────────

class DiseaseDetector:
    """
    YOLOv8-based plant disease detector.

    Usage:
        detector = DiseaseDetector()
        results = detector.detect(image_bytes)
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._disease_db, self._general_advice = _load_disease_knowledge()
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self):
        """
        Auto-discover and load the best available plant disease model.

        Priority:
          1. YOLO_MODEL_PATH env var (explicit user override)
          2. Pre-trained models in yolo_weights/ (preferred: plant_disease_38cls.pt)
          3. Auto-download from HuggingFace if not present locally
          4. Fall back to base yolov8n-cls.pt
        """
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return

        # ── Priority 1: explicit env override ─────────────────────────────
        if YOLO_MODEL_PATH:
            path = Path(YOLO_MODEL_PATH)
            if not path.is_absolute():
                path = WEIGHTS_DIR.parent / path
            if path.exists():
                try:
                    self._model = YOLO(str(path))
                    self._model_path = str(path)
                    self._model_loaded = True
                    logger.info(f"Loaded model from YOLO_MODEL_PATH: {path}")
                    return
                except Exception as e:
                    logger.warning(f"YOLO_MODEL_PATH model failed to load: {e}")
            else:
                logger.warning(f"YOLO_MODEL_PATH not found: {path}")

        # ── Priority 2+3: auto-discover pre-trained plant disease models ──
        WEIGHTS_DIR.mkdir(exist_ok=True)
        for entry in PRETRAINED_MODELS:
            model_file = WEIGHTS_DIR / entry["path"]

            # Try to download if missing and HF repo is specified
            if not model_file.exists() and entry.get("hf_repo"):
                logger.info(f"Downloading {entry['path']} from {entry['hf_repo']} ...")
                model_file = self._download_hf_model(
                    entry["hf_repo"], entry["hf_file"], model_file
                )

            if model_file and model_file.exists():
                try:
                    self._model = YOLO(str(model_file))
                    self._model_path = str(model_file.name)
                    self._model_loaded = True
                    num_classes = len(self._model.names)
                    logger.info(
                        f"Loaded plant disease model: {model_file.name} "
                        f"({num_classes} classes) — {entry['classes']}"
                    )
                    return
                except Exception as e:
                    logger.warning(f"Model {model_file.name} failed: {e}")
                    continue

        logger.error("No plant disease model could be loaded.")

    @staticmethod
    def _download_hf_model(repo_id: str, filename: str, dest: Path) -> Optional[Path]:
        """Download a model file from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest.parent),
            )
            src = Path(downloaded)
            if src.exists():
                if src.name != dest.name:
                    shutil.copy(src, dest)
                    return dest
                return src
        except Exception as e:
            logger.warning(f"HuggingFace download failed ({repo_id}): {e}")
        return None

    # ── Image preprocessing ────────────────────────────────────────────────

    @staticmethod
    def _preprocess_image(image_data: bytes | str | Path) -> Image.Image:
        """
        Convert various input types to a PIL Image.
        Resizes if larger than MAX_IMAGE_SIZE to avoid OOM.
        """
        if isinstance(image_data, (str, Path)):
            img = Image.open(image_data)
        elif isinstance(image_data, bytes):
            import io
            img = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            img = image_data
        else:
            raise ValueError(f"Unsupported image input type: {type(image_data)}")

        # Convert to RGB (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if too large
        w, h = img.size
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        return img

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(
        self,
        image_data: bytes | str | Path,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Run plant disease detection on an image.

        Args:
            image_data: Raw bytes, file path, or PIL Image.
            top_k: Number of top disease candidates to return.

        Returns:
            {
              "success": bool,
              "detections": [...],          # top_k results
              "primary_disease": {...},     # highest-confidence detection
              "treatment": {...},           # from disease_classes.json
              "general_advice": {...},
              "inference_time_ms": float,
              "model": str,
              "error": str | None
            }
        """
        if not self._model_loaded:
            # Try reloading
            self._load_model()
            if not self._model_loaded:
                return self._error_response("YOLOv8 model is not loaded. Check YOLO_MODEL_PATH configuration.")

        try:
            start = time.perf_counter()

            # Preprocess
            img = self._preprocess_image(image_data)

            # Save to temp file (ultralytics works best with file paths)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                img.save(tmp_path, format="JPEG", quality=95)

            try:
                results = self._model(tmp_path, verbose=False)
            finally:
                os.unlink(tmp_path)

            inference_ms = (time.perf_counter() - start) * 1000

            # Parse results
            detections = self._parse_results(results, top_k)

            if not detections:
                return {
                    "success": True,
                    "detections": [],
                    "primary_disease": None,
                    "treatment": None,
                    "general_advice": self._general_advice,
                    "inference_time_ms": round(inference_ms, 1),
                    "model": self._get_model_name(),
                    "message": "No diseases detected above confidence threshold. The plant appears healthy or the image quality may be insufficient.",
                    "error": None,
                }

            primary = detections[0]
            treatment = self._get_treatment(primary["class_name"])

            return {
                "success": True,
                "detections": detections,
                "primary_disease": primary,
                "treatment": treatment,
                "general_advice": self._general_advice,
                "inference_time_ms": round(inference_ms, 1),
                "model": self._get_model_name(),
                "message": self._build_summary(primary, treatment),
                "error": None,
            }

        except Exception as e:
            logger.exception(f"Disease detection error: {e}")
            return self._error_response(str(e))

    # ── Result parsing ─────────────────────────────────────────────────────

    def _parse_results(self, results, top_k: int) -> List[Dict[str, Any]]:
        """
        Parse Ultralytics YOLO results into structured detections.

        Handles both classification (results[0].probs) and
        detection (results[0].boxes) outputs.
        """
        detections = []

        for result in results:
            # ── Classification model output (yolov8-cls) ─────────────────
            if hasattr(result, "probs") and result.probs is not None:
                probs = result.probs
                # top5 gives indices and confidences
                top_indices = probs.top5
                top_confs = probs.top5conf.tolist() if hasattr(probs.top5conf, "tolist") else list(probs.top5conf)
                names = result.names  # {0: 'class_name', ...}

                for idx, conf in zip(top_indices, top_confs):
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    class_name = names.get(int(idx), f"class_{idx}")
                    detections.append({
                        "class_id": int(idx),
                        "class_name": class_name,
                        "confidence": round(float(conf), 4),
                        "display_name": self._get_display_name(class_name),
                        "hindi_name": self._get_hindi_name(class_name),
                        "severity": self._get_severity(class_name),
                        "bbox": None,  # classification has no bbox
                    })
                    if len(detections) >= top_k:
                        break

            # ── Object detection model output (yolov8-det) ───────────────
            elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes):
                boxes = result.boxes
                names = result.names

                # Sort by confidence descending
                confs = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else list(boxes.conf)
                cls_ids = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)
                xyxy = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else []

                sorted_indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)

                for i in sorted_indices[:top_k]:
                    conf = confs[i]
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    class_id = int(cls_ids[i])
                    class_name = names.get(class_id, f"class_{class_id}")
                    bbox = xyxy[i] if i < len(xyxy) else None

                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(float(conf), 4),
                        "display_name": self._get_display_name(class_name),
                        "hindi_name": self._get_hindi_name(class_name),
                        "severity": self._get_severity(class_name),
                        "bbox": {
                            "x1": round(bbox[0], 1),
                            "y1": round(bbox[1], 1),
                            "x2": round(bbox[2], 1),
                            "y2": round(bbox[3], 1),
                        } if bbox else None,
                    })

        return detections

    # ── Knowledge base lookups ─────────────────────────────────────────────

    def _get_treatment(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Look up treatment info from disease knowledge base."""
        # Try exact match
        if class_name in self._disease_db:
            return self._disease_db[class_name]

        # Try normalised match (case-insensitive, replace spaces)
        normalised = class_name.lower().replace(" ", "_").replace("-", "_")
        for key, val in self._disease_db.items():
            if key.lower().replace("-", "_") == normalised:
                return val

        # Partial match on common_name
        for key, val in self._disease_db.items():
            if class_name.lower() in val.get("common_name", "").lower():
                return val

        return None

    def _get_display_name(self, class_name: str) -> str:
        info = self._get_treatment(class_name)
        return info["common_name"] if info else class_name.replace("___", " — ").replace("_", " ").title()

    def _get_hindi_name(self, class_name: str) -> str:
        info = self._get_treatment(class_name)
        return info.get("hindi_name", "") if info else ""

    def _get_severity(self, class_name: str) -> str:
        info = self._get_treatment(class_name)
        return info.get("severity", "unknown") if info else "unknown"

    def _get_model_name(self) -> str:
        return getattr(self, "_model_path", None) or YOLO_MODEL_PATH or "yolov8_plant_disease.pt"

    def _build_summary(self, primary: Dict, treatment: Optional[Dict]) -> str:
        name = primary["display_name"]
        conf = primary["confidence"] * 100
        severity = primary["severity"].upper()

        summary = f"Detected: **{name}** ({conf:.1f}% confidence) — Severity: {severity}."
        if treatment:
            summary += f" Cause: {treatment.get('cause', 'N/A')}."
            summary += f" Recommended action: {treatment.get('organic_treatment', '')}"
        return summary

    @staticmethod
    def _error_response(message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "detections": [],
            "primary_disease": None,
            "treatment": None,
            "general_advice": {},
            "inference_time_ms": 0.0,
            "model": "",
            "message": message,
            "error": message,
        }

    # ── Utility ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if model is loaded and ready."""
        return self._model_loaded

    def get_supported_classes(self) -> List[str]:
        """Return list of disease class names from knowledge base."""
        return list(self._disease_db.keys())


# ── Module-level singleton ────────────────────────────────────────────────────

_detector_instance: Optional[DiseaseDetector] = None


def get_detector() -> DiseaseDetector:
    """Return the module-level singleton DiseaseDetector (lazy init)."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DiseaseDetector()
    return _detector_instance
