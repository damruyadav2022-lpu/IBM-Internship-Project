"""
End-to-end test: API endpoints + YOLOv8 disease model.
Run: python scripts/test_api.py
"""
import io
import json
import sys
import requests
from PIL import Image

BASE = "http://localhost:8000"
results = []

def check(name, ok, detail=""):
    icon = "PASS" if ok else "FAIL"
    results.append((icon, name, detail))
    print(f"  [{icon}] {name}: {detail}")

print("\n=== KrishiRakshak — API + Model Test ===\n")

# ── 1. Health endpoints ───────────────────────────────────────────────────────
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    check("GET /health", r.status_code == 200, str(r.json()))
except Exception as e:
    check("GET /health", False, f"ERROR: {e}")

try:
    r = requests.get(f"{BASE}/api/health", timeout=5)
    check("GET /api/health", r.status_code == 200, str(r.json()))
except Exception as e:
    check("GET /api/health", False, f"ERROR: {e}")

# ── 2. Data endpoints ─────────────────────────────────────────────────────────
try:
    r = requests.get(f"{BASE}/api/crops", timeout=5)
    crops = r.json().get("crops", [])
    check("GET /api/crops", r.status_code == 200, f"{len(crops)} crops returned")
except Exception as e:
    check("GET /api/crops", False, f"ERROR: {e}")

try:
    r = requests.get(f"{BASE}/api/markets", timeout=5)
    markets = r.json().get("markets", [])
    check("GET /api/markets", r.status_code == 200, f"{len(markets)} markets returned")
except Exception as e:
    check("GET /api/markets", False, f"ERROR: {e}")

try:
    r = requests.get(f"{BASE}/api/diseases", timeout=5)
    total = r.json().get("total", 0)
    check("GET /api/diseases", r.status_code == 200, f"{total} disease classes in KB")
except Exception as e:
    check("GET /api/diseases", False, f"ERROR: {e}")

# ── 3. YOLOv8 disease detection ───────────────────────────────────────────────
try:
    # Create a synthetic green leaf image
    img = Image.new("RGB", (224, 224), color=(34, 120, 40))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)

    r = requests.post(
        f"{BASE}/api/detect-disease",
        files={"file": ("test_leaf.jpg", buf, "image/jpeg")},
        params={"top_k": 3},
        timeout=30,
    )
    det = r.json()
    model_name = det.get("model", "?")[:35]
    n_det = len(det.get("detections", []))
    inf_ms = det.get("inference_time_ms", 0)
    check(
        "POST /api/detect-disease",
        r.status_code == 200 and det.get("success"),
        f"{n_det} detections | {inf_ms:.0f}ms | model={model_name}",
    )
    # Show detections
    for d in det.get("detections", []):
        conf = d.get("confidence", 0) * 100
        print(f"         -> {d['display_name']} ({conf:.1f}%) [{d.get('severity','?')}]")
except Exception as e:
    check("POST /api/detect-disease", False, f"ERROR: {e}")

# ── 4. Local YOLOv8 model direct check ───────────────────────────────────────
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    os.environ.setdefault("YOLO_MODEL_PATH", "yolo_weights/plant_disease_38cls.pt")
    from disease_detector.detector import DiseaseDetector
    det_local = DiseaseDetector()
    model_classes = len(det_local._model.names) if det_local._model else 0
    kb_classes = len(det_local.get_supported_classes())
    check(
        "YOLOv8 model (local)",
        det_local.is_ready,
        f"model={det_local._get_model_name()[:35]} | {model_classes} model classes | {kb_classes} KB entries",
    )
except Exception as e:
    check("YOLOv8 model (local)", False, f"ERROR: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"Results: {passed} passed, {failed} failed")

if failed == 0:
    print("\n✅  All tests passed — API and model are working correctly!")
else:
    print("\n⚠️  Some tests failed — check the errors above.")
    sys.exit(1)
