"""Quick smoke test for the YOLOv8 disease detector."""
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

os.environ['YOLO_MODEL_PATH'] = 'yolo_weights/yolov8n-cls.pt'
os.environ['YOLO_CONFIDENCE_THRESHOLD'] = '0.01'  # low threshold to force results

from disease_detector.detector import DiseaseDetector
from PIL import Image, ImageDraw

detector = DiseaseDetector()
print(f"Model ready: {detector.is_ready}")
print(f"Supported disease classes in KB: {len(detector.get_supported_classes())}")

# Create a synthetic test image (green leaf)
img = Image.new('RGB', (224, 224), color=(34, 120, 40))
draw = ImageDraw.Draw(img)
draw.ellipse([50, 50, 174, 174], fill=(20, 100, 25))
buf = io.BytesIO()
img.save(buf, format='JPEG')
img_bytes = buf.getvalue()

result = detector.detect(img_bytes, top_k=3)
print(f"Detection success: {result['success']}")
print(f"Inference time:    {result['inference_time_ms']:.1f}ms")
print(f"Detections found:  {len(result['detections'])}")
for d in result['detections']:
    print(f"  - {d['display_name']} ({d['confidence']*100:.1f}% confidence) [{d['severity']}]")

if result['treatment']:
    print(f"Primary treatment: {result['treatment'].get('organic_treatment', 'N/A')[:80]}...")

print("\nSmoke test PASSED")
