"""
server.py — connects the ANPR/OCR Python pipeline (YOLO + EasyOCR,
same detector model used by anpr.py / plate_detect.py) to the TrackNet website.

What it does:
  1. Serves this whole project folder as static files, so the website's
     existing relative fetches (../camera_config.json, ../map_ouput/routes.json,
     ../multi_camera_output/events.json, ../output/results.json, crop images,
     etc.) all resolve correctly when you open the site through this server
     instead of as a local file:// page.
  2. Exposes POST /api/anpr/upload — accepts an image from the browser,
     runs detect+OCR, saves the crop and an annotated frame, appends the
     result to output/results.json, and returns JSON the website can
     render immediately.

Note: this uses EasyOCR instead of PaddleOCR. PaddleOCR's CPU path is
known to be extremely slow (many minutes per image) on Apple Silicon
Macs; EasyOCR is CPU-friendly and typically finishes in a couple of
seconds per plate crop.

Run:
    pip install -r requirements.txt
    python server.py

Then open:
    http://localhost:5001/Website/sihweb4.html
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CROP_DIR = OUTPUT_DIR / "crops"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
RESULTS_JSON = OUTPUT_DIR / "results.json"

CROP_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.05

print("Loading YOLO plate detector...")
detector = YOLO(str(BASE_DIR / "model" / "plate_detector.pt"))
print("Loading EasyOCR (first run downloads model weights, ~1-2 min once)...")
ocr = easyocr.Reader(["en"], gpu=False)
print("Models loaded.\n")

# static_url_path="" serves this whole folder at the site root, so
# Website/sihweb4.html's "../camera_config.json" etc. resolve for free.
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
CORS(app)


def _next_plate_index():
    return len(list(CROP_DIR.glob("plate_*.jpg")))


def run_anpr(image):
    """Runs the plate detector + OCR on a single BGR image (numpy array).
    This mirrors anpr.py's loop, just packaged as a reusable function."""
    annotated = image.copy()
    h, w = image.shape[:2]

    results = detector.predict(source=image, conf=CONF_THRESHOLD, verbose=False)

    detections = []
    plate_count = _next_plate_index()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box.conf[0].cpu().item())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            plate_count += 1
            crop_name = f"plate_{plate_count:03d}.jpg"
            crop_path = CROP_DIR / crop_name
            cv2.imwrite(str(crop_path), plate_img)

            recognized_text, ocr_conf = "", 0.0
            # EasyOCR returns a list of (bbox, text, confidence) per detected
            # text region. A plate can have multiple regions (e.g. state code
            # on its own line), so join them left-to-right and average confidence.
            ocr_results = ocr.readtext(plate_img)
            if ocr_results:
                ocr_results.sort(key=lambda r: r[0][0][0])  # sort by left x-coord
                recognized_text = " ".join(r[1] for r in ocr_results).strip()
                ocr_conf = sum(r[2] for r in ocr_results) / len(ocr_results)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"{recognized_text} ({ocr_conf:.2f})",
                        (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            detections.append({
                "plate": recognized_text,
                "detector_confidence": det_conf,
                "ocr_confidence": ocr_conf,
                "bounding_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                # forward-slash, relative to project root, so it works as a URL
                "crop": crop_path.relative_to(BASE_DIR).as_posix(),
                "timestamp": datetime.now().isoformat(),
            })

    return detections, annotated


def _append_to_results(image_name, detections):
    data = {"detections": []}
    if RESULTS_JSON.exists():
        try:
            data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    data.setdefault("detections", [])
    for d in detections:
        entry = dict(d)
        entry["image"] = image_name
        data["detections"].append(entry)
    data["total_plates"] = len(data["detections"])
    data["processed_at"] = datetime.now().isoformat()
    RESULTS_JSON.write_text(json.dumps(data, indent=4), encoding="utf-8")


@app.route("/api/anpr/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "could not decode image"}), 400

    detections, annotated = run_anpr(image)

    annotated_name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    annotated_path = ANNOTATED_DIR / annotated_name
    cv2.imwrite(str(annotated_path), annotated)

    _append_to_results(file.filename, detections)

    return jsonify({
        "total_plates": len(detections),
        "detections": detections,
        "annotated": annotated_path.relative_to(BASE_DIR).as_posix(),
    })


@app.route("/api/anpr/results", methods=["GET"])
def get_results():
    if not RESULTS_JSON.exists():
        return jsonify({"total_plates": 0, "detections": []})
    return jsonify(json.loads(RESULTS_JSON.read_text(encoding="utf-8")))


if __name__ == "__main__":
    print("=" * 60)
    print("ANPR server running")
    print("=" * 60)
    print("\nOpen the site at:")
    print("  http://localhost:5001/Website/sihweb4.html\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
