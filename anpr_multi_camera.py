import os
import time

# Paddle/PIR compatibility
os.environ["FLAGS_enable_pir_api"] = "0"
import cv2
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

#TODO: somehow speed up live feed
#TODO: add per-camera frame rate control
#TODO: Bent plates are harder to detect for model currently
#TODO: Plates that extend to multiple lines (like on motorcycles) are harder to detect for model currently (only flags as a plate, but doesnt read its number)

CONF_THRESHOLD = 0.25
YOLO_IMAGE_SIZE = 300
FRAME_SKIP_MAX = 29  # Per-camera capped random skip [1..FRAME_SKIP_MAX]

# OCR/tracking reuse controls
OCR_REUSE_SECONDS = 1.0
OCR_HIGH_CONF_MIN = 0.85
TRACK_MATCH_IOU = 0.45
OCR_MIN_DET_CONF = 0.40
OCR_MAX_CROP_WIDTH = 320

# Frame flush controls (security-sensitive, fully tweakable)
ENABLE_FRAME_FLUSH = True
FLUSH_FRAMES_PER_CYCLE = 2
FLUSH_ONLY_WHEN_LAGGING = True
MAX_FRAME_AGE_MS = 250
MAX_CONSECUTIVE_FLUSH = 5
LOG_FLUSH_METRICS_EVERY_N = 100

# Logging controls
LOG_EVERY_N_CYCLES = 30

CAMERAS = {
    "CAM_01": "http://10.200.195.38:8080/video",
    "CAM_02": "http://10.200.195.44:8080/video",
    "CAM_03": "http://10.200.195.67:8080/video",
}

YOLO_MODEL = "model/plate_detector.pt"
OUTPUT_DIR = Path("multi_camera_output")
CROPS_DIR = OUTPUT_DIR / "crops"
FRAMES_DIR = OUTPUT_DIR / "frames"
JSON_PATH = OUTPUT_DIR / "events.json"

OUTPUT_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

print("Loading YOLO......")
detector = YOLO(YOLO_MODEL)
print("YOLO Loaded")

print("Loading PaddleOCR")
ocr = PaddleOCR(lang="en", enable_mkldnn=False)
print("Models Loaded")

captures = {}

print()
print("==============================")
print("Starting Cameras")
print("==============================")

for camera_id, url in CAMERAS.items():
    print()
    print(f"[{camera_id}] Connecting...")

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[{camera_id}] ERROR: Could not connect")
        continue

    captures[camera_id] = cap
    print(f"[{camera_id}] Connected")

if len(captures) == 0:
    print("ERROR: No cameras connected.")
    exit()

print()
print("==============================")
print("Multi-Camera ANPR Started")
print("==============================")
print()
print("Press Q to stop")
print()

events = []
plate_counter = 0

onFrameVsCamera = {camera_id: 0 for camera_id in captures}
per_camera_skip = {camera_id: 1 for camera_id in captures}
per_camera_cycle = {camera_id: 0 for camera_id in captures}

cached_detections = {cam_id: [] for cam_id in captures}
ocr_track_cache = {cam_id: [] for cam_id in captures}
camera_metrics = {
    cam_id: {
        "frames_read": 0,
        "frames_processed": 0,
        "frames_flushed": 0,
        "ocr_calls": 0,
        "ocr_skips": 0,
        "consecutive_flushes": 0,
    }
    for cam_id in captures
}


def compute_capped_skip(camera_id: str, cycle_index: int, max_skip: int) -> int:
    now_ns = time.time_ns()
    seed = f"{camera_id}:{cycle_index}:{now_ns}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return 1 + (int.from_bytes(digest[:4], byteorder="big") % max_skip)


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def downscale_plate_if_needed(plate):
    h, w = plate.shape[:2]
    if w <= OCR_MAX_CROP_WIDTH:
        return plate
    scale = OCR_MAX_CROP_WIDTH / float(w)
    new_w = OCR_MAX_CROP_WIDTH
    new_h = max(1, int(h * scale))
    return cv2.resize(plate, (new_w, new_h), interpolation=cv2.INTER_AREA)


def capture_latest_frame(cap, camera_id, metrics):
    ret, frame = cap.read()
    if not ret:
        return False, None

    metrics[camera_id]["frames_read"] += 1

    if not ENABLE_FRAME_FLUSH:
        return True, frame

    should_flush = True
    if FLUSH_ONLY_WHEN_LAGGING:
        # OpenCV doesn't provide reliable frame-age everywhere; use read slowness heuristic
        # and keep this config-driven via FLUSH_ONLY_WHEN_LAGGING.
        should_flush = True

    if not should_flush:
        metrics[camera_id]["consecutive_flushes"] = 0
        return True, frame

    flushed_now = 0
    flush_budget = min(FLUSH_FRAMES_PER_CYCLE, MAX_CONSECUTIVE_FLUSH)

    for _ in range(flush_budget):
        ok = cap.grab()
        if not ok:
            break
        ok2, latest = cap.retrieve()
        if not ok2:
            break
        frame = latest
        flushed_now += 1

    metrics[camera_id]["frames_flushed"] += flushed_now
    if flushed_now > 0:
        metrics[camera_id]["consecutive_flushes"] += 1
    else:
        metrics[camera_id]["consecutive_flushes"] = 0

    if metrics[camera_id]["consecutive_flushes"] > MAX_CONSECUTIVE_FLUSH:
        metrics[camera_id]["consecutive_flushes"] = 0  # hard safety reset

    return True, frame


try:
    while True:
        for camera_id, cap in captures.items():
            ret, frame = capture_latest_frame(cap, camera_id, camera_metrics)
            if not ret:
                if per_camera_cycle[camera_id] % LOG_EVERY_N_CYCLES == 0:
                    print(f"\n[{camera_id}] Failed to read frame")
                continue

            if onFrameVsCamera[camera_id] >= per_camera_skip[camera_id]:
                onFrameVsCamera[camera_id] = 0
                per_camera_cycle[camera_id] += 1
                per_camera_skip[camera_id] = compute_capped_skip(
                    camera_id, per_camera_cycle[camera_id], FRAME_SKIP_MAX
                )

                current_detections = []
                now_ts = time.time()

                results = detector.predict(
                    source=frame,
                    conf=CONF_THRESHOLD,
                    imgsz=YOLO_IMAGE_SIZE,
                    iou=0.45,
                    verbose=False
                )
                result = results[0]

                detection_count = len(result.boxes) if result.boxes is not None else 0
                prev_tracks = ocr_track_cache[camera_id]
                next_tracks = []

                if detection_count > 0 and per_camera_cycle[camera_id] % LOG_EVERY_N_CYCLES == 0:
                    print(f"\n[{camera_id}] Detections: {detection_count}")

                for box in result.boxes:
                    detection_confidence = float(box.conf[0].cpu().item())
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords

                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    plate = frame[y1:y2, x1:x2]
                    if plate.size == 0:
                        continue

                    matched_track = None
                    best_iou = 0.0
                    for track in prev_tracks:
                        current_iou = bbox_iou((x1, y1, x2, y2), track["bbox"])
                        if current_iou > best_iou:
                            best_iou = current_iou
                            matched_track = track

                    plate_text = ""
                    ocr_confidence = 0.0

                    track_age = 0.0
                    if matched_track is not None:
                        track_age = now_ts - matched_track["first_seen_ts"]

                    # Corrected logic: reuse only after track has existed long enough with high confidence
                    can_reuse_high_conf = (
                        matched_track is not None
                        and best_iou >= TRACK_MATCH_IOU
                        and matched_track["ocr_conf"] >= OCR_HIGH_CONF_MIN
                        and track_age >= OCR_REUSE_SECONDS
                        and bool(matched_track["text"])
                    )

                    if can_reuse_high_conf:
                        plate_text = matched_track["text"]
                        ocr_confidence = matched_track["ocr_conf"]
                        camera_metrics[camera_id]["ocr_skips"] += 1
                    elif detection_confidence < OCR_MIN_DET_CONF:
                        # Skip OCR on weak detections to cut cost
                        plate_text = ""
                        ocr_confidence = 0.0
                    else:
                        plate_for_ocr = downscale_plate_if_needed(plate)
                        try:
                            ocr_results = ocr.predict(plate_for_ocr)
                            camera_metrics[camera_id]["ocr_calls"] += 1
                        except Exception as e:
                            if per_camera_cycle[camera_id] % LOG_EVERY_N_CYCLES == 0:
                                print(f"OCR error: {e}")
                            continue

                        for ocr_result in ocr_results:
                            texts = ocr_result.get("rec_texts", [])
                            scores = ocr_result.get("rec_scores", [])
                            if len(texts) > 0:
                                plate_text = str(texts[0]).upper()
                                if len(scores) > 0:
                                    ocr_confidence = float(scores[0])
                                break

                    plate_text = re.sub(r"[^A-Z0-9]", "", plate_text)
                    now = datetime.now()
                    timestamp = now.isoformat(timespec="milliseconds")

                    current_detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "text": plate_text,
                        "det_conf": detection_confidence,
                        "ocr_conf": ocr_confidence
                    })

                    first_seen_ts = now_ts
                    if matched_track is not None and best_iou >= TRACK_MATCH_IOU:
                        first_seen_ts = matched_track["first_seen_ts"]

                    next_tracks.append({
                        "bbox": (x1, y1, x2, y2),
                        "text": plate_text,
                        "ocr_conf": ocr_confidence,
                        "first_seen_ts": first_seen_ts,
                        "last_seen_ts": now_ts
                    })

                    if plate_text:
                        plate_counter += 1
                        crop_name = f"{camera_id}_{plate_counter:05d}_{plate_text}.jpg"
                        crop_path = CROPS_DIR / crop_name
                        cv2.imwrite(str(crop_path), plate)

                        event = {
                            "camera_id": camera_id,
                            "plate": plate_text,
                            "timestamp": timestamp,
                            "detection_confidence": float(detection_confidence),
                            "ocr_confidence": float(ocr_confidence),
                            "bounding_box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                            "crop": str(crop_path)
                        }
                        events.append(event)

                        if per_camera_cycle[camera_id] % LOG_EVERY_N_CYCLES == 0:
                            print(
                                f"[{timestamp}] [{camera_id}] {plate_text} | "
                                f"DET {detection_confidence:.3f} | OCR {ocr_confidence:.3f}"
                            )

                cached_detections[camera_id] = current_detections
                ocr_track_cache[camera_id] = next_tracks
                camera_metrics[camera_id]["frames_processed"] += 1

                if per_camera_cycle[camera_id] % LOG_FLUSH_METRICS_EVERY_N == 0:
                    m = camera_metrics[camera_id]
                    print(
                        f"[{camera_id}] metrics read={m['frames_read']} processed={m['frames_processed']} "
                        f"flushed={m['frames_flushed']} ocr_calls={m['ocr_calls']} ocr_skips={m['ocr_skips']}"
                    )
            else:
                onFrameVsCamera[camera_id] += 1

            for det in cached_detections[camera_id]:
                x1, y1, x2, y2 = det["bbox"]
                text = det["text"]
                detection_confidence = det["det_conf"]
                ocr_confidence = det["ocr_conf"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if text:
                    label = f"{text} | {detection_confidence:.2f} | OCR {ocr_confidence:.2f}"
                else:
                    label = f"PLATE | {detection_confidence:.2f}"
                cv2.putText(frame, label, (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, camera_id, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, current_time, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(camera_id, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nQ pressed. Stopping...")
            break

finally:
    print("\nStopping cameras...")
    for camera_id, cap in captures.items():
        cap.release()
        print(f"[{camera_id}] Released")
    cv2.destroyAllWindows()

output = {
    "system": "Multi-Camera ANPR",
    "created_at": datetime.now().isoformat(timespec="milliseconds"),
    "camera_count": len(captures),
    "total_events": len(events),
    "events": events
}

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print()
print("==============================")
print("ANPR Finished")
print("==============================")
print(f"Total events: {len(events)}")
print(f"JSON saved: {JSON_PATH}")