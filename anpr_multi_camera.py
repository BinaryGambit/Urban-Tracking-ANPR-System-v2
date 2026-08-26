import os
# Paddle/PIR compatibility
os.environ["FLAGS_enable_pir_api"] = "0"
import cv2
import json
import re
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

CONF_THRESHOLD = 0.25
YOLO_IMAGE_SIZE = 640
FRAME_SKIP = 9 #Process 1 frame very FRAME_SKIP frames
CAMERAS = {
    "CAM_02": "http://10.200.231.125:8080/video",
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
    # Force reasonable buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
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

cached_detections = {cam_id: [] for cam_id in captures} #For caching the OCR readings during FRAME_SKIP frames #TODO: Implement caching logic

try:
    while True:
        for camera_id, cap in captures.items():
            ret, frame = cap.read()
            if not ret:
                print(f"\n[{camera_id}] Failed to read frame")
                continue
            results = detector.predict(source=frame,conf=CONF_THRESHOLD,imgsz=YOLO_IMAGE_SIZE,iou=0.45,verbose=False)
            result = results[0]

            detection_count = (len(result.boxes) if result.boxes is not None else 0)
            if detection_count > 0:
                print(f"\n[{camera_id}] Detections: {detection_count}")


            for box in result.boxes:
                detection_confidence = float(box.conf[0].cpu().item())
                coords = (box.xyxy[0].cpu().numpy())
                x1, y1, x2, y2 = (coords.astype(int))
                h, w = frame.shape[:2]
                x1 = max(0,min(x1, w - 1))
                y1 = max(0,min(y1, h - 1))
                x2 = max(0,min(x2, w))
                y2 = max(0,min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue
                plate = frame[y1:y2,x1:x2]
                if plate.size == 0:
                    continue
                try:
                    ocr_results = ocr.predict(plate)
                except Exception as e:
                    print(f"OCR error: {e}") #Make errors more precise if possible
                    continue

                plate_text = ""
                ocr_confidence = 0.0
                for ocr_result in ocr_results:
                    texts = ocr_result.get("rec_texts",[])
                    scores = ocr_result.get("rec_scores",[])
                    if len(texts) > 0:
                        plate_text = str(texts[0]).upper()
                        if len(scores) > 0:
                            ocr_confidence = float(scores[0])
                        break

                plate_text = re.sub(r"[^A-Z0-9]", "",plate_text) #TODO: See if regex can be changed to detect only number plate characters
                now = datetime.now()
                timestamp = now.isoformat(timespec="milliseconds")

                cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),3)

                if plate_text:
                    label = f"{plate_text} | {detection_confidence:.2f} | OCR {ocr_confidence:.2f}"
                else:
                    label = f"PLATE | {detection_confidence:.2f}"

                cv2.putText(frame,label, (x1,max(35,y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2)

                if plate_text:
                    plate_counter += 1
                    crop_name = f"{camera_id}_{plate_counter:05d}_{plate_text}.jpg"

                    crop_path = (CROPS_DIR /crop_name)
                    cv2.imwrite(str(crop_path),plate)

                    event = {
                        "camera_id":camera_id,
                        "plate":plate_text,
                        "timestamp": timestamp,
                        "detection_confidence": float(detection_confidence),
                        "ocr_confidence": float(ocr_confidence),
                        "bounding_box": {
                            "x1":int(x1),
                            "y1":int(y1),
                            "x2": int(x2),
                            "y2":int(y2)
                        },
                        "crop":str(crop_path)
                    }

                    events.append(event)
                    print(f"[{timestamp}] [{camera_id}] {plate_text} | DET {detection_confidence:.3f} | OCR {ocr_confidence:.3f}")

            cv2.putText(frame,camera_id, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, current_time, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(camera_id,frame)

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
    json.dump(output,f,indent=4)

print()
print("==============================")
print("ANPR Finished")
print("==============================")

print(f"Total events: {len(events)}")
print(f"JSON saved: {JSON_PATH}")