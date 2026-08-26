from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "model/plate_detector.pt"
IMAGE_PATH = "test_plate.jpg"

os.makedirs("output/crops", exist_ok=True)

model = YOLO(MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Couldn't read {IMAGE_PATH}")

results = model.predict(
    source=image,
    conf=0.25,
    verbose=False
)

plate_count = 0

for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        
        plate = image[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        plate_count += 1

        filename = f"output/crops/plate_{plate_count}.jpg"

        cv2.imwrite(filename, plate)

        print(
            f"Plate {plate_count}: "
            f"confidence={confidence:.3f}"
        )

        print(f"Saved: {filename}")

print(f"\nTotal plates detected: {plate_count}")