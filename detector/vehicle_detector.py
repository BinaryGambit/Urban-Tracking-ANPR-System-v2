import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
CAMERA_URL = "http://10.200.199.25:8080/video"


VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("Couldn't Connect to CAM_01")
    exit()

print("CAM_01 Connected")
print("Vehicle Tracking Started")
print("Press Q to Stop")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id not in VEHICLE_CLASSES:
                continue

            vehicle_type = VEHICLE_CLASSES[class_id]

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if box.id is not None:

                track_id = int(box.id[0])

            else:
                track_id = -1

            label = (
                f"ID {track_id} | "
                f"{vehicle_type} | "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            print(
                f"ID={track_id} | "
                f"{vehicle_type} | "
                f"confidence={confidence:.2f}"
            )

    cv2.imshow(
        "Urban Tracking - Vehicle Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()