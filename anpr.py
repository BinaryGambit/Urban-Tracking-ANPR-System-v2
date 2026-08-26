import os

# Paddle settings
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import json
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR


YOLO_MODEL = "model/plate_detector.pt"
IMAGE_PATH = "test_dh_car_front.jpg"

OUTPUT_DIR = "output"
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

CONF_THRESHOLD = 0.05


os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)



print("Loading YOLO......")

detector = YOLO(YOLO_MODEL)

print("Loading PaddleOCR")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

print("Models Loaded\n")



image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


annotated_image = image.copy()



results = detector.predict(
    source=image,
    conf=CONF_THRESHOLD,
    verbose=False
)



detections = []

plate_count = 0



for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

       
        coords = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = coords.astype(int)

        detection_confidence = float(
            box.conf[0].cpu().item()
        )


       

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)


        
        plate = image[y1:y2, x1:x2]

        if plate.size == 0:
            continue


        plate_count += 1


        

        crop_filename = f"plate_{plate_count:03d}.jpg"

        crop_path = os.path.join(
            CROP_DIR,
            crop_filename
        )

        cv2.imwrite(
            crop_path,
            plate
        )


        

        ocr_results = ocr.predict(plate)

        recognized_text = ""
        ocr_confidence = 0.0


        for ocr_result in ocr_results:

            texts = ocr_result.get(
                "rec_texts",
                []
            )

            scores = ocr_result.get(
                "rec_scores",
                []
            )

            if texts:

                recognized_text = texts[0]

                if scores:

                    ocr_confidence = float(
                        scores[0]
                    )

                break


        
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = (
            f"{recognized_text} "
            f"({ocr_confidence:.2f})"
        )


        cv2.putText(
            annotated_image,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        detection = {

            "plate": recognized_text,

            "detector_confidence":
                detection_confidence,

            "ocr_confidence":
                ocr_confidence,

            "bounding_box": {
                 "x1": int(x1),
                 "y1": int(y1),
                 "x2": int(x2),
                 "y2": int(y2)
            },

            "crop": crop_path,

            "timestamp":
                datetime.now().isoformat()

        }


        detections.append(
            detection
        )

        print(
            f"Plate {plate_count}: "
            f"{recognized_text}"
        )

        print(
            f"Detection confidence: "
            f"{detection_confidence:.3f}"
        )

        print(
            f"OCR confidence: "
            f"{ocr_confidence:.3f}"
        )

        print(
            f"Crop saved: "
            f"{crop_path}\n"
        )

annotated_path = os.path.join(
    ANNOTATED_DIR,
    "result.jpg"
)

cv2.imwrite(
    annotated_path,
    annotated_image
)

json_path = os.path.join(
    OUTPUT_DIR,
    "results.json"
)


output_data = {

    "image": IMAGE_PATH,

    "processed_at":
        datetime.now().isoformat(),

    "total_plates":
        plate_count,

    "detections":
        detections

}


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_data,
        f,
        indent=4
    )




print(
    f"Total Plates Detected: "
    f"{plate_count}"
)

print(
    f"Annotated image: "
    f"{annotated_path}"
)

print(
    f"JSON results: "
    f"{json_path}"
)