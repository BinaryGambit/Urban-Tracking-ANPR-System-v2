import os

os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR

IMAGE_PATH = r"output\\crops\\plate_1.jpg"

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

result = ocr.predict(IMAGE_PATH)

print("\nOCR RESULT:")

for res in result:
    print(res)