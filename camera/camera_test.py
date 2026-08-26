import cv2

URL = "http://10.200.95.28:8080/video"

cap = cv2.VideoCapture(URL)

if not cap.isOpened():
    print("Couldn't Connect to Cam_01")
    exit()

print("CAM_01 Connected")
print("Press Q to Stop")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to Recieve Frame")
        break

    cv2.imshow("Urban Tracking - CAM_01",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()