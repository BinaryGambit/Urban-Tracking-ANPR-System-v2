import cv2

CAMERAS = {
    "CAM_01":"http://10.200.199.25:8080/video",
    "CAM_02":"http://10.200.199.9:8080/video",
    "CAM_03":"http://10.200.199.35:8080/video",
}

def connect_cameras():
    cameras = {}

    for camera_id,url in CAMERAS.items():
        print(f"\nConnecting to {camera_id}")

        cap = cv2.VideoCapture(url)

        if cap.isOpened():
            print(f"{camera_id} Connected")
            cameras[camera_id] = cap

        else:
            print(f"{camera_id} failed to connect")

    return cameras

def main():
    cameras = connect_cameras()

    if not cameras:
        print("No Cameras Connected")
        return

    print("\nUrban Tracking")
    print("\nPress Q to Stop")

    while True:
        for camera_id, cap in cameras.items():
            ret,frame = cap.read()

            if not ret:
                print(f"{camera_id} Disconnected")
                continue

            cv2.imshow(camera_id,frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in cameras.values():
        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()