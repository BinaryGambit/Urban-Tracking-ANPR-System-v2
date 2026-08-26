import cv2
import threading
import time

class Camera:

    def __init__(self,camera_id,url):
        self.camera_id = camera_id
        self.url = url

        self.cap = None
        self.frame = None

        self.running = False
        self.thread = None

        self.Lock = threading.Lock()

    def start(self):
        print(f"[{self.camera_id}]",f"\nConnecting......")

        self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            print(f"[{self.camera_id}]",f"\nFailed")
            return False

        self.running = True

        self.thread = threading.Thread(
            target = self._update,
            daemon=True
        )

        self.thread.start()

        print(f"[{self.camera_id}]",f"\nConnected")

        return True

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                print(f"[{self.camera_id}]",f"\t\tFrame Failed")

                time.sleep(1)

                continue

            while self.Lock:
                self.frame = frame

    def get_frame(self):
        with self.Lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

        print(f"[{self.camera_id}]",f"Disconnected")

class CameraManager:
    def __init__(self,cameras):
        self.cameras = []
        for camera_id, url in cameras:
            camera = Camera(
                camera_id,
                url
            )

            self.cameras.append(camera)

    def start_all(self):
        print("\n\n==================\nStarting Camera Manager\n================\n\n")
        for camera in self.cameras:
            camera.start()

    def get_frames(self):

        frames = []

        for camera in self.cameras:
            frame = camera.get_frame()

            if frame is not None:
                frames.append((camera.camera_id,frame))

        return frames

    def stop_all(self):
        print("\nStopping Camera Feed")

        for camera in self.cameras:
            camera.stop()