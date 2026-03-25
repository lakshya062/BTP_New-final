import cv2
import time
from threading import Thread

class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)

        # Force MJPEG
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.ret, self.frame = self.cap.read()
        self.running = True

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()


cap = VideoCaptureThread(0).start()

prev_time = 0
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    fps_list.append(fps)
    if len(fps_list) > 10:
        fps_list.pop(0)

    avg_fps = sum(fps_list) / len(fps_list)

    cv2.putText(frame, f"FPS: {int(avg_fps)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Threaded MJPEG Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()