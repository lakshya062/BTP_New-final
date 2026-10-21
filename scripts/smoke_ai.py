"""Run real native AI inference on synthetic frames, without a camera or database."""

import os
from pathlib import Path
import sys
import tempfile


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Mirror_Code"))
    with tempfile.TemporaryDirectory(prefix="smart-gym-ai-smoke-") as directory:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(directory) / "matplotlib"))
        import cv2
        import face_recognition
        import mediapipe as mp
        import numpy as np

        from core.aruco_detection import ArucoDetector
        from core.face_run import FaceRecognizer

        detector = ArucoDetector()
        marker = cv2.aruco.generateImageMarker(detector.get_aruco_dict(), 7, 160)
        marker_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        marker_frame[160:320, 240:400] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        _, ids, _ = detector.detect_markers(marker_frame)
        if ids is None or 7 not in ids.flatten():
            raise RuntimeError("ArUco did not recognize the generated marker.")

        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Call the native detector directly as well: the app's live-frame boundary
        # deliberately catches inference errors and would otherwise mask failure.
        if face_recognition.face_locations(
            cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB),
            number_of_times_to_upsample=0,
            model="hog",
        ):
            raise RuntimeError("Native face detection produced a face on a blank frame.")
        recognizer = FaceRecognizer(str(Path(directory) / "empty-faces.hdf5"))
        try:
            _, locations, names = recognizer.recognize_faces(blank_frame)
            if locations or names:
                raise RuntimeError("Face inference produced a detection on a blank frame.")
        finally:
            recognizer.close()

        with mp.solutions.pose.Pose(model_complexity=1) as pose:
            result = pose.process(cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks is not None:
                raise RuntimeError("Pose inference produced a detection on a blank frame.")
        print("PASS: Native ArUco detection, dlib face inference, and MediaPipe pose inference")


if __name__ == "__main__":
    main()
