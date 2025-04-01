# core/aruco_detection.py
import cv2

class ArucoDetector:
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    }

    def __init__(self, dict_type="DICT_5X5_100"):
        """Initialize the Aruco detector with the specified dictionary type."""
        try:
            self.aruco_dict = cv2.aruco.Dictionary_get(self.ARUCO_DICT["DICT_5X5_100"])
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except Exception as e:
            print(f"Failed to initialize Aruco Detector: {e}")
            self.aruco_dict = None
            self.aruco_params = None

    def detect_markers(self, frame):
        """Detect Aruco markers in a given frame."""
        if not self.aruco_dict:
            print("Aruco dictionary is not initialized.")
            return [], None, []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
            # Optionally, draw detected markers for visualization
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            return corners, ids, rejected
        except Exception as e:
            print(f"Error detecting Aruco markers: {e}")
            return [], None, []
    def get_aruco_dict(self):
        """Get the Aruco dictionary."""
        return self.aruco_dict

    def get_aruco_params(self):
        return self.aruco_params    