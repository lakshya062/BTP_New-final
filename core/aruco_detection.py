# core/aruco_detection.py
import cv2
import logging

logger = logging.getLogger(__name__)

class ArucoDetector:
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    }
    ARUCO_ALIASES = {
        "4X4": "DICT_4X4_50",
        "DICT_4X4": "DICT_4X4_50",
        "4X4_50": "DICT_4X4_50",
        "5X5": "DICT_5X5_100",
        "DICT_5X5": "DICT_5X5_100",
        "5X5_100": "DICT_5X5_100",
    }

    @classmethod
    def normalize_dict_type(cls, dict_type):
        raw_type = str(dict_type or "").strip().upper().replace("-", "_").replace(" ", "")
        if raw_type in cls.ARUCO_DICT:
            return raw_type
        if raw_type in cls.ARUCO_ALIASES:
            return cls.ARUCO_ALIASES[raw_type]
        return "DICT_5X5_100"

    def __init__(self, dict_type="DICT_5X5_100"):
        """Initialize the Aruco detector with the specified dictionary type."""
        self.detector = None
        self.dict_type = "DICT_5X5_100"
        try:
            raw_type = str(dict_type or "").strip().upper().replace("-", "_").replace(" ", "")
            aruco_key = self.normalize_dict_type(dict_type)
            self.dict_type = aruco_key
            if raw_type and raw_type not in self.ARUCO_DICT and raw_type not in self.ARUCO_ALIASES:
                logger.warning(
                    "Unknown ArUco dictionary '%s'. Falling back to '%s'.",
                    dict_type,
                    aruco_key,
                )

            if hasattr(cv2.aruco, "getPredefinedDictionary"):
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[aruco_key])
            else:
                self.aruco_dict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[aruco_key])

            if hasattr(cv2.aruco, "DetectorParameters"):
                self.aruco_params = cv2.aruco.DetectorParameters()
            else:
                self.aruco_params = cv2.aruco.DetectorParameters_create()

            if hasattr(cv2.aruco, "ArucoDetector"):
                self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except Exception as exc:
            logger.error("Failed to initialize ArUco detector: %s", exc)
            self.aruco_dict = None
            self.aruco_params = None
            self.dict_type = "DICT_5X5_100"

    def detect_markers(self, frame):
        """Detect Aruco markers in a given frame."""
        if not self.aruco_dict:
            logger.warning("ArUco dictionary is not initialized.")
            return [], None, []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            if self.detector is not None:
                corners, ids, rejected = self.detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params
                )
            # Optionally, draw detected markers for visualization
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            return corners, ids, rejected
        except Exception as exc:
            logger.error("Error detecting ArUco markers: %s", exc)
            return [], None, []

    def get_aruco_dict(self):
        """Get the Aruco dictionary."""
        return self.aruco_dict

    def get_aruco_params(self):
        return self.aruco_params

    def get_dict_type(self):
        return self.dict_type
