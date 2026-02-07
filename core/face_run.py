# core/face_run.py

import cv2
import face_recognition
import os
import numpy as np
import h5py
import logging
import threading

logger = logging.getLogger(__name__)

FRAME_SCALE = 0.25
MATCH_TOLERANCE = 0.45
MIN_MATCH_MARGIN = 0.03
MAX_ENCODINGS_PER_USER = 80


def load_trained_model_hdf5(model_path="trained_model.hdf5"):
    """Load known face encodings and names from an HDF5 file."""
    known_face_encodings = []
    known_face_names = []
    if not os.path.exists(model_path):
        logger.warning("Face model file '%s' does not exist yet.", model_path)
        return known_face_encodings, known_face_names
    try:
        with h5py.File(model_path, 'r') as f:
            for person_name in f.keys():
                encodings = np.asarray(f[f"{person_name}/encodings"][:], dtype=np.float32)
                names = f[f"{person_name}/names"][:]
                if encodings.ndim == 1:
                    encodings = encodings.reshape(1, -1)
                if len(encodings) == 0:
                    continue
                if len(encodings) > MAX_ENCODINGS_PER_USER:
                    sample_idx = np.linspace(
                        0,
                        len(encodings) - 1,
                        num=MAX_ENCODINGS_PER_USER,
                        dtype=np.int32,
                    )
                    encodings = encodings[sample_idx]
                    names = names[sample_idx]
                known_face_encodings.extend(encodings)
                known_face_names.extend([
                    name.decode('utf-8') if isinstance(name, (bytes, bytearray)) else str(name)
                    for name in names
                ])
        logger.info("Loaded %s face encodings from %s.", len(known_face_encodings), model_path)
    except Exception as e:
        logger.error("Error loading trained model from %s: %s", model_path, e)
    return known_face_encodings, known_face_names


def _build_name_to_indices(known_face_names):
    name_to_indices = {}
    for idx, name in enumerate(known_face_names):
        name_to_indices.setdefault(name, []).append(idx)
    return {
        name: np.asarray(indices, dtype=np.int32)
        for name, indices in name_to_indices.items()
    }


def _resolve_identity(distances, name_to_indices, tolerance, min_margin):
    best_name = "Unknown"
    best_score = float("inf")
    second_score = float("inf")

    for name, indices in name_to_indices.items():
        user_distances = distances[indices]
        if user_distances.size == 0:
            continue
        user_min = float(np.min(user_distances))
        user_p25 = float(np.percentile(user_distances, 25))
        # Blend closest match with a robust user-level percentile to reduce identity flicker.
        user_score = (0.72 * user_min) + (0.28 * user_p25)
        if user_score < best_score:
            second_score = best_score
            best_score = user_score
            best_name = name
        elif user_score < second_score:
            second_score = user_score

    if best_name == "Unknown":
        return "Unknown"
    if best_score > tolerance:
        return "Unknown"
    if not np.isinf(second_score) and (second_score - best_score) < min_margin:
        return "Unknown"
    return best_name


def process_frame(args):
    """Process a single video frame for face recognition."""
    if len(args) < 3:
        raise ValueError("process_frame expects at least (frame, encodings, names).")

    frame, known_face_encodings, known_face_names = args[:3]
    name_to_indices = None
    inference_config = {}
    if len(args) >= 5:
        name_to_indices = args[3]
        inference_config = args[4] if isinstance(args[4], dict) else {}
    elif len(args) == 4 and isinstance(args[3], dict):
        inference_config = args[3]

    frame_scale = float(inference_config.get("frame_scale", FRAME_SCALE))
    if frame_scale <= 0 or frame_scale > 1:
        frame_scale = FRAME_SCALE
    tolerance = float(inference_config.get("tolerance", MATCH_TOLERANCE))
    min_margin = float(inference_config.get("min_margin", MIN_MATCH_MARGIN))
    face_location_model = inference_config.get("face_location_model", "hog")
    face_upsample = int(inference_config.get("face_upsample", 0))

    try:
        known_face_encodings_np = np.asarray(known_face_encodings, dtype=np.float32)
        if known_face_encodings_np.ndim == 1 and known_face_encodings_np.size:
            known_face_encodings_np = known_face_encodings_np.reshape(1, -1)
        if known_face_encodings_np.ndim != 2:
            known_face_encodings_np = np.empty((0, 128), dtype=np.float32)
        if name_to_indices is None:
            name_to_indices = _build_name_to_indices(known_face_names)

        if frame_scale == 1.0:
            frame_small = frame
        else:
            frame_small = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

        # face_recognition expects RGB frames; passing BGR increases misclassification/flicker.
        rgb_small_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            number_of_times_to_upsample=face_upsample,
            model=face_location_model,
        )
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            if known_face_encodings_np.size == 0 or not name_to_indices:
                name = "Unknown"
            else:
                distances = face_recognition.face_distance(known_face_encodings_np, face_encoding)
                name = _resolve_identity(distances, name_to_indices, tolerance, min_margin)
            face_names.append(name)

        return frame, face_locations, face_names
    except Exception as e:
        logger.error("Error processing frame: %s", e)
        return frame, [], []


def append_user_to_model_hdf5(user_name, face_encodings, model_path="trained_model.hdf5"):
    """Append a new user's encodings to the HDF5 model."""
    if not face_encodings:
        logger.warning("No face encodings provided for user %s.", user_name)
        return False
    try:
        mode = 'a' if os.path.exists(model_path) else 'w'
        with h5py.File(model_path, mode) as f:
            if user_name in f.keys():
                # User exists, append encodings and names
                existing_enc = f[f"{user_name}/encodings"][:]
                existing_names = f[f"{user_name}/names"][:]
                all_enc = np.concatenate((existing_enc, np.array(face_encodings)), axis=0)
                all_names = np.concatenate((existing_names, np.array([user_name] * len(face_encodings), dtype='S')), axis=0)
                
                # Delete old datasets
                del f[f"{user_name}/encodings"]
                del f[f"{user_name}/names"]
                
                # Create new datasets with appended data
                f.create_dataset(f"{user_name}/encodings", data=all_enc, compression="gzip")
                f.create_dataset(f"{user_name}/names", data=all_names, compression="gzip")
                logger.info("Appended %s encodings for user %s.", len(face_encodings), user_name)
            else:
                # Create new user group
                f.create_dataset(f"{user_name}/encodings", data=np.array(face_encodings), compression="gzip")
                f.create_dataset(f"{user_name}/names", data=np.array([user_name] * len(face_encodings), dtype='S'), compression="gzip")
                logger.info("Created new user %s with %s encodings.", user_name, len(face_encodings))
        try:
            os.chmod(model_path, 0o600)
        except OSError:
            pass
        return True
    except Exception as e:
        logger.error("Error appending user %s to model: %s", user_name, e)
        return False


def delete_user_from_model_hdf5(user_name, model_path="trained_model.hdf5"):
    """Delete a user and their encodings from the HDF5 model."""
    if not os.path.exists(model_path):
        logger.warning("Model file %s does not exist.", model_path)
        return False
    try:
        with h5py.File(model_path, 'a') as f:
            if user_name in f.keys():
                del f[user_name]
                logger.info("Deleted user %s from model.", user_name)
                return True
            else:
                logger.warning("User %s not found in model.", user_name)
                return False
    except Exception as e:
        logger.error("Error deleting user %s from model: %s", user_name, e)
        return False


class FaceRecognizer:
    def __init__(self, model_path="trained_model.hdf5"):
        self.model_path = model_path
        self._lock = threading.RLock()
        self.known_face_encodings = []
        self.known_face_names = []
        self._known_face_encodings_np = np.empty((0, 128), dtype=np.float32)
        self._name_to_indices = {}
        self._inference_config = {
            "frame_scale": FRAME_SCALE,
            "tolerance": MATCH_TOLERANCE,
            "min_margin": MIN_MATCH_MARGIN,
            "face_location_model": "hog",
            "face_upsample": 0,
        }
        self._load_model()

    def _load_model(self):
        """Load the trained model and build fast lookup structures."""
        try:
            self.known_face_encodings, self.known_face_names = load_trained_model_hdf5(self.model_path)
            if not self.known_face_encodings:
                logger.warning("No face encodings found in the model.")
            self._known_face_encodings_np = np.asarray(self.known_face_encodings, dtype=np.float32)
            if self._known_face_encodings_np.ndim == 1 and self._known_face_encodings_np.size:
                self._known_face_encodings_np = self._known_face_encodings_np.reshape(1, -1)
            if self._known_face_encodings_np.ndim != 2:
                self._known_face_encodings_np = np.empty((0, 128), dtype=np.float32)
            self._name_to_indices = _build_name_to_indices(self.known_face_names)
            logger.info(
                "Face model ready: %s users, %s encodings.",
                len(set(self.known_face_names)),
                len(self.known_face_names),
            )
        except Exception as e:
            logger.error("Failed to initialize FaceRecognizer: %s", e)
            self.known_face_encodings = []
            self.known_face_names = []
            self._known_face_encodings_np = np.empty((0, 128), dtype=np.float32)
            self._name_to_indices = {}

    def reload_model(self):
        """Reload the trained model."""
        with self._lock:
            self._load_model()

    def recognize_faces(self, frame):
        """Recognize faces in the given frame."""
        with self._lock:
            known_face_encodings_np = self._known_face_encodings_np
            known_face_names = tuple(self.known_face_names)
            name_to_indices = self._name_to_indices
            inference_config = dict(self._inference_config)
        try:
            _, face_locations, face_names = process_frame(
                (
                    frame,
                    known_face_encodings_np,
                    known_face_names,
                    name_to_indices,
                    inference_config,
                )
            )
            return frame, face_locations, face_names
        except Exception as e:
            logger.error("Error in recognize_faces: %s", e)
            return frame, [], []

    def register_new_user(self, user_name, face_encodings):
        """Register a new user by appending their encodings to the model."""
        with self._lock:
            success = append_user_to_model_hdf5(user_name, face_encodings, self.model_path)
            if success:
                self.reload_model()
                logger.info("User %s registered successfully.", user_name)
            return success

    def delete_user_from_model(self, user_name):
        """Delete a user from the model."""
        with self._lock:
            success = delete_user_from_model_hdf5(user_name, self.model_path)
            if success:
                self.reload_model()
            return success

    def close(self):
        """Close resources held by face recognizer."""
        logger.info("FaceRecognizer closed.")
