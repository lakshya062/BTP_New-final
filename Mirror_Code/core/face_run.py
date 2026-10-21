# core/face_run.py

import cv2
import face_recognition
import os
import numpy as np
import h5py
import logging
import shutil
import threading
import tempfile
from pathlib import Path

from core.paths import project_path, state_path

logger = logging.getLogger(__name__)

FRAME_SCALE = 0.25
MATCH_TOLERANCE = 0.45
MIN_MATCH_MARGIN = 0.03
MAX_ENCODINGS_PER_USER = 80
DEFAULT_FACE_MODEL_FILENAME = "trained_model.hdf5"
FACE_MODEL_PATH_ENV = "SMART_MIRROR_FACE_MODEL_PATH"
# h5py files are shared by registration, member deletion, and camera workers.
# Use one lock across recognizer instances as well as the module-level API.
_MODEL_FILE_LOCK = threading.RLock()


def _normalize_user_name(user_name):
    name = str(user_name or "").strip()
    if not name or name in {".", "..", "Unknown"} or "/" in name or any(ord(c) < 32 for c in name):
        raise ValueError("A nonempty user name without path separators or control characters is required.")
    return name


def _validated_encodings(encodings):
    values = np.asarray(encodings, dtype=np.float32)
    if values.ndim == 1 and values.size == 128:
        values = values.reshape(1, 128)
    if values.ndim != 2 or values.shape[1] != 128 or not len(values):
        raise ValueError("Face encodings must have shape (samples, 128).")
    if not np.isfinite(values).all():
        raise ValueError("Face encodings must contain only finite numbers.")
    return values


def _atomic_model_update(model_path, update, source_path=None):
    """Publish a complete model only after the HDF5 update succeeds."""
    with _MODEL_FILE_LOCK:
        fd, temporary_path = tempfile.mkstemp(prefix=".face-model-", suffix=".hdf5", dir=Path(model_path).parent)
        os.close(fd)
        try:
            source_path = model_path if source_path is None else source_path
            if os.path.exists(source_path):
                shutil.copyfile(source_path, temporary_path)
            with h5py.File(temporary_path, "a" if os.path.getsize(temporary_path) else "w") as model:
                update(model)
                model.flush()
            with open(temporary_path, "rb") as written_model:
                os.fsync(written_model.fileno())
            os.replace(temporary_path, model_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


def resolve_face_model_path(model_path=None):
    configured_path = str(model_path or os.getenv(FACE_MODEL_PATH_ENV) or "").strip()
    if configured_path:
        resolved_path = Path(configured_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = Path(project_path(configured_path))
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        return str(resolved_path)

    runtime_model_path = Path(state_path("models", DEFAULT_FACE_MODEL_FILENAME))
    legacy_model_path = Path(project_path(DEFAULT_FACE_MODEL_FILENAME))
    with _MODEL_FILE_LOCK:
        if not runtime_model_path.exists() and legacy_model_path.exists():
            try:
                runtime_model_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_model_update(str(runtime_model_path), lambda model: None, source_path=legacy_model_path)
                logger.info(
                    "Copied legacy face model from %s to %s.",
                    legacy_model_path,
                    runtime_model_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to copy legacy face model from %s to %s: %s",
                    legacy_model_path,
                    runtime_model_path,
                    exc,
                )
    return str(runtime_model_path)


def load_trained_model_hdf5(model_path=None):
    """Load known face encodings and names from an HDF5 file."""
    model_path = resolve_face_model_path(model_path)
    known_face_encodings = []
    known_face_names = []
    if not os.path.exists(model_path):
        logger.warning("Face model file '%s' does not exist yet.", model_path)
        return known_face_encodings, known_face_names
    try:
        with _MODEL_FILE_LOCK, h5py.File(model_path, 'r') as f:
            for person_name in f.keys():
                try:
                    normalized_name = _normalize_user_name(person_name)
                    encodings = _validated_encodings(f[f"{person_name}/encodings"][:])
                    names = f[f"{person_name}/names"][:]
                    decoded_names = [
                        name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
                        for name in names
                    ]
                    if len(decoded_names) != len(encodings) or any(name != normalized_name for name in decoded_names):
                        raise ValueError("Encoding labels do not match their user group.")
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    logger.warning("Skipping invalid face model group %r: %s", person_name, exc)
                    continue
                if len(encodings) > MAX_ENCODINGS_PER_USER:
                    sample_idx = np.linspace(
                        0,
                        len(encodings) - 1,
                        num=MAX_ENCODINGS_PER_USER,
                        dtype=np.int32,
                    )
                    encodings = encodings[sample_idx]
                known_face_encodings.extend(encodings)
                known_face_names.extend([normalized_name] * len(encodings))
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
    if not np.isfinite(distances).all() or np.any(np.asarray(distances) < 0):
        return "Unknown"
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


def process_frame(args): #this can be done on mobiel app.
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

    try:
        frame_scale = float(inference_config.get("frame_scale", FRAME_SCALE))
        if not np.isfinite(frame_scale) or frame_scale <= 0 or frame_scale > 1:
            frame_scale = FRAME_SCALE
        tolerance = float(inference_config.get("tolerance", MATCH_TOLERANCE))
        min_margin = float(inference_config.get("min_margin", MIN_MATCH_MARGIN))
        if not np.isfinite(tolerance) or not 0 < tolerance <= 1:
            tolerance = MATCH_TOLERANCE
        if not np.isfinite(min_margin) or min_margin < 0:
            min_margin = MIN_MATCH_MARGIN
        face_location_model = inference_config.get("face_location_model", "hog")
        face_upsample = max(0, min(2, int(inference_config.get("face_upsample", 0))))
        known_face_encodings_np = np.asarray(known_face_encodings, dtype=np.float32)
        if known_face_encodings_np.ndim == 1 and known_face_encodings_np.size:
            known_face_encodings_np = known_face_encodings_np.reshape(1, -1)
        if (
            known_face_encodings_np.ndim != 2
            or known_face_encodings_np.shape[1] != 128
            or len(known_face_encodings_np) != len(known_face_names)
            or not np.isfinite(known_face_encodings_np).all()
        ):
            known_face_encodings_np = np.empty((0, 128), dtype=np.float32)
            name_to_indices = {}
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


def append_user_to_model_hdf5(user_name, face_encodings, model_path=None):
    """Validate, bound, and atomically append a user's biometric samples."""
    try:
        user_name = _normalize_user_name(user_name)
        encodings = _validated_encodings(face_encodings)
        model_path = resolve_face_model_path(model_path)

        def update(model):
            all_encodings = encodings
            if user_name in model:
                existing = _validated_encodings(model[f"{user_name}/encodings"][:])
                all_encodings = np.concatenate((existing, encodings), axis=0)
                del model[user_name]
            if len(all_encodings) > MAX_ENCODINGS_PER_USER:
                indices = np.linspace(0, len(all_encodings) - 1, MAX_ENCODINGS_PER_USER, dtype=np.int32)
                all_encodings = all_encodings[indices]
            group = model.create_group(user_name)
            group.create_dataset("encodings", data=all_encodings, compression="gzip")
            group.create_dataset("names", data=[user_name] * len(all_encodings), dtype=h5py.string_dtype("utf-8"))

        _atomic_model_update(model_path, update)
        logger.info("Saved %s face samples for %s.", min(len(encodings), MAX_ENCODINGS_PER_USER), user_name)
        return True
    except Exception as e:
        logger.error("Error appending user %s to model: %s", user_name, e)
        return False


def delete_user_from_model_hdf5(user_name, model_path=None):
    """Delete a user and their encodings from the HDF5 model."""
    try:
        user_name = _normalize_user_name(user_name)
        model_path = resolve_face_model_path(model_path)
        with _MODEL_FILE_LOCK:
            if not os.path.exists(model_path):
                return False
            def update(model):
                if user_name not in model:
                    raise KeyError(user_name)
                del model[user_name]
            _atomic_model_update(model_path, update)
        logger.info("Deleted user %s from model.", user_name)
        return True
    except Exception as e:
        logger.error("Error deleting user %s from model: %s", user_name, e)
        return False


class FaceRecognizer:
    def __init__(self, model_path=None):
        self.model_path = resolve_face_model_path(model_path)
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

    def export_user_encodings(self, user_name):
        normalized_name = str(user_name or "").strip()
        if not normalized_name:
            return []

        with self._lock:
            if not os.path.exists(self.model_path):
                return []
            try:
                with _MODEL_FILE_LOCK, h5py.File(self.model_path, "r") as model_file:
                    if normalized_name not in model_file:
                        return []
                    encodings = np.asarray(
                        model_file[f"{normalized_name}/encodings"][:],
                        dtype=np.float32,
                    )
                    if encodings.ndim == 1 and encodings.size:
                        encodings = encodings.reshape(1, -1)
                    return [np.asarray(encoding, dtype=np.float32) for encoding in encodings]
            except Exception as exc:
                logger.error("Failed to export face encodings for %s: %s", normalized_name, exc)
                return None

    def restore_user_from_backup(self, user_name, face_encodings):
        normalized_name = str(user_name or "").strip()
        encodings = [] if face_encodings is None else list(face_encodings)
        if not normalized_name or not encodings:
            return True

        with self._lock:
            success = append_user_to_model_hdf5(normalized_name, encodings, self.model_path)
            if success:
                self.reload_model()
            return success

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
