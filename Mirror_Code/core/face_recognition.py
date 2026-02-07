# core/face_recognition.py
#
# Backward-compatibility wrapper around the canonical implementation in face_run.

from .face_run import (
    FaceRecognizer,
    append_user_to_model_hdf5,
    delete_user_from_model_hdf5,
    load_trained_model_hdf5,
    process_frame,
)

__all__ = [
    "FaceRecognizer",
    "append_user_to_model_hdf5",
    "delete_user_from_model_hdf5",
    "load_trained_model_hdf5",
    "process_frame",
]
