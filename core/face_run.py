# core/face_run.py

import cv2
import face_recognition
import os
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_trained_model_hdf5(model_path="trained_model.hdf5"):
    """Load known face encodings and names from an HDF5 file."""
    known_face_encodings = []
    known_face_names = []
    try:
        with h5py.File(model_path, 'r') as f:
            for person_name in f.keys():
                encodings = f[f"{person_name}/encodings"][:]
                names = f[f"{person_name}/names"][:]
                known_face_encodings.extend(encodings)
                known_face_names.extend([name.decode('utf-8') for name in names])
        logging.info(f"Loaded {len(known_face_encodings)} face encodings from {model_path}.")
    except Exception as e:
        logging.error(f"Error loading trained model from {model_path}: {e}")
    return known_face_encodings, known_face_names


def process_frame(args):
    """Process a single video frame for face recognition."""
    frame, known_face_encodings, known_face_names = args
    try:
        # Resize frame for faster processing
        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(frame_small)
        face_encodings = face_recognition.face_encodings(frame_small, face_locations)
        face_names = []
        tolerance = 0.4  # Stricter tolerance
        for face_encoding in face_encodings:
            # Compute distances
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < tolerance:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"
            face_names.append(name)
        return frame, face_locations, face_names
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return frame, [], []


def append_user_to_model_hdf5(user_name, face_encodings, model_path="trained_model.hdf5"):
    """Append a new user's encodings to the HDF5 model."""
    if not face_encodings:
        logging.warning(f"No face encodings provided for user {user_name}.")
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
                logging.info(f"Appended {len(face_encodings)} encodings for user {user_name}.")
            else:
                # Create new user group
                f.create_dataset(f"{user_name}/encodings", data=np.array(face_encodings), compression="gzip")
                f.create_dataset(f"{user_name}/names", data=np.array([user_name] * len(face_encodings), dtype='S'), compression="gzip")
                logging.info(f"Created new user {user_name} with {len(face_encodings)} encodings.")
        return True
    except Exception as e:
        logging.error(f"Error appending user {user_name} to model: {e}")
        return False


def delete_user_from_model_hdf5(user_name, model_path="trained_model.hdf5"):
    """Delete a user and their encodings from the HDF5 model."""
    if not os.path.exists(model_path):
        logging.warning(f"Model file {model_path} does not exist.")
        return False
    try:
        with h5py.File(model_path, 'a') as f:
            if user_name in f.keys():
                del f[user_name]
                logging.info(f"Deleted user {user_name} from model.")
                return True
            else:
                logging.warning(f"User {user_name} not found in model.")
                return False
    except Exception as e:
        logging.error(f"Error deleting user {user_name} from model: {e}")
        return False


class FaceRecognizer:
    def __init__(self, model_path="trained_model.hdf5"):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the trained model and initialize the multiprocessing pool."""
        try:
            self.known_face_encodings, self.known_face_names = load_trained_model_hdf5(self.model_path)
            if not self.known_face_encodings:
                logging.warning("No face encodings found in the model.")
            self.pool = Pool(processes=cpu_count())
            logging.info(f"Initialized multiprocessing pool with {cpu_count()} processes.")
        except Exception as e:
            logging.error(f"Failed to initialize FaceRecognizer: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.pool = None

    def reload_model(self):
        """Reload the trained model and restart the multiprocessing pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            logging.info("Closed existing multiprocessing pool.")
        self._load_model()

    def recognize_faces(self, frame):
        """Recognize faces in the given frame using multiprocessing."""
        if not self.pool:
            logging.error("Multiprocessing pool not initialized.")
            return frame, [], []
        try:
            # Submit the frame to the pool for processing
            result = self.pool.apply_async(process_frame, args=((frame, self.known_face_encodings, self.known_face_names),))
            _, face_locations, face_names = result.get(timeout=5)  # Adjust timeout as needed
            return frame, face_locations, face_names
        except Exception as e:
            logging.error(f"Error in recognize_faces: {e}")
            return frame, [], []

    def register_new_user(self, user_name, face_encodings):
        """Register a new user by appending their encodings to the model."""
        success = append_user_to_model_hdf5(user_name, face_encodings, self.model_path)
        if success:
            self.reload_model()
            logging.info(f"User {user_name} registered successfully.")
        return success

    def delete_user_from_model(self, user_name):
        """Delete a user from the model."""
        success = delete_user_from_model_hdf5(user_name, self.model_path)
        if success:
            self.reload_model()
        return success

    def close(self):
        """Close the multiprocessing pool."""
        if self.pool:
            try:
                self.pool.close()
                self.pool.join()
                logging.info("Multiprocessing pool closed successfully.")
            except Exception as e:
                logging.error(f"Error closing pool: {e}")