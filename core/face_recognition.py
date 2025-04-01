# core/face_recognition.py
# (Same as given in the previous response, no changes needed)
import cv2
import face_recognition
import os
import numpy as np
import h5py
from multiprocessing import Pool

def load_trained_model_hdf5(model_path="trained_model.hdf5"):
    known_face_encodings = []
    known_face_names = []
    try:
        with h5py.File(model_path, 'r') as f:
            for person_name in f.keys():
                encodings = f[f"{person_name}/encodings"][:]
                names = f[f"{person_name}/names"][:]
                known_face_encodings.extend(encodings)
                known_face_names.extend([name.decode('utf-8') for name in names])
    except Exception as e:
        print(f"Error loading trained model: {e}")
    return known_face_encodings, known_face_names

def process_frame(args):
    frame, known_face_encodings, known_face_names = args
    try:
        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(frame_small)
        face_encodings = face_recognition.face_encodings(frame_small, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
        return frame, face_locations, face_names
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, [], []

def append_user_to_model_hdf5(user_name, face_encodings, model_path="trained_model.hdf5"):
    if not face_encodings:
        return False
    mode = 'a' if os.path.exists(model_path) else 'w'
    with h5py.File(model_path, mode) as f:
        if user_name in f.keys():
            existing_enc = f[f"{user_name}/encodings"]
            existing_names = f[f"{user_name}/names"]
            all_enc = np.concatenate((existing_enc[:], np.array(face_encodings)), axis=0)
            all_names = np.concatenate((existing_names[:], np.array([user_name]*len(face_encodings), dtype='S')), axis=0)
            del f[f"{user_name}/encodings"]
            del f[f"{user_name}/names"]
            f.create_dataset(f"{user_name}/encodings", data=all_enc, compression="gzip")
            f.create_dataset(f"{user_name}/names", data=all_names, compression="gzip")
        else:
            f.create_dataset(f"{user_name}/encodings", data=np.array(face_encodings), compression="gzip")
            f.create_dataset(f"{user_name}/names", data=np.array([user_name]*len(face_encodings), dtype='S'), compression="gzip")
    return True

def delete_user_from_model_hdf5(user_name, model_path="trained_model.hdf5"):
    if not os.path.exists(model_path):
        return False
    try:
        with h5py.File(model_path, 'a') as f:
            if user_name in f.keys():
                del f[user_name]
                return True
            else:
                return False
    except Exception as e:
        print(f"Error deleting user from model: {e}")
        return False

class FaceRecognizer:
    def __init__(self, model_path="trained_model.hdf5"):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            self.known_face_encodings, self.known_face_names = load_trained_model_hdf5(self.model_path)
            if not self.known_face_encodings:
                print("No face encodings found in the model.")
            self.pool = Pool(processes=os.cpu_count())
        except Exception as e:
            print(f"Failed to initialize FaceRecognizer: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.pool = None

    def reload_model(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
        self._load_model()

    def recognize_faces(self, frame):
        if not self.pool:
            print("Multiprocessing pool not initialized.")
            return frame, [], []
        try:
            _, face_locations, face_names = process_frame((frame, self.known_face_encodings, self.known_face_names))
            return frame, face_locations, face_names
        except Exception as e:
            print(f"Error in recognize_faces: {e}")
            return frame, [], []

    def register_new_user(self, user_name, face_encodings):
        success = append_user_to_model_hdf5(user_name, face_encodings, self.model_path)
        if success:
            self.reload_model()
            print(f"User {user_name} registered successfully.")

    def delete_user_from_model(self, user_name):
        success = delete_user_from_model_hdf5(user_name, self.model_path)
        if success:
            self.reload_model()
        return success

    def close(self):
        if self.pool:
            try:
                self.pool.close()
                self.pool.join()
            except Exception as e:
                print(f"Error closing pool: {e}")