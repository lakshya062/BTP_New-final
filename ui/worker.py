# ui/worker.py

import cv2
import face_recognition as fr_lib
import mediapipe as mp 
import logging
from PySide6.QtCore import QThread, Signal
from core.aruco_detection import ArucoDetector
from core.pose_analysis import ExerciseAnalyzer
from core.face_run import FaceRecognizer
from core.config import exercise_config

logging.basicConfig(level=logging.INFO, filename='worker.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale=0.6, font_color=(255, 255, 255),
                             bg_color=(0, 0, 0), alpha=0.6, thickness=1):
    """
    Draw semi-transparent text on the given frame.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_height - baseline - 5),
                  (x + text_width + 10, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x + 5, y - 2), font, font_scale, font_color, thickness, cv2.LINE_AA)

class ExerciseWorker(QThread):
    frame_signal = Signal(object)
    thumbnail_frame_signal = Signal(object)
    status_signal = Signal(str)
    counters_signal = Signal(int, int)
    user_recognized_signal = Signal(dict)
    unknown_user_detected = Signal()
    data_updated = Signal()
    audio_feedback_signal = Signal(str)

    def __init__(self, db_handler, camera_index, exercise_choice='bicep_curl', face_recognizer=None, parent=None):
        super().__init__(parent)
        self.db_handler = db_handler
        self.exercise_choice = exercise_choice
        self.face_recognizer = face_recognizer
        self.stop_requested = False
        self.camera_index = camera_index

        self.unknown_frames = 0
        self.UNKNOWN_FRAME_THRESHOLD = 15
        self.KNOWN_FRAME_THRESHOLD = 30

        self.new_user_name = None
        self.new_user_encodings = []
        self.capturing_new_user_data = False
        self.frames_to_capture = 200

        self.face_recognition_active = True
        self.exercise_analysis_active = False
        self.exercise_analyzer = None
        self.current_user_info = None

        self.known_frames = 0
        self.last_recognized_user = None
        self.unknown_stable_detected = False

    def request_stop(self):
        self.stop_requested = True
        logging.info("Stop requested for ExerciseWorker.")

    def start_record_new_user(self, user_name):
        self.new_user_name = user_name
        self.new_user_encodings = []
        self.capturing_new_user_data = True
        self.status_signal.emit(f"Capturing face data for {user_name}. Please wait...")
        logging.info(f"Started capturing face data for new user: {user_name}")

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.status_signal.emit(f"Error: Could not open camera {self.camera_index}.")
            logging.error(f"Could not open camera {self.camera_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logging.info(f"Camera {self.camera_index} initialized.")

        mp_pose_solution = mp_pose.Pose(min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5,
                                        model_complexity=1)
        aruco_detector = ArucoDetector(dict_type="DICT_5X5_100")

        if self.exercise_choice not in exercise_config:
            self.status_signal.emit(f"Exercise '{self.exercise_choice}' not configured. Using default if available.")
            logging.warning(f"Exercise '{self.exercise_choice}' not found in config.")
            if exercise_config:
                self.exercise_choice = list(exercise_config.keys())[0]
            else:
                self.status_signal.emit("No exercises configured.")
                logging.error("No exercises configured. Worker terminating.")
                cap.release()
                mp_pose_solution.close()
                return

        occlusion_timeout = 90  # About 3 seconds at ~30 FPS

        while not self.stop_requested:
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("Failed to capture frame.")
                logging.error("Failed to capture frame from camera.")
                break

            display_frame = frame.copy()
            thumb = cv2.resize(display_frame, (160, 90), interpolation=cv2.INTER_AREA)

            # If exercise analysis is active, skip face recognition
            if self.exercise_analysis_active and not self.capturing_new_user_data:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = mp_pose_solution.process(frame_rgb)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks:
                        self.exercise_analyzer.no_pose_frames = 0
                        landmarks = results.pose_landmarks.landmark
                        # Analyze exercise form (which yields feedback_texts)
                        feedback_texts, _ = self.exercise_analyzer.analyze_exercise_form(landmarks, frame_bgr)

                        # Draw the skeleton lines
                        if not self.exercise_analyzer.stable_start_detected:
                            connection_color = (255, 0, 0)
                            landmark_color = (255, 0, 0)
                        elif self.exercise_analyzer.bend_warning_displayed:
                            connection_color = (0, 0, 255)
                            landmark_color = (0, 0, 255)
                        else:
                            connection_color = (0, 255, 0)
                            landmark_color = (0, 255, 0)

                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2)
                        )

                        # Show any short feedback (warnings, set complete, etc.) with alpha background
                        for i, text in enumerate(feedback_texts):
                            # We'll draw these at the bottom left
                            pos_x = 20
                            pos_y = frame_bgr.shape[0] - 100 + (i * 20)
                            if "Warning" in text:
                                color = (0, 0, 255)
                            elif "Set complete!" in text or "Ready to start!" in text:
                                color = (0, 255, 255)
                            else:
                                color = (255, 255, 255)

                            draw_text_with_background(
                                frame_bgr,
                                text,
                                (pos_x, pos_y),
                                font_scale=0.6,
                                font_color=color,
                                bg_color=(50, 50, 50),
                                alpha=0.6,
                                thickness=1
                            )

                            # Also trigger audio feedback for certain messages
                            if "Warning" in text:
                                self.audio_feedback_signal.emit(text)
                            elif "Set complete!" in text or "Ready to start!" in text:
                                self.audio_feedback_signal.emit(text)

                        # -----------------------------------------------------
                        #  REMOVE the direct overlays for reps / sets / progress
                        #  which were drawn on the camera feed. Now they're gone.
                        # -----------------------------------------------------

                        # However, keep special encouragement text for halfway
                        reps_per_set = exercise_config[self.exercise_choice].get('reps_per_set', 12)
                        if self.exercise_analyzer.rep_count == reps_per_set // 2:
                            encouragement_text = "Great halfway mark! Keep going!"
                            # Show it near top center
                            draw_text_with_background(
                                frame_bgr,
                                encouragement_text,
                                (frame_bgr.shape[1] // 2 - 150, 30),
                                font_scale=0.6,
                                font_color=(0, 255, 255),
                                bg_color=(50, 50, 50),
                                alpha=0.6,
                                thickness=1
                            )
                            self.audio_feedback_signal.emit(encouragement_text)

                        # If we detect set complete in feedback_texts, optionally show an extra message
                        if "Set complete!" in feedback_texts:
                            encouragement_text = "Excellent work! Take a short break."
                            draw_text_with_background(
                                frame_bgr,
                                encouragement_text,
                                (frame_bgr.shape[1] // 2 - 150, 60),
                                font_scale=0.6,
                                font_color=(0, 255, 255),
                                bg_color=(50, 50, 50),
                                alpha=0.6,
                                thickness=1
                            )
                            self.audio_feedback_signal.emit(encouragement_text)

                        # Emit final frame to the UI
                        final_frame = frame_bgr
                        self.frame_signal.emit(final_frame)
                        self.thumbnail_frame_signal.emit(cv2.resize(final_frame, (160, 90), interpolation=cv2.INTER_AREA))
                        # Let the counters be known for the black canvas
                        self.counters_signal.emit(self.exercise_analyzer.rep_count,
                                                  self.exercise_analyzer.set_count)
                    else:
                        self.exercise_analyzer.no_pose_frames += 1
                        logging.debug(f"No pose detected. no_pose_frames: {self.exercise_analyzer.no_pose_frames}")

                        if self.exercise_analyzer.no_pose_frames >= occlusion_timeout:
                            self.status_signal.emit("Person exited the frame, updating database...")
                            logging.info("Person exited frame. Updating database.")
                            record = self.exercise_analyzer.update_data()
                            if record:
                                insert_success = self.db_handler.insert_exercise_data_local(record)
                                if insert_success:
                                    self.status_signal.emit("Exercise data saved locally.")
                                    logging.info("Data saved locally.")
                                    self.data_updated.emit()
                                else:
                                    self.status_signal.emit("Failed to save exercise data locally.")
                                    logging.error("Failed to save exercise data locally.")

                            self.exercise_analyzer.reset_counters()
                            self.exercise_analyzer = None
                            self.exercise_analysis_active = False
                            self.face_recognition_active = True
                            self.status_signal.emit("Returning to Face Recognition Mode.")
                            logging.info("Switched back to Face Recognition Mode.")

                    self.msleep(10)
                    continue

                except Exception as e:
                    self.status_signal.emit("Exercise analysis error.")
                    logging.error(f"Exercise analysis error: {e}")
                    self.msleep(10)
                    continue

            # If capturing new user face data
            if self.capturing_new_user_data:
                try:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = fr_lib.face_locations(rgb_small_frame)
                    encodings = fr_lib.face_encodings(rgb_small_frame, face_locations)
                    self.new_user_encodings.extend(encodings)

                    self.frame_signal.emit(display_frame)
                    self.thumbnail_frame_signal.emit(thumb)

                    if len(self.new_user_encodings) >= self.frames_to_capture:
                        self.status_signal.emit(f"Registering user {self.new_user_name}...")
                        logging.info(f"Registering new user: {self.new_user_name}")
                        registration_success = self.face_recognizer.register_new_user(
                            self.new_user_name,
                            self.new_user_encodings
                        )
                        if registration_success:
                            self.status_signal.emit(f"User {self.new_user_name} registered successfully.")
                            logging.info(f"User {self.new_user_name} registered successfully.")
                        else:
                            self.status_signal.emit(f"Failed to register user {self.new_user_name}.")
                            logging.error(f"Failed to register user {self.new_user_name}.")

                        self.new_user_name = None
                        self.new_user_encodings = []
                        self.capturing_new_user_data = False
                        self.face_recognition_active = True
                        self.unknown_stable_detected = False
                        self.unknown_frames = 0
                        self.known_frames = 0
                        self.last_recognized_user = None

                    self.msleep(10)
                    continue
                except Exception as e:
                    self.status_signal.emit("New user registration error.")
                    logging.error(f"New user registration error: {e}")
                    self.msleep(10)
                    continue

            # Face Recognition Mode
            if self.face_recognition_active:
                try:
                    frame_face_processed, face_locations, face_names = self.face_recognizer.recognize_faces(frame)

                    for ((top, right, bottom, left), name) in zip(face_locations, face_names):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(display_frame, name, (left + 5, bottom - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # If at least one face found
                    if len(face_names) > 0:
                        # All unknown?
                        if all(n == "Unknown" for n in face_names):
                            self.unknown_frames += 1
                            self.known_frames = 0
                            if self.unknown_frames >= self.UNKNOWN_FRAME_THRESHOLD and not self.unknown_stable_detected:
                                self.status_signal.emit("Stable unknown user detected. Prompting for user name...")
                                logging.info("Stable unknown user detected.")
                                self.unknown_user_detected.emit()
                                self.unknown_stable_detected = True
                        else:
                            self.unknown_frames = 0
                            self.unknown_stable_detected = False
                            recognized_user = [n for n in face_names if n != "Unknown"][0]
                            if recognized_user == self.last_recognized_user:
                                self.known_frames += 1
                            else:
                                self.last_recognized_user = recognized_user
                                self.known_frames = 1

                            if self.known_frames >= self.KNOWN_FRAME_THRESHOLD:
                                logging.info(f"User recognized: {recognized_user}")
                                user_info = self.db_handler.get_member_info(recognized_user)
                                if user_info:
                                    self.user_recognized_signal.emit(user_info)
                                    self.status_signal.emit(f"Face recognized as {recognized_user}, starting exercise analysis.")
                                    logging.info(f"Starting exercise analysis for user: {recognized_user}")
                                    self.current_user_info = user_info
                                    self.exercise_analyzer = ExerciseAnalyzer(
                                        self.exercise_choice,
                                        aruco_detector,
                                        self.db_handler,
                                        user_id=user_info.get("user_id")
                                    )
                                    self.exercise_analysis_active = True
                                    self.face_recognition_active = False
                                else:
                                    self.status_signal.emit(f"Face recognized as {recognized_user}, but no member info found.")
                                    logging.warning(f"No member info found for recognized user {recognized_user}.")
                    else:
                        # No face found
                        self.unknown_frames = 0
                        self.known_frames = 0
                        self.last_recognized_user = None
                        self.unknown_stable_detected = False

                    self.frame_signal.emit(display_frame)
                    self.thumbnail_frame_signal.emit(thumb)
                except Exception as e:
                    self.status_signal.emit("Face recognition error.")
                    logging.error(f"Face recognition error: {e}")

            self.msleep(10)

        self.cleanup()

    def cleanup(self):
        """Handle cleanup operations when stopping the worker."""
        try:
            if self.exercise_analyzer:
                record = self.exercise_analyzer.update_data()
                if record:
                    insert_success = self.db_handler.insert_exercise_data_local(record)
                    if insert_success:
                        logging.info("Exercise data saved locally during cleanup.")
                    else:
                        logging.error("Failed to save exercise data locally during cleanup.")
                else:
                    logging.warning("No exercise data to save during cleanup.")

            self.data_updated.emit()
        except Exception as e:
            self.status_signal.emit("Error during cleanup.")
            logging.error(f"Error during cleanup: {e}")
        # No signal disconnect calls here
