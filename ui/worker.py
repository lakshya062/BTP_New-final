# ui/worker.py

import cv2
import face_recognition as fr_lib
import mediapipe as mp
import logging
from datetime import datetime

from PySide6.QtCore import QThread, Signal

from core.aruco_detection import ArucoDetector
from core.pose_analysis import ExerciseAnalyzer
from core.config import exercise_config

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_translucent_panel(
    frame,
    x,
    y,
    width,
    height,
    fill_color=(20, 28, 38),
    border_color=(103, 188, 255),
    alpha=0.62,
    border_thickness=2,
):
    """Draw a translucent rectangular panel with a border."""
    x = max(0, int(x))
    y = max(0, int(y))
    width = max(1, int(width))
    height = max(1, int(height))

    x2 = min(frame.shape[1] - 1, x + width)
    y2 = min(frame.shape[0] - 1, y + height)
    if x >= x2 or y >= y2:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), fill_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x2, y2), border_color, border_thickness)


class ExerciseWorker(QThread):
    frame_signal = Signal(object)
    thumbnail_frame_signal = Signal(object)
    status_signal = Signal(str)
    counters_signal = Signal(int, int)
    user_recognized_signal = Signal(dict)
    unknown_user_detected = Signal()
    data_updated = Signal()
    audio_feedback_signal = Signal(str)
    new_user_registration_signal = Signal(str, bool)

    def __init__(
        self,
        db_handler,
        camera_index,
        exercise_choice="bicep_curl",
        face_recognizer=None,
        assigned_user_name=None,
        aruco_dict_type="DICT_5X5_100",
        parent=None,
    ):
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

        self.assigned_user_name = (assigned_user_name or "Athlete").strip() or "Athlete"
        self.display_user_name = self.assigned_user_name
        self.last_audio_message = ""
        self.aruco_dict_type = ArucoDetector.normalize_dict_type(aruco_dict_type)

    def request_stop(self):
        self.stop_requested = True
        logging.info("Stop requested for ExerciseWorker.")

    def start_record_new_user(self, user_name):
        self.new_user_name = user_name
        self.new_user_encodings = []
        self.capturing_new_user_data = True
        self.status_signal.emit(f"Capturing face data for {user_name}. Please wait...")
        logging.info(f"Started capturing face data for new user: {user_name}")

    def _emit_important_audio(self, feedback_texts):
        critical_feedback = [
            text
            for text in feedback_texts
            if "Warning" in text or "Set complete!" in text or "Ready to start!" in text
        ]
        if critical_feedback:
            current = critical_feedback[0]
            if current != self.last_audio_message:
                self.audio_feedback_signal.emit(current)
                self.last_audio_message = current
        else:
            self.last_audio_message = ""

    def _resolve_user_name(self):
        if self.current_user_info and self.current_user_info.get("username"):
            return self.current_user_info.get("username")
        return self.display_user_name

    def _motivation_line(self, reps, sets_done, reps_per_set, feedback_texts, status_hint=""):
        if any("Warning" in text for text in feedback_texts):
            return "Fix your form and stay controlled on every rep."
        if "Set complete!" in feedback_texts:
            return "Excellent set. Breathe and reset for the next one."
        if "Ready to start!" in feedback_texts:
            return "You are locked in. Start your first rep."
        if sets_done > 0 and reps == 0:
            return "Strong set. Keep your rhythm going."
        if reps_per_set > 0 and reps >= max(1, int(reps_per_set * 0.75)):
            return "Final reps. Drive through with clean form."
        if reps_per_set > 0 and reps >= max(1, reps_per_set // 2):
            return "Halfway done. Keep the pace strong."
        if reps > 0:
            return "Smooth and steady. Own each repetition."
        if status_hint:
            return status_hint
        return "Stand centered and start when you are ready."

    def _format_angle_line(self, angles):
        if not angles:
            return "Angles: --"

        angle_items = list(angles.items())
        primary = []
        for key, value in angle_items[:2]:
            label = key.replace("_", " ").title()
            primary.append(f"{label}: {int(value)}deg")

        return "   |   ".join(primary)

    def _build_overlay_data(self, metrics=None, feedback_texts=None, status_hint=""):
        metrics = metrics or {}
        feedback_texts = feedback_texts or []

        angles = metrics.get("angles") or {}
        reps = int(
            metrics.get(
                "rep_count",
                self.exercise_analyzer.rep_count if self.exercise_analyzer else 0,
            )
        )
        sets_done = int(
            metrics.get(
                "set_count",
                self.exercise_analyzer.set_count if self.exercise_analyzer else 0,
            )
        )
        current_weight = int(
            metrics.get(
                "current_weight",
                self.exercise_analyzer.current_weight if self.exercise_analyzer else 0,
            )
        )

        cfg = exercise_config.get(self.exercise_choice, {})
        reps_per_set = int(cfg.get("reps_per_set", 12))

        return {
            "username": self._resolve_user_name(),
            "exercise": self.exercise_choice.replace("_", " ").title(),
            "rep_count": reps,
            "set_count": sets_done,
            "current_weight": current_weight,
            "reps_per_set": reps_per_set,
            "angles_line": self._format_angle_line(angles),
            "timestamp": datetime.now().strftime("%d %b %Y | %I:%M:%S %p"),
            "motivation": self._motivation_line(reps, sets_done, reps_per_set, feedback_texts, status_hint),
        }

    def _draw_live_overlay(self, frame, overlay_data, feedback_texts=None):
        feedback_texts = feedback_texts or []
        h, w = frame.shape[:2]
        scale = max(0.72, min(w / 1280.0, h / 720.0))

        margin = int(20 * scale)
        panel_w = int(470 * scale)
        panel_h = int(250 * scale)

        panel_x = margin
        panel_y = margin

        draw_translucent_panel(
            frame,
            panel_x,
            panel_y,
            panel_w,
            panel_h,
            fill_color=(22, 30, 43),
            border_color=(108, 201, 255),
            alpha=0.66,
            border_thickness=max(1, int(2 * scale)),
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.88 * scale
        label_scale = 0.53 * scale
        value_scale = 0.82 * scale

        cv2.putText(
            frame,
            "LIVE PERFORMANCE",
            (panel_x + int(18 * scale), panel_y + int(34 * scale)),
            font,
            title_scale,
            (240, 248, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"User: {overlay_data['username']}",
            (panel_x + int(18 * scale), panel_y + int(64 * scale)),
            font,
            0.64 * scale,
            (186, 227, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Exercise: {overlay_data['exercise']}",
            (panel_x + int(18 * scale), panel_y + int(88 * scale)),
            font,
            label_scale,
            (212, 223, 240),
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )

        chip_w = int((panel_w - int(52 * scale)) / 3)
        chip_h = int(90 * scale)
        chip_y = panel_y + int(102 * scale)
        chip_gap = int(14 * scale)

        stats = [
            ("REPS", str(overlay_data["rep_count"])),
            ("SETS", str(overlay_data["set_count"])),
            (
                "WEIGHT",
                f"{overlay_data['current_weight']} lbs" if overlay_data["current_weight"] > 0 else "--",
            ),
        ]

        for idx, (label, value) in enumerate(stats):
            chip_x = panel_x + int(16 * scale) + idx * (chip_w + chip_gap)
            draw_translucent_panel(
                frame,
                chip_x,
                chip_y,
                chip_w,
                chip_h,
                fill_color=(30, 41, 57),
                border_color=(92, 166, 236),
                alpha=0.7,
                border_thickness=max(1, int(1 * scale)),
            )
            cv2.putText(
                frame,
                label,
                (chip_x + int(12 * scale), chip_y + int(30 * scale)),
                font,
                0.48 * scale,
                (176, 210, 246),
                max(1, int(1 * scale)),
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                value,
                (chip_x + int(12 * scale), chip_y + int(68 * scale)),
                font,
                value_scale,
                (245, 250, 255),
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            overlay_data["angles_line"],
            (panel_x + int(18 * scale), panel_y + panel_h - int(20 * scale)),
            font,
            0.52 * scale,
            (178, 197, 218),
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )

        time_panel_w = int(390 * scale)
        time_panel_h = int(62 * scale)
        time_x = max(margin, w - time_panel_w - margin)
        time_y = margin
        draw_translucent_panel(
            frame,
            time_x,
            time_y,
            time_panel_w,
            time_panel_h,
            fill_color=(22, 30, 43),
            border_color=(108, 201, 255),
            alpha=0.64,
            border_thickness=max(1, int(2 * scale)),
        )
        cv2.putText(
            frame,
            overlay_data["timestamp"],
            (time_x + int(16 * scale), time_y + int(39 * scale)),
            font,
            0.60 * scale,
            (238, 246, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        motivation_h = int(72 * scale)
        motivation_x = margin
        motivation_y = h - motivation_h - margin
        motivation_w = max(100, w - (2 * margin))

        draw_translucent_panel(
            frame,
            motivation_x,
            motivation_y,
            motivation_w,
            motivation_h,
            fill_color=(17, 77, 88),
            border_color=(88, 223, 230),
            alpha=0.68,
            border_thickness=max(1, int(2 * scale)),
        )
        cv2.putText(
            frame,
            overlay_data["motivation"],
            (motivation_x + int(18 * scale), motivation_y + int(45 * scale)),
            font,
            0.68 * scale,
            (235, 255, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        warning_texts = [text for text in feedback_texts if "Warning" in text]
        if warning_texts:
            warning_text = warning_texts[0]
            warning_w = min(int(640 * scale), w - 2 * margin)
            warning_h = int(58 * scale)
            warning_x = max(margin, (w - warning_w) // 2)
            warning_y = margin + int(80 * scale)
            draw_translucent_panel(
                frame,
                warning_x,
                warning_y,
                warning_w,
                warning_h,
                fill_color=(61, 25, 32),
                border_color=(255, 124, 124),
                alpha=0.74,
                border_thickness=max(1, int(2 * scale)),
            )
            cv2.putText(
                frame,
                warning_text,
                (warning_x + int(16 * scale), warning_y + int(36 * scale)),
                font,
                0.62 * scale,
                (255, 230, 230),
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )

    def run(self):
        cap = None
        mp_pose_solution = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.status_signal.emit(f"Error: Could not open camera {self.camera_index}.")
                logging.error(f"Could not open camera {self.camera_index}.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            logging.info(f"Camera {self.camera_index} initialized.")

            mp_pose_solution = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
            )
            aruco_detector = ArucoDetector(dict_type=self.aruco_dict_type)
            logging.info(
                "ExerciseWorker cam_%s using ArUco dictionary: %s",
                self.camera_index,
                aruco_detector.get_dict_type(),
            )

            if self.exercise_choice not in exercise_config:
                self.status_signal.emit(
                    f"Exercise '{self.exercise_choice}' not configured. Using default if available."
                )
                logging.warning(f"Exercise '{self.exercise_choice}' not found in config.")
                if exercise_config:
                    self.exercise_choice = list(exercise_config.keys())[0]
                else:
                    self.status_signal.emit("No exercises configured.")
                    logging.error("No exercises configured. Worker terminating.")
                    return

            if self.face_recognizer is None:
                self.status_signal.emit("Face recognizer is not initialized.")
                logging.error("Face recognizer is not initialized. Worker terminating.")
                return

            occlusion_timeout = 90  # About 3 seconds at ~30 FPS

            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    self.status_signal.emit("Failed to capture frame.")
                    logging.error("Failed to capture frame from camera.")
                    break

                display_frame = frame.copy()

                # If exercise analysis is active, skip face recognition
                if self.exercise_analysis_active and not self.capturing_new_user_data:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = mp_pose_solution.process(frame_rgb)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                        if results.pose_landmarks:
                            self.exercise_analyzer.no_pose_frames = 0
                            landmarks = results.pose_landmarks.landmark
                            feedback_texts, overlay_metrics = self.exercise_analyzer.analyze_exercise_form(
                                landmarks,
                                frame_bgr,
                            )

                            if not self.exercise_analyzer.stable_start_detected:
                                connection_color = (255, 140, 90)
                                landmark_color = (255, 140, 90)
                            elif self.exercise_analyzer.bend_warning_displayed:
                                connection_color = (70, 70, 255)
                                landmark_color = (70, 70, 255)
                            else:
                                connection_color = (118, 242, 130)
                                landmark_color = (118, 242, 130)

                            mp_drawing.draw_landmarks(
                                frame_bgr,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(
                                    color=connection_color,
                                    thickness=2,
                                    circle_radius=2,
                                ),
                                mp_drawing.DrawingSpec(
                                    color=landmark_color,
                                    thickness=2,
                                    circle_radius=2,
                                ),
                            )

                            self._emit_important_audio(feedback_texts)

                            overlay_data = self._build_overlay_data(overlay_metrics, feedback_texts)
                            self._draw_live_overlay(frame_bgr, overlay_data, feedback_texts)

                            final_frame = frame_bgr
                            self.frame_signal.emit(final_frame)
                            self.thumbnail_frame_signal.emit(
                                cv2.resize(final_frame, (160, 90), interpolation=cv2.INTER_AREA)
                            )
                            self.counters_signal.emit(
                                self.exercise_analyzer.rep_count,
                                self.exercise_analyzer.set_count,
                            )
                        else:
                            self.exercise_analyzer.update_current_weight(frame_bgr)
                            self.exercise_analyzer.no_pose_frames += 1
                            logging.debug(
                                "No pose detected. no_pose_frames: %s",
                                self.exercise_analyzer.no_pose_frames,
                            )

                            status_hint = "Body not detected. Step back into frame."
                            overlay_data = self._build_overlay_data(
                                metrics={
                                    "rep_count": self.exercise_analyzer.rep_count,
                                    "set_count": self.exercise_analyzer.set_count,
                                    "current_weight": self.exercise_analyzer.current_weight,
                                },
                                feedback_texts=[],
                                status_hint=status_hint,
                            )
                            self._draw_live_overlay(frame_bgr, overlay_data, [])
                            self.frame_signal.emit(frame_bgr)
                            self.thumbnail_frame_signal.emit(
                                cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
                            )
                            self.counters_signal.emit(
                                self.exercise_analyzer.rep_count,
                                self.exercise_analyzer.set_count,
                            )

                            if self.exercise_analyzer.no_pose_frames >= occlusion_timeout:
                                self.status_signal.emit("Person exited the frame, updating database...")
                                logging.info("Person exited frame. Updating database.")
                                record = self.exercise_analyzer.update_data()
                                if record:
                                    self.status_signal.emit("Exercise data saved locally.")
                                    logging.info("Data saved locally.")
                                    self.data_updated.emit()
                                else:
                                    self.status_signal.emit("No exercise data to save.")

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

                        progress = len(self.new_user_encodings)
                        hint = f"Registering {self.new_user_name}: {progress}/{self.frames_to_capture}"
                        overlay_data = self._build_overlay_data(
                            metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                            feedback_texts=[],
                            status_hint=hint,
                        )
                        self._draw_live_overlay(display_frame, overlay_data, [])

                        self.frame_signal.emit(display_frame)
                        self.thumbnail_frame_signal.emit(
                            cv2.resize(display_frame, (160, 90), interpolation=cv2.INTER_AREA)
                        )

                        if len(self.new_user_encodings) >= self.frames_to_capture:
                            self.status_signal.emit(f"Registering user {self.new_user_name}...")
                            logging.info(f"Registering new user: {self.new_user_name}")
                            registered_name = self.new_user_name
                            registration_success = self.face_recognizer.register_new_user(
                                registered_name,
                                self.new_user_encodings,
                            )
                            if registration_success:
                                self.status_signal.emit(
                                    f"User {registered_name} registered successfully."
                                )
                                logging.info(
                                    f"User {registered_name} registered successfully."
                                )
                                self.display_user_name = registered_name
                            else:
                                self.status_signal.emit(
                                    f"Failed to register user {registered_name}."
                                )
                                logging.error(
                                    f"Failed to register user {registered_name}."
                                )
                            self.new_user_registration_signal.emit(
                                registered_name,
                                registration_success,
                            )

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
                        _, face_locations, face_names = self.face_recognizer.recognize_faces(frame)
                        status_hint = "Stand in front of the camera for recognition."

                        for ((top, right, bottom, left), name) in zip(face_locations, face_names):
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            cv2.rectangle(
                                display_frame,
                                (left, top),
                                (right, bottom),
                                (110, 255, 180),
                                2,
                            )
                            cv2.rectangle(
                                display_frame,
                                (left, bottom - 28),
                                (right, bottom),
                                (18, 32, 54),
                                cv2.FILLED,
                            )
                            cv2.putText(
                                display_frame,
                                name,
                                (left + 8, bottom - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.56,
                                (224, 250, 248),
                                2,
                                cv2.LINE_AA,
                            )

                        if len(face_names) > 0:
                            if all(n == "Unknown" for n in face_names):
                                self.unknown_frames += 1
                                self.known_frames = 0
                                status_hint = "Unknown user detected. Hold still for registration."
                                if (
                                    self.unknown_frames >= self.UNKNOWN_FRAME_THRESHOLD
                                    and not self.unknown_stable_detected
                                ):
                                    self.status_signal.emit(
                                        "Stable unknown user detected. Prompting for user name..."
                                    )
                                    logging.info("Stable unknown user detected.")
                                    self.unknown_user_detected.emit()
                                    self.unknown_stable_detected = True
                            else:
                                self.unknown_frames = 0
                                self.unknown_stable_detected = False
                                recognized_user = [n for n in face_names if n != "Unknown"][0]
                                self.display_user_name = recognized_user
                                status_hint = "User recognized. Preparing real-time feedback."
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
                                        self.status_signal.emit(
                                            f"Face recognized as {recognized_user}, starting exercise analysis."
                                        )
                                        logging.info(
                                            f"Starting exercise analysis for user: {recognized_user}"
                                        )
                                        self.current_user_info = user_info
                                        self.display_user_name = recognized_user
                                        self.exercise_analyzer = ExerciseAnalyzer(
                                            self.exercise_choice,
                                            aruco_detector,
                                            self.db_handler,
                                            user_id=user_info.get("user_id"),
                                        )
                                        self.exercise_analysis_active = True
                                        self.face_recognition_active = False
                                    else:
                                        self.status_signal.emit(
                                            f"Face recognized as {recognized_user}, but no member info found."
                                        )
                                        logging.warning(
                                            "No member info found for recognized user %s.",
                                            recognized_user,
                                        )
                        else:
                            self.unknown_frames = 0
                            self.known_frames = 0
                            self.last_recognized_user = None
                            self.unknown_stable_detected = False

                        overlay_data = self._build_overlay_data(
                            metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                            feedback_texts=[],
                            status_hint=status_hint,
                        )
                        self._draw_live_overlay(display_frame, overlay_data, [])

                        self.frame_signal.emit(display_frame)
                        self.thumbnail_frame_signal.emit(
                            cv2.resize(display_frame, (160, 90), interpolation=cv2.INTER_AREA)
                        )
                    except Exception as e:
                        self.status_signal.emit("Face recognition error.")
                        logging.error(f"Face recognition error: {e}")

                self.msleep(10)
        finally:
            if cap is not None:
                cap.release()
            if mp_pose_solution is not None:
                mp_pose_solution.close()
            self.cleanup()

    def cleanup(self):
        """Handle cleanup operations when stopping the worker."""
        try:
            if self.exercise_analyzer:
                record = self.exercise_analyzer.update_data()
                if record:
                    logging.info("Exercise data saved locally during cleanup.")
                else:
                    logging.warning("No exercise data to save during cleanup.")

            self.data_updated.emit()
        except Exception as e:
            self.status_signal.emit("Error during cleanup.")
            logging.error(f"Error during cleanup: {e}")
        # No signal disconnect calls here
