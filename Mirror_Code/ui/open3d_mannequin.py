import open3d as o3d
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QSize
from PySide6.QtGui import QFont
import datetime


class Open3DMannequinWidget(QWidget):
    """
    A custom widget that uses Open3D to visualize the user's 3D pose (a mannequin).
    This example attempts a real-time approach by updating the geometry in the Visualizer.
    """

    # Signal for logging or status updates if needed
    status_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setStyleSheet("background-color: black;")

        # Layout to hold top, center (3D view), bottom
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

        # Top bar: Name and Time
        self.top_bar = QHBoxLayout()
        self.top_left_label = QLabel("Name: Unknown")
        self.top_left_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.top_left_label.setStyleSheet("color: white;")
        self.top_right_label = QLabel("--:--:--")
        self.top_right_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.top_right_label.setStyleSheet("color: white;")
        self.top_bar.addWidget(self.top_left_label, alignment=Qt.AlignLeft)
        self.top_bar.addWidget(self.top_right_label, alignment=Qt.AlignRight)

        self.main_layout.addLayout(self.top_bar)

        # 3D view placeholder
        # Because direct embedding of open3d Visualizer is tricky in Qt,
        # we either do a placeholder or an offscreen rendering approach.
        #
        # We'll store references to the geometry and do an offscreen update,
        # then convert to QImage in a QTimer. (Simplified approach)
        self.viewer_label = QLabel()  # We will show the rendered frames here
        self.viewer_label.setAlignment(Qt.AlignCenter)
        self.viewer_label.setStyleSheet("background-color: black;")
        self.viewer_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.viewer_label)

        # Bottom bar: data (weight, height, exercise, sets, reps, BMI, etc.)
        self.bottom_layout = QVBoxLayout()

        self.exercise_label = QLabel("Exercise: -")
        self.exercise_label.setStyleSheet("color: white; font-size: 14px;")

        self.weight_label = QLabel("Weight (User): -")
        self.weight_label.setStyleSheet("color: white; font-size: 14px;")

        self.height_label = QLabel("Height: -")
        self.height_label.setStyleSheet("color: white; font-size: 14px;")

        self.exercise_weight_label = QLabel("Exercise Weight: -")
        self.exercise_weight_label.setStyleSheet("color: white; font-size: 14px;")

        self.reps_sets_label = QLabel("Reps: 0 | Sets: 0")
        self.reps_sets_label.setStyleSheet("color: white; font-size: 14px;")

        self.bmi_label = QLabel("BMI: -")
        self.bmi_label.setStyleSheet("color: white; font-size: 14px;")

        self.bottom_layout.addWidget(self.exercise_label)
        self.bottom_layout.addWidget(self.weight_label)
        self.bottom_layout.addWidget(self.height_label)
        self.bottom_layout.addWidget(self.exercise_weight_label)
        self.bottom_layout.addWidget(self.reps_sets_label)
        self.bottom_layout.addWidget(self.bmi_label)

        self.main_layout.addLayout(self.bottom_layout)

        # Create Open3D objects for skeleton lines
        self._create_open3d_scene()

        # QTimer to update the time on top_right_label
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_time)
        self.update_timer.start(1000)  # update every 1 second

    def _create_open3d_scene(self):
        # We create a single open3d Visualizer in "offscreen" mode
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480, visible=False)
        self.skeleton_lines = o3d.geometry.LineSet()
        self.skeleton_lines.points = o3d.utility.Vector3dVector([])
        self.skeleton_lines.lines = o3d.utility.Vector2iVector([])
        self.skeleton_lines.colors = o3d.utility.Vector3dVector([])
        self.vis.add_geometry(self.skeleton_lines)

        # Setup a black background
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])  # black
        opt.point_size = 5

        # We'll define connections as per some known pose skeleton
        # e.g., 0->1, 1->2, etc. (Mediapipe has 33 landmarks, you must define your edges)
        # Below is just an example subset for demonstration:
        self.body_edges = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),            # Shoulders
            (23, 24),            # Hips
            (11, 23), (12, 24),  # Torso
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]

    def update_time(self):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.top_right_label.setText(now)

    def update_user_info(self, name=None, user_weight=None, height=None, bmi=None):
        """Set user info fields."""
        if name is not None:
            self.top_left_label.setText(f"Name: {name}")
        if user_weight is not None:
            self.weight_label.setText(f"Weight (User): {user_weight} kg")
        if height is not None:
            self.height_label.setText(f"Height: {height} cm")
        if bmi is not None:
            self.bmi_label.setText(f"BMI: {bmi:.1f}")

    def update_exercise_info(self, exercise=None, exercise_weight=None, reps=0, sets_=0):
        """Set exercise info fields."""
        if exercise is not None:
            self.exercise_label.setText(f"Exercise: {exercise}")
        if exercise_weight is not None:
            self.exercise_weight_label.setText(f"Exercise Weight: {exercise_weight} lbs")
        self.reps_sets_label.setText(f"Reps: {reps} | Sets: {sets_}")

    @Slot(object)
    def update_pose(self, landmarks):
        """
        Receive the 3D landmarks from the worker.
        landmarks is expected to be a list of (x, y, z) for each Mediapipe body landmark.
        We update our open3d LineSet geometry.
        """
        if not landmarks or len(landmarks) < 33:
            return

        # Convert to Nx3 numpy array
        # Here we assume each landmark is (x, y, z) in some domain
        # Scale or shift as needed to keep it visible in the 3D view
        points = np.array(landmarks, dtype=np.float32)

        # We can do a simple scale so that the coordinates show up well
        # E.g., multiply by some factor
        points[:, :3] *= 2.0  # or a factor that looks good in your 3D viewer

        # Build the list of edges
        lines = []
        for (start, end) in self.body_edges:
            if start < len(points) and end < len(points):
                lines.append([start, end])

        # Update the line set
        self.skeleton_lines.points = o3d.utility.Vector3dVector(points)
        self.skeleton_lines.lines = o3d.utility.Vector2iVector(lines)
        # Color all lines white for now
        colors = [[1.0, 1.0, 1.0] for _ in lines]
        self.skeleton_lines.colors = o3d.utility.Vector3dVector(colors)

        # Update the geometry in the Visualizer
        self.vis.update_geometry(self.skeleton_lines)
        self.vis.poll_events()
        self.vis.update_renderer()

        # Now grab the rendered image from the Open3D Visualizer
        image = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        # Convert float buffer to uint8
        image = (255 * image).astype(np.uint8)

        # Convert the image to a QPixmap
        # We must convert from (H,W,3) to QImage, then QPixmap
        h, w, c = image.shape
        from PySide6.QtGui import QImage, QPixmap
        qimg = QImage(image.data, w, h, 3*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale the pixmap to fit in viewer_label
        scaled_pix = pixmap.scaled(self.viewer_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.viewer_label.setPixmap(scaled_pix)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # On resizing, we might want to re-render or scale the existing
        # pixmap. For simplicity, just call update_pose again if needed.
        # Or do nothing special – the scaled pixmap approach is enough.
