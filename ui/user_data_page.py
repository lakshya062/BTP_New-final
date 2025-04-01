# ui/user_data_page.py

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QIcon
from datetime import datetime
import os

class UserExerciseDataPage(QWidget):
    def __init__(self, db_handler, user_id, embedded=False):
        super().__init__()
        self.db_handler = db_handler
        self.user_id = user_id
        self.embedded = embedded

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("Exercise Data")
        self.header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("margin-bottom: 20px;")
        self.layout.addWidget(self.header)

        # Stacked Widget
        self.stacked = QStackedWidget()

        # Exercises Page
        self.exercises_page = QWidget()
        self.exercises_layout = QVBoxLayout()
        self.exercises_page.setLayout(self.exercises_layout)

        self.exercise_label = QLabel("Select an Exercise:")
        self.exercise_label.setFont(QFont("Segoe UI", 14))
        self.exercises_layout.addWidget(self.exercise_label)

        self.exercises_table = QTableWidget()
        self.exercises_table.setColumnCount(1)
        self.exercises_table.setHorizontalHeaderLabels(["Exercise"])
        self.exercises_table.horizontalHeader().setStretchLastSection(True)
        self.exercises_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.exercises_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.exercises_table.setAlternatingRowColors(True)
        self.exercises_table.setStyleSheet("QTableWidget { background-color: #2E2E2E; color: #C5C6C7; }")
        self.exercises_table.cellDoubleClicked.connect(self.on_exercise_double_clicked)
        self.exercises_layout.addWidget(self.exercises_table)

        # Back Button if Embedded
        if self.embedded:
            self.back_button = QPushButton(QIcon(os.path.join("resources", "icons", "back.png")), "Back to Members")
            self.back_button.setToolTip("Return to the members list")
            self.exercises_layout.addWidget(self.back_button, alignment=Qt.AlignRight)

        self.exercises_layout.addStretch()

        self.stacked.addWidget(self.exercises_page)

        # Sessions Page
        self.sessions_page = QWidget()
        self.sessions_layout = QVBoxLayout()
        self.sessions_page.setLayout(self.sessions_layout)

        self.sessions_label = QLabel("Select a Session:")
        self.sessions_label.setFont(QFont("Segoe UI", 14))
        self.sessions_layout.addWidget(self.sessions_label)

        self.sessions_table = QTableWidget()
        self.sessions_table.setColumnCount(6)
        self.sessions_table.setHorizontalHeaderLabels(["Exercise Name", "Sets", "Reps", "Date", "Timestamp", "Session ID"])
        self.sessions_table.setColumnHidden(5, True)
        self.sessions_table.horizontalHeader().setStretchLastSection(True)
        self.sessions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sessions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sessions_table.setAlternatingRowColors(True)
        self.sessions_table.setStyleSheet("QTableWidget { background-color: #2E2E2E; color: #C5C6C7; }")
        self.sessions_table.cellDoubleClicked.connect(self.on_session_double_clicked)
        self.sessions_layout.addWidget(self.sessions_table)

        self.back_to_exercises_button = QPushButton(QIcon(os.path.join("resources", "icons", "back.png")), "Back to Exercises")
        self.back_to_exercises_button.setToolTip("Return to the exercises list")
        self.back_to_exercises_button.clicked.connect(self.back_to_exercises)
        self.sessions_layout.addWidget(self.back_to_exercises_button, alignment=Qt.AlignRight)

        self.sessions_layout.addStretch()

        self.stacked.addWidget(self.sessions_page)

        # Rep Details Page
        self.rep_details_page = QWidget()
        self.rep_details_layout = QVBoxLayout()
        self.rep_details_page.setLayout(self.rep_details_layout)

        self.rep_label = QLabel("Rep Details:")
        self.rep_label.setFont(QFont("Segoe UI", 14))
        self.rep_details_layout.addWidget(self.rep_label)

        self.rep_table = QTableWidget()
        self.rep_table.setColumnCount(4)
        self.rep_table.setHorizontalHeaderLabels(["Rep Number", "Start Angle (°)", "End Angle (°)", "Weight (lbs)"])
        self.rep_table.horizontalHeader().setStretchLastSection(True)
        self.rep_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rep_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rep_table.setAlternatingRowColors(True)
        self.rep_table.setStyleSheet("QTableWidget { background-color: #2E2E2E; color: #C5C6C7; }")
        self.rep_details_layout.addWidget(self.rep_table)

        self.back_to_sessions_button = QPushButton(QIcon(os.path.join("resources", "icons", "back.png")), "Back to Sessions")
        self.back_to_sessions_button.setToolTip("Return to the sessions list")
        self.back_to_sessions_button.clicked.connect(self.back_to_sessions)
        self.rep_details_layout.addWidget(self.back_to_sessions_button, alignment=Qt.AlignRight)

        self.rep_details_layout.addStretch()

        self.stacked.addWidget(self.rep_details_page)

        self.layout.addWidget(self.stacked)

        # Load Exercises Initially
        self.load_exercises()

    def load_exercises(self):
        """Load exercises from the database and populate the table."""
        self.exercises_table.setRowCount(0)
        data = self.db_handler.get_exercise_data_for_user(self.user_id)
        exercises = sorted(list(set([entry['exercise'] for entry in data])))
        for exercise in exercises:
            row = self.exercises_table.rowCount()
            self.exercises_table.insertRow(row)
            self.exercises_table.setItem(row, 0, QTableWidgetItem(exercise))
        self.exercises_table.resizeColumnsToContents()
        self.stacked.setCurrentWidget(self.exercises_page)

    @Slot(int, int)
    def on_exercise_double_clicked(self, row, column):
        """Show sessions for the selected exercise."""
        exercise_item = self.exercises_table.item(row, 0)
        if exercise_item:
            self.selected_exercise = exercise_item.text()
            self.load_sessions(self.selected_exercise)
            self.stacked.setCurrentWidget(self.sessions_page)

    def load_sessions(self, exercise):
        """Load sessions for the selected exercise."""
        self.sessions_table.setRowCount(0)
        data = self.db_handler.get_exercise_data_for_user(self.user_id)
        sessions = [entry for entry in data if entry['exercise'] == exercise]
        # Sort sessions by timestamp descending
        sessions_sorted = sorted(sessions, key=lambda x: x['timestamp'] if x['timestamp'] else '', reverse=True)
        for session in sessions_sorted:
            row = self.sessions_table.rowCount()
            self.sessions_table.insertRow(row)
            self.sessions_table.setItem(row, 0, QTableWidgetItem(session['exercise']))
            self.sessions_table.setItem(row, 1, QTableWidgetItem(str(session['set_count'])))
            total_reps = sum(session['sets_reps']) if session['sets_reps'] else 0
            self.sessions_table.setItem(row, 2, QTableWidgetItem(str(total_reps)))
            self.sessions_table.setItem(row, 3, QTableWidgetItem(session['date'] if session['date'] else ''))
            self.sessions_table.setItem(row, 4, QTableWidgetItem(session['timestamp'] if session['timestamp'] else ''))
            self.sessions_table.setItem(row, 5, QTableWidgetItem(session['id']))
        self.sessions_table.resizeColumnsToContents()

    @Slot(int, int)
    def on_session_double_clicked(self, row, column):
        """Show rep details for the selected session."""
        session_id_item = self.sessions_table.item(row, 5)
        if session_id_item:
            self.selected_session_id = session_id_item.text()
            self.load_rep_details(self.selected_session_id)
            self.stacked.setCurrentWidget(self.rep_details_page)

    def load_rep_details(self, session_id):
        """Load rep details for the selected session."""
        self.rep_table.setRowCount(0)
        data = self.db_handler.get_exercise_data_for_user(self.user_id)
        session = next((entry for entry in data if entry['id'] == session_id), None)
        if not session:
            QMessageBox.warning(self, "Error", "Session data not found.")
            return
        rep_data = session.get('rep_data', [])
        for idx, rep in enumerate(rep_data, start=1):
            row = self.rep_table.rowCount()
            self.rep_table.insertRow(row)
            self.rep_table.setItem(row, 0, QTableWidgetItem(str(idx)))
            self.rep_table.setItem(row, 1, QTableWidgetItem(str(rep.get('start_angle', 'N/A'))))
            self.rep_table.setItem(row, 2, QTableWidgetItem(str(rep.get('end_angle', 'N/A'))))
            self.rep_table.setItem(row, 3, QTableWidgetItem(str(rep.get('weight', 'N/A'))))
        self.rep_table.resizeColumnsToContents()

    def back_to_exercises(self):
        """Navigate back to the Exercises Page."""
        self.stacked.setCurrentWidget(self.exercises_page)

    def back_to_sessions(self):
        """Navigate back to the Sessions Page."""
        self.stacked.setCurrentWidget(self.sessions_page)

    def go_back(self):
        """Navigate back to the Members List Page."""
        self.parent().go_back_to_members() if self.embedded else self.close()