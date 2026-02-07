# ui/member_list_page.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QPushButton, QHBoxLayout, QLabel, QStackedWidget, QMessageBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from ui.user_exercise_data_page import UserExerciseDataPage
from PySide6.QtGui import QIcon
from core.paths import resource_path

class MemberListPage(QWidget):
    # Signal to refresh members list externally if needed
    refresh_members_signal = Signal()

    def __init__(self, db_handler, face_recognizer):
        super().__init__()
        self.db_handler = db_handler
        self.face_recognizer = face_recognizer

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("Members")
        self.header.setObjectName("pageHeader")
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)

        # Search Bar
        self.search_layout = QHBoxLayout()
        self.search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter username or email...")
        self.search_input.textChanged.connect(self.search_members)
        self.search_layout.addWidget(self.search_label)
        self.search_layout.addWidget(self.search_input)
        self.layout.addLayout(self.search_layout)

        # Stacked Widget for Switching Views
        self.stacked = QStackedWidget()

        # Members List View
        self.members_widget = QWidget()
        self.members_layout = QVBoxLayout()
        self.members_widget.setLayout(self.members_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Username", "Email", "Membership", "Joined On", "User ID"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.members_layout.addWidget(self.table)

        # Buttons Layout
        self.members_buttons_layout = QHBoxLayout()
        self.refresh_button = QPushButton(QIcon(resource_path("icons", "refresh.png")), "Refresh")
        self.refresh_button.setToolTip("Refresh the members list")
        self.delete_button = QPushButton(QIcon(resource_path("icons", "delete.png")), "Delete Member")
        self.delete_button.setToolTip("Delete the selected member(s)")
        self.members_buttons_layout.addWidget(self.refresh_button)
        self.members_buttons_layout.addWidget(self.delete_button)
        self.members_buttons_layout.addStretch()

        self.members_layout.addLayout(self.members_buttons_layout)
        self.members_layout.addStretch()

        self.stacked.addWidget(self.members_widget)

        # User Exercise Data View
        self.user_data_container = QWidget()
        self.user_data_layout = QVBoxLayout()
        self.user_data_container.setLayout(self.user_data_layout)
        self.stacked.addWidget(self.user_data_container)

        self.layout.addWidget(self.stacked)

        # Connect Buttons
        self.refresh_button.clicked.connect(self.load_members)
        self.delete_button.clicked.connect(self.delete_member)

        # Load Members Initially
        self.load_members()

    def load_members(self):
        """Load members from the database and populate the table."""
        try:
            members = self.db_handler.get_all_members()  # Ensure this calls the local method
            self.table.setRowCount(0)
            for m in members:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(m.get("username","N/A")))
                self.table.setItem(row, 1, QTableWidgetItem(m.get("email","NA")))
                self.table.setItem(row, 2, QTableWidgetItem(m.get("membership","NA")))
                self.table.setItem(row, 3, QTableWidgetItem(m.get("joined_on","N/A")))
                self.table.setItem(row, 4, QTableWidgetItem(m.get("user_id","")))
            self.table.resizeColumnsToContents()
            self.stacked.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load members: {e}")

    def search_members(self, text):
        """Filter members based on search input."""
        for row in range(self.table.rowCount()):
            match = False
            for column in range(self.table.columnCount() -1):  # Exclude User ID from search
                item = self.table.item(row, column)
                if text.lower() in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(row, not match)

    def on_cell_double_clicked(self, row, column):
        """Show user exercise data in the same page using the stacked widget."""
        try:
            user_id = self.table.item(row,4).text()
            if user_id:
                # Clear previous widgets
                for i in reversed(range(self.user_data_layout.count())):
                    widget = self.user_data_layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)

                user_data_page = UserExerciseDataPage(self.db_handler, user_id, embedded=True)
                # Connect the back_button signal to go_back_to_members
                if hasattr(user_data_page, 'back_button'):
                    user_data_page.back_button.clicked.connect(self.go_back_to_members)
                else:
                    pass
                self.user_data_layout.addWidget(user_data_page)

                self.stacked.setCurrentWidget(self.user_data_container)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display user data: {e}")

    def go_back_to_members(self):
        """Navigate back to the Members List Page."""
        self.stacked.setCurrentIndex(0)
        self.load_members()

    def delete_member(self):
        """Delete the selected member(s) from DB and face recognition model."""
        try:
            selected_rows = self.table.selectionModel().selectedRows()
            if not selected_rows:
                QMessageBox.warning(self, "Delete Member", "Please select a member to delete.")
                return

            reply = QMessageBox.question(
                self, "Delete Member", "Are you sure you want to delete the selected member(s)? This will also delete all associated exercise data.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                for index in sorted(selected_rows, reverse=True):
                    username = self.table.item(index.row(), 0).text()
                    user_id = self.table.item(index.row(), 4).text()
                    if username and user_id:
                        success = self.db_handler.delete_member_local(username)  # Ensure this calls the local method
                        if success:
                            delete_success = self.face_recognizer.delete_user_from_model(username)
                            if delete_success:
                                QMessageBox.information(self, "Delete Member", f"Member '{username}' deleted successfully.")
                            else:
                                QMessageBox.warning(self, "Delete Member", f"Failed to delete member from face model: {username}")
                            self.table.removeRow(index.row())
                        else:
                            QMessageBox.warning(self, "Delete Member", f"Failed to delete member: {username}")
                self.load_members()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while deleting members: {e}")
