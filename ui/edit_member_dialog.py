# ui/edit_member_dialog.py

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QDateEdit,
    QPushButton, QHBoxLayout, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QIcon
from core.paths import resource_path

class EditMemberDialog(QDialog):
    def __init__(self, current_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Member")
        self.setModal(True)
        self.resize(450, 400)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        form_layout = QFormLayout()

        self.username_edit = QLineEdit(current_info.get("username", "N/A"))
        self.username_edit.setToolTip("Edit the username")
        self.email_edit = QLineEdit(current_info.get("email", "N/A"))
        self.email_edit.setToolTip("Edit the email address")
        self.membership_combo = QComboBox()
        self.membership_combo.addItems(["Basic", "Premium", "VIP"])
        current_membership = current_info.get("membership", "Basic")
        idx = self.membership_combo.findText(current_membership)
        if idx >= 0:
            self.membership_combo.setCurrentIndex(idx)
        self.membership_combo.setToolTip("Select membership level")

        self.joined_edit = QDateEdit()
        self.joined_edit.setDisplayFormat("yyyy-MM-dd")
        joined_on = current_info.get("joined_on", "2023-01-01")
        try:
            jdate = QDate.fromString(joined_on, "yyyy-MM-dd")
            if not jdate.isValid():
                jdate = QDate.currentDate()
        except Exception:
            jdate = QDate.currentDate()
        self.joined_edit.setDate(jdate)
        self.joined_edit.setCalendarPopup(True)
        self.joined_edit.setToolTip("Select the joining date")

        form_layout.addRow(QLabel("<b>Username:</b>"), self.username_edit)
        form_layout.addRow(QLabel("<b>Email:</b>"), self.email_edit)
        form_layout.addRow(QLabel("<b>Membership:</b>"), self.membership_combo)
        form_layout.addRow(QLabel("<b>Joined On:</b>"), self.joined_edit)

        self.layout.addLayout(form_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.save_button = QPushButton(QIcon(resource_path("icons", "save.png")), "Save")
        self.cancel_button = QPushButton(QIcon(resource_path("icons", "cancel.png")), "Cancel")
        self.save_button.setToolTip("Save changes")
        self.cancel_button.setToolTip("Cancel and close the dialog")
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.cancel_button)

        self.layout.addLayout(buttons_layout)

        # Connect Buttons
        self.save_button.clicked.connect(self.save_and_accept)
        self.cancel_button.clicked.connect(self.reject)

    def save_and_accept(self):
        """Validate and save the edited member information."""
        username = self.username_edit.text().strip()
        email = self.email_edit.text().strip()
        membership = self.membership_combo.currentText()
        joined_on = self.joined_edit.date().toString("yyyy-MM-dd")

        if not username:
            QMessageBox.warning(self, "Input Error", "Username cannot be empty.")
            return
        if not email:
            QMessageBox.warning(self, "Input Error", "Email cannot be empty.")
            return
        if "@" not in email or "." not in email:
            QMessageBox.warning(self, "Input Error", "Please enter a valid email address.")
            return

        self.updated_info = {
            "username": username,
            "email": email,
            "membership": membership,
            "joined_on": joined_on
        }
        self.accept()

    def get_updated_info(self):
        """Return the updated member information."""
        return self.updated_info
