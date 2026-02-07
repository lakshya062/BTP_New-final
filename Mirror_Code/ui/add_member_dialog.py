# ui/add_member_dialog.py

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QDateEdit,
    QPushButton, QMessageBox
)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QIcon
from core.paths import resource_path

class AddMemberDialog(QDialog):
    def __init__(self, db_handler, parent=None):
        super().__init__(parent)
        self.db_handler = db_handler
        self.setWindowTitle("Add Member")
        self.setModal(True)
        self.resize(450, 400)
    
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
    
        # Form Layout
        self.form_layout = QVBoxLayout()
    
        # Username
        self.username_label = QLabel("Username:")
        self.username_edit = QLineEdit()
        self.username_edit.setToolTip("Enter the member's username")
        self.form_layout.addWidget(self.username_label)
        self.form_layout.addWidget(self.username_edit)
    
        # Email
        self.email_label = QLabel("Email:")
        self.email_edit = QLineEdit()
        self.email_edit.setToolTip("Enter the member's email address")
        self.form_layout.addWidget(self.email_label)
        self.form_layout.addWidget(self.email_edit)
    
        # Membership
        self.membership_label = QLabel("Membership:")
        self.membership_combo = QComboBox()
        self.membership_combo.addItems(["Basic", "Premium", "VIP"])
        self.membership_combo.setToolTip("Select the membership level")
        self.form_layout.addWidget(self.membership_label)
        self.form_layout.addWidget(self.membership_combo)
    
        # Joined On
        self.joined_label = QLabel("Joined On:")
        self.joined_edit = QDateEdit()
        self.joined_edit.setDisplayFormat("yyyy-MM-dd")
        self.joined_edit.setDate(QDate.currentDate())
        self.joined_edit.setCalendarPopup(True)
        self.joined_edit.setToolTip("Select the joining date")
        self.form_layout.addWidget(self.joined_label)
        self.form_layout.addWidget(self.joined_edit)
    
        self.layout.addLayout(self.form_layout)
    
        # Buttons
        self.buttons_layout = QHBoxLayout()
        self.save_button = QPushButton(QIcon(resource_path("icons", "save.png")), "Save")
        self.cancel_button = QPushButton(QIcon(resource_path("icons", "cancel.png")), "Cancel")
        self.save_button.setToolTip("Save the new member")
        self.cancel_button.setToolTip("Cancel and close the dialog")
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.save_button)
        self.buttons_layout.addWidget(self.cancel_button)
    
        self.layout.addLayout(self.buttons_layout)
    
        # Connect Buttons
        self.save_button.clicked.connect(self.save_member)
        self.cancel_button.clicked.connect(self.reject)
    
    def save_member(self):
        """Validate inputs and save the new member."""
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
    
        # Check if username already exists
        existing_member = self.db_handler.get_member_info(username)
        if existing_member:
            QMessageBox.warning(self, "Input Error", f"Username '{username}' already exists.")
            return
    
        # Create a unique user_id
        import uuid
        user_id = str(uuid.uuid4())
    
        member_info = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "membership": membership,
            "joined_on": joined_on
        }
    
        success = self.db_handler.insert_member_local(member_info)
        if success:
            QMessageBox.information(self, "Success", f"Member '{username}' added successfully.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to add member '{username}'.")
