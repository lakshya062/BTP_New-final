from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.paths import resource_path


class RegisterUserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Register New User")
        self.setModal(True)
        self.resize(520, 620)

        layout = QVBoxLayout()
        self.setLayout(layout)

        intro = QLabel("Enter user details for registration. Username and email are required.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("unique username")
        form.addRow("Username*:", self.username_input)

        self.first_name_input = QLineEdit()
        form.addRow("First Name:", self.first_name_input)

        self.last_name_input = QLineEdit()
        form.addRow("Last Name:", self.last_name_input)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("name@example.com")
        form.addRow("Email*:", self.email_input)

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("+1xxxxxxxxxx")
        form.addRow("Phone:", self.phone_input)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["", "Male", "Female", "Other", "Prefer not to say"])
        form.addRow("Gender:", self.gender_combo)

        self.dob_known_check = QCheckBox("Provide Date of Birth")
        self.dob_known_check.setChecked(False)
        form.addRow("Date of Birth:", self.dob_known_check)

        self.dob_input = QDateEdit()
        self.dob_input.setDisplayFormat("yyyy-MM-dd")
        self.dob_input.setCalendarPopup(True)
        self.dob_input.setDate(QDate.currentDate())
        self.dob_input.setEnabled(False)
        self.dob_known_check.toggled.connect(self.dob_input.setEnabled)
        form.addRow("DOB Value:", self.dob_input)

        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("e.g. 175")
        form.addRow("Height (cm):", self.height_input)

        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("e.g. 72")
        form.addRow("Weight (kg):", self.weight_input)

        self.membership_combo = QComboBox()
        self.membership_combo.addItems(["Basic", "Premium", "VIP"])
        form.addRow("Membership:", self.membership_combo)

        self.joined_on_input = QDateEdit()
        self.joined_on_input.setDisplayFormat("yyyy-MM-dd")
        self.joined_on_input.setCalendarPopup(True)
        self.joined_on_input.setDate(QDate.currentDate())
        form.addRow("Joined On:", self.joined_on_input)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.save_btn = QPushButton(QIcon(resource_path("icons", "save.png")), "Save & Start Capture")
        self.cancel_btn = QPushButton(QIcon(resource_path("icons", "cancel.png")), "Cancel")
        self.save_btn.clicked.connect(self.validate_and_accept)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

    def validate_and_accept(self):
        username = self.username_input.text().strip()
        email = self.email_input.text().strip()

        if not username:
            QMessageBox.warning(self, "Input Error", "Username is required.")
            return
        if not email:
            QMessageBox.warning(self, "Input Error", "Email is required.")
            return
        if "@" not in email or "." not in email:
            QMessageBox.warning(self, "Input Error", "Enter a valid email address.")
            return

        for label, value in (("Height", self.height_input.text()), ("Weight", self.weight_input.text())):
            text = value.strip()
            if not text:
                continue
            try:
                float(text)
            except ValueError:
                QMessageBox.warning(self, "Input Error", f"{label} must be a number.")
                return

        self.accept()

    def get_member_info(self):
        info = {
            "username": self.username_input.text().strip(),
            "first_name": self.first_name_input.text().strip() or None,
            "last_name": self.last_name_input.text().strip() or None,
            "email": self.email_input.text().strip(),
            "phone": self.phone_input.text().strip() or None,
            "gender": self.gender_combo.currentText().strip() or None,
            "date_of_birth": self.dob_input.date().toString("yyyy-MM-dd") if self.dob_known_check.isChecked() else None,
            "height_cm": self.height_input.text().strip() or None,
            "weight_kg": self.weight_input.text().strip() or None,
            "membership": self.membership_combo.currentText().strip() or "Basic",
            "joined_on": self.joined_on_input.date().toString("yyyy-MM-dd"),
        }
        return info
