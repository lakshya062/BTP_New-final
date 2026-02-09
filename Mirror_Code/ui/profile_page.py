# ui/profile_page.py

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
import os
from core.paths import resource_path

class ProfilePage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("Gym Owner Profile")
        self.header.setObjectName("pageHeader")
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)

        # Profile Picture and Info
        profile_info_layout = QHBoxLayout()

        # Profile Picture
        self.profile_pic = QLabel()
        self.profile_pic.setFixedSize(200, 200)
        self.profile_pic.setStyleSheet("border: 2px solid #007ACC; border-radius: 100px;")
        # Load a default profile picture or a placeholder
        default_pic_path = resource_path("profiles", "owner.png")
        if os.path.exists(default_pic_path):
            pixmap = QPixmap(default_pic_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.profile_pic.setPixmap(pixmap)
        else:
            # Set a default colored circle as a placeholder
            pixmap = QPixmap(200, 200)
            pixmap.fill(Qt.transparent)
            from PySide6.QtGui import QPainter, QColor
            painter = QPainter(pixmap)
            painter.setBrush(QColor("#555555"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 200, 200)
            painter.end()
            self.profile_pic.setPixmap(pixmap)
        
        profile_info_layout.addWidget(self.profile_pic)

        # Owner Information
        self.info_frame = QFrame()
        self.info_layout = QVBoxLayout()
        self.info_frame.setLayout(self.info_layout)

        self.name_label = QLabel("Name: John Doe")
        # self.name_label.setFont(QFont("Segoe UI", 14))
        self.email_label = QLabel("Email: johndoe@example.com")
        # self.email_label.setFont(QFont("Segoe UI", 14))
        self.contact_label = QLabel("Contact: +1 234 567 8901")
        # self.contact_label.setFont(QFont("Segoe UI", 14))
        self.address_label = QLabel("Address: 123 Fitness Ave, Gym City")
        # self.address_label.setFont(QFont("Segoe UI", 14))
        self.membership_label = QLabel("Membership: VIP")
        # self.membership_label.setFont(QFont("Segoe UI", 14))

        self.info_layout.addWidget(self.name_label)
        self.info_layout.addWidget(self.email_label)
        self.info_layout.addWidget(self.contact_label)
        self.info_layout.addWidget(self.address_label)
        self.info_layout.addWidget(self.membership_label)
        self.info_layout.addStretch()

        profile_info_layout.addWidget(self.info_frame)

        self.layout.addLayout(profile_info_layout)

        self.layout.addStretch()

    def update_profile(self, user_info):
        """Update the profile display with user information."""
        username = user_info.get("username", "Unknown")
        first_name = user_info.get("first_name") or ""
        last_name = user_info.get("last_name") or ""
        full_name = f"{first_name} {last_name}".strip()
        email = user_info.get("email", "Unknown")
        membership = user_info.get("membership", "Unknown")
        phone = user_info.get("phone") or "N/A"
        date_of_birth = user_info.get("date_of_birth") or "N/A"
        height_cm = user_info.get("height_cm")
        weight_kg = user_info.get("weight_kg")

        body_stats = []
        if height_cm not in (None, ""):
            body_stats.append(f"Height: {height_cm} cm")
        if weight_kg not in (None, ""):
            body_stats.append(f"Weight: {weight_kg} kg")
        body_stats_text = " | ".join(body_stats) if body_stats else "Height/Weight: N/A"

        # Update Labels
        display_name = full_name if full_name else username
        self.name_label.setText(f"Name: {display_name} (@{username})")
        self.email_label.setText(f"Email: {email}")
        self.membership_label.setText(f"Membership: {membership}")
        self.contact_label.setText(f"Contact: {phone}")
        self.address_label.setText(f"DOB: {date_of_birth} | {body_stats_text}")

        # Update Profile Picture if available
        profile_pic_path = resource_path("profiles", f"{username}.png")
        if os.path.exists(profile_pic_path):
            pixmap = QPixmap(profile_pic_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.profile_pic.setPixmap(pixmap)
        else:
            # Keep the existing or default picture
            pass
