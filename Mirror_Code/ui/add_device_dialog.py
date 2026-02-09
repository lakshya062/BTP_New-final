import ipaddress

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.edge_device_manager import DiscoveredEdgeDevice


class AddDeviceDialog(QDialog):
    def __init__(self, edge_device_manager, parent=None):
        super().__init__(parent)
        self.edge_device_manager = edge_device_manager
        self.discovered_devices = []

        self.setWindowTitle("Add Edge Device")
        self.setModal(True)
        self.setMinimumWidth(620)

        root_layout = QVBoxLayout()
        self.setLayout(root_layout)

        intro = QLabel(
            "Ensure the Linux edge laptop is on the same local network and SSH access is enabled."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 12px;")
        root_layout.addWidget(intro)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Linux username on edge device")
        self.username_input.setText(self._default_username())
        form_layout.addRow("SSH Username:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Linux password on edge device")
        form_layout.addRow("SSH Password:", self.password_input)

        self.remote_dir_input = QLineEdit("~/smart_mirror_edge")
        form_layout.addRow("Remote Install Dir:", self.remote_dir_input)

        self.display_input = QLineEdit(":0")
        form_layout.addRow("Remote DISPLAY:", self.display_input)

        self.manual_ip_input = QLineEdit()
        self.manual_ip_input.setPlaceholderText("Optional manual edge IP, e.g. 192.168.1.42")
        form_layout.addRow("Manual IP:", self.manual_ip_input)

        self.install_deps_checkbox = QCheckBox(
            "Force dependency reinstall on Start All"
        )
        self.install_deps_checkbox.setChecked(False)
        form_layout.addRow("", self.install_deps_checkbox)

        self.setup_postgres_checkbox = QCheckBox(
            "Setup PostgreSQL on edge and apply edge schema during deploy (required)"
        )
        self.setup_postgres_checkbox.setChecked(True)
        self.setup_postgres_checkbox.setEnabled(False)
        form_layout.addRow("", self.setup_postgres_checkbox)

        root_layout.addLayout(form_layout)

        scan_layout = QHBoxLayout()
        self.scan_button = QPushButton("Scan Connected Devices")
        self.scan_button.clicked.connect(self.scan_devices)
        scan_layout.addWidget(self.scan_button)
        scan_layout.addStretch()
        root_layout.addLayout(scan_layout)

        self.device_dropdown = QComboBox()
        self.device_dropdown.setMinimumHeight(34)
        root_layout.addWidget(self.device_dropdown)

        self.status_label = QLabel("Press scan to discover Linux devices.")
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.add_button = QPushButton("Add Device")
        self.cancel_button = QPushButton("Cancel")
        self.add_button.clicked.connect(self.validate_and_accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.cancel_button)
        root_layout.addLayout(button_layout)

        QTimer.singleShot(100, self.scan_devices)

    def scan_devices(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        if not username:
            self.status_label.setText("Enter SSH username before scanning.")
            return

        self.scan_button.setEnabled(False)
        if password:
            self.status_label.setText("Scanning network interfaces for SSH devices (with Linux verification)...")
        else:
            self.status_label.setText("Scanning network interfaces for SSH devices (without Linux verification)...")
        try:
            devices = self.edge_device_manager.discover_devices(
                username=username,
                password=password,
            )
            self._set_discovered_devices(devices)
        except Exception as exc:
            self.device_dropdown.clear()
            self.discovered_devices = []
            self.status_label.setText(f"Device scan failed: {exc}")
        finally:
            self.scan_button.setEnabled(True)

    def _set_discovered_devices(self, devices):
        self.discovered_devices = list(devices)
        self.device_dropdown.clear()
        if not self.discovered_devices:
            self.status_label.setText(
                "No SSH devices found. Verify same LAN/Wi-Fi network + SSH access and scan again."
            )
            return

        linux_count = sum(1 for d in self.discovered_devices if d.linux_verified)
        for device in self.discovered_devices:
            self.device_dropdown.addItem(device.display_name, device.ip)

        self.status_label.setText(
            f"Found {len(self.discovered_devices)} device(s), Linux verified: {linux_count}."
        )

    def validate_and_accept(self):
        manual_ip = self.manual_ip_input.text().strip()
        if manual_ip:
            try:
                ipaddress.ip_address(manual_ip)
            except ValueError:
                QMessageBox.warning(self, "Invalid IP", "Manual IP address format is invalid.")
                return
        elif not self.discovered_devices:
            QMessageBox.warning(
                self,
                "No Device Selected",
                "No devices discovered. Scan again or provide a manual IP.",
            )
            return

        current_index = self.device_dropdown.currentIndex()
        if current_index < 0 and not manual_ip:
            QMessageBox.warning(
                self,
                "No Device Selected",
                "Select an edge device from the dropdown or enter a manual IP.",
            )
            return

        if not self.username_input.text().strip():
            QMessageBox.warning(self, "Missing Username", "SSH username is required.")
            return
        if not self.password_input.text().strip():
            QMessageBox.warning(self, "Missing Password", "SSH password is required.")
            return

        self.accept()

    def get_selection(self):
        manual_ip = self.manual_ip_input.text().strip()
        if manual_ip:
            selected_device = DiscoveredEdgeDevice(
                ip=manual_ip,
                interface="manual",
                hostname="manual-entry",
                linux_verified=False,
            )
        else:
            idx = self.device_dropdown.currentIndex()
            selected_device = self.discovered_devices[idx] if idx >= 0 else DiscoveredEdgeDevice(
                ip="",
                interface="",
            )
        return {
            "device": selected_device,
            "username": self.username_input.text().strip(),
            "password": self.password_input.text().strip(),
            "remote_dir": self.remote_dir_input.text().strip(),
            "display": self.display_input.text().strip(),
            "install_deps": self.install_deps_checkbox.isChecked(),
            "setup_postgres": True,
        }

    @staticmethod
    def _default_username():
        try:
            import getpass

            return getpass.getuser()
        except Exception:
            return ""
