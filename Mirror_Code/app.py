# app.py

import sys
import os
import faulthandler
import warnings
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt
from core.logging_config import configure_logging
from core.paths import project_path, resource_path
from ui.main_window import MainWindow

_fault_log_file = None


def _configure_runtime_diagnostics():
    global _fault_log_file
    try:
        fault_log_path = project_path("edge_fault.log")
        _fault_log_file = open(fault_log_path, "a", buffering=1)
        faulthandler.enable(file=_fault_log_file, all_threads=True)
    except Exception:
        _fault_log_file = None

    # Keep edge runtime logs clean; this warning is non-fatal and very noisy.
    warnings.filterwarnings(
        "ignore",
        message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
        category=UserWarning,
    )


def main():
    try:
        configure_logging()
        _configure_runtime_diagnostics()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        app = QApplication(sys.argv)
        app.setApplicationName("Smart Gym Client System")
        app.setOrganizationName("SmartGym")
        app.setOrganizationDomain("smartgym.com")

        # Set application icon
        icon_path = resource_path("icons", "app_icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))

        # Optional: Add Splash Screen
        splash_path = resource_path("icons", "splash.png")
        splash_pix = QPixmap(splash_path) if os.path.exists(splash_path) else QPixmap()
        if not splash_pix.isNull():
            splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
            splash.setMask(splash_pix.mask())
            splash.show()
            app.processEvents()

        stylesheet_path = resource_path("styles.qss")
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, "r") as f:
                app.setStyleSheet(f.read())
        else:
            # Enhanced basic styling if no QSS found
            app.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #C5C6C7;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background: #2E2E2E;
            }
            QTabBar::tab {
                background: #3C3F41;
                color: #C5C6C7;
                padding: 10px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #007ACC;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #007ACC;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
            QGroupBox {
                border: 1px solid #007ACC;
                border-radius: 5px;
                margin-top: 1em;
                font-weight: bold;
                background-color: #2E2E2E;
            }
            QLabel {
                color: #C5C6C7;
                font-size: 10pt;
            }
            QTableWidget {
                background-color: #2E2E2E;
                alternate-background-color: #3C3F41;
                font-size: 10pt;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #007ACC;
                color: #FFFFFF;
                font-size: 10pt;
                padding: 5px;
                border: none;
            }
            QLineEdit, QComboBox, QDateEdit {
                background-color: #3C3F41;
                color: #C5C6C7;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
            """)

        window = MainWindow()
        window.show()

        if 'splash' in locals():
            splash.finish(window)

        sys.exit(app.exec())
    
    except Exception as e:
        print(f"An error occurred: {e}")
        app = QApplication.instance()
        if app:
            QMessageBox.critical(None, "Error", f"An unexpected error occurred:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
