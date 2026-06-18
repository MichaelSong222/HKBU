"""
main.py — Entry point for the Spine Posture Analyzer GUI.
Run: python main.py
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from gui.app_window import MainWindow


def main():
    # High-DPI support (Qt6 handles this automatically, but explicit is safer)
    app = QApplication(sys.argv)
    app.setApplicationName("Spine Posture Analyzer")
    app.setOrganizationName("HKBU Dresio")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
