"""
camera_widget.py — Live webcam preview + 5-second countdown capture.

Emits the captured frame via the `frame_captured` signal so app_window.py
can run the analysis pipeline on it.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

import config


class _CameraThread(QThread):
    """Background thread that reads frames from the webcam."""
    frame_ready = Signal(np.ndarray)

    def __init__(self, camera_index: int = config.CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self._index = camera_index
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self._index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
        self._running = True
        while self._running:
            ok, frame = cap.read()
            if ok:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                self.frame_ready.emit(frame)
            self.msleep(33)   # ~30 fps
        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class CameraWidget(QWidget):
    """
    Widget that shows a live camera preview and allows the user to trigger a
    5-second countdown capture.  Emits `frame_captured` with the BGR numpy
    array of the captured frame.
    """
    frame_captured = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._latest_frame: np.ndarray | None = None
        self._countdown_value = 0
        self._countdown_active = False

        self._build_ui()
        self._thread = _CameraThread()
        self._thread.frame_ready.connect(self._on_frame)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_countdown)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("Camera not started")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 400)
        self.preview_label.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self.preview_label)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.clicked.connect(self.start_camera)
        btn_row.addWidget(self.btn_start)

        self.btn_capture = QPushButton(f"Take Photo ({config.COUNTDOWN_SECONDS}s countdown)")
        self.btn_capture.clicked.connect(self.start_countdown)
        self.btn_capture.setEnabled(False)
        btn_row.addWidget(self.btn_capture)

        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        layout.addLayout(btn_row)

    # ── Public API ────────────────────────────────────────────────────────

    def start_camera(self):
        if not self._thread.isRunning():
            self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)

    def stop_camera(self):
        self._timer.stop()
        self._countdown_active = False
        self._thread.stop()
        self.preview_label.setText("Camera stopped")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)

    def start_countdown(self):
        if self._countdown_active:
            return
        self._countdown_value = config.COUNTDOWN_SECONDS
        self._countdown_active = True
        self.btn_capture.setEnabled(False)
        self._timer.start()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        self._latest_frame = frame
        display = frame.copy()

        if self._countdown_active and self._countdown_value > 0:
            h, w = display.shape[:2]
            text = str(self._countdown_value)
            font_scale = min(w, h) / 120.0
            thickness = max(2, int(font_scale * 3))
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cx = (w - tw) // 2
            cy = (h + th) // 2
            # Shadow
            cv2.putText(display, text, (cx + 3, cy + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(display, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 255),
                        thickness, cv2.LINE_AA)

        self._display_frame(display)

    def _tick_countdown(self):
        self._countdown_value -= 1
        if self._countdown_value <= 0:
            self._timer.stop()
            self._countdown_active = False
            self.btn_capture.setEnabled(True)
            if self._latest_frame is not None:
                self.frame_captured.emit(self._latest_frame.copy())

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(
            pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
