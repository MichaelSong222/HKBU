"""
app_window.py — Main application window.

GUI framework choice: PySide6
  - Modern, well-maintained Qt6 binding with permissive LGPL license.
  - Better macOS native look than Tkinter.
  - Tkinter fallback is documented in README if PySide6 fails to install.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  [Upload Photo]  [Live Camera ▾]                    │
  ├──────────────────────────┬──────────────────────────┤
  │  Image / Preview panel   │  Results panel           │
  │                          │  (angles + classification│
  │                          │   + warnings)            │
  └──────────────────────────┴──────────────────────────┘
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit,
    QSplitter, QStackedWidget, QFrame, QMessageBox,
)

from spine.pipeline import run_analysis, AnalysisResult
from spine.visualization import draw_overlay
from gui.camera_widget import CameraWidget

# ─── Capture output directory ────────────────────────────────────────────────
_CAPTURE_DIR = Path(__file__).parent.parent / "captures"
_CAPTURE_DIR.mkdir(exist_ok=True)


# ─── Worker thread for analysis (keeps GUI responsive) ───────────────────────

class _AnalysisWorker(QThread):
    done = Signal(object, np.ndarray)   # result, annotated_image

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self._image = image_bgr

    def run(self):
        result = run_analysis(self._image)
        if result.success:
            annotated = draw_overlay(self._image, result)
        else:
            annotated = self._image.copy()
        self.done.emit(result, annotated)


# ─── Main window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spine Posture Analyzer — Thoracic & Lumbar Estimation")
        self.setMinimumSize(1100, 700)
        self._worker: _AnalysisWorker | None = None
        self._capture_stem: str | None = None  # set when a live frame is captured
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ── Toolbar ────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        self.btn_upload = QPushButton("📂  Upload Photo")
        self.btn_upload.setFixedHeight(36)
        self.btn_upload.clicked.connect(self._on_upload)
        toolbar.addWidget(self.btn_upload)

        self.btn_camera = QPushButton("📷  Live Camera")
        self.btn_camera.setFixedHeight(36)
        self.btn_camera.setCheckable(True)
        self.btn_camera.clicked.connect(self._toggle_camera)
        toolbar.addWidget(self.btn_camera)

        toolbar.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 13px;")
        toolbar.addWidget(self.status_label)
        root_layout.addLayout(toolbar)

        # ── Splitter: left = image/camera, right = results ─────────────
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # Left: stacked (image view / camera widget)
        self._stack = QStackedWidget()
        splitter.addWidget(self._stack)

        # Page 0: static image view
        self.image_label = QLabel("Upload a photo or start the live camera to begin.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background:#1a1a1a; color:#666; font-size:14px;")
        self._stack.addWidget(self.image_label)

        # Page 1: live camera widget
        self._camera_widget = CameraWidget()
        self._camera_widget.frame_captured.connect(self._on_frame_captured)
        self._stack.addWidget(self._camera_widget)

        # Right: results panel
        results_frame = QFrame()
        results_frame.setMinimumWidth(320)
        results_frame.setMaximumWidth(420)
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(8, 0, 0, 0)

        results_title = QLabel("Analysis Results")
        results_title.setFont(QFont("", 14, QFont.Bold))
        results_layout.addWidget(results_title)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            "background:#111; color:#ddd; font-size:13px; border-radius:6px; padding:6px;"
        )
        self.results_text.setPlaceholderText("No results yet.")
        results_layout.addWidget(self.results_text)

        splitter.addWidget(results_frame)
        splitter.setSizes([720, 360])

    # ── Actions ───────────────────────────────────────────────────────────

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Side-View Photo", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Load Error", f"Could not read image:\n{path}")
            return

        # Switch to static image page
        self._stack.setCurrentIndex(0)
        self.btn_camera.setChecked(False)

        self._run_analysis(image)

    def _toggle_camera(self, checked: bool):
        if checked:
            self._stack.setCurrentIndex(1)
            self._camera_widget.start_camera()
            self.status_label.setText("Camera active — press 'Take Photo' for 5-second countdown")
        else:
            self._camera_widget.stop_camera()
            self._stack.setCurrentIndex(0)
            self.status_label.setText("")

    def _on_frame_captured(self, frame: np.ndarray):
        self.status_label.setText("Analyzing captured frame…")
        self._stack.setCurrentIndex(0)
        self.btn_camera.setChecked(False)
        self._camera_widget.stop_camera()

        # Save original frame
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._capture_stem = ts  # share timestamp with _on_analysis_done
        cv2.imwrite(str(_CAPTURE_DIR / f"{ts}_original.jpg"), frame)

        self._run_analysis(frame)

    def _run_analysis(self, image_bgr: np.ndarray):
        self.status_label.setText("Analyzing…")
        self.btn_upload.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.results_text.setPlaceholderText("Computing…")
        self.results_text.clear()

        self._worker = _AnalysisWorker(image_bgr)
        self._worker.done.connect(self._on_analysis_done)
        self._worker.start()

    # ── Result display ────────────────────────────────────────────────────

    def _on_analysis_done(self, result: AnalysisResult, annotated: np.ndarray):
        self.btn_upload.setEnabled(True)
        self.btn_camera.setEnabled(True)

        # Save annotated image if this came from a live capture
        if hasattr(self, "_capture_stem") and self._capture_stem:
            cv2.imwrite(str(_CAPTURE_DIR / f"{self._capture_stem}_annotated.jpg"), annotated)
            self._capture_stem = None

        # Show annotated image
        self._show_image(annotated)

        if not result.success:
            self.status_label.setText("Analysis failed")
            self.results_text.setHtml(
                f'<span style="color:#f55;">&#x2717; {result.error}</span>'
            )
            return

        self.status_label.setText("Done")
        self.results_text.setHtml(self._format_results(result))

    def _show_image(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, w * ch, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _format_results(self, r: AnalysisResult) -> str:
        conf_flag = " ⚠ LOW CONFIDENCE" if r.low_confidence else ""
        lu_conf   = " ⚠ LOW CONFIDENCE" if r.lumbar_low_confidence else ""

        lines = []
        lines.append(f"<h3 style='color:#7af;'>Thoracic Kyphosis{conf_flag}</h3>")
        lines.append(
            f"<p><b style='font-size:18px;color:#7af;'>{r.thoracic_angle_deg:.1f}°</b></p>"
        )
        lines.append(f"<p>{r.thoracic_class_label}</p>")
        lines.append(f"<p style='color:#888;font-size:11px;'>Body axis deviation: "
                     f"{r.vertical_deviation_deg:.1f}°</p>")

        lines.append("<hr/>")
        lines.append(f"<h3 style='color:#f90;'>Lumbar Lordosis{lu_conf}</h3>")
        lines.append(
            f"<p><b style='font-size:18px;color:#f90;'>{r.lumbar_angle_deg:.1f}°</b></p>"
        )
        lines.append(f"<p>{r.lumbar_class_label}</p>")

        if r.warnings:
            lines.append("<hr/><p style='color:#fa0;'><b>Warnings:</b></p>")
            for w in r.warnings:
                lines.append(f"<p style='color:#fa0;font-size:12px;'>• {w}</p>")

        lines.append(
            "<hr/><p style='color:#555;font-size:11px;'>"
            "Estimation only. Not a medical diagnosis."
            "</p>"
        )
        return "".join(lines)

    def closeEvent(self, event):
        self._camera_widget.stop_camera()
        if self._worker and self._worker.isRunning():
            self._worker.wait(2000)
        super().closeEvent(event)
