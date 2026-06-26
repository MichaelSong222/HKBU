"""
app_window.py — Main application window (four-step mandatory workflow).

Step 1a — Contour photo (arms forward):
           Upload/capture side view with arms raised → kyphosis / lordosis analysis.
           Completed → auto-advance to Step 1b.

Step 1b — FHP photo (arms down, natural stand):
           Upload/capture side view with arms relaxed → Forward Head Position analysis.
           Completed → "Next: Front View" unlocked.

Step 2  — Front view:
           Upload/capture front view → head tilt, shoulder/pelvic/knee level.
           "View Full Report" button advances to Step 3.

Step 3  — Summary report: both annotated images + all metrics.
           "Export" saves PNG images + CSV report.
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
    QScrollArea,
)

from spine.pipeline import run_analysis, AnalysisResult
from spine.visualization import draw_overlay, draw_fhp_on_image
from posture.front_pipeline import analyze_front, FrontAnalysisResult
from gui.camera_widget import CameraWidget

_CAPTURE_DIR = Path(__file__).parent.parent / "captures"
_CAPTURE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class _ContourWorker(QThread):
    """Runs spine pipeline in contour mode (arms-forward photo)."""
    done = Signal(object, np.ndarray)

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self._image = image_bgr

    def run(self):
        result = run_analysis(self._image, mode="contour")
        annotated = draw_overlay(self._image, result) if result.success else self._image.copy()
        self.done.emit(result, annotated)


class _FhpWorker(QThread):
    """Runs spine pipeline in FHP mode (natural-standing photo)."""
    done = Signal(object, np.ndarray)

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self._image = image_bgr

    def run(self):
        result = run_analysis(self._image, mode="fhp")
        if result.success and result.forward_head_angle_deg is not None:
            # Build fhd_result dict from AnalysisResult fields
            fhd_result = None
            if result.fhd_inches is not None:
                fhd_result = {
                    "fhd_pixels":     result.fhd_pixels,
                    "fhd_cm":         result.fhd_cm,
                    "fhd_inches":     result.fhd_inches,
                    "spine_load_lbs": result.spine_load_lbs,
                    "ear_point":      result.fhd_ear_point,
                    "c7_point":       result.fhd_c7_point,
                    "severity":       result.fhd_severity,
                }
            annotated = draw_fhp_on_image(
                self._image,
                result._raw_landmarks,
                result.facing_direction,
                result.forward_head_angle_deg,
                fhd_result=fhd_result,
            )
        else:
            annotated = self._image.copy()
        self.done.emit(result, annotated)


class _FrontWorker(QThread):
    done = Signal(object, np.ndarray)

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self._image = image_bgr

    def run(self):
        result = analyze_front(self._image)
        annotated = result.annotated_image if result.annotated_image is not None else self._image.copy()
        self.done.emit(result, annotated)


# ---------------------------------------------------------------------------
# Helper: image → QPixmap
# ---------------------------------------------------------------------------

def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, w * ch, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Shared image panel (upload + live camera)
# ---------------------------------------------------------------------------

class _ImagePanel(QWidget):
    """Left-side panel used on Step 1 and Step 2."""

    image_ready = Signal(np.ndarray)   # emitted when a frame is confirmed

    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self._latest_image: np.ndarray | None = None
        self._label_text = label_text
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        self.btn_upload = QPushButton("Upload Photo")
        self.btn_upload.setFixedHeight(34)
        self.btn_upload.clicked.connect(self._on_upload)
        toolbar.addWidget(self.btn_upload)

        self.btn_camera = QPushButton("Live Camera")
        self.btn_camera.setFixedHeight(34)
        self.btn_camera.setCheckable(True)
        self.btn_camera.clicked.connect(self._toggle_camera)
        toolbar.addWidget(self.btn_camera)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)

        self.img_label = QLabel(self._label_text)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(580, 440)
        self.img_label.setStyleSheet("background:#1a1a1a; color:#666; font-size:13px;")
        self._stack.addWidget(self.img_label)

        self._cam = CameraWidget()
        self._cam.frame_captured.connect(self._on_frame_captured)
        self._stack.addWidget(self._cam)

    # -- public

    def show_image(self, image_bgr: np.ndarray):
        pix = _bgr_to_pixmap(image_bgr)
        self.img_label.setPixmap(
            pix.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._stack.setCurrentIndex(0)

    def stop_camera(self):
        self._cam.stop_camera()
        self._stack.setCurrentIndex(0)
        self.btn_camera.setChecked(False)

    # -- slots

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Photo", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Load Error", f"Could not read:\n{path}")
            return
        self._latest_image = img
        self.show_image(img)
        self.stop_camera()
        self.image_ready.emit(img)

    def _toggle_camera(self, checked: bool):
        if checked:
            self._stack.setCurrentIndex(1)
            self._cam.start_camera()
        else:
            self.stop_camera()

    def _on_frame_captured(self, frame: np.ndarray):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(_CAPTURE_DIR / f"{ts}_original.jpg"), frame)
        self._latest_image = frame
        self.show_image(frame)
        self.stop_camera()
        self.image_ready.emit(frame)


# ---------------------------------------------------------------------------
# Step pages
# ---------------------------------------------------------------------------

class _StepPage(QWidget):
    """Base for Step 1 and Step 2 pages."""

    advance = Signal()   # emitted when user clicks the "Next" button

    def __init__(self, title: str, img_hint: str, btn_label: str, parent=None):
        super().__init__(parent)
        self._worker = None
        self._annotated: np.ndarray | None = None
        self._result = None
        self._build(title, img_hint, btn_label)

    def _build(self, title: str, img_hint: str, btn_label: str):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        hdr = QLabel(title)
        hdr.setFont(QFont("", 15, QFont.Bold))
        root.addWidget(hdr)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        self._img_panel = _ImagePanel(img_hint)
        self._img_panel.image_ready.connect(self._on_image_ready)
        splitter.addWidget(self._img_panel)

        right = QFrame()
        right.setMinimumWidth(300)
        right.setMaximumWidth(400)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)

        res_title = QLabel("Results")
        res_title.setFont(QFont("", 13, QFont.Bold))
        rl.addWidget(res_title)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#888; font-size:12px;")
        rl.addWidget(self.status_lbl)

        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setStyleSheet(
            "background:#111; color:#ddd; font-size:12px; border-radius:4px; padding:4px;"
        )
        self.results_box.setPlaceholderText("No results yet.")
        rl.addWidget(self.results_box, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([660, 340])

        self.btn_next = QPushButton(btn_label)
        self.btn_next.setFixedHeight(38)
        self.btn_next.setEnabled(False)
        self.btn_next.setStyleSheet(
            "QPushButton{background:#1a6fbc; color:white; border-radius:5px; font-size:14px;}"
            "QPushButton:disabled{background:#333; color:#666;}"
        )
        self.btn_next.clicked.connect(self.advance)
        root.addWidget(self.btn_next, alignment=Qt.AlignRight)

    def _on_image_ready(self, image: np.ndarray):
        self.status_lbl.setText("Analyzing…")
        self.results_box.clear()
        self.btn_next.setEnabled(False)
        self._img_panel.btn_upload.setEnabled(False)
        self._img_panel.btn_camera.setEnabled(False)
        self._run_worker(image)

    def _run_worker(self, image: np.ndarray):
        raise NotImplementedError

    def _finish(self, result, annotated: np.ndarray):
        self._result    = result
        self._annotated = annotated
        self._img_panel.show_image(annotated)
        self._img_panel.btn_upload.setEnabled(True)
        self._img_panel.btn_camera.setEnabled(True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(_CAPTURE_DIR / f"{ts}_annotated.jpg"), annotated)
        if result.success:
            self.status_lbl.setText("Done ✓")
            self.results_box.setHtml(self._format_result(result))
            self.btn_next.setEnabled(True)
        elif result.error and result.error.startswith("WRONG_VIEW:"):
            # Format: "WRONG_VIEW:<expected>:<human message>"
            parts = result.error.split(":", 2)
            expected = parts[1] if len(parts) > 1 else "?"
            msg      = parts[2] if len(parts) > 2 else result.error
            icon = "↕" if expected == "side" else "↔"
            self.status_lbl.setText("⚠ Wrong view")
            self.results_box.setHtml(
                f"<p style='color:#ff9900;font-size:15px;font-weight:bold;'>"
                f"{icon} Wrong Pose Detected</p>"
                f"<p style='color:#ffbb44;font-size:13px;'>{msg}</p>"
                f"<hr/>"
                f"<p style='color:#aaa;font-size:11px;'>"
                f"Please retake the photo and try again.</p>"
            )
        else:
            self.status_lbl.setText("Failed ✗")
            self.results_box.setHtml(
                f'<span style="color:#f55;">Error: {result.error}</span>'
            )

    def _format_result(self, result) -> str:
        raise NotImplementedError

    def get_result(self):
        return self._result

    def get_annotated(self) -> np.ndarray | None:
        return self._annotated


class SpineContourPage(_StepPage):
    """
    Step 1a — Contour photo.
    User raises arms forward; pipeline runs in mode='contour' to extract
    the back curve without arm-occlusion and compute Cobb angles.
    On success the page auto-signals advance so MainWindow can show Step 1b.
    """
    def __init__(self, parent=None):
        super().__init__(
            "Step 1a — 轮廓照  (Arms Forward)",
            "请将双臂向前平举，侧身站立，保持背部挺直。\n"
            "Extend both arms forward, stand sideways, keep back straight.",
            "下一步：自然站立照  →",
            parent,
        )

    def _run_worker(self, image: np.ndarray):
        self._worker = _ContourWorker(image)
        self._worker.done.connect(self._finish)
        self._worker.start()

    def _format_result(self, r: AnalysisResult) -> str:
        conf = " ⚠ LOW CONFIDENCE" if r.low_confidence else ""
        lu_c = " ⚠ LOW CONFIDENCE" if r.lumbar_low_confidence else ""
        dir_icon = "←" if r.facing_direction == "left" else "→"
        lines = [
            f"<p style='color:#aaa;font-size:11px;'>Facing: {dir_icon} {r.facing_direction}</p>",
            f"<h3 style='color:#7af;'>Thoracic Kyphosis{conf}</h3>",
            f"<p><b style='font-size:18px;color:#7af;'>{r.thoracic_angle_deg:.1f}°</b></p>",
            f"<p>{r.thoracic_class_label}</p>",
            "<hr/>",
            f"<h3 style='color:#f90;'>Lumbar Lordosis{lu_c}</h3>",
            f"<p><b style='font-size:18px;color:#f90;'>{r.lumbar_angle_deg:.1f}°</b></p>",
            f"<p>{r.lumbar_class_label}</p>",
        ]
        if r.warnings:
            lines.append("<hr/><p style='color:#fa0;'><b>Warnings:</b></p>")
            for w in r.warnings:
                lines.append(f"<p style='color:#fa0;font-size:11px;'>• {w}</p>")
        lines.append(
            "<hr/><p style='color:#555;font-size:10px;'>Estimation only. Not a medical diagnosis.</p>"
        )
        return "".join(lines)


class SpineFhpPage(_StepPage):
    """
    Step 1b — FHP photo.
    User stands naturally with arms down; pipeline runs in mode='fhp'
    to compute Forward Head Position from the ear-shoulder geometry.
    On success the 'Next: Front View' button is unlocked.
    """
    def __init__(self, parent=None):
        super().__init__(
            "Step 1b — 自然站立照  (Natural Stand)",
            "请放下手臂，自然站立，保持头部正常位置。\n"
            "Lower your arms, stand naturally, keep your head in a neutral position.",
            "Next: Front View  →",
            parent,
        )

    def _run_worker(self, image: np.ndarray):
        self._worker = _FhpWorker(image)
        self._worker.done.connect(self._finish)
        self._worker.start()

    def _format_result(self, r: AnalysisResult) -> str:
        lines = []
        if r.forward_head_angle_deg is not None:
            lines += [
                "<h3 style='color:#0be;'>Forward Head Position (FHP)</h3>",
                f"<p>Craniovertebral angle: <b style='font-size:18px;color:#0be;'>"
                f"{r.forward_head_angle_deg:.1f}°</b></p>",
                f"<p>Neck inclination: {r.neck_inclination_deg:.1f}°</p>",
            ]
        else:
            lines.append("<p style='color:#fa0;'>FHP not detected.</p>")

        # ── FHD block ──────────────────────────────────────────────────────
        if r.fhd_inches is not None:
            sev = r.fhd_severity or "Unknown"
            sev_colors = {
                "Normal":   "#55cc55",
                "Mild":     "#ffdd00",
                "Moderate": "#ff8800",
                "Severe":   "#ff3333",
            }
            sev_color = sev_colors.get(sev, "#aaa")
            lines += [
                "<hr/>",
                "<h3 style='color:#adf;'>Forward Head Distance (FHD)</h3>",
                f"<p>Distance: <b style='font-size:16px;color:#adf;'>"
                f"{r.fhd_inches:.2f} in</b>  ({r.fhd_cm:.1f} cm)</p>",
                f"<p>Severity: <b style='color:{sev_color};font-size:14px;'>{sev}</b></p>",
                f"<p>Est. spine load: <b style='color:#f90;'>{r.spine_load_lbs:.0f} lbs</b>"
                f"  <span style='color:#666;font-size:10px;'>"
                f"(Kapandji: 12 + {r.fhd_inches:.1f}×10)</span></p>",
            ]

        if r.warnings:
            lines.append("<hr/><p style='color:#fa0;'><b>Warnings:</b></p>")
            for w in r.warnings:
                lines.append(f"<p style='color:#fa0;font-size:11px;'>• {w}</p>")
        lines.append(
            "<hr/><p style='color:#555;font-size:10px;'>Estimation only. Not a medical diagnosis.</p>"
        )
        return "".join(lines)


class FrontStepPage(_StepPage):
    def __init__(self, parent=None):
        super().__init__(
            "Step 2 — Front View",
            "Upload or capture a FRONT-VIEW photo (facing camera).",
            "View Full Report  →",
            parent,
        )

    def _run_worker(self, image: np.ndarray):
        self._worker = _FrontWorker(image)
        self._worker.done.connect(self._finish)
        self._worker.start()

    def _format_result(self, r: FrontAnalysisResult) -> str:
        def _val(v, unit="°"):
            return f"{v:.1f}{unit}" if v is not None else "N/A"

        def _dir(d):
            labels = {
                "left_high": "Left high", "right_high": "Right high", "level": "Level",
                "left_low": "Left low", "right_low": "Right low",
                "unknown": "Unknown",
            }
            return labels.get(d, d)

        lines = [
            "<h3 style='color:#9f9;'>Head Tilt</h3>",
            f"<p>Tilt: <b style='color:#9f9;'>{_val(r.head_tilt_deg)}</b>  {_dir(r.head_tilt_direction)}</p>",
            "<hr/>",
            "<h3 style='color:#fc9;'>Shoulder Level</h3>",
            f"<p>Diff: <b style='color:#fc9;'>{_val(r.shoulder_diff_px, 'px')}</b>  {_dir(r.shoulder_direction)}</p>",
            "<hr/>",
            "<h3 style='color:#f9c;'>Pelvic Level</h3>",
            f"<p>Diff: <b style='color:#f9c;'>{_val(r.pelvic_diff_px, 'px')}</b>  {_dir(r.pelvic_direction)}</p>",
            "<hr/>",
            "<h3 style='color:#cf9;'>Knee Level</h3>",
            f"<p>Diff: <b style='color:#cf9;'>{_val(r.knee_diff_px, 'px')}</b>  {_dir(r.knee_direction)}</p>",
        ]
        if r.warnings:
            lines.append("<hr/><p style='color:#fa0;'><b>Warnings:</b></p>")
            for w in r.warnings:
                lines.append(f"<p style='color:#fa0;font-size:11px;'>• {w}</p>")
        return "".join(lines)


# ---------------------------------------------------------------------------
# Step 3 — Summary page
# ---------------------------------------------------------------------------

class SummaryPage(QWidget):
    restart = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spine_result = None
        self._front_result = None
        self._spine_annotated: np.ndarray | None = None
        self._front_annotated: np.ndarray | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        hdr = QLabel("Full Report")
        hdr.setFont(QFont("", 16, QFont.Bold))
        root.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, stretch=1)

        container = QWidget()
        scroll.setWidget(container)
        self._content_layout = QVBoxLayout(container)

        btn_row = QHBoxLayout()

        self.btn_export = QPushButton("Export (PNG + CSV)")
        self.btn_export.setFixedHeight(36)
        self.btn_export.setStyleSheet(
            "QPushButton{background:#2a8a3c; color:white; border-radius:5px; font-size:13px;}"
        )
        self.btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self.btn_export)

        self.btn_restart = QPushButton("New Analysis")
        self.btn_restart.setFixedHeight(36)
        self.btn_restart.setStyleSheet(
            "QPushButton{background:#555; color:white; border-radius:5px; font-size:13px;}"
        )
        self.btn_restart.clicked.connect(self.restart)
        btn_row.addWidget(self.btn_restart)
        btn_row.addStretch()

        root.addLayout(btn_row)

    def populate(
        self,
        spine_result: AnalysisResult,
        spine_annotated: np.ndarray,
        front_result: FrontAnalysisResult,
        front_annotated: np.ndarray,
        fhp_annotated: np.ndarray | None = None,
    ):
        self._spine_result    = spine_result
        self._front_result    = front_result
        self._spine_annotated = spine_annotated
        self._front_annotated = front_annotated
        self._fhp_annotated   = fhp_annotated

        # Clear previous content
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        splitter = QSplitter(Qt.Horizontal)
        self._content_layout.addWidget(splitter)

        splitter.addWidget(self._make_side_panel(spine_result, spine_annotated, fhp_annotated))
        splitter.addWidget(self._make_front_panel(front_result, front_annotated))
        splitter.setSizes([500, 500])

    # -- panel builders

    def _make_side_panel(
        self,
        r: AnalysisResult,
        img: np.ndarray,
        fhp_img: np.ndarray | None = None,
    ) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)

        t = QLabel("Side View — Spine Analysis")
        t.setFont(QFont("", 13, QFont.Bold))
        t.setStyleSheet("color:#7af;")
        vl.addWidget(t)

        # ── Contour photo (arms-forward) ──────────────────────────────────
        lbl_contour = QLabel("Contour photo (arms forward):")
        lbl_contour.setStyleSheet("color:#888; font-size:11px;")
        vl.addWidget(lbl_contour)

        il_contour = QLabel()
        pix = _bgr_to_pixmap(img)
        il_contour.setPixmap(pix.scaledToWidth(480, Qt.SmoothTransformation))
        il_contour.setAlignment(Qt.AlignCenter)
        vl.addWidget(il_contour)

        # ── FHP photo (natural stand) — shown only when available ─────────
        if fhp_img is not None:
            lbl_fhp = QLabel("FHP photo (natural stand):")
            lbl_fhp.setStyleSheet("color:#888; font-size:11px; margin-top:6px;")
            vl.addWidget(lbl_fhp)

            il_fhp = QLabel()
            pix2 = _bgr_to_pixmap(fhp_img)
            il_fhp.setPixmap(pix2.scaledToWidth(480, Qt.SmoothTransformation))
            il_fhp.setAlignment(Qt.AlignCenter)
            vl.addWidget(il_fhp)

        # ── Text summary ──────────────────────────────────────────────────
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFixedHeight(260)
        txt.setStyleSheet("background:#111; color:#ddd; font-size:12px;")
        dir_icon = "← left" if r.facing_direction == "left" else "→ right"

        fhp_str = ""
        if r.forward_head_angle_deg is not None:
            fhp_str = (
                f"\n\nForward Head Position (FHP)\n"
                f"  CV angle:     {r.forward_head_angle_deg:.1f}°\n"
                f"  Neck tilt:    {r.neck_inclination_deg:.1f}°"
            )

        fhd_str = ""
        if getattr(r, "fhd_inches", None) is not None:
            fhd_str = (
                f"\n\nForward Head Distance (FHD)\n"
                f"  Distance:     {r.fhd_inches:.2f} in  ({r.fhd_cm:.1f} cm)\n"
                f"  Severity:     {r.fhd_severity}\n"
                f"  Spine load:   {r.spine_load_lbs:.0f} lbs  (Kapandji model)"
            )

        body = (
            f"Facing:  {dir_icon}\n\n"
            f"Thoracic Kyphosis\n"
            f"  Angle:  {r.thoracic_angle_deg:.1f}°\n"
            f"  Class:  {r.thoracic_class_label}\n\n"
            f"Lumbar Lordosis\n"
            f"  Angle:  {r.lumbar_angle_deg:.1f}°\n"
            f"  Class:  {r.lumbar_class_label}"
            f"{fhp_str}"
            f"{fhd_str}"
        )
        txt.setPlainText(body)
        vl.addWidget(txt)
        return w

    def _make_front_panel(self, r: FrontAnalysisResult, img: np.ndarray) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        t = QLabel("Front View — Postural Symmetry")
        t.setFont(QFont("", 13, QFont.Bold))
        t.setStyleSheet("color:#9f9;")
        vl.addWidget(t)

        il = QLabel()
        pix = _bgr_to_pixmap(img)
        il.setPixmap(pix.scaledToWidth(480, Qt.SmoothTransformation))
        il.setAlignment(Qt.AlignCenter)
        vl.addWidget(il)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFixedHeight(220)
        txt.setStyleSheet("background:#111; color:#ddd; font-size:12px;")

        def _v(val, unit="°"):
            return f"{val:.1f}{unit}" if val is not None else "N/A"

        def _d(d):
            return {"left_high": "Left high", "right_high": "Right high",
                    "level": "Level", "left_low": "Left low",
                    "right_low": "Right low", "unknown": "—"}.get(d, d)

        body = (
            f"Head Tilt\n"
            f"  Angle:  {_v(r.head_tilt_deg)}  ({_d(r.head_tilt_direction)})\n\n"
            f"Shoulder Level\n"
            f"  Diff:   {_v(r.shoulder_diff_px, 'px')}  ({_d(r.shoulder_direction)})\n\n"
            f"Pelvic Level\n"
            f"  Diff:   {_v(r.pelvic_diff_px, 'px')}  ({_d(r.pelvic_direction)})\n\n"
            f"Knee Level\n"
            f"  Diff:   {_v(r.knee_diff_px, 'px')}  ({_d(r.knee_direction)})"
        )
        txt.setPlainText(body)
        vl.addWidget(txt)
        return w

    # -- export

    def _on_export(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder",
                                                  str(_CAPTURE_DIR))
        if not folder:
            return
        try:
            from export.report_exporter import export_csv, export_images
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_images(self._spine_annotated, self._front_annotated, folder)
            export_csv(self._spine_result, self._front_result,
                       str(Path(folder) / f"report_{ts}.csv"))
            QMessageBox.information(self, "Export Complete",
                                    f"Files saved to:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spine & Posture Analyzer")
        self.setMinimumSize(1050, 700)
        self._build_ui()

    def _build_ui(self):
        self._pages = QStackedWidget()
        self.setCentralWidget(self._pages)

        self._contour_page = SpineContourPage()   # Step 1a
        self._fhp_page     = SpineFhpPage()        # Step 1b
        self._front_page   = FrontStepPage()       # Step 2
        self._summary_page = SummaryPage()         # Step 3

        self._pages.addWidget(self._contour_page)   # index 0
        self._pages.addWidget(self._fhp_page)       # index 1
        self._pages.addWidget(self._front_page)     # index 2
        self._pages.addWidget(self._summary_page)   # index 3

        self._contour_page.advance.connect(self._go_to_fhp)
        self._fhp_page.advance.connect(self._go_to_front)
        self._front_page.advance.connect(self._go_to_summary)
        self._summary_page.restart.connect(self._restart)

        self._pages.setCurrentIndex(0)

    def _go_to_fhp(self):
        self._pages.setCurrentIndex(1)

    def _go_to_front(self):
        self._pages.setCurrentIndex(2)

    def _go_to_summary(self):
        contour_result  = self._contour_page.get_result()
        fhp_result      = self._fhp_page.get_result()
        front_result    = self._front_page.get_result()

        if contour_result is None or fhp_result is None or front_result is None:
            QMessageBox.warning(self, "Incomplete",
                                "Please complete all three side/front analyses first.")
            return

        # Merge contour + fhp into a single AnalysisResult for display
        merged = contour_result
        merged.forward_head_angle_deg  = fhp_result.forward_head_angle_deg
        merged.forward_displacement_px = fhp_result.forward_displacement_px
        merged.neck_inclination_deg    = fhp_result.neck_inclination_deg

        # Use contour annotated image for spine; FHP annotated for head position
        spine_annotated = self._contour_page.get_annotated()
        fhp_annotated   = self._fhp_page.get_annotated()
        front_annotated = self._front_page.get_annotated()

        self._summary_page.populate(
            merged,       spine_annotated,
            front_result, front_annotated,
            fhp_annotated=fhp_annotated,
        )
        self._pages.setCurrentIndex(3)

    def _restart(self):
        for page in (self._contour_page, self._fhp_page,
                     self._front_page, self._summary_page):
            page.deleteLater()

        self._contour_page = SpineContourPage()
        self._fhp_page     = SpineFhpPage()
        self._front_page   = FrontStepPage()
        self._summary_page = SummaryPage()

        self._pages.addWidget(self._contour_page)
        self._pages.addWidget(self._fhp_page)
        self._pages.addWidget(self._front_page)
        self._pages.addWidget(self._summary_page)

        self._contour_page.advance.connect(self._go_to_fhp)
        self._fhp_page.advance.connect(self._go_to_front)
        self._front_page.advance.connect(self._go_to_summary)
        self._summary_page.restart.connect(self._restart)

        self._pages.setCurrentWidget(self._contour_page)

    def closeEvent(self, event):
        for page in (self._contour_page, self._fhp_page, self._front_page):
            if hasattr(page, "_img_panel"):
                page._img_panel.stop_camera()
        super().closeEvent(event)
