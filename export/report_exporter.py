"""
report_exporter.py — Export analysis results to CSV and save annotated images.
"""

import csv
import datetime
import os
import cv2
import numpy as np


def export_csv(spine_result, front_result, output_path: str) -> None:
    """
    Export spine + front analysis results to a single-row CSV file.

    Parameters
    ----------
    spine_result : AnalysisResult  (from spine/pipeline.py)
    front_result : FrontAnalysisResult  (from posture/front_pipeline.py)
    output_path  : full path including filename, e.g. "captures/report.csv"
    """
    fields = {
        "timestamp": datetime.datetime.now().isoformat(),
        # ── Side-view / spine ──────────────────────────────────────────
        "facing_direction":        getattr(spine_result, "facing_direction", ""),
        "kyphosis_angle_deg":      getattr(spine_result, "thoracic_angle_deg",    None),
        "kyphosis_class":          getattr(spine_result, "thoracic_class_label",  ""),
        "lordosis_angle_deg":      getattr(spine_result, "lumbar_angle_deg",      None),
        "lordosis_class":          getattr(spine_result, "lumbar_class_label",    ""),
        "forward_head_angle_deg":  getattr(spine_result, "forward_head_angle_deg",  None),
        "forward_displacement_px": getattr(spine_result, "forward_displacement_px", None),
        "neck_inclination_deg":    getattr(spine_result, "neck_inclination_deg",    None),
        # ── Front-view ────────────────────────────────────────────────
        "head_tilt_deg":            getattr(front_result, "head_tilt_deg",            None),
        "head_tilt_direction":      getattr(front_result, "head_tilt_direction",      ""),
        "shoulder_diff_px":         getattr(front_result, "shoulder_diff_px",         None),
        "shoulder_diff_normalized": getattr(front_result, "shoulder_diff_normalized", None),
        "shoulder_direction":       getattr(front_result, "shoulder_direction",       ""),
        "pelvic_diff_px":           getattr(front_result, "pelvic_diff_px",           None),
        "pelvic_diff_normalized":   getattr(front_result, "pelvic_diff_normalized",   None),
        "pelvic_direction":         getattr(front_result, "pelvic_direction",         ""),
        "knee_diff_px":             getattr(front_result, "knee_diff_px",             None),
        "knee_diff_normalized":     getattr(front_result, "knee_diff_normalized",     None),
        "knee_direction":           getattr(front_result, "knee_direction",           ""),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields.keys()))
        writer.writeheader()
        writer.writerow(fields)


def export_images(
    spine_img: np.ndarray,
    front_img: np.ndarray,
    output_dir: str,
) -> tuple[str, str]:
    """
    Save the two annotated images to *output_dir*.

    Returns
    -------
    (spine_path, front_path) — the absolute paths of saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    spine_path = os.path.join(output_dir, "spine_annotated.png")
    front_path = os.path.join(output_dir, "front_annotated.png")
    cv2.imwrite(spine_path, spine_img)
    cv2.imwrite(front_path, front_img)
    return spine_path, front_path
