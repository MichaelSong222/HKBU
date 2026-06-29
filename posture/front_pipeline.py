"""
front_pipeline.py — Front-view analysis pipeline.

Mirrors the structure of spine/pipeline.py but for frontal posture:
  1. Detect pose landmarks (reuses spine/pose_detector.detect_pose)
  2. Compute four front-view metrics
  3. Draw annotation overlay
  4. Return a FrontAnalysisResult dataclass
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from spine.pose_detector import detect_pose, PoseDetectionError
from posture.front_metrics import (
    calculate_head_tilt,
    calculate_shoulder_level,
    calculate_pelvic_level,
    calculate_knee_level,
)
from posture.front_visualization import draw_front_overlay


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class FrontAnalysisResult:
    # ── Input ─────────────────────────────────────────────────────────────
    image_bgr: Optional[np.ndarray] = None

    # ── Metrics (raw dict from each calculator) ────────────────────────────
    head_tilt:      Optional[dict] = None
    shoulder_level: Optional[dict] = None
    pelvic_level:   Optional[dict] = None
    knee_level:     Optional[dict] = None

    # ── Convenience scalar accessors ──────────────────────────────────────
    # (filled by pipeline so GUI can read them directly)
    head_tilt_deg:            Optional[float] = None
    head_tilt_direction:      str = "unknown"
    shoulder_diff_cm:         Optional[float] = None
    shoulder_diff_normalized: Optional[float] = None
    shoulder_direction:       str = "unknown"
    pelvic_diff_cm:           Optional[float] = None
    pelvic_diff_normalized:   Optional[float] = None
    pelvic_direction:         str = "unknown"
    knee_diff_cm:             Optional[float] = None
    knee_diff_normalized:     Optional[float] = None
    knee_direction:           str = "unknown"

    # ── Annotated image ────────────────────────────────────────────────────
    annotated_image: Optional[np.ndarray] = None

    # ── Raw landmarks (for visualization / export) ────────────────────────
    _raw_landmarks: object = field(default=None, repr=False)

    # ── Error ──────────────────────────────────────────────────────────────
    error: Optional[str] = None
    success: bool = True
    warnings: list[str] = field(default_factory=list)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def analyze_front(image_bgr: np.ndarray) -> FrontAnalysisResult:
    """
    Run the full front-view analysis on *image_bgr* and return a
    FrontAnalysisResult.  Never raises — errors are captured in result.error.
    """
    result = FrontAnalysisResult(image_bgr=image_bgr)

    # ── Step 1: Pose detection ─────────────────────────────────────────────
    try:
        pose = detect_pose(image_bgr)
    except PoseDetectionError as e:
        result.success = False
        result.error = str(e)
        return result
    except Exception as e:
        result.success = False
        result.error = f"Pose detection failed: {e}"
        return result

    lms       = pose["raw_landmarks"]
    world_lms = pose.get("world_landmarks")
    result._raw_landmarks = lms
    img_h, img_w = image_bgr.shape[:2]

    # ── View-type gate ────────────────────────────────────────────────────
    # Front pipeline expects a FRONT view.  If the image looks like a side
    # view, abort with a structured error so the GUI can show a retake prompt.
    is_side_view = pose.get("is_side_view", False)
    angle_deg    = pose.get("shoulder_line_angle_deg", 0.0)
    if is_side_view:
        result.success = False
        result.error   = (
            f"WRONG_VIEW:front:"
            f"This looks like a SIDE-VIEW photo (shoulder line angle {angle_deg:.0f}°). "
            "Please face the camera directly and retake."
        )
        return result

    # ── Step 2: Front metrics ──────────────────────────────────────────────
    try:
        ht = calculate_head_tilt(lms, img_w, img_h)
        result.head_tilt           = ht
        result.head_tilt_deg       = ht.get("tilt_abs_deg")
        result.head_tilt_direction = ht.get("direction", "unknown")
    except Exception as e:
        result.warnings.append(f"Head tilt failed: {e}")

    try:
        sl = calculate_shoulder_level(lms, world_lms, img_w, img_h)
        result.shoulder_level            = sl
        result.shoulder_diff_cm          = sl.get("shoulder_diff_cm")
        result.shoulder_diff_normalized  = sl.get("shoulder_diff_normalized")
        result.shoulder_direction        = sl.get("direction", "unknown")
    except Exception as e:
        result.warnings.append(f"Shoulder level failed: {e}")

    try:
        pl = calculate_pelvic_level(lms, world_lms, img_w, img_h)
        result.pelvic_level            = pl
        result.pelvic_diff_cm          = pl.get("pelvic_diff_cm")
        result.pelvic_diff_normalized  = pl.get("pelvic_diff_normalized")
        result.pelvic_direction        = pl.get("direction", "unknown")
    except Exception as e:
        result.warnings.append(f"Pelvic level failed: {e}")

    try:
        kl = calculate_knee_level(lms, world_lms, img_w, img_h)
        result.knee_level            = kl
        result.knee_diff_cm          = kl.get("knee_diff_cm")
        result.knee_diff_normalized  = kl.get("knee_diff_normalized")
        result.knee_direction        = kl.get("direction", "unknown")
    except Exception as e:
        result.warnings.append(f"Knee level failed: {e}")

    # ── Step 3: Annotate image ─────────────────────────────────────────────
    try:
        result.annotated_image = draw_front_overlay(image_bgr, result)
    except Exception as e:
        result.warnings.append(f"Visualization failed: {e}")
        result.annotated_image = image_bgr.copy()

    return result
