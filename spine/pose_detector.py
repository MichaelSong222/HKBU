"""
pose_detector.py — MediaPipe Pose: shoulder/hip detection + verticality check.

Uses mp.solutions.pose (model_complexity=2), identical to the Spine/ reference.

Added vs reference:
  - detect_pose() returns "facing_direction" ("right" or "left") determined by
    comparing left_shoulder.x vs right_shoulder.x.
"""

import math
import cv2
import numpy as np
import mediapipe as mp

import config

mp_pose = mp.solutions.pose


class PoseDetectionError(Exception):
    """Raised when shoulder or hip landmarks cannot be reliably detected."""


def _landmark_to_px(lm, w: int, h: int) -> tuple[float, float]:
    return lm.x * w, lm.y * h


def _avg_visible(lm_a, lm_b, w: int, h: int, threshold: float = 0.5):
    """Return (x, y) averaged over visible landmarks; raise if none visible."""
    pts = []
    for lm in (lm_a, lm_b):
        if lm.visibility >= threshold:
            pts.append(_landmark_to_px(lm, w, h))
    if not pts:
        raise PoseDetectionError("No visible landmark pair found.")
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    return x, y


def detect_pose(image_bgr: np.ndarray) -> dict:
    """
    Run MediaPipe Pose on *image_bgr* and return a dict with:
        shoulder             : (x, y) pixel coords
        hip                  : (x, y) pixel coords
        vertical_deviation_deg
        is_vertical          : bool
        raw_landmarks        : the full MediaPipe landmark list
        facing_direction     : "right" if person faces right (back on left edge),
                               "left"  if person faces left  (back on right edge)

    Raises PoseDetectionError if landmarks cannot be found.
    """
    h, w = image_bgr.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

    if results.pose_landmarks is None:
        raise PoseDetectionError(
            "MediaPipe Pose could not detect a person in this image."
        )

    lms = results.pose_landmarks.landmark
    shoulder = _avg_visible(
        lms[mp_pose.PoseLandmark.LEFT_SHOULDER],
        lms[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        w, h,
    )
    hip = _avg_visible(
        lms[mp_pose.PoseLandmark.LEFT_HIP],
        lms[mp_pose.PoseLandmark.RIGHT_HIP],
        w, h,
    )

    # Verticality: angle between Shoulder→Hip vector and the downward vertical
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]   # positive = downward in image coords
    if dy == 0:
        vertical_deviation = 90.0
    else:
        vertical_deviation = abs(math.degrees(math.atan2(dx, dy)))

    is_vertical = vertical_deviation <= config.VERTICAL_DEVIATION_THRESHOLD_DEG

    # ── View classification ───────────────────────────────────────────────
    # See config.py for full rationale.  Two signals combined:
    #   1. Shoulder Z-depth difference (primary, orientation-independent)
    #   2. Shoulder-span / hip-span ratio (guard against tilted front views)
    l_sh_lm  = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_sh_lm  = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip_lm = lms[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip_lm = lms[mp_pose.PoseLandmark.RIGHT_HIP]

    sh_z_diff     = abs(l_sh_lm.z - r_sh_lm.z)
    sh_span_norm  = abs(l_sh_lm.x - r_sh_lm.x)
    hip_span_norm = abs(l_hip_lm.x - r_hip_lm.x)
    sh_hip_ratio  = sh_span_norm / (hip_span_norm + 0.001)

    # Shoulder-line angle (normalised coords) — fallback for ambiguous z zone
    sh_dx_n = r_sh_lm.x - l_sh_lm.x
    sh_dy_n = r_sh_lm.y - l_sh_lm.y
    shoulder_line_angle_deg = abs(math.degrees(math.atan2(sh_dy_n, sh_dx_n)))
    if shoulder_line_angle_deg > 90:
        shoulder_line_angle_deg = 180 - shoulder_line_angle_deg

    if sh_z_diff >= config.SIDE_VIEW_Z_DIFF_MIN:
        # High z_diff likely = side view, unless sh/hip ratio is very large
        # (which signals a tilted-front pose where hips are nearly hidden)
        is_side_view = sh_hip_ratio < config.SIDE_VIEW_SH_HIP_RATIO_MAX
    elif sh_z_diff < 0.10:
        is_side_view = False
    else:
        # Ambiguous z zone: fall back to shoulder-line angle
        is_side_view = shoulder_line_angle_deg >= config.SIDE_VIEW_SHOULDER_ANGLE_MIN_DEG

    # Facing direction: compare nose x with shoulder midpoint x.
    # In a side-view image:
    #   nose to the RIGHT of shoulder midpoint → person faces right (back on left)
    #   nose to the LEFT  of shoulder midpoint → person faces left  (back on right)
    nose       = lms[mp_pose.PoseLandmark.NOSE]
    sh_mid_x   = (lms[mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                  lms[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
    facing_direction = "right" if nose.x > sh_mid_x else "left"

    return {
        "shoulder":                 shoulder,
        "hip":                      hip,
        "vertical_deviation_deg":   vertical_deviation,
        "is_vertical":              is_vertical,
        "raw_landmarks":            lms,
        "facing_direction":         facing_direction,
        "is_side_view":             is_side_view,
        "shoulder_line_angle_deg":  shoulder_line_angle_deg,
    }
