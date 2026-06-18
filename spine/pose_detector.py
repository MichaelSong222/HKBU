"""
pose_detector.py — MediaPipe Pose: shoulder/hip detection + verticality check.

Landmark choice: Person faces RIGHT (back on LEFT). We use the LEFT landmarks
(MediaPipe IDs 11/23) which correspond to the FAR side from camera, typically
more occulded, but for a true sagittal view both sides should be similar.
In practice we average left & right landmarks when both are detected with
reasonable visibility, giving a midline estimate that is robust to slight
camera offset. Visibility threshold is 0.5.
"""

import math
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
        shoulder   : (x, y) pixel coords
        hip        : (x, y) pixel coords
        vertical_deviation_deg : angle of Shoulder→Hip from true vertical
        is_vertical : bool — True if deviation <= VERTICAL_DEVIATION_THRESHOLD_DEG
        raw_landmarks : the full MediaPipe landmark list (for debugging)

    Raises PoseDetectionError if landmarks cannot be found.
    """
    h, w = image_bgr.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        import cv2
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

    return {
        "shoulder": shoulder,
        "hip": hip,
        "vertical_deviation_deg": vertical_deviation,
        "is_vertical": is_vertical,
        "raw_landmarks": lms,
    }
