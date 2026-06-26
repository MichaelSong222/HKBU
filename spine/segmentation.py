"""
segmentation.py — Silhouette extraction and back-contour edge detection.

Uses MediaPipe Selfie Segmentation (model 1), identical to the Spine/ reference.

Added vs reference:
  - extract_back_contour() accepts facing_direction="right"|"left"
    - "right": back is on LEFT edge  → take xs[0]  (original logic)
    - "left":  back is on RIGHT edge → take xs[-1]
"""

import cv2
import numpy as np
import mediapipe as mp

import config

mp_selfie = mp.solutions.selfie_segmentation


class SegmentationError(Exception):
    """Raised when the segmentation mask is empty or too small."""


def get_segmentation_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary uint8 mask (255 = person, 0 = background) using
    MediaPipe Selfie Segmentation model 1 (landscape, more accurate).
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(rgb)

    condition = result.segmentation_mask >= config.SEGMENTATION_THRESHOLD
    mask = np.where(condition, 255, 0).astype(np.uint8)

    # Light morphological cleanup: close small holes, remove tiny islands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    if mask.sum() == 0:
        raise SegmentationError("Segmentation mask is empty — no person detected.")

    return mask


def extract_back_contour(
    mask: np.ndarray,
    shoulder_y: float,
    hip_y: float,
    facing_direction: str = "right",
) -> np.ndarray:
    """
    Extract the back boundary of the torso silhouette between shoulder_y and
    hip_y as an (N, 2) float32 array of (x, y) points ordered top→bottom.

    facing_direction="right": back is on LEFT edge  → xs[0]  (leftmost pixel per row)
    facing_direction="left":  back is on RIGHT edge → xs[-1] (rightmost pixel per row)

    If a row has multiple disjoint mask segments, xs[0]/xs[-1] picks the
    outermost pixel of the outermost segment — i.e. the true back edge.

    Returns shape (N, 2) with N == number of valid rows.
    Raises SegmentationError if fewer than 10 rows are valid.
    """
    y_top = max(0, int(round(min(shoulder_y, hip_y))))
    y_bot = min(mask.shape[0] - 1, int(round(max(shoulder_y, hip_y))))

    points = []
    for y in range(y_top, y_bot + 1):
        row = mask[y]
        xs = np.where(row > 0)[0]
        if xs.size == 0:
            continue
        if facing_direction == "left":
            back_x = int(xs[-1])   # rightmost pixel = back edge when facing left
        else:
            back_x = int(xs[0])    # leftmost pixel  = back edge when facing right
        points.append((float(back_x), float(y)))

    if len(points) < 10:
        raise SegmentationError(
            f"Back contour has only {len(points)} valid rows — "
            "segmentation may have failed."
        )

    return np.array(points, dtype=np.float32)


def smooth_contour(contour: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Apply a 1-D moving-average smoothing to the x-coordinates of the contour
    to reduce pixel-level noise before curvature computation.
    Window size is clipped to an odd value and must be < len(contour).
    """
    if len(contour) < window * 2:
        return contour.copy()

    window = max(3, window | 1)  # ensure odd
    xs = contour[:, 0]
    kernel = np.ones(window) / window
    xs_smooth = np.convolve(xs, kernel, mode="same")

    # Fix boundary effects: keep original values at edges
    half = window // 2
    xs_smooth[:half] = xs[:half]
    xs_smooth[-half:] = xs[-half:]

    result = contour.copy()
    result[:, 0] = xs_smooth
    return result
