"""
segmentation.py — Silhouette extraction and back-contour (left edge) detection.

Library choice: MediaPipe Selfie Segmentation.
  - Bundled with mediapipe, zero extra install.
  - Produces a clean binary mask at full resolution.
  - rembg (U2Net) would give sharper edges but requires a ~170 MB model download
    and onnxruntime; chosen against to keep setup simple.

Person faces RIGHT → back is on the LEFT side of the silhouette.
We extract the LEFT boundary of the torso mask as an ordered list of (x, y)
pixel positions from top to bottom.  This is the "back contour" A→B.
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
) -> np.ndarray:
    """
    Extract the LEFT boundary of the torso silhouette between shoulder_y and
    hip_y as an (N, 2) float32 array of (x, y) points ordered top→bottom.

    'Left boundary' = smallest x-coordinate among mask pixels on each row.
    This corresponds to the back of a right-facing person.

    If a row has multiple disjoint mask segments (rare with clean segmentation),
    we take the leftmost segment's left edge.

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
        # Left edge of the leftmost foreground run
        left_x = int(xs[0])
        points.append((float(left_x), float(y)))

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
