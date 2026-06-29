"""
front_visualization.py — Draw front-view postural annotations on an image.

Draws four horizontal reference lines (ear, shoulder, hip, knee) with:
  - Color coding: green (≤1% img_h diff) → yellow (1–3%) → red (>3%)
  - Angle/difference labels next to each line
  - Small dots at each landmark used
"""

import cv2
import math
import numpy as np

# Landmark index constants (MediaPipe 33-point schema)
_LEFT_EAR       = 7
_RIGHT_EAR      = 8
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_LEFT_KNEE      = 25
_RIGHT_KNEE     = 26

# ── Color thresholds (normalized diff relative to image height) ───────────────
_THRESH_NORMAL  = 0.010   # ≤ 1 %  → green
_THRESH_MILD    = 0.030   # 1–3 %  → yellow
# > 3 % → red

_COLOR_NORMAL  = (  0, 200,  80)   # green
_COLOR_MILD    = (  0, 200, 255)   # yellow
_COLOR_CONCERN = ( 30,  30, 220)   # red (BGR)
_COLOR_DOT     = (255, 255, 255)   # white


def _lm_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(round(lm.x * w)), int(round(lm.y * h))


def _line_color(normalized_diff) -> tuple:
    if normalized_diff is None:
        return _COLOR_NORMAL
    v = abs(normalized_diff)
    if v <= _THRESH_NORMAL:
        return _COLOR_NORMAL
    if v <= _THRESH_MILD:
        return _COLOR_MILD
    return _COLOR_CONCERN


def _draw_level_line(
    img: np.ndarray,
    pt_left: tuple[int, int],
    pt_right: tuple[int, int],
    color: tuple,
    label: str,
    thickness: int = 2,
) -> None:
    """Draw a line between two landmarks and a text label to the right."""
    cv2.line(img, pt_left, pt_right, color, thickness, cv2.LINE_AA)
    cv2.circle(img, pt_left,  5, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, pt_left,  4, color,        -1, cv2.LINE_AA)
    cv2.circle(img, pt_right, 5, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, pt_right, 4, color,        -1, cv2.LINE_AA)

    # Label to the right of the rightmost point
    lx = max(pt_left[0], pt_right[0]) + 8
    ly = (pt_left[1] + pt_right[1]) // 2
    cv2.putText(img, label, (lx + 1, ly + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def draw_front_overlay(image_bgr: np.ndarray, front_result) -> np.ndarray:
    """
    Draw front-view postural lines on a copy of *image_bgr*.

    Expects a FrontAnalysisResult (or any object with the same fields).
    Returns the annotated image.
    """
    img = image_bgr.copy()
    r   = front_result
    lms = r._raw_landmarks

    if lms is None:
        return img

    h, w = img.shape[:2]

    # ── Head tilt ─────────────────────────────────────────────────────────
    le = lms[_LEFT_EAR]
    re = lms[_RIGHT_EAR]
    if le.visibility >= 0.4 and re.visibility >= 0.4:
        ht = r.head_tilt
        color = _line_color(
            (ht.get("tilt_abs_deg", 0) / 90.0) if ht else None
        )
        label = f"Head: {ht['tilt_abs_deg']:.1f}° {ht['direction']}" if ht else "Head: N/A"
        _draw_level_line(img, _lm_px(le, w, h), _lm_px(re, w, h), color, label)

    # ── Shoulder level ────────────────────────────────────────────────────
    ls = lms[_LEFT_SHOULDER]
    rs = lms[_RIGHT_SHOULDER]
    if ls.visibility >= 0.4 and rs.visibility >= 0.4:
        sl = r.shoulder_level
        color = _line_color(sl.get("shoulder_diff_normalized") if sl else None)
        label = (f"Sho: {sl['diff_abs_cm']:.1f}cm {sl['direction']}"
                 if sl and sl.get("diff_abs_cm") is not None else "Sho: N/A")
        _draw_level_line(img, _lm_px(ls, w, h), _lm_px(rs, w, h), color, label)

    # ── Pelvic level ──────────────────────────────────────────────────────
    lh = lms[_LEFT_HIP]
    rh = lms[_RIGHT_HIP]
    if lh.visibility >= 0.4 and rh.visibility >= 0.4:
        pl = r.pelvic_level
        color = _line_color(pl.get("pelvic_diff_normalized") if pl else None)
        label = (f"Hip: {pl['diff_abs_cm']:.1f}cm {pl['direction']}"
                 if pl and pl.get("diff_abs_cm") is not None else "Hip: N/A")
        _draw_level_line(img, _lm_px(lh, w, h), _lm_px(rh, w, h), color, label)

    # ── Knee level ────────────────────────────────────────────────────────
    lk = lms[_LEFT_KNEE]
    rk = lms[_RIGHT_KNEE]
    if lk.visibility >= 0.4 and rk.visibility >= 0.4:
        kl = r.knee_level
        color = _line_color(kl.get("knee_diff_normalized") if kl else None)
        label = (f"Knee: {kl['diff_abs_cm']:.1f}cm {kl['direction']}"
                 if kl and kl.get("diff_abs_cm") is not None else "Knee: N/A")
        _draw_level_line(img, _lm_px(lk, w, h), _lm_px(rk, w, h), color, label)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        (_COLOR_NORMAL,  "Normal (<=1%)"),
        (_COLOR_MILD,    "Mild (1-3%)"),
        (_COLOR_CONCERN, "Notable (>3%)"),
    ]
    for i, (c, text) in enumerate(legend_items):
        lx, ly = 10, h - 14 - i * 20
        cv2.rectangle(img, (lx, ly - 10), (lx + 12, ly + 2), c, -1)
        cv2.putText(img, text, (lx + 16, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return img
