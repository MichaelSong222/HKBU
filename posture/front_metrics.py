"""
front_metrics.py — Front-view postural metrics.

Four indicators:
  - Head tilt      (lateral flexion): ear-to-ear line angle from horizontal
  - Shoulder level : vertical height difference left vs right shoulder
  - Pelvic level   : vertical height difference left vs right hip
  - Knee level     : vertical height difference left vs right knee

All functions accept the raw MediaPipe landmark list (from detect_pose())
and image dimensions.  No body-height input required; normalization uses
image height as the reference.

MediaPipe landmark indices (hard-coded to avoid API version dependency):
  NOSE=0, LEFT_EAR=7, RIGHT_EAR=8
  LEFT_SHOULDER=11, RIGHT_SHOULDER=12
  LEFT_HIP=23,      RIGHT_HIP=24
  LEFT_KNEE=25,     RIGHT_KNEE=26
"""

import math

# Landmark index constants (MediaPipe 33-point schema)
_NOSE           = 0
_LEFT_EAR       = 7
_RIGHT_EAR      = 8
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_LEFT_KNEE      = 25
_RIGHT_KNEE     = 26

# ── Visibility threshold ──────────────────────────────────────────────────────
_VIS_MIN = 0.4


def _lm_px(lm, img_w: int, img_h: int) -> tuple[float, float]:
    """Convert normalised landmark to pixel coordinates."""
    return lm.x * img_w, lm.y * img_h


def _visible(lm) -> bool:
    return lm.visibility >= _VIS_MIN


# ── Head tilt ─────────────────────────────────────────────────────────────────

def calculate_head_tilt(lms, img_w: int, img_h: int) -> dict:
    """
    Head tilt (lateral flexion).

    Computes the angle of the left-ear → right-ear line relative to
    true horizontal.  Positive = left ear higher; negative = right ear higher.

    Returns:
        head_tilt_deg       : signed angle from horizontal (degrees)
        tilt_abs_deg        : absolute tilt magnitude
        direction           : "left_high" | "right_high" | "level"
        landmarks_available : bool
    """
    left_ear  = lms[_LEFT_EAR]
    right_ear = lms[_RIGHT_EAR]

    if not (_visible(left_ear) and _visible(right_ear)):
        return {"head_tilt_deg": None, "tilt_abs_deg": None,
                "direction": "unknown", "landmarks_available": False}

    lx, ly = _lm_px(left_ear,  img_w, img_h)
    rx, ry = _lm_px(right_ear, img_w, img_h)

    # Angle of left→right vector relative to horizontal.
    # atan2 range is (-180, 180].  When the two ears are near-horizontal
    # the angle is close to 0° — but if MediaPipe swaps their x order (right
    # ear reported to the left of left ear, which happens on some front-facing
    # images) the raw result jumps to ±179.9°.
    # Normalise to (−90°, 90°] so the displayed value always reflects the
    # true tilt magnitude.
    angle_deg = math.degrees(math.atan2(ry - ly, rx - lx))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    if abs(angle_deg) < 1.5:
        direction = "level"
    elif angle_deg > 0:
        direction = "left_high"   # right ear lower in image → left side higher
    else:
        direction = "right_high"

    return {
        "head_tilt_deg": round(angle_deg, 2),
        "tilt_abs_deg":  round(abs(angle_deg), 2),
        "direction":     direction,
        "landmarks_available": True,
    }


# ── Shoulder level ────────────────────────────────────────────────────────────

def calculate_shoulder_level(lms, img_w: int, img_h: int) -> dict:
    """
    Shoulder level asymmetry.

    diff_px = left_shoulder_y − right_shoulder_y  (image coords, +y downward)
    Positive diff → left shoulder is LOWER (higher y) in the image.

    Returns:
        shoulder_diff_px         : signed pixel difference (left y − right y)
        shoulder_diff_normalized : diff / img_h
        diff_abs_px              : absolute difference
        direction                : "left_low" | "right_low" | "level"
        landmarks_available      : bool
    """
    left_sh  = lms[_LEFT_SHOULDER]
    right_sh = lms[_RIGHT_SHOULDER]

    if not (_visible(left_sh) and _visible(right_sh)):
        return {"shoulder_diff_px": None, "shoulder_diff_normalized": None,
                "diff_abs_px": None, "direction": "unknown",
                "landmarks_available": False}

    _, ly = _lm_px(left_sh,  img_w, img_h)
    _, ry = _lm_px(right_sh, img_w, img_h)

    diff = ly - ry
    normalized = diff / img_h if img_h > 0 else 0.0

    if abs(diff) < img_h * 0.01:
        direction = "level"
    elif diff > 0:
        direction = "left_low"
    else:
        direction = "right_low"

    return {
        "shoulder_diff_px":         round(diff, 1),
        "shoulder_diff_normalized": round(normalized, 4),
        "diff_abs_px":              round(abs(diff), 1),
        "direction":                direction,
        "landmarks_available":      True,
    }


# ── Pelvic level ──────────────────────────────────────────────────────────────

def calculate_pelvic_level(lms, img_w: int, img_h: int) -> dict:
    """
    Pelvic level asymmetry (left vs right hip height).

    Returns:
        pelvic_diff_px         : signed pixel difference (left hip y − right hip y)
        pelvic_diff_normalized : diff / img_h
        diff_abs_px            : absolute difference
        direction              : "left_low" | "right_low" | "level"
        landmarks_available    : bool
    """
    left_hip  = lms[_LEFT_HIP]
    right_hip = lms[_RIGHT_HIP]

    if not (_visible(left_hip) and _visible(right_hip)):
        return {"pelvic_diff_px": None, "pelvic_diff_normalized": None,
                "diff_abs_px": None, "direction": "unknown",
                "landmarks_available": False}

    _, ly = _lm_px(left_hip,  img_w, img_h)
    _, ry = _lm_px(right_hip, img_w, img_h)

    diff = ly - ry
    normalized = diff / img_h if img_h > 0 else 0.0

    if abs(diff) < img_h * 0.01:
        direction = "level"
    elif diff > 0:
        direction = "left_low"
    else:
        direction = "right_low"

    return {
        "pelvic_diff_px":         round(diff, 1),
        "pelvic_diff_normalized": round(normalized, 4),
        "diff_abs_px":            round(abs(diff), 1),
        "direction":              direction,
        "landmarks_available":    True,
    }


# ── Knee level ────────────────────────────────────────────────────────────────

def calculate_knee_level(lms, img_w: int, img_h: int) -> dict:
    """
    Knee level asymmetry (left vs right knee height).

    Returns:
        knee_diff_px         : signed pixel difference (left knee y − right knee y)
        knee_diff_normalized : diff / img_h
        diff_abs_px          : absolute difference
        direction            : "left_low" | "right_low" | "level"
        landmarks_available  : bool
    """
    left_knee  = lms[_LEFT_KNEE]
    right_knee = lms[_RIGHT_KNEE]

    if not (_visible(left_knee) and _visible(right_knee)):
        return {"knee_diff_px": None, "knee_diff_normalized": None,
                "diff_abs_px": None, "direction": "unknown",
                "landmarks_available": False}

    _, ly = _lm_px(left_knee,  img_w, img_h)
    _, ry = _lm_px(right_knee, img_w, img_h)

    diff = ly - ry
    normalized = diff / img_h if img_h > 0 else 0.0

    if abs(diff) < img_h * 0.01:
        direction = "level"
    elif diff > 0:
        direction = "left_low"
    else:
        direction = "right_low"

    return {
        "knee_diff_px":         round(diff, 1),
        "knee_diff_normalized": round(normalized, 4),
        "diff_abs_px":          round(abs(diff), 1),
        "direction":            direction,
        "landmarks_available":  True,
    }
