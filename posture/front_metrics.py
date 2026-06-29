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

cm conversion method (方案B):
  diff_cm = (left_y_px − right_y_px) / sh_hip_px_dist × sh_hip_world_cm
  where sh_hip_px_dist  = pixel distance from shoulder midpoint to hip midpoint
        sh_hip_world_cm = Euclidean distance of same segment in world coords × 100
  This avoids using world Y absolute values directly, which are noisy for
  distal joints (knees) and sensitive to camera distance.
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

# ── Visibility thresholds ─────────────────────────────────────────────────────
_VIS_MIN        = 0.4   # drawing / direction logic
_VIS_MIN_WORLD  = 0.5   # minimum to trust a cm measurement

# Fallback assumed shoulder-to-hip real-world distance when world_lms absent.
# Average adult torso (acromion to greater trochanter) ≈ 55 cm.
_FALLBACK_SH_HIP_CM = 55.0


def _lm_px(lm, img_w: int, img_h: int) -> tuple[float, float]:
    """Convert normalised landmark to pixel coordinates."""
    return lm.x * img_w, lm.y * img_h


def _visible(lm) -> bool:
    return lm.visibility >= _VIS_MIN


def _sh_hip_reference(lms, world_lms, img_w: int, img_h: int) -> tuple[float, float]:
    """
    Return (sh_hip_px_dist, sh_hip_world_cm): the pixel and real-world length
    of the shoulder-midpoint → hip-midpoint segment.

    sh_hip_px_dist  — vertical pixel distance (Y only, same axis as the tilt diffs)
    sh_hip_world_cm — Euclidean 3-D distance in world coords converted to cm;
                      falls back to _FALLBACK_SH_HIP_CM when world_lms is None.
    """
    # Pixel Y midpoints
    sh_y_px  = (_lm_px(lms[_LEFT_SHOULDER], img_w, img_h)[1] +
                _lm_px(lms[_RIGHT_SHOULDER], img_w, img_h)[1]) / 2
    hip_y_px = (_lm_px(lms[_LEFT_HIP],      img_w, img_h)[1] +
                _lm_px(lms[_RIGHT_HIP],      img_w, img_h)[1]) / 2
    sh_hip_px = abs(hip_y_px - sh_y_px)

    if sh_hip_px < 1:      # degenerate frame — avoid division by zero
        sh_hip_px = 1.0

    if world_lms is not None:
        wsh_x  = (world_lms[_LEFT_SHOULDER].x + world_lms[_RIGHT_SHOULDER].x) / 2
        wsh_y  = (world_lms[_LEFT_SHOULDER].y + world_lms[_RIGHT_SHOULDER].y) / 2
        wsh_z  = (world_lms[_LEFT_SHOULDER].z + world_lms[_RIGHT_SHOULDER].z) / 2
        whip_x = (world_lms[_LEFT_HIP].x      + world_lms[_RIGHT_HIP].x)      / 2
        whip_y = (world_lms[_LEFT_HIP].y      + world_lms[_RIGHT_HIP].y)      / 2
        whip_z = (world_lms[_LEFT_HIP].z      + world_lms[_RIGHT_HIP].z)      / 2
        sh_hip_world_cm = math.sqrt(
            (wsh_x - whip_x)**2 +
            (wsh_y - whip_y)**2 +
            (wsh_z - whip_z)**2
        ) * 100
        if sh_hip_world_cm < 1:    # implausibly small — use fallback
            sh_hip_world_cm = _FALLBACK_SH_HIP_CM
    else:
        sh_hip_world_cm = _FALLBACK_SH_HIP_CM

    return sh_hip_px, sh_hip_world_cm


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

def calculate_shoulder_level(lms, world_lms, img_w: int, img_h: int) -> dict:
    """
    Shoulder level asymmetry.

    diff_cm = (left_y_px − right_y_px) / sh_hip_px_dist × sh_hip_world_cm
    This scales the pixel difference by the real-world reference segment length,
    avoiding direct use of absolute world Y values.

    Returns:
        shoulder_diff_cm         : signed cm difference
        shoulder_diff_normalized : pixel diff / img_h  (kept for colour-coding)
        diff_abs_cm              : absolute difference in cm, or None if low visibility
        direction                : "left_low" | "right_low" | "level"
        landmarks_available      : bool
    """
    left_sh  = lms[_LEFT_SHOULDER]
    right_sh = lms[_RIGHT_SHOULDER]

    if not (_visible(left_sh) and _visible(right_sh)):
        return {"shoulder_diff_cm": None, "shoulder_diff_normalized": None,
                "diff_abs_cm": None, "direction": "unknown",
                "landmarks_available": False}

    _, ly = _lm_px(left_sh,  img_w, img_h)
    _, ry = _lm_px(right_sh, img_w, img_h)
    diff_px    = ly - ry
    normalized = diff_px / img_h if img_h > 0 else 0.0

    if abs(normalized) < 0.01:
        direction = "level"
    elif diff_px > 0:
        direction = "left_low"
    else:
        direction = "right_low"

    # cm conversion: scale pixel diff by the shoulder-hip reference segment
    low_vis = (left_sh.visibility < _VIS_MIN_WORLD or
               right_sh.visibility < _VIS_MIN_WORLD)
    if low_vis:
        diff_cm = None
    else:
        sh_hip_px, sh_hip_world_cm = _sh_hip_reference(lms, world_lms, img_w, img_h)
        diff_cm = round(diff_px / sh_hip_px * sh_hip_world_cm, 2)

    return {
        "shoulder_diff_cm":         diff_cm,
        "shoulder_diff_normalized": round(normalized, 4),
        "diff_abs_cm":              round(abs(diff_cm), 2) if diff_cm is not None else None,
        "direction":                direction,
        "landmarks_available":      True,
    }


# ── Pelvic level ──────────────────────────────────────────────────────────────

def calculate_pelvic_level(lms, world_lms, img_w: int, img_h: int) -> dict:
    """
    Pelvic level asymmetry (left vs right hip height).

    diff_cm = (left_y_px − right_y_px) / sh_hip_px_dist × sh_hip_world_cm

    Returns:
        pelvic_diff_cm         : signed cm difference
        pelvic_diff_normalized : pixel diff / img_h  (kept for colour-coding)
        diff_abs_cm            : absolute difference in cm, or None if low visibility
        direction              : "left_low" | "right_low" | "level"
        landmarks_available    : bool
    """
    left_hip  = lms[_LEFT_HIP]
    right_hip = lms[_RIGHT_HIP]

    if not (_visible(left_hip) and _visible(right_hip)):
        return {"pelvic_diff_cm": None, "pelvic_diff_normalized": None,
                "diff_abs_cm": None, "direction": "unknown",
                "landmarks_available": False}

    _, ly = _lm_px(left_hip,  img_w, img_h)
    _, ry = _lm_px(right_hip, img_w, img_h)
    diff_px    = ly - ry
    normalized = diff_px / img_h if img_h > 0 else 0.0

    if abs(normalized) < 0.01:
        direction = "level"
    elif diff_px > 0:
        direction = "left_low"
    else:
        direction = "right_low"

    low_vis = (left_hip.visibility < _VIS_MIN_WORLD or
               right_hip.visibility < _VIS_MIN_WORLD)
    if low_vis:
        diff_cm = None
    else:
        sh_hip_px, sh_hip_world_cm = _sh_hip_reference(lms, world_lms, img_w, img_h)
        diff_cm = round(diff_px / sh_hip_px * sh_hip_world_cm, 2)

    return {
        "pelvic_diff_cm":         diff_cm,
        "pelvic_diff_normalized": round(normalized, 4),
        "diff_abs_cm":            round(abs(diff_cm), 2) if diff_cm is not None else None,
        "direction":              direction,
        "landmarks_available":    True,
    }


# ── Knee level ────────────────────────────────────────────────────────────────

def calculate_knee_level(lms, world_lms, img_w: int, img_h: int) -> dict:
    """
    Knee level asymmetry (left vs right knee height).

    diff_cm = (left_y_px − right_y_px) / sh_hip_px_dist × sh_hip_world_cm

    Returns:
        knee_diff_cm         : signed cm difference
        knee_diff_normalized : pixel diff / img_h  (kept for colour-coding)
        diff_abs_cm          : absolute difference in cm, or None if low visibility
        direction            : "left_low" | "right_low" | "level"
        landmarks_available  : bool
    """
    left_knee  = lms[_LEFT_KNEE]
    right_knee = lms[_RIGHT_KNEE]

    if not (_visible(left_knee) and _visible(right_knee)):
        return {"knee_diff_cm": None, "knee_diff_normalized": None,
                "diff_abs_cm": None, "direction": "unknown",
                "landmarks_available": False}

    _, ly = _lm_px(left_knee,  img_w, img_h)
    _, ry = _lm_px(right_knee, img_w, img_h)
    diff_px    = ly - ry
    normalized = diff_px / img_h if img_h > 0 else 0.0

    if abs(normalized) < 0.01:
        direction = "level"
    elif diff_px > 0:
        direction = "left_low"
    else:
        direction = "right_low"

    low_vis = (left_knee.visibility < _VIS_MIN_WORLD or
               right_knee.visibility < _VIS_MIN_WORLD)
    if low_vis:
        diff_cm = None
    else:
        sh_hip_px, sh_hip_world_cm = _sh_hip_reference(lms, world_lms, img_w, img_h)
        diff_cm = round(diff_px / sh_hip_px * sh_hip_world_cm, 2)

    return {
        "knee_diff_cm":         diff_cm,
        "knee_diff_normalized": round(normalized, 4),
        "diff_abs_cm":          round(abs(diff_cm), 2) if diff_cm is not None else None,
        "direction":            direction,
        "landmarks_available":  True,
    }
