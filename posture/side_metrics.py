"""
side_metrics.py — Side-view postural metrics (beyond Cobb angles).

Currently implements:
  - Forward Head Position (craniovertebral angle)

All functions accept the raw MediaPipe landmark list returned by
spine/pose_detector.py's detect_pose() and the image dimensions,
so they can be called directly inside the pipeline without re-running
MediaPipe.
"""

import math

# Landmark index constants (MediaPipe 33-point schema, stable across API versions)
_LEFT_EAR       = 7
_RIGHT_EAR      = 8
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12


def _lm_px(lm, img_w: int, img_h: int) -> tuple[float, float]:
    """Convert a normalised MediaPipe landmark to pixel coordinates."""
    return lm.x * img_w, lm.y * img_h


def _angle_3pts(p1: tuple, vertex: tuple, p2: tuple) -> float:
    """
    Return the angle (degrees, 0–180) at *vertex* formed by p1–vertex–p2.
    Uses the dot-product formula; safe against degenerate cases.
    """
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    len1 = math.hypot(*v1)
    len2 = math.hypot(*v2)
    if len1 < 1e-9 or len2 < 1e-9:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_val = max(-1.0, min(1.0, dot / (len1 * len2)))
    return math.degrees(math.acos(cos_val))


def calculate_forward_head_position(
    lms,
    facing_direction: str,
    img_w: int,
    img_h: int,
) -> dict:
    """
    Calculate Forward Head Position (craniovertebral angle) from the raw
    MediaPipe landmark list.

    Landmark selection is automatic:
      facing_direction="right" → left side visible → LEFT_EAR + LEFT_SHOULDER
      facing_direction="left"  → right side visible → RIGHT_EAR + RIGHT_SHOULDER

    Craniovertebral (CV) angle:
        vertex  = shoulder
        point1  = ear
        point2  = vertical_point = (ear_x, shoulder_y)
    A larger CV angle means the head is less forward; normal is ≥ 50°.
    Forward displacement (pixels) = abs(ear_x − shoulder_x).

    Returns:
        {
            "craniovertebral_angle_deg": float,
            "forward_displacement_px":  float,
            "neck_inclination_deg":     float,   # deviation from vertical
        }
    """
    if lms is None:
        return {
            "craniovertebral_angle_deg": None,
            "forward_displacement_px": None,
            "neck_inclination_deg": None,
        }

    if facing_direction == "left":
        ear_lm = lms[_RIGHT_EAR]
        sh_lm  = lms[_RIGHT_SHOULDER]
    else:
        ear_lm = lms[_LEFT_EAR]
        sh_lm  = lms[_LEFT_SHOULDER]

    ear      = _lm_px(ear_lm, img_w, img_h)
    shoulder = _lm_px(sh_lm,  img_w, img_h)

    # Vertical reference: directly below the ear at the same height as the shoulder
    vertical_point = (ear[0], shoulder[1])

    cv_angle = _angle_3pts(ear, shoulder, vertical_point)

    forward_displacement = abs(ear[0] - shoulder[0])

    # Neck inclination = angle of ear→shoulder line relative to true vertical
    dx = shoulder[0] - ear[0]
    dy = shoulder[1] - ear[1]   # positive downward
    if abs(dy) < 1e-6:
        neck_inclination = 90.0
    else:
        neck_inclination = abs(math.degrees(math.atan2(dx, dy)))

    return {
        "craniovertebral_angle_deg": round(cv_angle, 2),
        "forward_displacement_px":  round(forward_displacement, 1),
        "neck_inclination_deg":     round(neck_inclination, 2),
    }


# ─── Forward Head Distance (FHD) ─────────────────────────────────────────────

def _fhd_severity(fhd_inches: float) -> str:
    """Classify FHD into severity tiers per Kapandji biomechanical model."""
    if fhd_inches < 1.0:
        return "Normal"
    elif fhd_inches < 2.0:
        return "Mild"
    elif fhd_inches < 3.0:
        return "Moderate"
    else:
        return "Severe"


def calculate_fhd(
    lms,
    img_w: int,
    img_h: int,
    facing_direction: str = "right",
    ref_ear_to_c7_cm: float = 24.0,
) -> dict:
    """
    Calculate Forward Head Distance (FHD) from raw MediaPipe landmarks.

    Measurement
    -----------
    - Ear point  : visible-side ear landmark
    - C7 proxy   : midpoint of LEFT_SHOULDER and RIGHT_SHOULDER
    - FHD pixels : |ear_x − c7_x|  (horizontal offset only)

    Pixel → cm calibration
    ----------------------
    On a side-view image the two shoulders are nearly stacked → shoulder-span
    is only ~20 px and is useless as a ruler.  Instead we use the **vertical
    distance from the ear to C7** (ear_y → c7_y), which is unaffected by the
    side-view perspective.  Average anatomical value ≈ 24 cm (configurable).

        px_per_cm = |ear_y − c7_y| / ref_ear_to_c7_cm

    Kapandji (2009) spine-load model
    ---------------------------------
        spine_load_lbs = 12.0 + (fhd_inches × 10.0)

    Severity tiers
    --------------
      < 1 inch  → Normal   (green)
      1–2 inch  → Mild     (yellow)
      2–3 inch  → Moderate (orange)
      ≥ 3 inch  → Severe   (red)

    Parameters
    ----------
    lms               : MediaPipe landmark list (33 points)
    img_w, img_h      : image dimensions in pixels
    facing_direction  : "right" or "left"
    ref_ear_to_c7_cm  : assumed anatomical ear-to-C7 distance in cm (default 24 cm)

    Returns
    -------
    dict with keys:
        fhd_pixels    : float   horizontal pixel offset
        fhd_cm        : float
        fhd_inches    : float
        spine_load_lbs: float
        ear_point     : (int, int)   pixel coords of ear
        c7_point      : (int, int)   pixel coords of C7 proxy
        severity      : str
    """
    if lms is None:
        return {
            "fhd_pixels": None, "fhd_cm": None, "fhd_inches": None,
            "spine_load_lbs": None,
            "ear_point": None, "c7_point": None, "severity": "Unknown",
        }

    # Select visible-side ear
    ear_lm = lms[_RIGHT_EAR] if facing_direction == "left" else lms[_LEFT_EAR]

    l_sh = lms[_LEFT_SHOULDER]
    r_sh = lms[_RIGHT_SHOULDER]

    # C7 proxy = shoulder midpoint
    c7_x = ((l_sh.x + r_sh.x) / 2) * img_w
    c7_y = ((l_sh.y + r_sh.y) / 2) * img_h
    ear_x = ear_lm.x * img_w
    ear_y = ear_lm.y * img_h

    # Calibration: use ear→C7 vertical distance as the ruler.
    # This is robust on a side-view image (both landmarks have stable y coords).
    ear_c7_vertical_px = abs(ear_y - c7_y)
    if ear_c7_vertical_px < 5.0:
        # Degenerate: ear and shoulder at same height, cannot calibrate
        return {
            "fhd_pixels": None, "fhd_cm": None, "fhd_inches": None,
            "spine_load_lbs": None,
            "ear_point": (int(round(ear_x)), int(round(ear_y))),
            "c7_point":  (int(round(c7_x)),  int(round(c7_y))),
            "severity": "Unknown",
        }

    px_per_cm  = ear_c7_vertical_px / ref_ear_to_c7_cm
    fhd_px     = abs(ear_x - c7_x)
    fhd_cm     = fhd_px / px_per_cm
    fhd_inches = fhd_cm / 2.54
    spine_load = 12.0 + (fhd_inches * 10.0)

    return {
        "fhd_pixels":     round(fhd_px,     1),
        "fhd_cm":         round(fhd_cm,     2),
        "fhd_inches":     round(fhd_inches, 2),
        "spine_load_lbs": round(spine_load, 1),
        "ear_point":      (int(round(ear_x)), int(round(ear_y))),
        "c7_point":       (int(round(c7_x)),  int(round(c7_y))),
        "severity":       _fhd_severity(fhd_inches),
    }
