"""
visualization.py — Overlay matching the hand-drawn reference diagram.

Style (from the reference sketch):
  • Spine curve A→B              — solid green, medium weight
  • Shoulder/Hip reference lines — dashed grey
  • Landmark dots: A, apex_K (T4/T5), inflect (T12/L1), apex_L (L3/L4), B
  • Four angle-endpoint dots:
      thoracic_upper (~T2/T3), thoracic_lower (~T10/T11)  — blue family
      lumbar_upper (~L2),      lumbar_lower (~L5)          — orange family
  • At each of the four endpoints:
      - Short DASHED tangent line (direction of local curve tangent)
      - Solid NORMAL line (perpendicular to tangent — the Cobb construction line)
  • Thoracic angle x:
      - Arc drawn at the intersection of the two thoracic normal lines
      - Label "x=N.N°" near the arc
  • Lumbar angle y:
      - Arc drawn at the intersection of the two lumbar normal lines
      - Label "y=N.N°" near the arc

Geometry note
─────────────
The angle between two NORMAL lines equals the angle between the two TANGENT
lines (90° rotation preserves angle).  Both are drawn; the arc is placed at
the normal-line intersection, which is the clearest visual anchor.
"""

import cv2
import math
import numpy as np

import config


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_int(pt) -> tuple[int, int]:
    return (int(round(float(pt[0]))), int(round(float(pt[1]))))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _normal(v) -> np.ndarray:
    """Return the unit normal (90° CCW rotation) of unit vector v."""
    u = _unit(v)
    return np.array([-u[1], u[0]])


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple,
    thickness: int = 1,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        sx = int(x1 + ux * pos)
        sy = int(y1 + uy * pos)
        ex = int(x1 + ux * min(pos + dash, length))
        ey = int(y1 + uy * min(pos + dash, length))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += dash + gap


def _draw_dashed_ray(
    img: np.ndarray,
    centre: np.ndarray,
    direction: np.ndarray,
    color: tuple,
    half_len: int,
    thickness: int = 1,
    dash: int = 10,
    gap: int = 7,
) -> None:
    """Dashed line centred on *centre* extending ±half_len in *direction*."""
    d = _unit(direction)
    pt1 = _to_int(np.array(centre, float) - d * half_len)
    pt2 = _to_int(np.array(centre, float) + d * half_len)
    _draw_dashed_line(img, pt1, pt2, color, thickness=thickness, dash=dash, gap=gap)


def _draw_solid_ray(
    img: np.ndarray,
    centre: np.ndarray,
    direction: np.ndarray,
    color: tuple,
    half_len: int,
    thickness: int = 2,
) -> None:
    """Solid line centred on *centre* extending ±half_len in *direction*."""
    d = _unit(direction)
    pt1 = _to_int(np.array(centre, float) - d * half_len)
    pt2 = _to_int(np.array(centre, float) + d * half_len)
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


# ─── Normal-line intersection ─────────────────────────────────────────────────

def _line_intersect(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> np.ndarray | None:
    """
    Find the intersection of two infinite lines:
      L1: p1 + t·d1
      L2: p2 + s·d2

    Returns the intersection point, or None if lines are parallel.
    """
    # Solve: p1 + t*d1 = p2 + s*d2  →  [d1 | -d2] [t; s] = p2 - p1
    A = np.array([[d1[0], -d2[0]],
                  [d1[1], -d2[1]]], dtype=float)
    b = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    t = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
    return np.array(p1, dtype=float) + t * np.array(d1, dtype=float)


# ─── Angle arc ────────────────────────────────────────────────────────────────

def _draw_angle_arc(
    img: np.ndarray,
    centre: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    angle_deg: float,
    label_char: str,
    color: tuple,
    arc_r: int = 32,
) -> None:
    """
    Draw a small filled arc at *centre* marking the acute angle between d1
    and d2, then place the label just outside the arc midpoint.

    Picks the ±d1, ±d2 combination whose enclosed sweep is closest to
    angle_deg so the arc always marks the correct (small) wedge.
    """
    C  = np.array(centre, dtype=float)
    u1 = _unit(d1)
    u2 = _unit(d2)

    best_start, best_sweep, best_mid_vec = None, None, None
    best_err = 1e9

    for s1 in (1, -1):
        for s2 in (1, -1):
            v1 = s1 * u1
            v2 = s2 * u2
            a1 = math.degrees(math.atan2(-v1[1], v1[0]))
            a2 = math.degrees(math.atan2(-v2[1], v2[0]))
            sw_ccw = (a2 - a1) % 360
            sw = sw_ccw if sw_ccw <= 180 else 360 - sw_ccw
            err = abs(sw - angle_deg)
            if err < best_err:
                best_err = err
                if sw_ccw <= 180:
                    best_start = a1;  best_sweep = sw_ccw
                else:
                    best_start = a2;  best_sweep = 360 - sw_ccw
                mid_rad = math.radians(best_start + best_sweep / 2)
                best_mid_vec = np.array([math.cos(mid_rad), -math.sin(mid_rad)])

    start_ang = best_start
    sweep     = best_sweep
    mid_vec   = best_mid_vec

    # Filled translucent wedge
    overlay = img.copy()
    n_steps = max(20, int(abs(sweep)))
    poly_pts = [_to_int(C)]
    for i in range(n_steps + 1):
        frac = i / n_steps
        rad  = math.radians(start_ang + frac * sweep)
        poly_pts.append(_to_int(C + arc_r * np.array([math.cos(rad), -math.sin(rad)])))
    poly = np.array(poly_pts, dtype=np.int32)
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)

    # Arc border
    cv2.ellipse(img, _to_int(C), (arc_r, arc_r), 0,
                -(start_ang + sweep), -start_ang,
                color, config.OVERLAY_LINE_THICKNESS, cv2.LINE_AA)

    # Label
    label_r = arc_r + 16
    lx = int(C[0] + mid_vec[0] * label_r)
    ly = int(C[1] - mid_vec[1] * label_r)
    text = f"{label_char}={angle_deg:.1f}"
    cv2.putText(img, text, (lx + 1, ly + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


# ─── Per-endpoint drawing ─────────────────────────────────────────────────────

def _draw_endpoint(
    img: np.ndarray,
    pt: np.ndarray,
    tangent: np.ndarray,
    color: tuple,
    tangent_half: int,
    normal_half: int,
    dot_radius: int = 4,
) -> None:
    """
    Draw one angle endpoint:
      • dashed tangent line (short, centred on pt)
      • solid normal line (centred on pt)
      • filled dot
    """
    norm = _normal(tangent)
    _draw_dashed_ray(img, pt, tangent, color, tangent_half, thickness=1)
    _draw_solid_ray(img, pt, norm, color, normal_half, thickness=2)
    c_pt = _to_int(pt)
    cv2.circle(img, c_pt, dot_radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, c_pt, dot_radius, color, -1, cv2.LINE_AA)


# ─── Public entry point ───────────────────────────────────────────────────────

def draw_overlay(
    image_bgr: np.ndarray,
    result,
) -> np.ndarray:
    """
    Draw the reference-style overlay on a copy of *image_bgr* and return it.

    Drawing order (back → front):
      1. Grey axis line Shoulder→Hip + horizontal projection dashes
      2. Spine contour A→B (solid green)
      3. Thoracic construction (blue):
           - dashed tangent + solid normal at thoracic_upper (~T2/T3)
           - dashed tangent + solid normal at thoracic_lower (~T10/T11)
           - arc + "x=N.N°" label at normal-line intersection
      4. Lumbar construction (orange):
           - dashed tangent + solid normal at lumbar_upper (~L2)
           - dashed tangent + solid normal at lumbar_lower (~L5)
           - arc + "y=N.N°" label at normal-line intersection
      5. Anatomical landmark dots: A, apex_K, inflect, apex_L, B
      6. Warnings
    """
    img = image_bgr.copy()
    r   = result
    t   = config.OVERLAY_LINE_THICKNESS

    # arm lengths (pixels)
    tang_half   = config.TANGENT_ARM_PX // 2      # short dashed tangent arm
    normal_half = config.TANGENT_ARM_PX            # longer solid normal arm

    shoulder = np.array(r.shoulder, float) if r.shoulder is not None else None
    hip      = np.array(r.hip,      float) if r.hip      is not None else None

    # ── 1. Reference lines ────────────────────────────────────────────────
    if shoulder is not None and hip is not None:
        cv2.line(img, _to_int(shoulder), _to_int(hip),
                 config.COLOR_AXIS_LINE, 1, cv2.LINE_AA)
    if shoulder is not None and r.A is not None:
        _draw_dashed_line(img, _to_int(shoulder), _to_int(r.A),
                          config.COLOR_AXIS_LINE, thickness=1, dash=8, gap=5)
    if hip is not None and r.B is not None:
        _draw_dashed_line(img, _to_int(hip), _to_int(r.B),
                          config.COLOR_AXIS_LINE, thickness=1, dash=8, gap=5)

    # ── 2. Spine contour A→B ──────────────────────────────────────────────
    if r.back_contour is not None and len(r.back_contour) > 1:
        pts = r.back_contour.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False,
                      config.COLOR_CONTOUR, t + 1, cv2.LINE_AA)

    # ── 3. Thoracic construction (blue) ──────────────────────────────────
    has_thoracic = (
        r.ep_thoracic_upper is not None
        and r.ep_thoracic_lower is not None
        and r.T1 is not None and r.T2 is not None
        and r.thoracic_angle_deg is not None
    )
    if has_thoracic:
        ep_tu = np.array(r.ep_thoracic_upper, float)
        ep_tl = np.array(r.ep_thoracic_lower, float)

        _draw_endpoint(img, ep_tu, r.T1, config.COLOR_THORACIC,
                       tang_half, normal_half)
        _draw_endpoint(img, ep_tl, r.T2, config.COLOR_THORACIC,
                       tang_half, normal_half)

        # Arc at intersection of the two normal lines
        n1 = _normal(r.T1)
        n2 = _normal(r.T2)
        intersect_T = _line_intersect(ep_tu, n1, ep_tl, n2)
        arc_centre_T = intersect_T if intersect_T is not None else ep_tu
        _draw_angle_arc(img, arc_centre_T, n1, n2,
                        r.thoracic_angle_deg, "x",
                        config.COLOR_THORACIC, arc_r=34)

    # ── 4. Lumbar construction (orange) ───────────────────────────────────
    has_lumbar = (
        r.ep_lumbar_upper is not None
        and r.ep_lumbar_lower is not None
        and r.T3 is not None and r.T4 is not None
        and r.lumbar_angle_deg is not None
    )
    if has_lumbar:
        ep_lu = np.array(r.ep_lumbar_upper, float)
        ep_ll = np.array(r.ep_lumbar_lower, float)

        _draw_endpoint(img, ep_lu, r.T3, config.COLOR_LUMBAR,
                       tang_half, normal_half)
        _draw_endpoint(img, ep_ll, r.T4, config.COLOR_LUMBAR,
                       tang_half, normal_half)

        n3 = _normal(r.T3)
        n4 = _normal(r.T4)
        intersect_L = _line_intersect(ep_lu, n3, ep_ll, n4)
        arc_centre_L = intersect_L if intersect_L is not None else ep_lu
        _draw_angle_arc(img, arc_centre_L, n3, n4,
                        r.lumbar_angle_deg, "y",
                        config.COLOR_LUMBAR, arc_r=34)

    # ── 5. Anatomical landmark dots + labels ──────────────────────────────
    rad = config.OVERLAY_POINT_RADIUS
    landmarks = [
        (r.shoulder,  "Sho.",   ( 8,  -6)),
        (r.hip,       "Hip",    ( 8,  -6)),
        (r.A,         "A",      (-16, -6)),
        (r.apex_K,    "T6/T7",  (-50, -6)),
        (r.inflect,   "T12/L1", (-60, -6)),
        (r.apex_L,    "L3/L4",  (-60, -6)),
        (r.B,         "B",      (-16,  14)),
    ]
    for pt, label, (ox, oy) in landmarks:
        if pt is None:
            continue
        c_pt = _to_int(pt)
        cv2.circle(img, c_pt, rad + 1, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, c_pt, rad, config.COLOR_LANDMARK, -1, cv2.LINE_AA)
        tx, ty = c_pt[0] + ox, c_pt[1] + oy
        cv2.putText(img, label, (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    config.COLOR_LANDMARK, 1, cv2.LINE_AA)

    # ── 5b. Fitted lumbar curve + L5(est) dot ────────────────────────────
    if getattr(r, "estimated_l5", None) is not None:
        el5 = _to_int(r.estimated_l5)

        # Fitted quadratic curve from apex_L to L5(est): drawn as a polyline so
        # the viewer can confirm it follows the lordotic arc, not the buttock slope.
        fit_curve = getattr(r, "lumbar_fit_curve", None)
        if fit_curve is not None and len(fit_curve) > 1:
            pts_curve = fit_curve.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts_curve], False,
                          config.COLOR_L5_EST, 1, cv2.LINE_AA)
        elif r.ep_lumbar_upper is not None:
            # Fallback: straight line when curve unavailable
            cv2.line(img, _to_int(r.ep_lumbar_upper), el5,
                     config.COLOR_L5_EST, 1, cv2.LINE_AA)

        # Corrected T4 tangent + normal at ep_lumbar_lower (= estimated_l5 / virtual_B)
        if r.T4 is not None and r.ep_lumbar_lower is not None:
            _draw_endpoint(img, np.array(r.ep_lumbar_lower, float),
                           r.T4, config.COLOR_L5_EST,
                           tang_half, normal_half, dot_radius=4)

        # Yellow dot + label on top
        cv2.circle(img, el5, rad + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, el5, rad + 1, config.COLOR_L5_EST, -1, cv2.LINE_AA)
        lx, ly = el5[0] + 8, el5[1] - 6
        cv2.putText(img, "L5(est)", (lx + 1, ly + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, "L5(est)", (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, config.COLOR_L5_EST, 1, cv2.LINE_AA)

        # ── Shoulder/hip contour projections (arc-length method debug dots) ──
        for proj, label in [
            (getattr(r, "l5_shoulder_proj", None), "Sho.proj"),
            (getattr(r, "l5_hip_proj",      None), "Hip.proj"),
        ]:
            if proj is not None:
                pp = _to_int(proj)
                cv2.circle(img, pp, rad + 1, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, pp, rad, (0, 200, 255), -1, cv2.LINE_AA)  # cyan
                cv2.putText(img, label, (pp[0] + 6, pp[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

    # ── 6. Warnings ───────────────────────────────────────────────────────
    if r.warnings:
        h = img.shape[0]
        for i, msg in enumerate(r.warnings):
            y = h - 14 - i * 22
            cv2.putText(img, f"! {msg}", (11, y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(img, f"! {msg}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        config.COLOR_WARNING_TEXT, 1, cv2.LINE_AA)

    # ── 7. Forward Head Position overlay ─────────────────────────────────
    _draw_fhp_overlay(img, r)

    return img


def draw_fhp_on_image(
    image_bgr: np.ndarray,
    lms,
    facing_direction: str,
    forward_head_angle_deg: float,
    fhd_result: dict | None = None,
) -> np.ndarray:
    """
    Draw FHP (CV angle) and optionally FHD construction on *image_bgr*.
    Called when the FHP photo is separate from the contour photo.
    Returns the annotated copy.
    """
    img = image_bgr.copy()
    _draw_fhp_overlay_raw(img, lms, facing_direction, forward_head_angle_deg,
                          image_bgr.shape[1], image_bgr.shape[0])
    if fhd_result is not None:
        _draw_fhd_overlay(img, fhd_result, facing_direction)
    return img


def _draw_fhp_overlay(img: np.ndarray, r) -> None:
    """Thin wrapper that pulls fields from an AnalysisResult and delegates."""
    if getattr(r, "forward_head_angle_deg", None) is None:
        return
    if r._raw_landmarks is None:
        return
    _draw_fhp_overlay_raw(
        img,
        r._raw_landmarks,
        getattr(r, "facing_direction", "right"),
        r.forward_head_angle_deg,
        img.shape[1],
        img.shape[0],
    )


def _draw_fhp_overlay_raw(
    img: np.ndarray,
    lms,
    facing_direction: str,
    cv_angle_deg: float,
    img_w: int,
    img_h: int,
) -> None:
    """
    Draw Forward Head Position construction lines directly onto *img*.

    Geometry:
      - ear_px     = landmark pixel coords of the visible ear
      - shoulder_px = landmark pixel coords of the visible shoulder
      - vert_pt    = (ear_px[0], shoulder_px[1])   ← directly below ear, at shoulder height
      - CV angle   = angle at vertex=shoulder between ray→ear and ray→vert_pt
        (matches calculate_forward_head_position in side_metrics.py)

    Drawn elements:
      • Solid cyan line   : ear_px → shoulder_px
      • Dashed white line : ear_px → vert_pt  (vertical reference)
      • Filled arc at shoulder_px marking the CV angle
      • Dot at ear_px and shoulder_px
      • Label "FHP: N.N°" near the shoulder
    """
    # ── Landmark selection ────────────────────────────────────────────────
    _LEFT_EAR       = 7
    _RIGHT_EAR      = 8
    _LEFT_SHOULDER  = 11
    _RIGHT_SHOULDER = 12

    if facing_direction == "left":
        ear_lm = lms[_RIGHT_EAR]
    else:
        ear_lm = lms[_LEFT_EAR]

    # ── Pixel coordinates ────────────────────────────────────────────────
    # Ear: visible-side ear landmark
    # C7 proxy: midpoint of BOTH shoulders — this is the correct anatomical
    #           reference for C7 and matches calculate_fhd() in side_metrics.py.
    #           Using a single-side shoulder produces a laterally-offset endpoint.
    ear_px  = np.array([ear_lm.x * img_w,  ear_lm.y * img_h], dtype=float)

    l_sh_lm = lms[_LEFT_SHOULDER]
    r_sh_lm = lms[_RIGHT_SHOULDER]
    sh_px   = np.array([
        (l_sh_lm.x + r_sh_lm.x) / 2 * img_w,
        (l_sh_lm.y + r_sh_lm.y) / 2 * img_h,
    ], dtype=float)

    vert_pt = np.array([ear_px[0], sh_px[1]], dtype=float)

    COLOR_FHP  = (230, 160, 0)    # blue-ish (BGR)
    COLOR_VERT = (200, 200, 200)  # light grey

    # ── Lines ──────────────────────────────────────────────────────────────
    # Solid: ear → shoulder
    cv2.line(img, _to_int(ear_px), _to_int(sh_px), COLOR_FHP, 2, cv2.LINE_AA)

    # Dashed vertical reference: ear → vert_pt
    _draw_dashed_line(img, _to_int(ear_px), _to_int(vert_pt),
                      COLOR_VERT, thickness=1, dash=8, gap=5)

    # ── Dots ───────────────────────────────────────────────────────────────
    for pt in (ear_px, sh_px):
        cv2.circle(img, _to_int(pt), 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, _to_int(pt), 5, COLOR_FHP,    -1, cv2.LINE_AA)

    # ── Arc at shoulder vertex ─────────────────────────────────────────────
    # Vectors FROM shoulder TOWARD ear and toward vert_pt
    try:
        v_ear  = _unit(ear_px  - sh_px)
        v_vert = _unit(vert_pt - sh_px)
        _draw_angle_arc(img, sh_px, v_ear, v_vert,
                        cv_angle_deg, "cv", COLOR_FHP, arc_r=30)
    except Exception:
        pass

    # ── Label ──────────────────────────────────────────────────────────────
    label = f"FHP: {cv_angle_deg:.1f}"
    lx = int(sh_px[0]) + (10 if facing_direction == "left" else -130)
    ly = int(sh_px[1]) - 12
    cv2.putText(img, label, (lx + 1, ly + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FHP, 1, cv2.LINE_AA)


# ─── FHD overlay ─────────────────────────────────────────────────────────────

# Severity → BGR colour
_FHD_COLORS = {
    "Normal":   (60,  200,  60),    # green
    "Mild":     (0,   210, 255),    # yellow
    "Moderate": (0,   140, 255),    # orange
    "Severe":   (40,   40, 220),    # red
}


def draw_fhd_on_image(
    image_bgr: np.ndarray,
    fhd_result: dict,
    facing_direction: str = "right",
) -> np.ndarray:
    """
    Draw Forward Head Distance construction on a copy of *image_bgr*.
    Reads pre-computed values from the dict returned by calculate_fhd().
    Returns the annotated image (does not modify the original).
    """
    img = image_bgr.copy()
    _draw_fhd_overlay(img, fhd_result, facing_direction)
    return img


def _draw_fhd_overlay_from_result(img: np.ndarray, r) -> None:
    """
    Thin wrapper: pull FHD fields from an AnalysisResult and delegate.
    Called at the end of draw_overlay() when FHD data is present.
    """
    if getattr(r, "fhd_inches", None) is None:
        return
    if r.fhd_ear_point is None or r.fhd_c7_point is None:
        return
    fhd_dict = {
        "fhd_pixels":     r.fhd_pixels,
        "fhd_cm":         r.fhd_cm,
        "fhd_inches":     r.fhd_inches,
        "spine_load_lbs": r.spine_load_lbs,
        "ear_point":      r.fhd_ear_point,
        "c7_point":       r.fhd_c7_point,
        "severity":       r.fhd_severity,
    }
    _draw_fhd_overlay(img, fhd_dict, getattr(r, "facing_direction", "right"))


def _draw_fhd_overlay(
    img: np.ndarray,
    fhd: dict,
    facing_direction: str,
) -> None:
    """
    Draw FHD construction lines onto *img* (in-place).

    Visual elements
    ---------------
    1. Vertical reference line  — dashed white, from C7-proxy straight down
       to the height of the ear (or to a fixed length), showing the "plumb line"
    2. Horizontal distance arrow — solid coloured line from C7-proxy x to ear x,
       at ear height, with arrow tips at both ends
    3. C7 dot — white filled circle at the shoulder midpoint
    4. Ear dot — coloured filled circle at the ear landmark
    5. Label box — "FHD: N.N in  |  Mild  |  XX lbs" near the ear,
       background-filled for readability
    """
    if fhd.get("fhd_inches") is None:
        return

    ear_pt  = fhd["ear_point"]    # (x, y) int
    c7_pt   = fhd["c7_point"]     # (x, y) int
    severity = fhd.get("severity", "Normal")
    color   = _FHD_COLORS.get(severity, _FHD_COLORS["Normal"])

    ear_x, ear_y = ear_pt
    c7_x,  c7_y  = c7_pt

    # ── 1. Vertical plumb line from C7 down to ear height ─────────────────
    plumb_top = (c7_x, min(c7_y, ear_y))
    plumb_bot = (c7_x, max(c7_y, ear_y))
    _draw_dashed_line(img, plumb_top, plumb_bot,
                      (220, 220, 220), thickness=1, dash=8, gap=5)

    # ── 2. Horizontal distance indicator at ear height ─────────────────────
    h_left  = (min(c7_x, ear_x), ear_y)
    h_right = (max(c7_x, ear_x), ear_y)

    if abs(ear_x - c7_x) > 6:
        cv2.line(img, h_left, h_right, color, 2, cv2.LINE_AA)
        # Arrow tips (short diagonal ticks)
        tip_len = 7
        for px, direction in [(h_left[0], +1), (h_right[0], -1)]:
            cv2.line(img, (px, ear_y),
                     (px + direction * tip_len, ear_y - tip_len),
                     color, 2, cv2.LINE_AA)
            cv2.line(img, (px, ear_y),
                     (px + direction * tip_len, ear_y + tip_len),
                     color, 2, cv2.LINE_AA)

    # ── 3. C7 dot ──────────────────────────────────────────────────────────
    cv2.circle(img, c7_pt,  7, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, c7_pt,  5, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.putText(img, "C7", (c7_x + 6, c7_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # ── 4. Ear dot ─────────────────────────────────────────────────────────
    cv2.circle(img, ear_pt, 7, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, ear_pt, 5, color,        -1, cv2.LINE_AA)

    # ── 5. Label ───────────────────────────────────────────────────────────
    fhd_in  = fhd["fhd_inches"]
    sp_load = fhd["spine_load_lbs"]
    line1   = f"FHD: {fhd_in:.1f} in  ({fhd['fhd_cm']:.1f} cm)"
    line2   = f"{severity}  |  Spine load: {sp_load:.0f} lbs"

    # Place label to the left or right depending on facing direction
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness  = 1
    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
    (w2, _),  _ = cv2.getTextSize(line2, font, font_scale, thickness)
    box_w = max(w1, w2) + 10
    box_h = h1 * 2 + 16

    if facing_direction == "left":
        bx = ear_x + 12
    else:
        bx = ear_x - box_w - 12
    by = ear_y - box_h // 2

    # Dark background for legibility
    overlay = img.copy()
    cv2.rectangle(overlay, (bx - 4, by - 4),
                  (bx + box_w, by + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    # Text lines
    cv2.putText(img, line1, (bx, by + h1 + 2),
                font, font_scale, color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, line1, (bx, by + h1 + 2),
                font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
    cv2.putText(img, line2, (bx, by + h1 * 2 + 10),
                font, font_scale, color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, line2, (bx, by + h1 * 2 + 10),
                font, font_scale, color, thickness, cv2.LINE_AA)
