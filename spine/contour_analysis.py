"""
contour_analysis.py — Find A/B endpoints, curvature-based anatomical markers,
and handle lumbar occlusion from the buttock protrusion.

Coordinate convention:
  - Image x increases LEFT→RIGHT
  - Image y increases TOP→BOTTOM
  - Person faces RIGHT → back contour is on the LEFT (smaller x values)
  - "Outward" (dorsal direction for thoracic kyphosis) = more negative x = leftward

Key outputs:
  A        : back-contour point at shoulder Y level (top of curve)
  B        : back-contour point at hip Y level (bottom of curve)
  apex_K   : thoracic apex ~T4/T5  — leftmost (most-outward) point in upper region
  inflect  : thoracolumbar inflection ~T12/L1 — curvature sign change
  apex_L   : lumbar apex ~L3/L4  — rightmost (most-inward) point in lower region
  S_bottom_virtual : extrapolated lumbar lower endpoint if buttock occludes B
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from typing import Optional

import config


# ─── Contour smoothing ────────────────────────────────────────────────────────

def smooth_back_contour(contour: np.ndarray, smooth_factor: float) -> np.ndarray:
    """
    Smooth the back contour using a cubic UnivariateSpline: x = f(y).
    - contour: Nx2 array of [x, y] points along the back curve, ordered A -> B
    - smooth_factor: scales the spline's s= parameter as s = N * smooth_factor;
      larger = smoother. 0 leaves the contour unchanged.
    Returns a smoothed Nx2 contour with the same number of points.
    Endpoints are re-pinned to the original A and B to prevent drift.

    UnivariateSpline is used instead of splprep because the back contour has
    monotone y (one point per image row), making x = f(y) well-posed. splprep
    treats x and y as equal parametric components: on a nearly-vertical contour
    the large y-variation consumes the s= budget, leaving x unregulated and
    producing oscillations rather than smoothing.
    """
    n = len(contour)
    if n < 4 or smooth_factor == 0.0:
        return contour.copy()

    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)

    # UnivariateSpline requires strictly increasing knot positions.
    # Deduplicate rows with the same y (rare but possible at mask boundaries).
    _, unique_idx = np.unique(y, return_index=True)
    if len(unique_idx) < 4:
        return contour.copy()
    y_u, x_u = y[unique_idx], x[unique_idx]

    try:
        spl = UnivariateSpline(y_u, x_u, s=n * smooth_factor, k=3)
    except Exception:
        return contour.copy()

    x_s = spl(y)

    # Re-anchor endpoints exactly to original A and B
    x_s[0]  = x[0]
    x_s[-1] = x[-1]

    return np.column_stack([x_s, y]).astype(np.float32)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def arc_lengths(contour: np.ndarray) -> np.ndarray:
    """Return cumulative arc-length array (shape N) starting from 0."""
    diffs = np.diff(contour, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate(([0.0], np.cumsum(seg_lens)))


def _find_nearest_y_idx(contour: np.ndarray, target_y: float) -> int:
    """Return the index in *contour* whose y-value is closest to *target_y*."""
    return int(np.argmin(np.abs(contour[:, 1] - target_y)))


def _find_contour_point_at_y(
    contour: np.ndarray, target_y: float, prefer: str = "leftmost"
) -> tuple[np.ndarray, int]:
    """
    Find the point on *contour* whose y-coordinate equals (or is nearest to)
    *target_y*.  If multiple rows match (within ±1 px), pick the leftmost
    (back-contour convention) or the one specified by *prefer*.

    Returns (point_xy, index).
    Logs a warning (via return) when multiple candidates exist.
    """
    ys = contour[:, 1]
    tol = 1.0
    candidates = np.where(np.abs(ys - target_y) <= tol)[0]

    if candidates.size == 0:
        idx = _find_nearest_y_idx(contour, target_y)
        return contour[idx], idx

    if prefer == "leftmost":
        # Most outward = smallest x
        best = candidates[np.argmin(contour[candidates, 0])]
    else:
        best = candidates[len(candidates) // 2]

    return contour[best], int(best)


# ─── Endpoints A and B ────────────────────────────────────────────────────────

def find_ab_endpoints(
    contour: np.ndarray,
    shoulder_y: float,
    hip_y: float,
) -> tuple[np.ndarray, int, np.ndarray, int]:
    """
    Project Shoulder and Hip Y-coordinates onto the back contour.

    Returns (A, idx_A, B, idx_B) where A is near shoulder_y (top) and
    B is near hip_y (bottom).
    """
    A, idx_A = _find_contour_point_at_y(contour, shoulder_y)
    B, idx_B = _find_contour_point_at_y(contour, hip_y)

    # Ensure A is above B (smaller y index)
    if idx_A > idx_B:
        idx_A, idx_B = idx_B, idx_A
        A, B = B, A

    return A, idx_A, B, idx_B


# ─── Curvature computation ────────────────────────────────────────────────────

def compute_curvature(
    contour: np.ndarray, smooth_window: int | None = None
) -> np.ndarray:
    """
    Compute the signed curvature κ at each point of *contour* using the
    standard formula:  κ = (x'·y'' - y'·x'') / (x'² + y'²)^(3/2)

    Positive κ = curves to the LEFT (back bulges out = thoracic kyphosis).
    Negative κ = curves to the RIGHT (back curves in = lumbar lordosis).

    A Savitzky-Golay filter is applied to x(y) before differentiation for
    numerical stability.
    """
    n = len(contour)
    if smooth_window is None:
        smooth_window = max(5, int(n * config.CURVATURE_SMOOTH_FRAC) | 1)
    smooth_window = min(smooth_window, n - 2 if n % 2 == 0 else n - 1)
    smooth_window = max(5, smooth_window | 1)

    xs = contour[:, 0]
    ys = contour[:, 1]

    # Smooth x as a function of arc-index
    xs_s = savgol_filter(xs, window_length=smooth_window, polyorder=3)

    # Parametric derivatives w.r.t. index t
    dx  = np.gradient(xs_s)
    dy  = np.gradient(ys)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-8, 1e-8, denom)
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa


# ─── Anatomical markers ───────────────────────────────────────────────────────

def find_anatomical_markers(
    contour: np.ndarray,
    idx_A: int,
    idx_B: int,
) -> dict:
    """
    Locate apex_K, inflect, and apex_L on the A→B sub-contour.

    Strategy:
      1. Split A→B at 50% arc-length (rough thoracic/lumbar split).
      2. In the upper half (±APEX_SEARCH_WINDOW_FRAC), find the point with
         minimum x (most outward = thoracic apex_K).
      3. Compute signed curvature along A→B and find the sign-change point
         closest to the 50% split → inflect (thoracolumbar junction).
      4. In the lower half (±APEX_SEARCH_WINDOW_FRAC), find the point with
         maximum x (most inward = lumbar apex_L).

    Returns a dict with keys:
      apex_K, idx_apex_K,
      inflect, idx_inflect,
      apex_L, idx_apex_L
    """
    sub = contour[idx_A : idx_B + 1]
    n = len(sub)
    if n < 10:
        raise ValueError("A→B sub-contour too short for marker detection.")

    # Arc-length fractions for search windows
    arc = arc_lengths(sub)
    total = arc[-1]
    fracs = arc / total if total > 0 else np.linspace(0, 1, n)

    w = config.APEX_SEARCH_WINDOW_FRAC / 2   # half-window

    # ── apex_K: leftmost x in upper third ─────────────────────────────────
    upper_mask = fracs <= (0.33 + w)
    if upper_mask.sum() < 3:
        upper_mask = np.ones(n, bool)
    local_x = sub[:, 0].copy()
    local_x[~upper_mask] = np.inf
    idx_apex_K_local = int(np.argmin(local_x))

    # ── inflect: curvature sign change near the 50% arc split ─────────────
    kappa = compute_curvature(sub)

    # Search for sign change of kappa in middle 20–80% of arc
    mid_mask = (fracs >= 0.20) & (fracs <= 0.80)
    mid_indices = np.where(mid_mask)[0]

    # Collect ALL sign-change candidates in the mid region, then pick the one
    # closest to the 50% arc position.  Taking the first one is wrong when a
    # spurious sign change from the thoracic apex impulse appears before the
    # true thoracolumbar junction.
    mid_half_idx = int(np.searchsorted(fracs, 0.50))   # index of ~50% arc point
    candidates = []
    if len(mid_indices) > 1:
        for j in mid_indices[:-1]:
            if kappa[j] * kappa[j + 1] < 0:
                t = abs(kappa[j]) / (abs(kappa[j]) + abs(kappa[j + 1]) + 1e-12)
                cand = j if t <= 0.5 else j + 1
                candidates.append(cand)

    if candidates:
        # Pick the candidate closest to the 50% arc position
        inflect_idx_local = min(candidates, key=lambda c: abs(c - mid_half_idx))
    else:
        # Fallback: minimum absolute curvature near midpoint
        mid_kappa = np.abs(kappa)
        mid_kappa[~mid_mask] = np.inf
        inflect_idx_local = int(np.argmin(mid_kappa))

    # ── apex_L: rightmost x in lower region ───────────────────────────────
    lower_mask = fracs >= (0.66 - w)
    if lower_mask.sum() < 3:
        lower_mask = np.ones(n, bool)
    local_x2 = sub[:, 0].copy()
    local_x2[~lower_mask] = -np.inf
    idx_apex_L_local = int(np.argmax(local_x2))

    # Map local indices back to full contour indices
    offset = idx_A
    return {
        "apex_K":       sub[idx_apex_K_local],
        "idx_apex_K":   idx_apex_K_local + offset,
        "inflect":      sub[inflect_idx_local],
        "idx_inflect":  inflect_idx_local + offset,
        "apex_L":       sub[idx_apex_L_local],
        "idx_apex_L":   idx_apex_L_local + offset,
    }


# ─── Buttock occlusion handling ───────────────────────────────────────────────

def find_reliable_lumbar_end(
    contour: np.ndarray,
    idx_inflect: int,
    idx_B: int,
    image_height: int,
) -> tuple[int, bool, np.ndarray | None]:
    """
    Scan upward from B to find the last reliable lumbar point before the
    buttock protrusion causes the contour to swing outward (rightward).

    Heuristic:
      - Compute dx/dy (horizontal change per vertical pixel) along the
        lower portion of the lumbar sub-contour (inflect→B).
      - Scan from B upward; the first point where dx/dy exceeds
        BUTTOCK_SLOPE_CHANGE_THRESHOLD (contour rapidly going left→right
        = buttock bulge) marks the occlusion start.
      - Everything BELOW (higher y) that point is discarded.
      - The reliable end is the point just above that transition.

    Returns:
      (idx_reliable_end,  is_extrapolated,  virtual_B_point_or_None)

    If the extrapolation distance exceeds LUMBAR_EXTRAPOLATION_MAX_FRAC
    of image height, the second return value is True (low confidence).
    """
    sub = contour[idx_inflect : idx_B + 1]
    n = len(sub)

    # Default: no occlusion
    reliable_end_local = n - 1
    occlusion_found = False

    # We need at least a few points to scan
    if n >= 5:
        # dxdy: how much x changes per unit y, smoothed with a small window
        xs = sub[:, 0]
        ys = sub[:, 1]
        dy = np.gradient(ys)
        dx = np.gradient(xs)
        # Avoid divide-by-zero; sign: positive = rightward as y increases
        dxdy = np.where(np.abs(dy) > 0.5, dx / (np.abs(dy) + 1e-6), 0.0)

        # Smooth to avoid single-pixel spikes
        win = max(3, min(11, n // 4) | 1)
        dxdy_s = savgol_filter(dxdy, window_length=win, polyorder=1)

        # Scan from bottom (index n-1) upward; find the first point
        # where dxdy_s exceeds the threshold (contour going rightward)
        thresh = config.BUTTOCK_SLOPE_CHANGE_THRESHOLD
        for i in range(n - 1, 0, -1):
            if dxdy_s[i] > thresh:
                occlusion_found = True
                reliable_end_local = i - 1
                break

    idx_reliable_end = idx_inflect + reliable_end_local
    reliable_pt = contour[idx_reliable_end]

    if not occlusion_found:
        return idx_reliable_end, False, None

    # Extrapolation: fit a line to the last few reliable points and extend
    # to the y-level of B (contour[idx_B])
    seg_start = max(0, reliable_end_local - max(4, int(n * 0.15)))
    seg = sub[seg_start : reliable_end_local + 1]
    if len(seg) < 2:
        return idx_reliable_end, False, None

    # Fit x = m*y + b  (y is roughly monotone → safe as predictor)
    coeffs = np.polyfit(seg[:, 1], seg[:, 0], 1)
    target_y = contour[idx_B][1]
    virtual_x = np.polyval(coeffs, target_y)
    virtual_B = np.array([virtual_x, target_y], dtype=np.float32)

    # Distance check
    extrap_dist = abs(target_y - reliable_pt[1])
    low_conf = extrap_dist / image_height > config.LUMBAR_EXTRAPOLATION_MAX_FRAC

    return idx_reliable_end, low_conf, virtual_B


# ─── Arc-length helpers for L5 estimation ────────────────────────────────────

def _calc_arc_length(contour: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Sum of Euclidean distances between consecutive contour points [start_idx, end_idx]."""
    seg = contour[start_idx:end_idx + 1].astype(np.float64)
    if len(seg) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))


def _contour_point_at_arc(
    contour: np.ndarray, start_idx: int, target_arc: float
) -> np.ndarray:
    """
    Walk along contour from start_idx, accumulating arc length, and return
    the interpolated point at exactly target_arc distance.
    Clamped to the last contour point if target_arc exceeds total arc.
    """
    pts = contour[start_idx:].astype(np.float64)
    accumulated = 0.0
    for i in range(1, len(pts)):
        seg_len = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if accumulated + seg_len >= target_arc:
            t = (target_arc - accumulated) / seg_len if seg_len > 1e-9 else 0.0
            return (pts[i - 1] + t * (pts[i] - pts[i - 1])).astype(np.float32)
        accumulated += seg_len
    # target_arc exceeded contour — clamp to last point
    return pts[-1].astype(np.float32)


# ─── Estimated L5 position (sagittal-view, hip-landmark method) ───────────────

def estimate_l5_position(
    back_contour: np.ndarray,
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    apex_L: Optional[np.ndarray] = None,
    idx_apex_L: int = -1,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate the L5 anchor point on the dorsal surface.

    Primary method (when apex_L and idx_apex_L are provided):
        1. Project shoulder_y and hip_y onto the contour (nearest-y points).
        2. Compute spine_arc = arc length from shoulder_proj to hip_proj.
        3. Compute apex_arc  = arc length from shoulder_proj to apex_L.
        4. l5_arc = apex_arc + 0.16 * spine_arc  (1.5 lumbar segments ≈ 16%
           of T3→L4/L5 span; based on: hip ≈ L4/L5, shoulder ≈ T2/T3,
           that span covers ~14 vertebral levels; 1.5 / 14 ≈ 0.107, rounded
           to 0.16 to account for the larger size of lumbar vertebrae).
        5. Walk that arc distance from shoulder_proj along the contour.

    Fallback (apex_L unavailable):
        target_y = hip_y − L5_ABOVE_HIP_FRAC * torso_height

    Parameters
    ----------
    back_contour : (N, 2) ndarray, ordered top → bottom
    shoulder     : (x, y) pixel coords of the mid-shoulder landmark
    hip          : (x, y) pixel coords of the mid-hip landmark
    apex_L       : (2,) ndarray pixel coords of the lumbar apex (L3/L4), or None
    idx_apex_L   : contour index of apex_L (-1 = unknown → triggers fallback)

    Returns
    -------
    (l5_point, shoulder_proj, hip_proj)
        l5_point      : (2,) float32 — estimated L5 position on the contour
        shoulder_proj : (2,) float32 — contour projection of shoulder landmark
        hip_proj      : (2,) float32 — contour projection of hip landmark
        (projections are None when the fallback path is used)
    """
    if apex_L is not None and idx_apex_L >= 0:
        # ── Primary: arc-length method ─────────────────────────────────────
        ys = back_contour[:, 1].astype(np.float64)

        # 1. Project shoulder and hip onto contour by nearest y
        shoulder_idx = int(np.argmin(np.abs(ys - float(shoulder[1]))))
        hip_idx      = int(np.argmin(np.abs(ys - float(hip[1]))))

        # Ensure shoulder_idx < apex_idx < hip_idx (contour is top→bottom)
        shoulder_idx = min(shoulder_idx, idx_apex_L)
        hip_idx      = max(hip_idx, idx_apex_L)

        shoulder_proj = back_contour[shoulder_idx].astype(np.float32)
        hip_proj      = back_contour[hip_idx].astype(np.float32)

        # 2–3. Arc lengths from shoulder_proj
        spine_arc = _calc_arc_length(back_contour, shoulder_idx, hip_idx)
        apex_arc  = _calc_arc_length(back_contour, shoulder_idx, idx_apex_L)

        # 4–5. Walk 16% of spine_arc past the apex
        l5_arc = apex_arc + config.L5_SPINE_ARC_FRAC * spine_arc
        l5_pt  = _contour_point_at_arc(back_contour, shoulder_idx, l5_arc)

        return l5_pt, shoulder_proj, hip_proj

    # ── Fallback: torso-height fraction ────────────────────────────────────
    torso_h  = float(hip[1] - shoulder[1])
    target_y = float(hip[1]) - config.L5_ABOVE_HIP_FRAC * torso_h
    diffs    = np.abs(back_contour[:, 1].astype(np.float64) - target_y)
    raw_pt   = back_contour[int(np.argmin(diffs))].astype(np.float32)
    return raw_pt, None, None
