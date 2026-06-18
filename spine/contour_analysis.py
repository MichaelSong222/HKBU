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

import config


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
