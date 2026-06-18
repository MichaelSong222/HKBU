"""
angles.py — Arc-length-fraction endpoint location + tangent/normal Cobb-style angle.

Clinical geometry overview
──────────────────────────
Standard Cobb angle is defined by perpendiculars (normals) to the superior and
inferior endplates of the tilted vertebral column.  We approximate endplate
direction with a tangent line fitted to a SHORT local window of the back
contour centred on each of four anatomical target points.

The angle between two NORMAL lines equals the angle between the two TANGENT
lines (both pairs of lines differ only by a 90° rotation, which preserves the
enclosed angle).  We compute it from the tangent directions for numerical
convenience, but the result is the Cobb-style normal-intersection angle.

Four target endpoints, each located by an arc-length fraction within a
sub-segment defined by an apex and a boundary:

  thoracic_upper  (~T2/T3)   : apex_K → A,       frac = THORACIC_UPPER_FRAC
  thoracic_lower  (~T10/T11) : apex_K → inflect,  frac = THORACIC_LOWER_FRAC
  lumbar_upper    (~L2)      : apex_L → inflect,  frac = LUMBAR_UPPER_FRAC
  lumbar_lower    (~L5)      : apex_L → B,        frac = LUMBAR_LOWER_FRAC

Constraints (anatomical sanity):
  thoracic_lower index must stay ABOVE (≤) idx_inflect  (clamped if needed)
  lumbar_upper   index must stay BELOW (≥) idx_inflect  (clamped if needed)

Thoracic Kyphosis angle  X = angle between tangents at thoracic_upper & thoracic_lower
Lumbar Lordosis angle    Y = angle between tangents at lumbar_upper   & lumbar_lower
"""

import math
import numpy as np

import config
from spine.contour_analysis import arc_lengths


# ─── Arc-length helpers ───────────────────────────────────────────────────────

def _arc_length_index(
    contour: np.ndarray,
    idx_from: int,
    idx_to: int,
    frac: float,
) -> int:
    """
    Return the contour index that lies at arc-length fraction *frac* of the
    sub-segment contour[idx_from : idx_to+1], measured FROM idx_from TOWARD
    idx_to.

    frac=0.0 → idx_from,  frac=1.0 → idx_to.
    The index is chosen as the point whose cumulative arc-length is closest to
    frac * total_arc_of_sub_segment.
    """
    # Handle reversed sub-segments (idx_from > idx_to) by always walking low→high
    lo, hi = (idx_from, idx_to) if idx_from <= idx_to else (idx_to, idx_from)
    sub = contour[lo : hi + 1]
    if len(sub) < 2:
        return lo

    arcs = arc_lengths(sub)          # cumulative, starts at 0
    target = frac * arcs[-1]
    local_idx = int(np.argmin(np.abs(arcs - target)))

    # Map back to absolute contour index, respecting original direction
    if idx_from <= idx_to:
        return lo + local_idx
    else:
        # Walking from hi→lo; local_idx=0 corresponds to hi, etc.
        return hi - local_idx


# ─── Tangent fitting ──────────────────────────────────────────────────────────

def _fit_tangent_at(
    contour: np.ndarray,
    idx: int,
    half_window: int | None = None,
) -> np.ndarray:
    """
    Fit a tangent to *contour* centred on *idx* using a symmetric local window
    of ±half_window points (clamped to contour bounds).

    Fitting strategy: polyfit x = f(y) avoids the infinite-slope singularity
    for near-vertical lines.  Returns a unit direction vector (dx, dy) in
    image-pixel space (dy > 0 means pointing downward).
    """
    hw = half_window if half_window is not None else config.TANGENT_LOCAL_HALF_WINDOW
    hw = max(hw, config.TANGENT_MIN_POINTS // 2)

    lo = max(0, idx - hw)
    hi = min(len(contour) - 1, idx + hw)
    seg = contour[lo : hi + 1]

    if len(seg) < 2:
        # Degenerate: use nearest neighbours
        lo2 = max(0, idx - 1)
        hi2 = min(len(contour) - 1, idx + 1)
        seg = contour[lo2 : hi2 + 1]

    ys = seg[:, 1].astype(float)
    xs = seg[:, 0].astype(float)

    if np.ptp(ys) < 1e-6:
        # Perfectly horizontal — tangent points right
        return np.array([1.0, 0.0])

    m, _ = np.polyfit(ys, xs, 1)
    # Direction vector: increasing y (downward) → (m·Δy, Δy), i.e. (m, 1) normalised
    vec = np.array([m, 1.0])
    return vec / (np.linalg.norm(vec) + 1e-12)


def _angle_between_lines_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Return the acute angle (0–90°) between two lines defined by direction
    vectors v1 and v2.

    Uses the identity:
        θ = arctan2(|v1 × v2|, |v1 · v2|)

    The angle between two NORMAL lines equals the angle between the two
    TANGENT lines (rotating both by 90° preserves the enclosed angle), so
    this function accepts either pair and returns the same Cobb-style value.
    """
    n1 = v1 / (np.linalg.norm(v1) + 1e-12)
    n2 = v2 / (np.linalg.norm(v2) + 1e-12)

    cross = abs(float(n1[0] * n2[1] - n1[1] * n2[0]))
    dot   = abs(float(np.dot(n1, n2)))

    return math.degrees(math.atan2(cross, dot))


# ─── Main angle computation ───────────────────────────────────────────────────

def compute_angles(
    contour: np.ndarray,
    idx_A: int,
    idx_B: int,
    idx_apex_K: int,
    idx_inflect: int,
    idx_apex_L: int,
    idx_reliable_lumbar_end: int,
    virtual_B: np.ndarray | None,
) -> dict:
    """
    Compute thoracic kyphosis angle X and lumbar lordosis angle Y using the
    arc-length-fraction endpoint method.

    Endpoint location
    -----------------
    Each endpoint index is found by walking a fraction of the arc length of a
    sub-segment FROM the relevant apex TOWARD a boundary point:

      thoracic_upper : apex_K → A,             frac = config.THORACIC_UPPER_FRAC
      thoracic_lower : apex_K → inflect,        frac = config.THORACIC_LOWER_FRAC
      lumbar_upper   : apex_L → inflect,        frac = config.LUMBAR_UPPER_FRAC
      lumbar_lower   : apex_L → reliable_B/B,   frac = config.LUMBAR_LOWER_FRAC

    Anatomical constraints (clamped)
    ---------------------------------
      thoracic_lower must be ≤ idx_inflect   (stays in thoracic region)
      lumbar_upper   must be ≥ idx_inflect   (stays in lumbar region)

    Tangent & angle
    ---------------
    A tangent is fitted to a symmetric local window centred on each endpoint.
    Thoracic angle X = angle between the two thoracic tangents.
    Lumbar angle Y   = angle between the two lumbar tangents.
    (Tangent-angle == Normal-angle by the 90°-rotation identity.)

    Returns dict with:
      thoracic_angle_deg, lumbar_angle_deg,
      T1 (thoracic_upper tangent), T2 (thoracic_lower tangent),
      T3 (lumbar_upper tangent),   T4 (lumbar_lower tangent),
      anchor_T1_idx, anchor_T2_idx, anchor_T3_idx, anchor_T4_idx,
      ep_thoracic_upper, ep_thoracic_lower,  (contour xy points)
      ep_lumbar_upper,   ep_lumbar_lower,
      lumbar_lower_extrapolated  (bool — True when virtual_B was used)
    """
    # ── 1. Thoracic upper endpoint: apex_K → A ───────────────────────────
    t1_idx = _arc_length_index(
        contour, idx_apex_K, idx_A, config.THORACIC_UPPER_FRAC
    )

    # ── 2. Thoracic lower endpoint: apex_K → inflect ─────────────────────
    t2_idx_raw = _arc_length_index(
        contour, idx_apex_K, idx_inflect, config.THORACIC_LOWER_FRAC
    )
    # Constraint: must stay above (≤) inflect
    t2_idx = min(t2_idx_raw, idx_inflect)

    # ── 3. Lumbar upper endpoint: apex_L → inflect ────────────────────────
    t3_idx_raw = _arc_length_index(
        contour, idx_apex_L, idx_inflect, config.LUMBAR_UPPER_FRAC
    )
    # Constraint: must stay below (≥) inflect
    t3_idx = max(t3_idx_raw, idx_inflect)

    # ── 4. Lumbar lower endpoint: apex_L → reliable_B (or virtual_B) ─────
    lumbar_lower_extrapolated = False
    if virtual_B is not None:
        # Build a two-point synthetic segment from apex_L to virtual_B and walk
        # the fraction along that.  We snap to the reliable end for the tangent.
        reliable_pt = contour[idx_reliable_lumbar_end]
        # Linear interpolation along apex_L → virtual_B
        p_apex = contour[idx_apex_L].astype(float)
        p_virt = virtual_B.astype(float)
        frac_pt = p_apex + config.LUMBAR_LOWER_FRAC * (p_virt - p_apex)
        # Use the reliable end index for tangent fitting (no real contour beyond it)
        t4_idx = idx_reliable_lumbar_end
        lumbar_lower_extrapolated = True
        # Build a synthetic contour segment for the tangent: reliable end + virtual_B
        _synth = np.vstack([reliable_pt, virtual_B])
        T4 = _fit_tangent_at(_synth, 0, half_window=1)
    else:
        t4_idx = _arc_length_index(
            contour, idx_apex_L, idx_reliable_lumbar_end, config.LUMBAR_LOWER_FRAC
        )
        T4 = _fit_tangent_at(contour, t4_idx)

    # ── 5. Fit tangents at the three real-contour endpoints ───────────────
    T1 = _fit_tangent_at(contour, t1_idx)
    T2 = _fit_tangent_at(contour, t2_idx)
    T3 = _fit_tangent_at(contour, t3_idx)

    # ── 6. Angles ─────────────────────────────────────────────────────────
    thoracic_angle = _angle_between_lines_deg(T1, T2)
    lumbar_angle   = _angle_between_lines_deg(T3, T4)

    return {
        "thoracic_angle_deg": thoracic_angle,
        "lumbar_angle_deg":   lumbar_angle,
        # Tangent unit vectors (kept as T1–T4 for pipeline/visualization compat)
        "T1": T1, "T2": T2, "T3": T3, "T4": T4,
        "anchor_T1_idx": t1_idx,
        "anchor_T2_idx": t2_idx,
        "anchor_T3_idx": t3_idx,
        "anchor_T4_idx": t4_idx,
        # Endpoint coordinate points (for visualization)
        "ep_thoracic_upper": contour[t1_idx],
        "ep_thoracic_lower": contour[t2_idx],
        "ep_lumbar_upper":   contour[t3_idx],
        "ep_lumbar_lower":   (virtual_B if lumbar_lower_extrapolated
                              else contour[t4_idx]),
        "lumbar_lower_extrapolated": lumbar_lower_extrapolated,
    }


# ─── Classification ───────────────────────────────────────────────────────────

def classify_thoracic(angle_deg: float) -> tuple[str, str]:
    """Return (key, human-readable label) for the thoracic kyphosis angle."""
    th = config.THORACIC_THRESHOLDS
    if angle_deg < th["normal"][0]:
        key = "hypokyphosis"
    elif angle_deg > th["normal"][1]:
        key = "hyperkyphosis"
    else:
        key = "normal"
    return key, config.THORACIC_LABELS[key]


def classify_lumbar(angle_deg: float) -> tuple[str, str]:
    """Return (key, human-readable label) for the lumbar lordosis angle."""
    th = config.LUMBAR_THRESHOLDS
    if angle_deg < th["normal"][0]:
        key = "hypolordosis"
    elif angle_deg > th["normal"][1]:
        key = "hyperlordosis"
    else:
        key = "normal"
    return key, config.LUMBAR_LABELS[key]
