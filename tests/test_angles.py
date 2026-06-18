"""
test_angles.py — Unit tests for the arc-length-fraction angle computation.

Tests use SYNTHETIC back-contour curves with analytically known tangent angles
so we can verify the computed angles are within ±5° of the ground truth.

Coordinate convention (same as production code):
  x increases LEFT→RIGHT (rightward = inward = lordotic direction)
  y increases TOP→BOTTOM
  Back contour runs from A (top, small y) to B (bottom, large y).
  Thoracic kyphosis = back bulges LEFT (decreasing x).
  Lumbar lordosis   = back curves RIGHT (increasing x).

Synthetic contour design
─────────────────────────
We build a SMOOTH sinusoidal back contour so that:
  - The thoracic region (A→inflect) has a known maximum outward deflection
    whose tangent angles at the arc-fraction endpoints can be computed
    analytically.
  - The lumbar region (inflect→B) has the same structure, inward.

For a half-sine bump of amplitude A over a span of L pixels:
    x(t) = x_center ± A·sin(π·t/L),   t in [0, L]
  The slope dx/dt = ±A·π/L·cos(π·t/L).
  At arc-fraction f from the apex toward a boundary:
    The apex is at t=L/2 (dx/dt=0).
    Moving toward the START boundary (t=0) by fraction f of L/2:
      t_ep = L/2 · (1 - f)
      slope at t_ep = ±A·π/L·cos(π·(1-f)/2)
    The tangent direction vector is (slope, 1) normalised.
    The angle between two tangents at fractions f_upper and f_lower is:
      θ = atan(slope_upper) + atan(slope_lower)  [both measured from vertical]

For the tests, rather than computing the exact expected angle through the
full fraction logic, we instead verify:
  1. The computed angle is in the right ballpark (within ±10° for smoothed curves)
  2. A higher-amplitude curve gives a higher angle (monotonicity)
  3. A near-flat curve gives a near-zero angle
  4. Angles stay acute (< 90°)
  5. Classification thresholds are respected

For EXACT ground-truth tests (±5°), we build piecewise-linear contours where
the tangent at any point is precisely known, and pass anchor indices that fall
exactly on the linear segments — bypassing the arc-fraction location step to
isolate the tangent-fitting + angle math.
"""

import sys
import os
import math
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spine.angles import (
    compute_angles,
    classify_thoracic,
    classify_lumbar,
    _fit_tangent_at,
    _angle_between_lines_deg,
    _arc_length_index,
)
from spine.contour_analysis import (
    find_anatomical_markers,
    find_ab_endpoints,
    find_reliable_lumbar_end,
)


# ─── Synthetic contour builders ───────────────────────────────────────────────

def _make_sinusoidal_contour(
    thoracic_amplitude: float,
    lumbar_amplitude: float,
    height: int = 400,
    x_center: float = 200.0,
) -> tuple[np.ndarray, dict]:
    """
    Build a smooth back contour using half-sine bumps.

    Thoracic region (rows 0..H/2): x = x_center - A_th · sin(π·i/(H/2))
      → bulges LEFT (outward) with amplitude A_th at the midpoint (apex_K)

    Lumbar region (rows H/2..H): x = x_center + A_lu · sin(π·(i-H/2)/(H/2))
      → bulges RIGHT (inward) with amplitude A_lu at the midpoint (apex_L)

    Returns (contour, anchors) where anchors has known ground-truth indices.
    """
    H    = height
    H_th = H // 2
    H_lu = H - H_th
    ys   = np.arange(H, dtype=np.float64)
    xs   = np.empty(H, dtype=np.float64)

    for i in range(H_th):
        xs[i] = x_center - thoracic_amplitude * math.sin(math.pi * i / H_th)

    for i in range(H_lu):
        row = H_th + i
        xs[row] = x_center + lumbar_amplitude * math.sin(math.pi * i / H_lu)

    contour = np.column_stack([xs, ys]).astype(np.float32)

    anchors = {
        "idx_A":       0,
        "idx_B":       H - 1,
        "idx_apex_K":  H_th // 2,
        "idx_inflect": H_th,
        "idx_apex_L":  H_th + H_lu // 2,
    }
    return contour, anchors


def _make_piecewise_linear_contour(
    thoracic_angle_deg: float,
    lumbar_angle_deg: float,
    height: int = 400,
    x_center: float = 200.0,
) -> tuple[np.ndarray, dict]:
    """
    Build a piecewise-linear back contour with EXACT known tangent angles at
    each quarter-point (the arc-fraction-derived endpoints for fracs=0.5).

    Construction (thoracic region, rows 0..H/2):
      - Arm 1 (rows 0..H/4):   slope = -tan(θ_th/2)  (tilts left = outward)
      - Arm 2 (rows H/4..H/2): slope = +tan(θ_th/2)  (tilts right back)
    Tangent angle between arm1 and arm2 = θ_th EXACTLY.

    Similarly for lumbar (rows H/2..H), mirrored rightward.

    Light Gaussian smoothing (σ=2) is applied to give the tangent fitter a
    numerically stable signal without changing the slope significantly.

    The quarter-points are the EXACT locations of thoracic_upper, thoracic_lower,
    lumbar_upper, lumbar_lower when THORACIC_UPPER_FRAC = THORACIC_LOWER_FRAC
    = LUMBAR_UPPER_FRAC = LUMBAR_LOWER_FRAC = 0.5, so we pass the quarter-point
    indices directly as anchor indices to compute_angles() — bypassing arc-fraction
    location — to test tangent-fitting + angle math in isolation.
    """
    H    = height
    H_th = H // 2
    H_lu = H - H_th

    th = math.radians(thoracic_angle_deg)
    lu = math.radians(lumbar_angle_deg)
    sl_th = math.tan(th / 2)
    sl_lu = math.tan(lu / 2)

    xs = np.empty(H, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)

    # Thoracic: V going left
    for i in range(H_th):
        if i <= H_th // 2:
            xs[i] = x_center - sl_th * i
        else:
            xs[i] = x_center - sl_th * (H_th - i)

    # Lumbar: V going right
    for i in range(H_lu):
        row = H_th + i
        if i <= H_lu // 2:
            xs[row] = x_center + sl_lu * i
        else:
            xs[row] = x_center + sl_lu * (H_lu - i)

    xs = gaussian_filter1d(xs, sigma=2)
    contour = np.column_stack([xs, ys]).astype(np.float32)

    anchors = {
        "idx_A":             0,
        "idx_B":             H - 1,
        "idx_apex_K":        H_th // 2,
        "idx_inflect":       H_th,
        "idx_apex_L":        H_th + H_lu // 2,
        # Quarter-points: exact endpoints for frac=0.5
        "idx_th_upper":      H_th // 4,           # midpoint of arm1
        "idx_th_lower":      3 * H_th // 4,       # midpoint of arm2
        "idx_lu_upper":      H_th + H_lu // 4,    # midpoint of lumbar arm1
        "idx_lu_lower":      H_th + 3 * H_lu // 4,# midpoint of lumbar arm2
    }
    return contour, anchors


def _run_full_pipeline(
    contour: np.ndarray,
    anchors: dict,
    height: int,
) -> dict:
    """Run compute_angles() with known anchor indices (no auto-detection)."""
    idx_rel, _, virtual_B = find_reliable_lumbar_end(
        contour,
        idx_inflect=anchors["idx_inflect"],
        idx_B=anchors["idx_B"],
        image_height=height,
    )
    return compute_angles(
        contour,
        idx_A=anchors["idx_A"],
        idx_B=anchors["idx_B"],
        idx_apex_K=anchors["idx_apex_K"],
        idx_inflect=anchors["idx_inflect"],
        idx_apex_L=anchors["idx_apex_L"],
        idx_reliable_lumbar_end=idx_rel,
        virtual_B=virtual_B,
    )


# ─── Exact-angle tests (piecewise-linear, anchor indices at quarter-points) ───

class TestExactTangentAngles:
    """
    Test tangent fitting + angle math in isolation by passing known anchor
    indices directly to _fit_tangent_at() and _angle_between_lines_deg().

    For a piecewise-linear arm with slope s = tan(θ/2), the tangent direction
    vector is (±s, 1) normalised; the angle between the two arms is θ exactly.
    """

    TOLERANCE = 5.0   # ±5° tolerance as specified

    def _check_pair(self, contour, idx_u, idx_l, expected_deg):
        T_u = _fit_tangent_at(contour, idx_u)
        T_l = _fit_tangent_at(contour, idx_l)
        got = _angle_between_lines_deg(T_u, T_l)
        assert abs(got - expected_deg) <= self.TOLERANCE, (
            f"Expected {expected_deg:.1f}°, got {got:.1f}° "
            f"(err={abs(got-expected_deg):.1f}°)"
        )

    def test_near_straight_thoracic(self):
        contour, a = _make_piecewise_linear_contour(5.0, 25.0)
        self._check_pair(contour, a["idx_th_upper"], a["idx_th_lower"], 5.0)

    def test_near_straight_lumbar(self):
        contour, a = _make_piecewise_linear_contour(25.0, 5.0)
        self._check_pair(contour, a["idx_lu_upper"], a["idx_lu_lower"], 5.0)

    def test_normal_thoracic(self):
        contour, a = _make_piecewise_linear_contour(30.0, 30.0)
        self._check_pair(contour, a["idx_th_upper"], a["idx_th_lower"], 30.0)

    def test_normal_lumbar(self):
        contour, a = _make_piecewise_linear_contour(25.0, 30.0)
        self._check_pair(contour, a["idx_lu_upper"], a["idx_lu_lower"], 30.0)

    def test_hyperkyphosis(self):
        contour, a = _make_piecewise_linear_contour(55.0, 25.0)
        self._check_pair(contour, a["idx_th_upper"], a["idx_th_lower"], 55.0)

    def test_hyperlordosis(self):
        contour, a = _make_piecewise_linear_contour(25.0, 55.0)
        self._check_pair(contour, a["idx_lu_upper"], a["idx_lu_lower"], 55.0)


# ─── Full pipeline tests (arc-length endpoint + tangent + angle) ──────────────

class TestFullPipeline:
    """
    Test the complete compute_angles() path including arc-length-fraction
    endpoint location, on sinusoidal contours.  Tolerances are looser (±10°)
    because smoothing shifts the effective tangent angle slightly.
    """

    TOLERANCE = 10.0

    def _run(self, th_amp, lu_amp, height=400):
        contour, anchors = _make_sinusoidal_contour(th_amp, lu_amp, height=height)
        result = _run_full_pipeline(contour, anchors, height)
        return result["thoracic_angle_deg"], result["lumbar_angle_deg"]

    def test_near_straight_spine(self):
        th, lu = self._run(5.0, 5.0)
        assert th < 20.0, f"Near-flat thoracic should be small, got {th:.1f}°"
        assert lu < 20.0, f"Near-flat lumbar should be small, got {lu:.1f}°"

    def test_monotone_thoracic(self):
        th_lo, _ = self._run(20.0, 30.0)
        th_hi, _ = self._run(80.0, 30.0)
        assert th_hi > th_lo, (
            f"Larger amplitude should give larger thoracic angle: "
            f"{th_lo:.1f}° vs {th_hi:.1f}°"
        )

    def test_monotone_lumbar(self):
        # Amplitudes must stay below the buttock-occlusion threshold (~55px)
        # so find_reliable_lumbar_end doesn't truncate the lower sub-segment.
        _, lu_lo = self._run(30.0, 20.0)
        _, lu_hi = self._run(30.0, 45.0)
        assert lu_hi > lu_lo, (
            f"Larger amplitude should give larger lumbar angle: "
            f"{lu_lo:.1f}° vs {lu_hi:.1f}°"
        )

    def test_angles_acute(self):
        for th_amp, lu_amp in [(10, 20), (40, 50), (80, 70)]:
            th, lu = self._run(th_amp, lu_amp)
            assert 0 <= th < 90, f"Thoracic {th:.1f}° is not acute"
            assert 0 <= lu < 90, f"Lumbar {lu:.1f}° is not acute"

    def test_hyperkyphosis_classification(self):
        contour, anchors = _make_piecewise_linear_contour(55.0, 25.0)
        result = _run_full_pipeline(contour, anchors, 400)
        th = result["thoracic_angle_deg"]
        assert th > 45.0, f"Expected hyperkyphosis (>45°), got {th:.1f}°"
        key, _ = classify_thoracic(th)
        assert key == "hyperkyphosis"

    def test_hyperlordosis_classification(self):
        contour, anchors = _make_piecewise_linear_contour(25.0, 55.0)
        result = _run_full_pipeline(contour, anchors, 400)
        lu = result["lumbar_angle_deg"]
        assert lu > 40.0, f"Expected hyperlordosis (>40°), got {lu:.1f}°"
        key, _ = classify_lumbar(lu)
        assert key == "hyperlordosis"

    def test_hypolordosis_classification(self):
        contour, anchors = _make_piecewise_linear_contour(25.0, 5.0)
        result = _run_full_pipeline(contour, anchors, 400)
        lu = result["lumbar_angle_deg"]
        assert lu < 20.0, f"Expected hypolordosis (<20°), got {lu:.1f}°"
        key, _ = classify_lumbar(lu)
        assert key == "hypolordosis"


# ─── Arc-length index helper tests ────────────────────────────────────────────

class TestArcLengthIndex:

    def test_frac_zero_returns_from(self):
        contour = np.column_stack([np.zeros(20), np.arange(20)]).astype(np.float32)
        idx = _arc_length_index(contour, 0, 19, 0.0)
        assert idx == 0

    def test_frac_one_returns_to(self):
        contour = np.column_stack([np.zeros(20), np.arange(20)]).astype(np.float32)
        idx = _arc_length_index(contour, 0, 19, 1.0)
        assert idx == 19

    def test_frac_half_midpoint(self):
        contour = np.column_stack([np.zeros(21), np.arange(21)]).astype(np.float32)
        idx = _arc_length_index(contour, 0, 20, 0.5)
        assert abs(idx - 10) <= 1   # allow ±1 for rounding

    def test_reversed_segment(self):
        contour = np.column_stack([np.zeros(20), np.arange(20)]).astype(np.float32)
        idx_fwd = _arc_length_index(contour, 5, 15, 0.5)
        idx_rev = _arc_length_index(contour, 15, 5, 0.5)
        # Both should land near the midpoint of [5, 15]
        assert abs(idx_fwd - 10) <= 1
        assert abs(idx_rev - 10) <= 1


# ─── Endpoint constraint tests ────────────────────────────────────────────────

class TestEndpointConstraints:
    """
    Verify that thoracic_lower stays ≤ inflect and lumbar_upper stays ≥ inflect.
    """

    def test_thoracic_lower_above_inflect(self):
        contour, anchors = _make_sinusoidal_contour(60.0, 40.0, height=400)
        result = _run_full_pipeline(contour, anchors, 400)
        assert result["anchor_T2_idx"] <= anchors["idx_inflect"], (
            f"thoracic_lower {result['anchor_T2_idx']} must be ≤ inflect "
            f"{anchors['idx_inflect']}"
        )

    def test_lumbar_upper_below_inflect(self):
        contour, anchors = _make_sinusoidal_contour(40.0, 60.0, height=400)
        result = _run_full_pipeline(contour, anchors, 400)
        assert result["anchor_T3_idx"] >= anchors["idx_inflect"], (
            f"lumbar_upper {result['anchor_T3_idx']} must be ≥ inflect "
            f"{anchors['idx_inflect']}"
        )


# ─── Result dict shape ────────────────────────────────────────────────────────

class TestResultShape:

    def test_all_keys_present(self):
        contour, anchors = _make_sinusoidal_contour(40.0, 35.0, height=400)
        result = _run_full_pipeline(contour, anchors, 400)
        expected_keys = {
            "thoracic_angle_deg", "lumbar_angle_deg",
            "T1", "T2", "T3", "T4",
            "anchor_T1_idx", "anchor_T2_idx", "anchor_T3_idx", "anchor_T4_idx",
            "ep_thoracic_upper", "ep_thoracic_lower",
            "ep_lumbar_upper",   "ep_lumbar_lower",
            "lumbar_lower_extrapolated",
        }
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_tangent_vectors_are_unit(self):
        contour, anchors = _make_sinusoidal_contour(40.0, 35.0, height=400)
        result = _run_full_pipeline(contour, anchors, 400)
        for key in ("T1", "T2", "T3", "T4"):
            norm = np.linalg.norm(result[key])
            assert abs(norm - 1.0) < 1e-6, f"{key} norm={norm:.6f}, expected 1.0"

    def test_endpoint_coords_shape(self):
        contour, anchors = _make_sinusoidal_contour(40.0, 35.0, height=400)
        result = _run_full_pipeline(contour, anchors, 400)
        for key in ("ep_thoracic_upper", "ep_thoracic_lower",
                    "ep_lumbar_upper", "ep_lumbar_lower"):
            assert result[key].shape == (2,), f"{key} shape wrong: {result[key].shape}"


# ─── Classification tests ─────────────────────────────────────────────────────

class TestClassification:

    def test_thoracic_boundaries(self):
        assert classify_thoracic(10.0)[0] == "hypokyphosis"
        assert classify_thoracic(20.0)[0] == "normal"
        assert classify_thoracic(32.0)[0] == "normal"
        assert classify_thoracic(45.0)[0] == "normal"
        assert classify_thoracic(46.0)[0] == "hyperkyphosis"

    def test_lumbar_boundaries(self):
        assert classify_lumbar(10.0)[0] == "hypolordosis"
        assert classify_lumbar(20.0)[0] == "normal"
        assert classify_lumbar(30.0)[0] == "normal"
        assert classify_lumbar(40.0)[0] == "normal"
        assert classify_lumbar(41.0)[0] == "hyperlordosis"

    def test_labels_are_strings(self):
        for angle in [5, 30, 60]:
            _, label = classify_thoracic(float(angle))
            assert isinstance(label, str) and len(label) > 0
            _, label = classify_lumbar(float(angle))
            assert isinstance(label, str) and len(label) > 0


# ─── Contour marker tests (unchanged logic) ───────────────────────────────────

class TestContourMarkers:

    def _contour(self):
        contour, _ = _make_sinusoidal_contour(50.0, 40.0, height=400)
        return contour

    def test_markers_within_ab(self):
        c = self._contour()
        markers = find_anatomical_markers(c, 0, len(c) - 1)
        for key in ["idx_apex_K", "idx_inflect", "idx_apex_L"]:
            idx = markers[key]
            assert 0 <= idx < len(c), f"{key}={idx} out of range"

    def test_marker_ordering(self):
        c = self._contour()
        markers = find_anatomical_markers(c, 0, len(c) - 1)
        assert markers["idx_apex_K"] < markers["idx_inflect"], \
            "apex_K must be above inflect"
        assert markers["idx_inflect"] < markers["idx_apex_L"], \
            "inflect must be above apex_L"

    def test_apex_k_is_outward(self):
        c = self._contour()
        markers = find_anatomical_markers(c, 0, len(c) - 1)
        assert c[markers["idx_apex_K"], 0] < c[0, 0], \
            "apex_K should be to the left of A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
