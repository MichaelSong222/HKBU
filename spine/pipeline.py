"""
pipeline.py — Orchestrates the full analysis flow and returns an AnalysisResult.

Flow:
  1. Detect shoulder / hip landmarks (pose_detector)
  2. Posture validity gate (vertical deviation check)
  3. Segment silhouette → extract back contour (segmentation)
  4. Find A/B endpoints, anatomical markers, buttock occlusion (contour_analysis)
  5. Fit tangents, compute angles, classify (angles)
  6. Bundle everything into an AnalysisResult dataclass
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import config
from spine.pose_detector import detect_pose, PoseDetectionError
from spine.segmentation import (
    get_segmentation_mask,
    extract_back_contour,
    smooth_contour,
    SegmentationError,
)
from spine.contour_analysis import (
    find_ab_endpoints,
    find_anatomical_markers,
    find_reliable_lumbar_end,
    smooth_back_contour,
    estimate_l5_position,
)
from spine.angles import compute_angles, classify_thoracic, classify_lumbar


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    # ── Input ─────────────────────────────────────────────────────────────
    image_bgr: Optional[np.ndarray] = None   # original image (not annotated)

    # ── Landmarks ─────────────────────────────────────────────────────────
    shoulder: Optional[tuple[float, float]] = None
    hip:      Optional[tuple[float, float]] = None
    vertical_deviation_deg: float = 0.0
    is_vertical: bool = True
    _raw_landmarks: object = field(default=None, repr=False)  # MediaPipe landmark list

    # ── Contour ───────────────────────────────────────────────────────────
    back_contour: Optional[np.ndarray] = None   # (N,2) ordered top→bottom
    A:  Optional[np.ndarray] = None             # top contour point (shoulder Y)
    B:  Optional[np.ndarray] = None             # bottom contour point (hip Y)
    idx_A: int = 0
    idx_B: int = 0

    # ── Anatomical markers ─────────────────────────────────────────────────
    apex_K:  Optional[np.ndarray] = None        # T4/T5
    inflect: Optional[np.ndarray] = None        # T12/L1
    apex_L:  Optional[np.ndarray] = None        # L3/L4
    idx_apex_K:  int = 0
    idx_inflect: int = 0
    idx_apex_L:  int = 0

    # ── Lumbar occlusion ───────────────────────────────────────────────────
    idx_reliable_lumbar_end: int = 0
    virtual_B: Optional[np.ndarray] = None      # extrapolated lower end if occluded

    # ── Estimated L5 (hip-landmark SVD method) ────────────────────────────
    estimated_l5: Optional[np.ndarray] = None   # None if feature off or fallback used
    estimated_l5_used: bool = False             # True when this point replaced virtual_B
    l5_shoulder_proj: Optional[np.ndarray] = None  # contour projection of shoulder landmark
    l5_hip_proj: Optional[np.ndarray] = None       # contour projection of hip landmark

    # ── Tangent vectors (unit 2-D) ────────────────────────────────────────
    T1: Optional[np.ndarray] = None             # upper thoracic tangent
    T2: Optional[np.ndarray] = None             # lower thoracic tangent
    T3: Optional[np.ndarray] = None             # upper lumbar tangent
    T4: Optional[np.ndarray] = None             # lower lumbar tangent / extrapolated
    anchor_T1_idx: int = 0
    anchor_T2_idx: int = 0
    anchor_T3_idx: int = 0
    anchor_T4_idx: int = 0

    # ── Angle endpoint coordinates (for visualization) ────────────────────
    ep_thoracic_upper: Optional[np.ndarray] = None   # ~T2/T3
    ep_thoracic_lower: Optional[np.ndarray] = None   # ~T10/T11
    ep_lumbar_upper:   Optional[np.ndarray] = None   # ~L2
    ep_lumbar_lower:   Optional[np.ndarray] = None   # ~L5
    lumbar_lower_extrapolated: bool = False
    lumbar_fit_curve: Optional[np.ndarray] = None    # (N,2) fitted curve apex_L→L5(est)

    # ── Angles & classification ────────────────────────────────────────────
    thoracic_angle_deg: Optional[float] = None
    lumbar_angle_deg:   Optional[float] = None
    thoracic_class_key:   Optional[str] = None
    thoracic_class_label: Optional[str] = None
    lumbar_class_key:     Optional[str] = None
    lumbar_class_label:   Optional[str] = None

    # ── Confidence flags & warnings ───────────────────────────────────────
    low_confidence: bool = False
    lumbar_low_confidence: bool = False
    warnings: list[str] = field(default_factory=list)

    # ── Error (set if pipeline aborted) ───────────────────────────────────
    error: Optional[str] = None
    success: bool = True


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_analysis(image_bgr: np.ndarray) -> AnalysisResult:
    """
    Run the full spine analysis pipeline on *image_bgr* and return an
    AnalysisResult.  Never raises — errors are captured in result.error.
    """
    result = AnalysisResult(image_bgr=image_bgr)

    # ── Step 1: Pose detection ─────────────────────────────────────────────
    try:
        pose = detect_pose(image_bgr)
    except PoseDetectionError as e:
        result.success = False
        result.error = str(e)
        return result
    except Exception as e:
        result.success = False
        result.error = f"Pose detection failed: {e}"
        return result

    result.shoulder = pose["shoulder"]
    result.hip      = pose["hip"]
    result.vertical_deviation_deg = pose["vertical_deviation_deg"]
    result.is_vertical = pose["is_vertical"]
    result._raw_landmarks = pose["raw_landmarks"]

    if not result.is_vertical:
        result.low_confidence = True
        result.warnings.append(
            f"Body not vertical enough ({result.vertical_deviation_deg:.1f}° off). "
            "Please stand straighter and retake."
        )

    # ── Step 2: Segmentation & back contour ───────────────────────────────
    try:
        mask = get_segmentation_mask(image_bgr)
        raw_contour = extract_back_contour(
            mask,
            shoulder_y=result.shoulder[1],
            hip_y=result.hip[1],
        )
        contour = smooth_contour(raw_contour, window=9)
        if config.CONTOUR_SMOOTH_FACTOR > 0:
            contour = smooth_back_contour(contour, smooth_factor=config.CONTOUR_SMOOTH_FACTOR)
    except SegmentationError as e:
        result.success = False
        result.error = str(e)
        return result
    except Exception as e:
        result.success = False
        result.error = f"Segmentation failed: {e}"
        return result

    result.back_contour = contour

    # ── Step 3: Endpoints A and B ──────────────────────────────────────────
    try:
        A, idx_A, B, idx_B = find_ab_endpoints(
            contour,
            shoulder_y=result.shoulder[1],
            hip_y=result.hip[1],
        )
    except Exception as e:
        result.success = False
        result.error = f"Could not find A/B endpoints: {e}"
        return result

    result.A, result.idx_A = A, idx_A
    result.B, result.idx_B = B, idx_B

    # ── Step 4: Anatomical markers ─────────────────────────────────────────
    try:
        markers = find_anatomical_markers(contour, idx_A, idx_B)
    except Exception as e:
        result.success = False
        result.error = f"Marker detection failed: {e}"
        return result

    result.apex_K    = markers["apex_K"]
    result.idx_apex_K = markers["idx_apex_K"]
    result.inflect   = markers["inflect"]
    result.idx_inflect = markers["idx_inflect"]
    result.apex_L    = markers["apex_L"]
    result.idx_apex_L = markers["idx_apex_L"]

    # ── Step 5: Buttock occlusion ──────────────────────────────────────────
    try:
        img_h = image_bgr.shape[0]
        idx_rel_end, lumbar_low_conf, virtual_B = find_reliable_lumbar_end(
            contour,
            idx_inflect=result.idx_inflect,
            idx_B=idx_B,
            image_height=img_h,
        )
    except Exception as e:
        idx_rel_end   = idx_B
        lumbar_low_conf = False
        virtual_B       = None
        result.warnings.append(f"Buttock occlusion check failed: {e}")

    result.idx_reliable_lumbar_end = idx_rel_end
    result.virtual_B = virtual_B
    result.lumbar_low_confidence = lumbar_low_conf

    if virtual_B is not None:
        result.warnings.append(
            "Lumbar lower end extrapolated (buttock occludes L5/S1)."
        )
    if lumbar_low_conf:
        result.warnings.append(
            "Lumbar angle LOW CONFIDENCE — extrapolation distance too large."
        )

    # ── Step 5b: Estimated L5 (hip-landmark SVD method) ──────────────────
    # Only runs when USE_ESTIMATED_L5 is True and both hip landmarks are
    # visible enough.  On success, estimated_l5 overrides virtual_B for the
    # angle computation below.  The original virtual_B path (slope-based) is
    # preserved byte-for-byte above and used as the fallback.
    if config.USE_ESTIMATED_L5 and result._raw_landmarks is not None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose
        lms = result._raw_landmarks
        lh = lms[_mp_pose.PoseLandmark.LEFT_HIP]
        rh = lms[_mp_pose.PoseLandmark.RIGHT_HIP]
        hips_visible = (
            lh.visibility >= config.L5_HIP_VISIBILITY_MIN
            or rh.visibility >= config.L5_HIP_VISIBILITY_MIN
        )
        if hips_visible:
            try:
                est, shoulder_proj, hip_proj = estimate_l5_position(
                    contour,
                    shoulder=result.shoulder,
                    hip=result.hip,
                    apex_L=result.apex_L,
                    idx_apex_L=result.idx_apex_L,
                )
                result.estimated_l5 = est
                result.estimated_l5_used = True
                result.l5_shoulder_proj = shoulder_proj
                result.l5_hip_proj = hip_proj
                # Replace virtual_B with the estimated point so compute_angles
                # uses it for T4 / ep_lumbar_lower.
                virtual_B = est
            except Exception as e:
                result.warnings.append(f"L5 estimation failed, using fallback: {e}")

    # ── Step 6: Angles ────────────────────────────────────────────────────
    try:
        angle_data = compute_angles(
            contour,
            idx_A=idx_A,
            idx_B=idx_B,
            idx_apex_K=result.idx_apex_K,
            idx_inflect=result.idx_inflect,
            idx_apex_L=result.idx_apex_L,
            idx_reliable_lumbar_end=idx_rel_end,
            virtual_B=virtual_B,
            estimated_l5=result.estimated_l5,
        )
    except Exception as e:
        result.success = False
        result.error = f"Angle computation failed: {e}"
        return result

    result.thoracic_angle_deg = angle_data["thoracic_angle_deg"]
    result.lumbar_angle_deg   = angle_data["lumbar_angle_deg"]
    result.T1 = angle_data["T1"]
    result.T2 = angle_data["T2"]
    result.T3 = angle_data["T3"]
    result.T4 = angle_data["T4"]
    result.anchor_T1_idx = angle_data["anchor_T1_idx"]
    result.anchor_T2_idx = angle_data["anchor_T2_idx"]
    result.anchor_T3_idx = angle_data["anchor_T3_idx"]
    result.anchor_T4_idx = angle_data["anchor_T4_idx"]
    result.ep_thoracic_upper = angle_data["ep_thoracic_upper"]
    result.ep_thoracic_lower = angle_data["ep_thoracic_lower"]
    result.ep_lumbar_upper   = angle_data["ep_lumbar_upper"]
    result.ep_lumbar_lower   = angle_data["ep_lumbar_lower"]
    result.lumbar_lower_extrapolated = angle_data["lumbar_lower_extrapolated"]
    result.lumbar_fit_curve  = angle_data["lumbar_fit_curve"]

    # ── Step 7: Classification ─────────────────────────────────────────────
    th_key, th_label = classify_thoracic(result.thoracic_angle_deg)
    lu_key, lu_label = classify_lumbar(result.lumbar_angle_deg)

    result.thoracic_class_key   = th_key
    result.thoracic_class_label = th_label
    result.lumbar_class_key     = lu_key
    result.lumbar_class_label   = lu_label

    return result
