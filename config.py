"""
All tunable constants for the Spine Posture Analyzer.
Edit values here to tune the analysis without touching algorithm code.
"""

# ─── Clinical thresholds (Mejia 1996 / Tüzün 1999) ───────────────────────────
THORACIC_THRESHOLDS = {
    "hypokyphosis": (None, 20),   # < 20°
    "normal":       (20,   45),   # 20°–45°
    "hyperkyphosis":(45,   None), # > 45°
}

LUMBAR_THRESHOLDS = {
    "hypolordosis": (None, 20),   # < 20°
    "normal":       (20,   40),   # 20°–40°
    "hyperlordosis":(40,   None), # > 40°
}

THORACIC_LABELS = {
    "hypokyphosis":  "Thoracic Hypokyphosis (flat back)",
    "normal":        "Normal Thoracic Kyphosis",
    "hyperkyphosis": "Thoracic Hyperkyphosis",
}

LUMBAR_LABELS = {
    "hypolordosis":  "Lumbar Hypolordosis",
    "normal":        "Normal Lumbar Lordosis",
    "hyperlordosis": "Lumbar Hyperlordosis",
}

# ─── Posture validity ─────────────────────────────────────────────────────────
# Maximum allowed deviation (degrees) of Shoulder→Hip line from true vertical
# before the result is flagged LOW CONFIDENCE.
VERTICAL_DEVIATION_THRESHOLD_DEG = 8.0

# ─── Contour smoothing ───────────────────────────────────────────────────────
# B-spline smoothing applied to the back contour after extraction, before angle
# computation. Larger values = smoother curve. Set to 0 to disable smoothing entirely.
CONTOUR_SMOOTH_FACTOR = 50.0

# ─── Contour extraction ───────────────────────────────────────────────────────
# Fraction of image width to keep when cropping the torso column for contour
TORSO_X_MARGIN = 0.05

# Minimum segmentation mask confidence (0–1) for MediaPipe Selfie Segmentation
SEGMENTATION_THRESHOLD = 0.5

# Number of rows to average when projecting shoulder/hip onto back contour
CONTOUR_PROJECTION_BAND = 3   # pixels

# ─── Curve analysis ───────────────────────────────────────────────────────────
# Fraction of A→B arc length used as the sliding-window for curvature smoothing
CURVATURE_SMOOTH_FRAC = 0.05   # ~5% of curve

# Search window (fraction of curve) centered on 33% / 66% arc positions
APEX_SEARCH_WINDOW_FRAC = 0.25  # ±12.5% around the split

# ─── Tangent fitting ──────────────────────────────────────────────────────────
# Number of curve points to use for each local tangent fit (as fraction of
# total A→B contour points).  8–12% of curve is a sensible default.
TANGENT_SEGMENT_FRAC = 0.10

# Minimum absolute number of points for a tangent fit (safety floor)
TANGENT_MIN_POINTS = 6

# Half-window (in contour points) centred on each endpoint for tangent fitting.
# The fit uses points in [endpoint_idx - w, endpoint_idx + w].
TANGENT_LOCAL_HALF_WINDOW = 8

# ---- Angle endpoint locators (arc-length fraction WITHIN each sub-segment) ----
# Each fraction = (vertebrae from apex to target) / (vertebrae from apex to boundary),
# assuming vertebrae are ~equally spaced along the back contour arc length.
# Landmarks: A~T1, apex_K~T4/T5(4.5), inflect~T12/L1(12.5),
#            apex_L~L3/L4(3.5 within lumbar, L1=1), B~L5/S1(5.5)
#
# Thoracic upper target T2/T3(2.5): |4.5-2.5|/|4.5-1|   = 2.0/3.5 ≈ 0.57
THORACIC_UPPER_FRAC = 0.82   # from apex_K toward A       -> ~T2/T3
# Thoracic lower target T10/T11(10.5): |10.5-4.5|/|12.5-4.5| = 6.0/8.0 = 0.75
THORACIC_LOWER_FRAC = 0.75   # from apex_K toward inflect -> ~T10/T11
# Lumbar upper target L2: |3.5-2|/|3.5-1|               = 1.5/2.5 = 0.60
LUMBAR_UPPER_FRAC   = 0.60   # from apex_L toward inflect -> ~L2
# Lumbar lower target L5: |5-3.5|/|5.5-3.5|             = 1.5/2.0 = 0.75
LUMBAR_LOWER_FRAC   = 0.75   # from apex_L toward B       -> ~L5

# ─── Buttock / lumbar occlusion detection ─────────────────────────────────────
# Scan upward from B; mark the first point where the horizontal slope changes
# by more than this many pixels-per-pixel (contour swings outward sharply).
BUTTOCK_SLOPE_CHANGE_THRESHOLD = 0.8   # Δx/Δy

# If the extrapolation distance (in pixels, normalised to image height) exceeds
# this fraction, the lumbar lower tangent is flagged LOW CONFIDENCE.
LUMBAR_EXTRAPOLATION_MAX_FRAC = 0.12

# ─── Estimated L5 position (sagittal-view hip-landmark method) ───────────────
# Master switch: set False to revert entirely to the contour slope-extrapolation
# method (virtual_B) and ignore all parameters below.
USE_ESTIMATED_L5 = True

# Minimum MediaPipe visibility score for each hip landmark to be considered
# reliable enough to use the estimation.  Below this → fallback to virtual_B.
L5_HIP_VISIBILITY_MIN = 0.5

# L5 sits this fraction of torso height ABOVE the mid-hip point.
# Torso height = hip_y − shoulder_y (pixels).  Sagittal view only — do NOT
# use left/right hip separation here; in a true side-on shot it is near zero.
# NOTE: Used only as fallback when apex_L is unavailable.
L5_ABOVE_HIP_FRAC = 0.31

# Fraction along the apex_L → hip vector where L5 is placed.
# 0.62 means 62% of the way from apex_L (L3/L4) toward hip.
# NOTE: legacy parameter, superseded by L5_SPINE_ARC_FRAC below.
L5_APEXL_HIP_FRAC = 0.62

# Arc-length fraction of the shoulder→hip spine arc added past apex_L to reach L5.
# Anatomical basis: shoulder ≈ T2/T3, hip ≈ L4/L5 → ~14 vertebral levels;
# 1.5 lumbar segments (apex L3/L4 → L5 centre) ÷ 14 ≈ 0.107,
# scaled up to 0.16 to account for lumbar vertebrae being larger than thoracic.
L5_SPINE_ARC_FRAC = 0.12

# Use the last 1/N of the spine contour points for the SVD direction fit.
# Larger N = shorter tail segment = tighter local direction estimate.
# Recommended range 4–8; default 6 uses roughly the last 17% of the curve.
L5_TAIL_FIT_RATIO = 6

# Fraction of the apex_L → reliable_lumbar_end arc used as clean contour input
# for the quadratic tangent fit.  Only the first L5_FIT_CLEAN_FRAC of that
# segment is used, avoiding the buttock-contaminated tail.
# 0.40 ≈ take the first 40% of apex_L→reliable_end, i.e. roughly the top 1/3
# of the lumbar curve where the contour is still undeformed.
L5_FIT_CLEAN_FRAC = 0.40

# Maximum allowed angle (degrees) between the fitted T4 tangent and the vertical
# Y-axis.  Anatomically L5's endplate is nearly vertical; if the fit yields a
# steeper tilt the result is likely polluted and we fall back to the apex_L →
# estimated_l5 chord direction instead.
L5_TANGENT_MAX_FROM_VERTICAL = 30.0

# ─── Visualization colors (BGR for OpenCV) ───────────────────────────────────
COLOR_AXIS_LINE     = (160, 160, 160)   # faint grey — Shoulder→Hip reference
COLOR_CONTOUR       = (  0, 210,  80)   # green — spine curve A→B
COLOR_THORACIC      = (220, 110,  50)   # blue  — thoracic tangents/arc/label
COLOR_LUMBAR        = ( 30, 150, 255)   # orange — lumbar tangents/arc/label
COLOR_LANDMARK      = (255, 255, 255)   # white dots for all spine landmarks
COLOR_LANDMARK_OUTLINE = (30, 30, 30)   # thin dark outline on landmark dots
COLOR_WARNING_TEXT  = (  0,  60, 220)   # red warning text
COLOR_L5_EST        = (  0, 220, 255)   # yellow — estimated L5 landmark dot

# ─── Camera / GUI ─────────────────────────────────────────────────────────────
COUNTDOWN_SECONDS   = 5
CAMERA_INDEX        = 0   # default webcam
CAMERA_FRAME_WIDTH  = 1280
CAMERA_FRAME_HEIGHT = 720

# ─── Display ──────────────────────────────────────────────────────────────────
OVERLAY_LINE_THICKNESS  = 2
OVERLAY_POINT_RADIUS    = 5
# Half-length of each tangent arm drawn from the apex (pixels).
# Each tangent is drawn as two arms of this length meeting at the intersection.
TANGENT_ARM_PX          = 120
# Alpha of the filled translucent wedge arc (0=invisible, 1=opaque).
ARC_FILL_ALPHA          = 0.25
