# Spine & Posture Analyzer

A Python desktop application for comprehensive postural assessment from photos or live webcam. Analyzes both **side-view** (sagittal) and **front-view** (coronal) images through a guided four-step workflow.

> **Estimation tool only — not a medical diagnostic device.**
> Results are approximate and depend on image quality, posture, and clothing.

---

## What it measures

**Side view (sagittal)**
| Metric | Description |
|--------|-------------|
| Thoracic Kyphosis angle | Cobb-style angle, T1–T12 (arms-forward contour photo) |
| Lumbar Lordosis angle | Cobb-style angle, L1–S1 (arms-forward contour photo) |
| Forward Head Position (FHP) | Craniovertebral angle — ear vs. shoulder vertical |
| Forward Head Distance (FHD) | Horizontal ear-to-C7 offset in cm / inches |
| Estimated spine load | Kapandji model: 12 + (FHD_inches × 10) lbs |

**Front view (coronal)**
| Metric | Description |
|--------|-------------|
| Head tilt | Left/right ear-line angle from horizontal |
| Shoulder level | Left/right shoulder height difference (cm) |
| Pelvic level | Left/right hip height difference (cm) |
| Knee level | Left/right knee height difference (cm) |

---

## System requirements

| Item | Requirement |
|------|-------------|
| Python | 3.10 or 3.11 (MediaPipe 0.10.x is not yet compatible with 3.12+) |
| OS | macOS 12+ (Intel or Apple Silicon), Linux, or Windows 10+ |
| Webcam | Optional — required for Live Camera mode |

---

## Setup

### 1. Create and activate a virtual environment

```bash
cd "~/Desktop/Dresio/HKBU/HKBU Final"
python3.11 -m venv .venv
source .venv/bin/activate
```

> On Apple Silicon Macs, if `python3.11` is not found:
> ```bash
> brew install python@3.11
> python3.11 -m venv .venv
> source .venv/bin/activate
> ```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `mediapipe`, `opencv-python`, `numpy`, `scipy`, `Pillow`, `PySide6`, and `pytest`.
Total download is approximately 500 MB (MediaPipe bundles its own TensorFlow Lite runtime).

### 3. Run the application

```bash
python main.py
```

---

## macOS webcam permission

The first time Live Camera mode is used, macOS will prompt for camera access.
If the prompt does not appear or the camera feed is black:

1. Open **System Settings → Privacy & Security → Camera**.
2. Enable access for **Terminal** (or your IDE / Python launcher).
3. Restart the app.

---

## How to use

The app enforces a **four-step workflow**. Each step must be completed before the next unlocks.

### Step 1a — Contour photo (arms forward)

Used to extract the back silhouette for Cobb angle measurement.

1. Raise both arms straight forward (clears the back contour from arm shadow).
2. Stand **sideways** — either side facing the camera is fine; the app auto-detects direction.
3. Click **Upload Photo** or **Live Camera → Take Photo**.
4. Results show thoracic kyphosis and lumbar lordosis angles with clinical classification.
5. On success, the app automatically advances to Step 1b.

### Step 1b — Natural stand photo (arms down)

Used for Forward Head Position and Forward Head Distance measurement.

1. Lower arms to your sides, stand naturally with head in a neutral position.
2. Remain in the same sideways orientation as Step 1a.
3. Upload or capture the photo.
4. Results show CV angle, neck inclination, FHD in cm/inches, and estimated spine load.
5. On success, **Next: Front View** is unlocked.

### Step 2 — Front view

Used for bilateral symmetry assessment.

1. Turn to face the camera directly.
2. Upload or capture a front-view photo.
3. Results show head tilt, shoulder level, pelvic level, and knee level, all in cm with left/right direction.
4. Click **View Full Report** to advance to the summary.

### Step 3 — Full report & export

- Displays annotated images from all three photos side by side.
- Click **Export (PNG + CSV)** to save:
  - `spine_annotated.png` — side-view overlay
  - `front_annotated.png` — front-view overlay
  - `report_YYYYMMDDHHMMSS.csv` — all metrics in a single row

---

## Reading the results

### Thoracic / Lumbar clinical ranges

| Curve | Range | Classification |
|-------|-------|---------------|
| Thoracic Kyphosis | < 20° | Hypokyphosis (flat back) |
| | 20° – 45° | Normal |
| | > 45° | Hyperkyphosis |
| Lumbar Lordosis | < 20° | Hypolordosis |
| | 20° – 40° | Normal |
| | > 40° | Hyperlordosis |

*Reference: Mejia EA et al. 1996; Tüzün C et al. 1999.*

### Forward Head Distance severity

| FHD | Severity |
|-----|----------|
| < 1 inch | Normal |
| 1 – 2 inch | Mild |
| 2 – 3 inch | Moderate |
| ≥ 3 inch | Severe |

*Kapandji (2009) spine-load model: baseline 12 lbs + 10 lbs per inch of forward displacement.*

### Confidence flags

| Flag | Meaning |
|------|---------|
| LOW CONFIDENCE (body) | Shoulder–Hip axis deviated > 8° from vertical |
| LOW CONFIDENCE (lumbar) | Buttock occlusion extrapolation distance too large |

---

## Measurement geometry

### Spine (side view) — Cobb-style tangent method

1. **Segmentation** — MediaPipe Selfie Segmentation extracts the person's silhouette.
2. **Back contour** — The posterior boundary of the silhouette is extracted from shoulder level to hip level (curve **A→B**).
3. **Anatomical markers** — Curvature analysis locates:
   - `apex_K` (~T4/T5): most outward point of the thoracic curve
   - `inflect` (~T12/L1): inflection point where curvature reverses
   - `apex_L` (~L3/L4): most inward point of the lumbar curve
4. **Tangent lines** — Four local segments are fitted at arc-length-fraction positions:
   - T1 near A (~T2/T3 upper thoracic)
   - T2 below apex_K (~T10/T11 lower thoracic)
   - T3 above apex_L (~L2 upper lumbar)
   - T4 near L5 (estimated via hip-landmark SVD projection)
5. **Angles** — Thoracic angle = angle between T1 & T2; Lumbar angle = angle between T3 & T4.

### Lumbar occlusion & L5 estimation

The buttocks protrude outward at L5/S1, making the lower back contour unreliable below that point. The app uses two complementary strategies:

- **Slope-change detection** — scans upward from the hip to find where the contour swings outward; below that point the contour is discarded.
- **Hip-landmark SVD projection** — MediaPipe hip landmarks are projected onto the spine arc to estimate L5 via arc-length extrapolation. This is the primary method when hip visibility is sufficient (`L5_HIP_VISIBILITY_MIN = 0.5`).

Results are flagged LOW CONFIDENCE if the extrapolation distance exceeds 12% of image height.

### Forward Head Position (side view)

- **Craniovertebral (CV) angle**: angle at the shoulder between the ear and the vertical line through the ear. Normal ≥ 50°; lower values indicate more forward head position.
- **Forward Head Distance (FHD)**: horizontal pixel offset between the ear and the shoulder midpoint (C7 proxy), converted to cm using the vertical ear-to-C7 distance as the calibration ruler.

### Front-view metrics

All bilateral differences are converted to centimetres using the shoulder-to-hip segment as a real-world reference (world coordinates from MediaPipe, fallback 55 cm when unavailable):

```
diff_cm = (left_y_px − right_y_px) / sh_hip_px_dist × sh_hip_world_cm
```

### View-type detection

The app automatically classifies each uploaded photo as side view or front view using two signals:

1. **Shoulder Z-depth difference** (primary) — `|z_L − z_R| ≥ 0.30` indicates a side view.
2. **Shoulder-span / hip-span ratio** (guard) — a ratio ≥ 5.0 overrides Z-depth for front-view photos with large shoulder span.

If the wrong view is submitted for a step, the app shows a retake prompt without running the analysis.

---

## Project structure

```
HKBU Final/
  main.py                    GUI entry point
  config.py                  All tunable constants and clinical thresholds
  requirements.txt
  README.md
  spine/
    pose_detector.py         MediaPipe pose + view-type classification
    segmentation.py          Silhouette → back contour extraction
    contour_analysis.py      A/B endpoints, markers, L5 estimation, occlusion
    angles.py                Tangent fitting + angle computation + classification
    visualization.py         Side-view annotated overlay drawing
    pipeline.py              Full analysis orchestration → AnalysisResult
  posture/
    front_metrics.py         Head tilt, shoulder/pelvic/knee level calculators
    front_pipeline.py        Front-view analysis orchestration → FrontAnalysisResult
    front_visualization.py   Front-view annotated overlay drawing
    side_metrics.py          FHP (craniovertebral angle) + FHD (Kapandji model)
  gui/
    app_window.py            Main window — four-step workflow pages + summary
    camera_widget.py         Live webcam preview + countdown capture
  export/
    report_exporter.py       CSV export + annotated PNG saving
  tests/
    test_angles.py           Unit tests with synthetic contours (exact + pipeline)
  captures/                  Auto-saved originals and annotated frames
```

---

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests use synthetic parametric curves with known curvature to verify:
- Near-straight spines yield angles < 20°
- Larger curve amplitude always yields larger angle (monotonicity)
- Hyperkyphotic / hyperlordotic curves are classified correctly
- All computed angles are acute (< 90°), confirming the tangent method
- Clinical classification boundaries are exact
- Tangent vectors are unit-length
- Contour smoothing reduces noise while preserving endpoints
- Arc-length index helper is correct at 0%, 50%, and 100% fractions

---

## Library choices

| Purpose | Library | Why |
|---------|---------|-----|
| Pose landmarks | MediaPipe Pose | Bundled, no extra model download, accurate shoulder/hip/ear/knee |
| Segmentation | MediaPipe Selfie Segmentation | Bundled with MediaPipe, clean torso mask |
| GUI | PySide6 (Qt6) | LGPL, Qt6 native macOS look, image display, threaded workers |
| Signal processing | SciPy (Savitzky-Golay / B-spline) | Smooth curvature computation and contour fitting |

`rembg` (U2Net) was considered for segmentation but requires a ~170 MB ONNX model download; MediaPipe Selfie Segmentation achieves comparable results with zero extra downloads.

---

## Known limitations

1. **Side-view orientation** — the spine analysis is undefined for front/back/oblique views. The app detects and rejects wrong-view submissions.
2. **Near-vertical posture required** — Shoulder→Hip axis deviation > 8° triggers a LOW CONFIDENCE flag. Severely stooped postures may need the threshold relaxed in `config.py`.
3. **Lumbar lower end is estimated** — the buttock protrusion occludes L5/S1. The lower lumbar tangent is derived from a hip-landmark SVD projection, with slope-extrapolation as fallback. This is the largest source of lumbar angle error.
4. **Clothing** — loose or baggy clothing obscures the natural back curve. Tight-fitting clothing gives more accurate results.
5. **Approximate anatomy** — T4/T5, T12/L1, and L3/L4 positions are estimated geometrically from contour curvature, not from X-ray endplate identification.
6. **Single-person images only** — multiple people in frame may confuse the segmentation.
7. **No camera calibration** — no correction for perspective distortion. Camera should be approximately at waist height and 2–3 m away for best results.
8. **FHD calibration** — pixel-to-cm conversion uses the assumed anatomical ear-to-C7 distance (default 24 cm). Actual anatomy varies; this introduces a fixed systematic offset that affects absolute FHD values but not the severity classification trend.

---

## Tuning

All thresholds are in `config.py`. Key values to adjust:

| Constant | Default | Effect |
|----------|---------|--------|
| `VERTICAL_DEVIATION_THRESHOLD_DEG` | 8° | Strictness of body-verticality gate |
| `TANGENT_SEGMENT_FRAC` | 0.10 | Tangent segment length (10% of curve) |
| `BUTTOCK_SLOPE_CHANGE_THRESHOLD` | 0.8 | Sensitivity of lumbar occlusion detection |
| `LUMBAR_EXTRAPOLATION_MAX_FRAC` | 0.12 | Max allowed extrapolation before LOW CONFIDENCE |
| `USE_ESTIMATED_L5` | True | Switch between hip-SVD method and slope-extrapolation fallback |
| `L5_SPINE_ARC_FRAC` | 0.12 | Arc-length fraction added past apex_L to reach L5 |
| `CONTOUR_SMOOTH_FACTOR` | 50.0 | B-spline smoothing strength (0 = disabled) |
| `SIDE_VIEW_Z_DIFF_MIN` | 0.30 | Minimum shoulder Z-depth difference to classify as side view |
| `COUNTDOWN_SECONDS` | 5 | Camera countdown duration |
