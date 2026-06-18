# Spine Posture Analyzer

A Python desktop application that estimates **thoracic kyphosis angle** and **lumbar lordosis angle** from a side-view (sagittal) photo of a standing person.

> **Estimation tool only — not a medical diagnostic device.**
> Results are approximate and depend on image quality, posture, and clothing.

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
cd ~/Desktop/Dresio/HKBU/Spine
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

### Upload Photo mode
1. Click **Upload Photo**.
2. Select a side-view JPEG or PNG. The person must face **RIGHT** (back on the left side of the image).
3. The analysis runs automatically and displays the annotated image plus a text summary.

### Live Camera mode
1. Click **Live Camera** — the webcam preview starts.
2. Stand side-on (facing right) in front of the camera.
3. Click **Take Photo** — a **5-second countdown** overlay appears.
4. Hold still; the frame is captured automatically and analyzed.

### Reading the results
| Field | Meaning |
|-------|---------|
| Thoracic angle | Cobb-style kyphosis angle (T1–T12) |
| Lumbar angle | Cobb-style lordosis angle (L1–S1) |
| LOW CONFIDENCE (body) | Shoulder–Hip axis deviated >8° from vertical |
| LOW CONFIDENCE (lumbar) | Buttock occlusion extrapolation distance too large |

---

## Clinical reference ranges

| Curve | Range | Classification |
|-------|-------|---------------|
| Thoracic Kyphosis | < 20° | Hypokyphosis (flat back) |
| | 20° – 45° | Normal |
| | > 45° | Hyperkyphosis |
| Lumbar Lordosis | < 20° | Hypolordosis |
| | 20° – 40° | Normal |
| | > 40° | Hyperlordosis |

*Reference: Mejia EA et al. 1996; Tüzün C et al. 1999.*

---

## Measurement geometry

### How angles are computed

This app uses a **Cobb-style tangent-line method**, not a vertex/two-point angle:

1. **Segmentation** — MediaPipe Selfie Segmentation extracts the person's silhouette.
2. **Back contour** — The left (posterior) boundary of the silhouette is extracted from shoulder level to hip level. This is curve **A→B**.
3. **Anatomical markers** — Curvature analysis locates:
   - `apex_K` (T4/T5): most outward point of upper curve
   - `inflect` (T12/L1): inflection point where curvature changes sign
   - `apex_L` (L3/L4): most inward point of lower curve
4. **Tangent lines** — Four short local segments are fitted:
   - T1 near A (upper thoracic endplate direction)
   - T2 near inflect going upward (lower thoracic)
   - T3 near inflect going downward (upper lumbar)
   - T4 near reliable lumbar end / extrapolated (lower lumbar / S1)
5. **Angles** — Thoracic angle = angle between T1 & T2; Lumbar angle = angle between T3 & T4.

### Landmark choice

MediaPipe detects both left and right landmarks. Since the person faces right:
- Both sides are averaged when visibility ≥ 0.5 on each side.
- This gives a midline estimate robust to slight camera offset.

### Lumbar occlusion

The buttocks protrude outward at L5/S1, making the lower back contour unreliable.
The app detects the protrusion onset by scanning the horizontal slope of the contour
upward from hip level. Below the protrusion point, the contour is discarded and the
lower lumbar tangent is **extrapolated** from the reliable portion. Results are
flagged LOW CONFIDENCE if the extrapolation distance exceeds 12% of image height.

---

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests use synthetic parametric curves with known curvature to verify:
- Near-straight spines yield angles < 15°
- Hyperkyphotic curves are classified correctly
- All computed angles are < 90° (verifying tangent method, not vertex method)
- Clinical classification boundaries are correct

---

## Project structure

```
Spine/
  main.py              GUI entry point
  config.py            All tunable constants and clinical thresholds
  requirements.txt
  README.md
  spine/
    pose_detector.py   MediaPipe shoulder/hip + verticality check
    segmentation.py    Silhouette → back contour extraction
    contour_analysis.py  A/B endpoints, curvature, markers, occlusion
    angles.py          Tangent fitting + angle computation + classification
    visualization.py   Annotated overlay drawing
    pipeline.py        Full analysis orchestration → AnalysisResult
  gui/
    app_window.py      Main window (upload + camera + results)
    camera_widget.py   Live webcam preview + countdown capture
  tests/
    test_angles.py     Unit tests with synthetic contours
```

---

## Library choices

| Purpose | Library | Why |
|---------|---------|-----|
| Pose landmarks | MediaPipe Pose | Bundled, no extra model download, accurate shoulder/hip |
| Segmentation | MediaPipe Selfie Segmentation | Bundled with MediaPipe, clean torso mask, no extra install vs rembg |
| GUI | PySide6 (Qt6) | LGPL, Qt6 native macOS look, better than Tkinter for image display |
| Signal processing | SciPy (Savitzky-Golay) | Smooth curvature computation without manual rolling average |

`rembg` (U2Net) was considered for segmentation but requires a ~170 MB ONNX model
download and `onnxruntime`; MediaPipe Selfie Segmentation achieves comparable
results for standing-person silhouettes with zero extra downloads.

---

## Known limitations

1. **Side-view only** — the analysis is undefined for front/back/oblique views.
2. **Near-vertical posture required** — if the Shoulder→Hip axis deviates > 8° from vertical, results are flagged as low confidence. For severely stooped postures this threshold may need to be relaxed in `config.py`.
3. **Lumbar lower end is extrapolated** — the buttock protrusion occludes L5/S1. The lower lumbar tangent is estimated by extrapolating from the reliable portion of the lumbar curve. This is the largest source of lumbar angle error.
4. **Clothing affects contour quality** — loose or baggy clothing obscures the natural back curve. Tight-fitting clothing gives more accurate results.
5. **Approximate anatomy** — T4/T5, T12/L1, and L3/L4 positions are estimated geometrically from the contour curvature, not from X-ray endplate identification.
6. **Single-person images only** — multiple people in the frame may confuse the segmentation.
7. **Camera calibration** — no correction for perspective distortion or camera-to-subject distance. The camera should be at approximately waist height and ~2–3 m away for best results.

---

## Tuning

All thresholds are in `config.py`. Key values to adjust:

| Constant | Default | Effect |
|----------|---------|--------|
| `VERTICAL_DEVIATION_THRESHOLD_DEG` | 8° | Strictness of verticality gate |
| `TANGENT_SEGMENT_FRAC` | 0.10 | Tangent segment length (10% of curve) |
| `BUTTOCK_SLOPE_CHANGE_THRESHOLD` | 0.8 | Sensitivity of occlusion detection |
| `LUMBAR_EXTRAPOLATION_MAX_FRAC` | 0.12 | Max allowed extrapolation (% of image height) |
| `COUNTDOWN_SECONDS` | 5 | Camera countdown duration |
