"""
pdf_exporter.py — Generate a branded PDF posture report.

Layout (A4 portrait):
  - Header: Dresio logo top-left, report title top-right
  - Three sections: photo (left) + metrics table (right) for each view
  - Summary table spanning full width
  - Footer: Dresio logo bottom-right + disclaimer
"""

import io
import os
import datetime
from pathlib import Path

import cv2
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent
_LOGO = _HERE / "Dresio LOGO.jpg"

# ── Dimensions ─────────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4          # 595 x 842 pt
L_MARGIN = 14 * mm
R_MARGIN = 14 * mm
BODY_W   = PAGE_W - L_MARGIN - R_MARGIN   # ≈ 516 pt

PHOTO_W  = 200.0             # pt — left column (photo)
META_W   = BODY_W - PHOTO_W  # pt — right column (metrics)
PHOTO_MAX_H = 185.0          # pt — max rendered photo height

# Metrics sub-columns inside META_W
M_LABEL  = META_W * 0.44
M_VALUE  = META_W * 0.20
M_STATUS = META_W * 0.36

# ── Brand colours ──────────────────────────────────────────────────────────────
C_ACCENT  = HexColor("#0F3460")
C_TEAL    = HexColor("#00B4D8")
C_GREEN   = HexColor("#4CAF50")
C_ORANGE  = HexColor("#FF9800")
C_RED     = HexColor("#F44336")
C_LIGHT   = HexColor("#F5F5F5")
C_RULE    = HexColor("#0F3460")
C_GREY    = HexColor("#666666")
C_DARK    = HexColor("#1A1A2E")
C_BG      = HexColor("#FFFFFF")

# ── Styles ─────────────────────────────────────────────────────────────────────
def _s(name, **kw):
    base = dict(fontName="Helvetica", fontSize=10, textColor=black,
                leading=14, spaceAfter=0)
    base.update(kw)
    return ParagraphStyle(name, **base)

ST_TITLE   = _s("title",  fontName="Helvetica-Bold", fontSize=20,
                textColor=C_DARK, leading=24)
ST_DATE    = _s("date",   fontName="Helvetica",      fontSize=10,
                textColor=C_GREY, leading=14)
ST_SECT    = _s("sect",   fontName="Helvetica-Bold", fontSize=12,
                textColor=white, leading=16)
ST_PHOTO_L = _s("pl",     fontName="Helvetica-Bold", fontSize=9,
                textColor=C_ACCENT, leading=12)
ST_ML      = _s("ml",     fontName="Helvetica",      fontSize=9,
                textColor=C_DARK, leading=13)
ST_MV      = _s("mv",     fontName="Helvetica-Bold", fontSize=10,
                textColor=C_TEAL, leading=13, alignment=TA_RIGHT)
ST_NOTE    = _s("note",   fontName="Helvetica-Oblique", fontSize=8,
                textColor=C_GREY, leading=11)
ST_FOOT    = _s("foot",   fontName="Helvetica",      fontSize=7,
                textColor=C_GREY, leading=10, alignment=TA_CENTER)
ST_WARN    = _s("warn",   fontName="Helvetica-Oblique", fontSize=8,
                textColor=C_ORANGE, leading=11)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _img_flowable(img_bgr: np.ndarray, max_w: float, max_h: float) -> RLImage:
    h_px, w_px = img_bgr.shape[:2]
    scale = min(max_w / w_px, max_h / h_px)
    # Encode as BGR — cv2.imencode writes BGR byte order into the JPEG stream,
    # and JPEG/ReportLab readers interpret that stream as RGB, giving correct colours.
    # A prior cvtColor BGR→RGB followed by imencode would double-swap the channels.
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return RLImage(io.BytesIO(buf.tobytes()), width=w_px * scale, height=h_px * scale)


def _logo(max_w: float, max_h: float) -> RLImage | None:
    if not _LOGO.exists():
        return None
    img = cv2.imread(str(_LOGO))
    return _img_flowable(img, max_w, max_h) if img is not None else None


def _status_color(key: str | None) -> HexColor:
    if not key:
        return C_GREY
    k = key.lower()
    if "normal" in k or "level" in k:
        return C_GREEN
    if "severe" in k:
        return C_RED
    if any(x in k for x in ("mild", "moderate", "hypo", "hyper", "forward",
                             "tilt", "asymm", "high", "low")):
        return C_ORANGE
    return C_GREY


def _status_p(text: str, key: str | None) -> Paragraph:
    st = ParagraphStyle("sp", fontName="Helvetica-Bold", fontSize=9,
                        textColor=_status_color(key), leading=13)
    return Paragraph(text or "", st)


def _section_bar(title: str) -> Table:
    t = Table([[Paragraph(title, ST_SECT)]], colWidths=[BODY_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t


def _metrics_rows_table(rows: list[tuple]) -> Table:
    """
    rows: (label, value, status_text, status_key)
    Column widths sum to META_W.
    """
    data = [
        [Paragraph(lbl, ST_ML), Paragraph(str(val), ST_MV), _status_p(st or "", sk)]
        for lbl, val, st, sk in rows
    ]
    t = Table(data, colWidths=[M_LABEL, M_VALUE, M_STATUS])
    cmds = [
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("ALIGN",         (1, 0), (1, -1),  "RIGHT"),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, HexColor("#DDDDDD")),
    ]
    for i in range(0, len(data), 2):
        cmds.append(("BACKGROUND", (0, i), (-1, i), C_LIGHT))
    t.setStyle(TableStyle(cmds))
    return t


def _photo_section(
    img_bgr: np.ndarray,
    section_title: str,
    sublabel: str,
    rows: list[tuple],
    notes: list[str] | None = None,
) -> KeepTogether:
    """
    Returns a KeepTogether block: section bar + side-by-side [photo | metrics].
    Right column is a plain list — ReportLab Table handles list cells natively,
    which avoids the height-miscalculation that happens with nested Tables.
    """
    photo = _img_flowable(img_bgr, PHOTO_W - 8, PHOTO_MAX_H)

    # Right column: list of flowables — ReportLab lays them out top-to-bottom
    right_items: list = [
        Paragraph(sublabel, ST_PHOTO_L),
        Spacer(1, 4),
        _metrics_rows_table(rows),
    ]
    if notes:
        right_items.append(Spacer(1, 3))
        for n in notes:
            right_items.append(Paragraph(f"⚠  {n}", ST_WARN))

    outer = Table([[photo, right_items]], colWidths=[PHOTO_W, META_W])
    outer.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",    (0, 0), (-1, -1), C_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (0, -1),  4),
        ("RIGHTPADDING",  (0, 0), (0, -1),  0),
        ("LEFTPADDING",   (1, 0), (1, -1),  8),
        ("RIGHTPADDING",  (1, 0), (1, -1),  4),
    ]))

    return KeepTogether([
        _section_bar(section_title),
        Spacer(1, 2),
        outer,
        Spacer(1, 5 * mm),
    ])


def _summary_table(rows: list[tuple]) -> Table:
    """Full-width summary table — wider label/status columns."""
    lw = BODY_W * 0.38
    vw = BODY_W * 0.18
    sw = BODY_W * 0.44
    data = [
        [Paragraph(lbl, ST_ML), Paragraph(str(val), ST_MV), _status_p(st or "", sk)]
        for lbl, val, st, sk in rows
    ]
    t = Table(data, colWidths=[lw, vw, sw])
    cmds = [
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ALIGN",         (1, 0), (1, -1),  "RIGHT"),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, HexColor("#DDDDDD")),
        ("BOX",           (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ("BACKGROUND",    (0, 0), (-1, -1), C_BG),
    ]
    for i in range(0, len(data), 2):
        cmds.append(("BACKGROUND", (0, i), (-1, i), C_LIGHT))
    t.setStyle(TableStyle(cmds))
    return t


# ── Public API ─────────────────────────────────────────────────────────────────

def export_pdf(
    spine_result,
    fhp_result,
    front_result,
    contour_annotated: np.ndarray,
    fhp_annotated: np.ndarray | None,
    front_annotated: np.ndarray,
    output_path: str,
) -> None:
    """
    Generate a branded A4 PDF report.

    spine_result      : AnalysisResult  (contour step — Cobb angles)
    fhp_result        : AnalysisResult  (FHP step — forward head metrics)
    front_result      : FrontAnalysisResult
    contour_annotated : BGR image from Step 1a
    fhp_annotated     : BGR image from Step 1b (None if unavailable)
    front_annotated   : BGR image from Step 2
    output_path       : destination .pdf path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=14 * mm,   bottomMargin=18 * mm,
        title="Spine & Posture Analysis Report",
        author="Dresio",
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    logo = _logo(75 * mm, 34 * mm)
    hdr_left  = [logo] if logo else [Paragraph("Dresio", ST_TITLE)]
    hdr_right = [
        Paragraph("Spine &amp; Posture Analysis Report", ST_TITLE),
        Spacer(1, 3),
        Paragraph(f"Generated: {ts}", ST_DATE),
    ]
    hdr = Table([[hdr_left, hdr_right]],
                colWidths=[80 * mm, BODY_W - 80 * mm])
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ALIGN",         (1, 0), (1, -1),  "RIGHT"),
    ]))
    story += [
        hdr,
        HRFlowable(width="100%", thickness=2, color=C_RULE,
                   spaceBefore=4, spaceAfter=5 * mm),
    ]

    # ── Helper lambdas ────────────────────────────────────────────────────────
    def _v(val, unit="°"):
        return f"{val:.1f}{unit}" if val is not None else "N/A"

    def _d(d):
        return {"left_high": "Left High", "right_high": "Right High",
                "level": "Level", "left_low": "Left Low",
                "right_low": "Right Low", "unknown": "—"
                }.get(d or "unknown", d or "—")

    def _asym(diff):
        if diff is None:
            return None
        return "normal" if abs(diff) < 0.5 else ("mild" if abs(diff) < 1.5 else "moderate")

    def _cerv_class(deg):
        if deg is None:
            return ("—", None)
        if 20 <= deg <= 40:
            return ("Healthy (20–40°)", "normal")
        if (10 <= deg < 20) or (40 < deg <= 50):
            return ("Acceptable (10–19° / 41–50°)", "mild")
        return ("High Risk (<10° or >50°)", "severe")

    # ── Section 1: Spinal curvature ───────────────────────────────────────────
    th  = spine_result.thoracic_angle_deg
    lu  = spine_result.lumbar_angle_deg
    thl = spine_result.thoracic_class_label or "—"
    lul = spine_result.lumbar_class_label   or "—"
    thk = spine_result.thoracic_class_key   or ""
    luk = spine_result.lumbar_class_key     or ""

    cobb_notes = []
    if spine_result.low_confidence:
        cobb_notes.append("LOW CONFIDENCE — body not fully vertical")
    if spine_result.lumbar_low_confidence:
        cobb_notes.append("LOW CONFIDENCE — lumbar extrapolation too large")

    story.append(_photo_section(
        contour_annotated,
        "Side View — Spinal Curvature  (Arms Forward)",
        "Cobb-style angle measurement",
        [
            ("Thoracic Kyphosis", f"{th:.1f}°" if th is not None else "N/A", thl, thk),
            ("Lumbar Lordosis",   f"{lu:.1f}°" if lu is not None else "N/A", lul, luk),
            ("Facing Direction",
             spine_result.facing_direction.title() if spine_result.facing_direction else "—",
             "", None),
        ],
        cobb_notes or None,
    ))

    # ── Section 2: Neck Posture / Cervical Lordosis ───────────────────────────
    if fhp_annotated is not None and fhp_result is not None:
        cerv = getattr(fhp_result, "cervical_flexion_deg", None)
        if cerv is None:
            cerv = fhp_result.forward_head_angle_deg

        cerv_label, cerv_key = _cerv_class(cerv)

        nk = fhp_result.neck_inclination_deg

        fhp_rows = [
            ("Cervical Flexion Angle",
             f"{cerv:.1f}°" if cerv is not None else "N/A",
             cerv_label, cerv_key),
            ("Neck Inclination",
             f"{nk:.1f}°" if nk is not None else "N/A",
             "", None),
        ]

        fhp_notes = [w for w in (getattr(fhp_result, "warnings", None) or []) if w]
        story.append(_photo_section(
            fhp_annotated,
            "Side View — Neck Posture  (Natural Stand)",
            "Cervical lordosis — ear · C7 · vertical angle",
            fhp_rows,
            fhp_notes or None,
        ))

    # ── Section 3: Front view ─────────────────────────────────────────────────
    ht = front_result.head_tilt_deg
    sh = front_result.shoulder_diff_cm
    pe = front_result.pelvic_diff_cm
    kn = front_result.knee_diff_cm
    sh_px = getattr(front_result, "shoulder_diff_px", None)
    pe_px = getattr(front_result, "pelvic_diff_px",   None)
    kn_px = getattr(front_result, "knee_diff_px",     None)

    def _bilat(diff_px, diff_cm):
        """Format as 'X px  /  X.X cm' — no direction or status label."""
        parts = []
        if diff_px is not None:
            parts.append(f"{diff_px:.0f} px")
        if diff_cm is not None:
            parts.append(f"{abs(diff_cm):.1f} cm")
        return "  /  ".join(parts) if parts else "N/A"

    front_notes = [w for w in (getattr(front_result, "warnings", None) or []) if w]
    story.append(_photo_section(
        front_annotated,
        "Front View — Postural Symmetry",
        "Bilateral level differences",
        [
            ("Head Tilt",
             f"{abs(ht):.1f}°  ({_d(front_result.head_tilt_direction)})" if ht is not None else "N/A",
             "", None),
            ("Shoulder Level", _bilat(sh_px, sh), "", None),
            ("Pelvic Level",   _bilat(pe_px, pe), "", None),
            ("Knee Level",     _bilat(kn_px, kn), "", None),
        ],
        front_notes or None,
    ))

    # ── Summary ───────────────────────────────────────────────────────────────
    sum_rows: list[tuple] = [
        ("Thoracic Kyphosis",
         f"{th:.1f}°" if th else "N/A", thl, thk),
        ("Lumbar Lordosis",
         f"{lu:.1f}°" if lu else "N/A", lul, luk),
    ]
    if fhp_annotated is not None and fhp_result is not None:
        cerv2 = getattr(fhp_result, "cervical_flexion_deg", None)
        if cerv2 is None:
            cerv2 = fhp_result.forward_head_angle_deg
        cerv_lbl2, cerv_key2 = _cerv_class(cerv2)
        sum_rows.append(
            ("Cervical Flexion",
             f"{cerv2:.1f}°" if cerv2 else "N/A",
             cerv_lbl2, cerv_key2),
        )
    sum_rows += [
        ("Head Tilt",
         f"{abs(ht):.1f}°  ({_d(front_result.head_tilt_direction)})" if ht is not None else "N/A",
         "", None),
        ("Shoulder Level", _bilat(sh_px, sh), "", None),
        ("Pelvic Level",   _bilat(pe_px, pe), "", None),
        ("Knee Level",     _bilat(kn_px, kn), "", None),
    ]

    story += [
        _section_bar("Summary"),
        Spacer(1, 2),
        _summary_table(sum_rows),
        Spacer(1, 5 * mm),
    ]

    # ── Footer ────────────────────────────────────────────────────────────────
    logo_f = _logo(42 * mm, 20 * mm)
    disc   = Paragraph(
        "Estimation tool only — not a medical diagnostic device. "
        "Results depend on image quality, clothing, and posture at capture time. "
        "Consult a qualified healthcare professional for clinical assessment.",
        ST_FOOT,
    )
    footer_items = [disc, Spacer(1, 3), Paragraph(f"Dresio  ·  {ts}", ST_FOOT)]

    if logo_f:
        foot_tbl = Table([[logo_f, footer_items]],
                         colWidths=[46 * mm, BODY_W - 48 * mm])
        foot_tbl.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
        ]))
    else:
        foot_tbl = Table([[footer_items]], colWidths=[BODY_W])
        foot_tbl.setStyle(TableStyle([
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ]))

    story += [
        HRFlowable(width="100%", thickness=1, color=C_RULE,
                   spaceBefore=2 * mm, spaceAfter=3 * mm),
        foot_tbl,
    ]

    doc.build(story)
