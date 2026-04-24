"""
PDF clinical summary export for HCC risk predictions.

Generates a one-page PDF report containing:
  1. Patient risk gauge (colour-coded Low / Medium / High)
  2. Predicted class probabilities
  3. FIB-4 index and APRI score with interpretations
  4. Top SHAP-driven risk factors and protective factors
  5. Standard clinical disclaimer

Usage:
    from src.pdf_report import generate_clinical_pdf
    from src.predict import PredictionResult

    pdf_bytes = generate_clinical_pdf(result, patient_dict)
    # In Streamlit:
    st.download_button("Download PDF", pdf_bytes, "hcc_report.pdf", "application/pdf")

Dependencies:
    reportlab (added to requirements.txt)
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Optional

# reportlab is listed in requirements.txt; import deferred to function body
# so the module stays importable in test environments without reportlab.


def _rl_imports():
    """Lazy-import reportlab components once (keeps module importable without RL)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm
    from reportlab.lib.colors import (
        HexColor, white, black, Color,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Wedge
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics import renderPDF
    return (
        A4, cm, mm, HexColor, white, black, Color,
        getSampleStyleSheet, ParagraphStyle, TA_CENTER, TA_LEFT, TA_RIGHT,
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        Drawing, Rect, String, Line, Wedge, renderPDF,
    )


# ── Colour palette ────────────────────────────────────────────────────────────

_RISK_HEX = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
}

_RISK_BG = {
    "Low":    "#d5f5e3",
    "Medium": "#fef9e7",
    "High":   "#fde8e6",
}


# ── Semi-circular gauge drawing ───────────────────────────────────────────────

def _build_gauge(risk_category: str, prob: float, w: float = 200, h: float = 110):
    """
    Return a reportlab Drawing of a semi-circular risk gauge.

    Args:
        risk_category: "Low" | "Medium" | "High"
        prob:          Probability of the predicted class (0–1).
        w, h:          Width and height of the Drawing canvas.
    """
    (
        A4, cm, mm, HexColor, white, black, Color,
        getSampleStyleSheet, ParagraphStyle, TA_CENTER, TA_LEFT, TA_RIGHT,
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        Drawing, Rect, String, Line, Wedge, renderPDF,
    ) = _rl_imports()

    d = Drawing(w, h)
    cx, cy = w / 2, 15          # centre of the semi-circle arc
    r_outer, r_inner = 75, 48   # ring thickness

    # Three coloured arc segments covering 0° … 180° (Low / Med / High)
    arc_colours = ["#2ecc71", "#f39c12", "#e74c3c"]
    seg_degrees = [60, 60, 60]     # equal thirds of 180°
    start = 0

    for colour_hex, degrees in zip(arc_colours, seg_degrees):
        colour = HexColor(colour_hex)
        # Draw a filled wedge for outer circle then overdraw inner as white
        w_outer = Wedge(cx, cy, r_outer, start, start + degrees, fillColor=colour, strokeColor=None)
        d.add(w_outer)
        start += degrees

    # White inner circle to create the donut
    from reportlab.graphics.shapes import Circle
    inner = Circle(cx, cy, r_inner, fillColor=white, strokeColor=white, strokeWidth=2)
    d.add(inner)

    # Needle — points to the predicted probability angle in 0..180°
    angle_deg = prob * 180.0        # 0% → left (0°), 100% → right (180°)
    angle_rad = math.radians(angle_deg)
    needle_len = r_inner - 4
    nx = cx + needle_len * math.cos(angle_rad)
    ny = cy + needle_len * math.sin(angle_rad)
    needle = Line(cx, cy, nx, ny, strokeColor=HexColor("#2c3e50"), strokeWidth=2.5)
    d.add(needle)

    # Centre dot
    dot = Circle(cx, cy, 5, fillColor=HexColor("#2c3e50"), strokeColor=white, strokeWidth=1)
    d.add(dot)

    # Label
    label_colour = HexColor(_RISK_HEX.get(risk_category, "#888"))
    label = String(
        cx, cy + r_outer + 6,
        f"{risk_category.upper()} RISK  ({prob*100:.0f}%)",
        textAnchor="middle",
        fontSize=12,
        fillColor=label_colour,
        fontName="Helvetica-Bold",
    )
    d.add(label)

    # Low / High axis labels
    d.add(String(cx - r_outer - 2, cy - 12, "Low", fontSize=8,
                 fillColor=HexColor("#27ae60"), textAnchor="end"))
    d.add(String(cx + r_outer + 2, cy - 12, "High", fontSize=8,
                 fillColor=HexColor("#c0392b"), textAnchor="start"))

    return d


# ── Main public function ──────────────────────────────────────────────────────

def generate_clinical_pdf(
    result,
    patient: dict,
    title: str = "HCC Risk Assessment — Clinical Summary",
) -> bytes:
    """
    Generate a one-page PDF clinical summary for an HCC risk prediction.

    Args:
        result:  PredictionResult dataclass from src.predict.predict_patient().
        patient: Raw patient dict (lab values) as supplied to the predictor.
        title:   Document title shown in the header.

    Returns:
        PDF file as bytes — suitable for st.download_button() in Streamlit.

    Example:
        >>> pdf_bytes = generate_clinical_pdf(result, patient_dict)
        >>> with open("report.pdf", "wb") as f:
        ...     f.write(pdf_bytes)
    """
    (
        A4, cm, mm, HexColor, white, black, Color,
        getSampleStyleSheet, ParagraphStyle, TA_CENTER, TA_LEFT, TA_RIGHT,
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        Drawing, Rect, String, Line, Wedge, renderPDF,
    ) = _rl_imports()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 4 * cm    # usable page width

    # ── Custom styles ──────────────────────────────────────────────────────────
    header_style = ParagraphStyle(
        "Header",
        parent=styles["Title"],
        fontSize=16,
        textColor=HexColor("#2c3e50"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9,
        textColor=HexColor("#7f8c8d"),
        spaceAfter=6,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=HexColor("#2c3e50"),
        spaceBefore=10,
        spaceAfter=4,
        borderPad=2,
    )
    normal_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=HexColor("#7f8c8d"),
        leading=11,
        borderPad=4,
    )

    # ── Helper: coloured badge table ──────────────────────────────────────────
    risk_cat = result.risk_category
    risk_hex = _RISK_HEX.get(risk_cat, "#888")
    risk_bg  = _RISK_BG.get(risk_cat, "#eee")

    # ── Build content ──────────────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph(title, header_style))
    ts_str = datetime.now().strftime("%d %b %Y, %H:%M")
    story.append(Paragraph(f"Generated: {ts_str} &nbsp;|&nbsp; <b>Research prototype — not for clinical use</b>", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HexColor("#2c3e50"), spaceAfter=8))

    # ── Risk gauge ────────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Classification", section_style))
    gauge_drawing = _build_gauge(
        risk_cat,
        result.probabilities.get(risk_cat, 0.5),
    )
    # Embed gauge in a single-cell table so it centres properly
    gauge_table = Table([[gauge_drawing]], colWidths=[W])
    gauge_table.setStyle(TableStyle([
        ("ALIGN",    (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(gauge_table)
    story.append(Spacer(1, 4))

    # ── Probability table ──────────────────────────────────────────────────────
    story.append(Paragraph("Predicted Probabilities", section_style))
    prob_data = [["Risk Class", "Probability", "Interpretation"]]
    interpretations = {
        "Low":    "≤30% HCC probability — routine surveillance recommended",
        "Medium": "30–60% HCC probability — enhanced monitoring advised",
        "High":   "≥60% HCC probability — urgent hepatology referral warranted",
    }
    for cls, prob in result.probabilities.items():
        prob_data.append([cls, f"{prob*100:.1f}%", interpretations.get(cls, "")])

    prob_table = Table(prob_data, colWidths=[2.5 * cm, 2.5 * cm, W - 5 * cm])
    prob_ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (1, 0), (1, -1),  "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f9f9f9"), white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("PADDING",     (0, 0), (-1, -1), 5),
    ])
    # Highlight predicted class row
    for i, cls in enumerate(result.probabilities.keys(), start=1):
        if cls == risk_cat:
            prob_ts.add("BACKGROUND", (0, i), (-1, i), HexColor(risk_bg))
            prob_ts.add("TEXTCOLOR",  (0, i), (0, i),  HexColor(risk_hex))
            prob_ts.add("FONTNAME",   (0, i), (0, i),  "Helvetica-Bold")
    prob_table.setStyle(prob_ts)
    story.append(prob_table)
    story.append(Spacer(1, 6))

    # ── Clinical composite scores ──────────────────────────────────────────────
    story.append(Paragraph("Derived Clinical Scores", section_style))
    age_val = patient.get("age", 0)
    ast_val = patient.get("ast", 0)
    alt_val = max(patient.get("alt", 0.1), 0.1)
    plt_val = max(patient.get("platelets", 1), 1)
    fib4    = (age_val * ast_val) / (plt_val * math.sqrt(max(alt_val, 0.1)))
    apri    = (ast_val / 40.0) / plt_val * 100

    def _fib4_interp(v):
        if v > 3.25: return "Advanced fibrosis likely ⚠️"
        if v > 1.45: return "Intermediate — consider biopsy"
        return "Low fibrosis risk ✓"

    def _apri_interp(v):
        if v > 1.5: return "Significant fibrosis likely ⚠️"
        if v > 0.5: return "Intermediate"
        return "Low fibrosis risk ✓"

    scores_data = [
        ["Score", "Value", "Interpretation"],
        ["FIB-4 Index",  f"{fib4:.2f}",  _fib4_interp(fib4)],
        ["APRI Score",   f"{apri:.2f}",  _apri_interp(apri)],
        ["AST:ALT Ratio",f"{ast_val / alt_val:.2f}", "> 2.0 suggests alcohol / cirrhosis" if ast_val / alt_val > 2.0 else "Normal range"],
    ]
    scores_table = Table(scores_data, colWidths=[3.5 * cm, 2.5 * cm, W - 6 * cm])
    scores_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f9f9f9"), white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("PADDING",     (0, 0), (-1, -1), 5),
    ]))
    story.append(scores_table)
    story.append(Spacer(1, 6))

    # ── SHAP factors ───────────────────────────────────────────────────────────
    story.append(Paragraph("Key Predictive Factors (SHAP)", section_style))

    shap_rows = [["Factor", "SHAP Contribution", "Direction"]]
    for name, val in result.explanation:
        direction = "↑ Risk-increasing" if val > 0 else "↓ Protective"
        shap_rows.append([name.upper(), f"{val:+.4f}", direction])

    shap_table = Table(shap_rows, colWidths=[4 * cm, 3.5 * cm, W - 7.5 * cm])
    shap_ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f9f9f9"), white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("PADDING",     (0, 0), (-1, -1), 5),
    ])
    for i, (_, val) in enumerate(result.explanation, start=1):
        c = HexColor("#fde8e6") if val > 0 else HexColor("#d5f5e3")
        shap_ts.add("BACKGROUND", (2, i), (2, i), c)
    shap_table.setStyle(shap_ts)
    story.append(shap_table)
    story.append(Spacer(1, 10))

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc"), spaceAfter=6))
    disclaimer_text = (
        "<b>⚠️  DISCLAIMER:</b>  This report is generated by a research prototype model trained on "
        "synthetic data.  It has NOT been validated for clinical use and must NOT be used to inform "
        "clinical decisions without review by a qualified hepatologist.  All risk estimates are "
        "probabilistic approximations.  Always consult a licensed healthcare professional for "
        "medical diagnosis and treatment decisions.  For emergencies, contact emergency services."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(story)
    return buf.getvalue()
