"""
PDF evidence report.

Turns a finished analysis into a one-page document a person can attach to a
cybercrime complaint or a platform takedown request. It states the verdict, the
file's SHA-256 so the exact file analysed can be identified later, every signal
with its score and plain-language note, and the strongest heatmap.
The footer says plainly that this is triage, not a certified forensic finding.
"""

import os

import config

# Verdict colours, chosen to survive black and white printing as light/dark too.
VERDICT_STYLE = {
    "likely_synthetic": ("#B00020", "LIKELY SYNTHETIC"),
    "likely_authentic": ("#1B7F3B", "LIKELY AUTHENTIC"),
    "inconclusive": ("#B8860B", "INCONCLUSIVE"),
}


def _pct(value):
    """
    Format a 0..1 score for display.

    value (float|None): the score.

    Returns (str): e.g. "87%", or "n/a" when the signal produced no score.
    """
    if value is None:
        return "n/a"
    return f"{int(round(value * 100))}%"


def _pick_heatmap(result):
    """
    Choose the single most useful image to embed.

    result (dict): the pipeline response.

    Returns (tuple): (absolute path, kind) where kind is "heatmap" or "ela",
    or (None, None) when neither was produced. The kind is returned because the
    two images mean different things and must not share a caption.
    Only one image is embedded, to keep the report to a single page.
    """
    media = result.get("media") or {}
    candidates = [(url, "heatmap") for url in (media.get("heatmaps") or [])]
    if media.get("ela_image"):
        candidates.append((media["ela_image"], "ela"))

    for url, kind in candidates:
        # URLs are "/static/...", map back to a real path for embedding.
        relative = str(url).replace("/static/", "", 1).replace("/", os.sep)
        path = os.path.join(config.STATIC_DIR, relative)
        if os.path.exists(path):
            return path, kind
    return None, None


# Status wording for the score column. The raw enum values are too long for a
# 14 mm column and wrap into nonsense like "not_appli cable".
STATUS_LABEL = {"error": "failed", "not_applicable": "n/a", "ok": ""}


def build_report(result, job_id):
    """
    Render the analysis to a PDF on disk.

    result (dict): the pipeline response, containing "file", "verdict",
        "signals" and "media".
    job_id (str): used for the filename, so the PDF can be matched to the job.

    Returns (str): absolute path to the written PDF.
    Raises whatever reportlab raises if the file cannot be written; pipeline.py
    catches that and simply omits report_url from the response.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    out_path = os.path.join(config.REPORT_DIR, f"nazarai_{job_id}.pdf")
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=14 * mm,
                            title=f"NazarAI report {job_id}")

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.5,
                          leading=11, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, textColor=colors.grey)

    file_block = result.get("file") or {}
    verdict_block = result.get("verdict") or {}
    verdict_key = verdict_block.get("verdict", "inconclusive")
    colour_hex, verdict_label = VERDICT_STYLE.get(verdict_key,
                                                  VERDICT_STYLE["inconclusive"])

    story = []

    # ---- header
    story.append(Paragraph(
        "<font size=17><b>NazarAI</b></font>  "
        "<font size=9 color='#666666'>Media Verification Report</font>",
        styles["Normal"]))
    story.append(Spacer(1, 3 * mm))

    # ---- file identity
    identity = [
        ["File", str(file_block.get("name", ""))[:60]],
        ["Type", str(file_block.get("type", ""))],
        ["SHA-256", str(file_block.get("sha256", ""))],
        ["Analyzed", str(file_block.get("analyzed_at", ""))],
    ]
    id_table = Table(identity, colWidths=[26 * mm, 148 * mm])
    id_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 2), (1, 2), "Courier"),   # hash in a fixed-width face
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
    ]))
    story.append(id_table)
    story.append(Spacer(1, 4 * mm))

    # ---- verdict box
    score_text = _pct(verdict_block.get("overall_score"))
    verdict_rows = [[Paragraph(
        f"<font size=13 color='{colour_hex}'><b>{verdict_label}</b></font>"
        f"&nbsp;&nbsp;<font size=9>overall synthetic score {score_text}</font>"
        f"<br/><br/><font size=9>{verdict_block.get('headline', '')}</font>", body)]]
    verdict_table = Table(verdict_rows, colWidths=[174 * mm])
    verdict_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(colour_hex)),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(verdict_table)

    if verdict_block.get("reasoning"):
        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(f"Basis: {verdict_block['reasoning']}", small))

    story.append(Spacer(1, 4 * mm))

    # ---- signal table
    story.append(Paragraph("<b>Signals</b>", body))
    story.append(Spacer(1, 1.5 * mm))

    rows = [[Paragraph("<b>Check</b>", small), Paragraph("<b>Score</b>", small),
             Paragraph("<b>Conf.</b>", small), Paragraph("<b>Finding</b>", small)]]
    for sig in result.get("signals", []):
        if sig.get("status") == "ok":
            score_cell = _pct(sig.get("synthetic_score"))
            conf_cell = _pct(sig.get("confidence"))
        else:
            score_cell = STATUS_LABEL.get(sig.get("status", ""), "n/a")
            conf_cell = "-"
        rows.append([
            Paragraph(str(sig.get("display_name", "")), small),
            Paragraph(score_cell, small),
            Paragraph(conf_cell, small),
            Paragraph(str(sig.get("human_note", ""))[:400], small),
        ])

    sig_table = Table(rows, colWidths=[36 * mm, 14 * mm, 14 * mm, 110 * mm],
                      repeatRows=1)
    sig_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(sig_table)

    # ---- one supporting image
    heatmap, heatmap_kind = _pick_heatmap(result)
    if heatmap:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("<b>Visual evidence</b>", body))
        story.append(Spacer(1, 1.5 * mm))
        try:
            # Fixed width, height derived from the real aspect ratio so nothing
            # is squashed. Capped so the report stays on one page.
            from PIL import Image as PILImage
            with PILImage.open(heatmap) as probe:
                width_px, height_px = probe.size
            draw_w = 78 * mm
            draw_h = min(60 * mm, draw_w * (height_px / max(1, width_px)))
            story.append(Image(heatmap, width=draw_w, height=draw_h))
            caption = (
                "Warmer colours mark the regions that most influenced the model."
                if heatmap_kind == "heatmap" else
                "Error level analysis. Evenly lit areas share one save history; "
                "isolated bright patches can indicate a locally edited region.")
            story.append(Paragraph(caption, small))
        except Exception as exc:
            story.append(Paragraph(f"(image could not be embedded: {exc})", small))

    # ---- footer
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "Automated multi-signal analysis for triage and complaint filing. "
        "Not a certified forensic conclusion.", small))
    story.append(Paragraph(
        f"Generated by NazarAI | job {job_id} | "
        f"{file_block.get('analyzed_at', '')}", small))

    doc.build(story)
    return out_path
