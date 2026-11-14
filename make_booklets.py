"""Build identical blind PDF/TXT batches for manual testing, never answer keys."""

from __future__ import annotations

import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, KeepTogether, PageBreak, Paragraph,
                               SimpleDocTemplate, Spacer)

from research import BOOKLETS, ROOT, load_cases

ACCENT = colors.HexColor("#4F46E5")
INK = colors.HexColor("#161B33")
MUTED = colors.HexColor("#596579")

# Embed proper Unicode text fonts. The built-in Helvetica can render with
# distorted glyph spacing on some PDF viewers (notably macOS/Poppler builds).
pdfmetrics.registerFont(TTFont("FinSans", str(ROOT / "assets" / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("FinSans-Bold", str(ROOT / "assets" / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("FinSans", normal="FinSans", bold="FinSans-Bold")


def batches(rows: list[dict], size: int = 10):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def page_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#D5DBEA"))
    canvas.line(17 * mm, 17 * mm, w - 17 * mm, 17 * mm)
    canvas.setFont("FinSans", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 12.5 * mm, "FinLingo  |  Blind model evaluation  |  Manual testing")
    canvas.drawRightString(w - 17 * mm, 12.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def pstyle(name, font_size, leading, text_color=INK, *, space_before=0, space_after=0):
    return ParagraphStyle(name, fontName="FinSans", fontSize=font_size, leading=leading,
                          textColor=text_color, spaceBefore=space_before, spaceAfter=space_after,
                          alignment=TA_LEFT, splitLongWords=1, allowWidows=0, allowOrphans=0)


H1 = pstyle("Title", 23, 28, space_after=16)
H2 = pstyle("Question", 13, 18, ACCENT, space_after=9)
BODY = pstyle("Body", 10.6, 16, space_after=12)
SMALL = pstyle("Notes", 9.6, 14, MUTED, space_after=10)


def paragraph_safe(value: str):
    return html.escape(value).replace("\n", "<br/>")


def instructions(rows: list[dict], label: str, booklet_id: str) -> str:
    ids = [r["run_id"] for r in rows]
    prompt = rows[0]["system_prompt"]
    return (f"FINLINGO MANUAL LLM EVALUATION\n\n"
            f"Booklet: {booklet_id}\n"
            f"Dataset: {label}\n"
            f"Questions in this booklet: {len(rows)}\n"
            f"Run IDs: {', '.join(ids)}\n\n"
            "EVALUATION INSTRUCTIONS - apply to every question below:\n"
            f"{prompt}\n\n"
            "Respond to ALL questions, in the order shown. Use the exact paired markers, "
            "one pair around each answer. Do not include an introduction or summary. "
            "Keep each ID unchanged. Example:\n\n"
            f"[[{ids[0]}]]\nYour complete answer here\n[[/{ids[0]}]]\n\n"
            "Do not include model-generated ratings or evaluation reference answers. "
            "If an answer cannot be determined from the given information, explain what is missing.\n\n"
            "Question pages follow.\n")


def make_pdf(output: Path, rows: list[dict], label: str, booklet_id: str):
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=19*mm,
                            leftMargin=19*mm, topMargin=21*mm, bottomMargin=24*mm,
                            title=f"FinLingo blind questions - {booklet_id}",
                            author="FinLingo research benchmark")
    story = [Paragraph("FinLingo / Blind Question Booklet", H1),
             Paragraph(html.escape(booklet_id), H2),
             Paragraph(f"<b>Group:</b> {html.escape(label)} | "
                       f"<b>Questions:</b> {len(rows)}", BODY),
             HRFlowable(width="100%", thickness=1, color=ACCENT), Spacer(1, 11)]
    instr = rows[0]["system_prompt"]
    story.extend([Paragraph("Instructions for every question", H2),
                  Paragraph(paragraph_safe(instr), BODY),
                  Paragraph("Answer all questions. Use one exact marker pair per answer, as shown:", BODY),
                  Paragraph(f"<b>[[{rows[0]['run_id']}]]</b><br/>Your answer here<br/>"
                            f"<b>[[/{rows[0]['run_id']}]]</b>", BODY),
                  Paragraph("Do not browse or call external tools. Use a new chat for each booklet. "
                            "Provide no answer-key or grading commentary.", SMALL)])
    for number, row in enumerate(rows, 1):
        rid = row["run_id"]
        story.extend([PageBreak(),
                      Paragraph(f"Question {number} of {len(rows)}", SMALL),
                      Paragraph(f"{html.escape(rid)}", H2),
                      Paragraph(f"Category: {html.escape(row['category'])}  |  "
                                f"Difficulty: {html.escape(row['difficulty'])}", SMALL),
                      HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CED3E1")),
                      Spacer(1, 12),
                      Paragraph(paragraph_safe(row["user_prompt"]), BODY),
                      Spacer(1, 12),
                      Paragraph(f"Reply marker: [[{html.escape(rid)}]] ... "
                                f"[[/{html.escape(rid)}]]", SMALL)])
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def make_txt(output: Path, rows: list[dict], label: str, booklet_id: str):
    blocks = [instructions(rows, label, booklet_id)]
    for index, row in enumerate(rows, 1):
        blocks.append(f"\n{'='*50}\nQUESTION {index}/{len(rows)} | {row['run_id']} | "
                      f"{row['category']} | {row['difficulty']}\n{'='*50}\n"
                      f"{row['user_prompt']}\n")
    output.write_text("\n".join(blocks), encoding="utf-8")


def build() -> list[dict]:
    pilot = load_cases("pilot")
    final = load_cases("final")
    groups = [
        ("pilot", "pilot_no_added_context", [r for r in pilot], "pilot", "PILOT ONLY"),
        ("final", "no_added_context", [r for r in final if r["condition"] == "no_added_context"],
         "no_context", "No added context - 120 question baseline"),
        ("final", "verified_context", [r for r in final if r["condition"] == "verified_context"],
         "with_context", "Context-provided - 80 additional cases"),
    ]
    manifest = []
    for phase, condition, cases, folder, label in groups:
        for number, batch in enumerate(batches(cases), 1):
            prefix = f"{folder}_{number:02d}"
            pdf = BOOKLETS / folder / f"{prefix}.pdf"
            txt = BOOKLETS / folder / f"{prefix}.txt"
            make_pdf(pdf, batch, label, prefix)
            make_txt(txt, batch, label, prefix)
            manifest.append({"booklet_id": prefix, "phase": phase, "condition": condition,
                             "run_ids": [r["run_id"] for r in batch],
                             "pdf": str(pdf.relative_to(BOOKLETS)),
                             "txt": str(txt.relative_to(BOOKLETS))})
    (BOOKLETS / "booklet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(manifest)} PDF/TXT booklet pairs: "
          f"{sum(len(x['run_ids']) for x in manifest)} blind prompt cases.")
    return manifest


if __name__ == "__main__":
    build()
