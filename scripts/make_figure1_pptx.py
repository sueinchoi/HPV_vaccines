"""
Editable PowerPoint version of Figure 1 (cohort selection flow).

Each step is a rounded rectangle with editable text; arrows connect them.
The slide mirrors the layout/colours of Data/Figure1_CohortSelection.png so
that text edits in PowerPoint produce a consistent figure.

Output: Data/Figure1_CohortSelection.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---------------------------------------------------------------------------
# Layout (inches) — landscape slide 13.33 × 7.5 (16:9 widescreen)
# ---------------------------------------------------------------------------
SLIDE_W, SLIDE_H = 13.33, 7.5
BOX_W, BOX_H_S, BOX_H_M, BOX_H_L = 4.7, 0.7, 0.85, 1.6
LEFT_X, RIGHT_X = 3.0, 8.6  # centres of the two columns
CENTER_X = (LEFT_X + RIGHT_X) / 2

# Vertical row centres
ROW_SOURCE  = 0.55
ROW_VACC    = 1.30
ROW_HEADER  = 2.20
ROW_STEP1   = 3.10
ROW_STEP2   = 4.10
ROW_FINAL   = 5.65

# Colours
RGB_SOURCE  = RGBColor(0xE8, 0xF4, 0xF8)
RGB_SRC_BR  = RGBColor(0x1F, 0x6F, 0x97)
RGB_VACC    = RGBColor(0xFF, 0xF3, 0xCD)
RGB_VAC_BR  = RGBColor(0x85, 0x64, 0x04)
RGB_A_HEAD  = RGBColor(0xD4, 0xED, 0xDA)
RGB_A_BR    = RGBColor(0x15, 0x57, 0x24)
RGB_A_STEP  = RGBColor(0xEA, 0xF6, 0xEE)
RGB_A_FIN   = RGBColor(0xA8, 0xD5, 0xB5)
RGB_B_HEAD  = RGBColor(0xFD, 0xE2, 0xE4)
RGB_B_BR    = RGBColor(0x9B, 0x22, 0x26)
RGB_B_STEP  = RGBColor(0xFD, 0xED, 0xEE)
RGB_B_FIN   = RGBColor(0xF4, 0xA4, 0xA8)
RGB_TEXT    = RGBColor(0x22, 0x22, 0x22)
RGB_ARROW   = RGBColor(0x44, 0x44, 0x44)

# ---------------------------------------------------------------------------
# Build presentation
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width  = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout


def add_box(cx, cy, w, h, lines, fill, edge, *, fs_first=14, fs_rest=12,
            bold_first=False, bold_all=False):
    """Add a rounded rectangle whose text is the given list of lines."""
    left = Inches(cx - w / 2); top = Inches(cy - h / 2)
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = edge
    shape.line.width = Pt(1.25)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.10); tf.margin_right = Inches(0.10)
    tf.margin_top  = Inches(0.05); tf.margin_bottom = Inches(0.05)
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run(); run.text = line
        run.font.name = 'Calibri'
        run.font.size = Pt(fs_first if i == 0 else fs_rest)
        run.font.color.rgb = RGB_TEXT
        run.font.bold = bold_all or (i == 0 and bold_first)
    return shape


def add_arrow(x1, y1, x2, y2):
    """Add a straight downward arrow between two (x, y) points (inches)."""
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = RGB_ARROW
    line.line.width = Pt(1.3)
    # Add arrowhead at end
    from pptx.oxml.ns import qn
    ln = line.line._get_or_add_ln()
    tail = ln.find(qn('a:tailEnd'))
    if tail is None:
        from lxml import etree
        tail = etree.SubElement(ln, qn('a:tailEnd'))
    tail.set('type', 'triangle'); tail.set('w', 'med'); tail.set('len', 'med')


# ---------- Source + ascertainment ----------
add_box(CENTER_X, ROW_SOURCE, 5.0, BOX_H_S,
        ['Source population', 'N = 32,969'],
        RGB_SOURCE, RGB_SRC_BR, fs_first=15, fs_rest=14, bold_first=True)
add_arrow(CENTER_X, ROW_SOURCE + BOX_H_S/2,
          CENTER_X, ROW_VACC   - BOX_H_S/2)

add_box(CENTER_X, ROW_VACC, 5.6, BOX_H_S,
        ['HPV vaccine ascertained',
         'Vaccinated 2,156  /  Unvaccinated 30,813'],
        RGB_VACC, RGB_VAC_BR, fs_first=14, fs_rest=13)

# Branch arrows
add_arrow(CENTER_X, ROW_VACC + BOX_H_S/2,
          LEFT_X,   ROW_HEADER - BOX_H_S/2)
add_arrow(CENTER_X, ROW_VACC + BOX_H_S/2,
          RIGHT_X,  ROW_HEADER - BOX_H_S/2)

# ---------- Cohort headers ----------
add_box(LEFT_X, ROW_HEADER, BOX_W, BOX_H_S,
        ['COHORT A — Safety'],
        RGB_A_HEAD, RGB_A_BR, fs_first=16, bold_first=True)
add_box(RIGHT_X, ROW_HEADER, BOX_W, BOX_H_S,
        ['COHORT B — Efficacy'],
        RGB_B_HEAD, RGB_B_BR, fs_first=16, bold_first=True)

# ---------- Cohort A steps ----------
add_arrow(LEFT_X, ROW_HEADER + BOX_H_S/2, LEFT_X, ROW_STEP1 - BOX_H_S/2)
add_box(LEFT_X, ROW_STEP1, BOX_W, BOX_H_S,
        ['Eligibility check', '+ pseudo index date'],
        RGB_A_STEP, RGB_A_BR, fs_first=14, fs_rest=14)
add_arrow(LEFT_X, ROW_STEP1 + BOX_H_S/2, LEFT_X, ROW_STEP2 - BOX_H_M/2)
add_box(LEFT_X, ROW_STEP2, BOX_W, BOX_H_M,
        ['Propensity-score 1:1 matching',
         '+ \u22652 doses + 3-mo landmark',
         '(matched-pair integrity)'],
        RGB_A_STEP, RGB_A_BR, fs_first=14, fs_rest=14)
add_arrow(LEFT_X, ROW_STEP2 + BOX_H_M/2, LEFT_X, ROW_FINAL - BOX_H_L/2)
add_box(LEFT_X, ROW_FINAL, BOX_W, BOX_H_L,
        ['Final Cohort A   n = 2,776',
         'Vaccinated 1,396',
         'Unvaccinated 1,380',
         '',
         'Outcomes: 5 chronic conditions'],
        RGB_A_FIN, RGB_A_BR, fs_first=15, fs_rest=13, bold_first=True)

# ---------- Cohort B steps ----------
add_arrow(RIGHT_X, ROW_HEADER + BOX_H_S/2, RIGHT_X, ROW_STEP1 - BOX_H_S/2)
add_box(RIGHT_X, ROW_STEP1, BOX_W, BOX_H_S,
        ['Cervical surgery', 'n = 6,890'],
        RGB_B_STEP, RGB_B_BR, fs_first=14, fs_rest=14)
add_arrow(RIGHT_X, ROW_STEP1 + BOX_H_S/2, RIGHT_X, ROW_STEP2 - BOX_H_M/2)
add_box(RIGHT_X, ROW_STEP2, BOX_W, BOX_H_M,
        ['Variable-ratio fine matching',
         '+ \u22652 doses + 3-mo landmark',
         '(matched-set integrity)'],
        RGB_B_STEP, RGB_B_BR, fs_first=14, fs_rest=14)
add_arrow(RIGHT_X, ROW_STEP2 + BOX_H_M/2, RIGHT_X, ROW_FINAL - BOX_H_L/2)
add_box(RIGHT_X, ROW_FINAL, BOX_W, BOX_H_L,
        ['Final Cohort B   n = 912',
         'Vaccinated 203',
         'Unvaccinated 709',
         '',
         'Outcomes: recurrence, HPV'],
        RGB_B_FIN, RGB_B_BR, fs_first=15, fs_rest=13, bold_first=True)

# ---------- Save ----------
out_path = 'Data/Figure1_CohortSelection.pptx'
prs.save(out_path)
print(f'Saved: {out_path}')
