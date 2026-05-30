"""
Editable PowerPoint version of Figure 1 (cohort selection flow diagram).

Academic CONSORT-style layout — monochrome (black borders, white fill,
black text) for journal submission. Each step is a rounded rectangle
with editable text; arrows connect them.

Output: Data/Figure1_CohortSelection.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---------------------------------------------------------------------------
# Layout (inches) — landscape slide 13.33 × 7.5 (16:9 widescreen)
# ---------------------------------------------------------------------------
SLIDE_W, SLIDE_H = 13.33, 7.5
BOX_W, BOX_H_S, BOX_H_M, BOX_H_L = 4.7, 0.7, 0.95, 1.65
LEFT_X, RIGHT_X = 3.0, 8.6  # centres of the two columns
CENTER_X = (LEFT_X + RIGHT_X) / 2

# Vertical row centres
ROW_SOURCE  = 0.55
ROW_VACC    = 1.30
ROW_HEADER  = 2.20
ROW_STEP1   = 3.10
ROW_STEP2   = 4.20
ROW_FINAL   = 5.80

# ---------------------------------------------------------------------------
# Academic monochrome palette
# ---------------------------------------------------------------------------
FILL_WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
FILL_LIGHT_GRAY  = RGBColor(0xF2, 0xF2, 0xF2)
FILL_FINAL_GRAY  = RGBColor(0xE6, 0xE6, 0xE6)
EDGE_BLACK       = RGBColor(0x1A, 0x1A, 0x1A)
RGB_TEXT         = RGBColor(0x00, 0x00, 0x00)
RGB_ARROW        = RGBColor(0x1A, 0x1A, 0x1A)

FONT_NAME = 'Arial'

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
    shape.line.width = Pt(1.0)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.10); tf.margin_right = Inches(0.10)
    tf.margin_top  = Inches(0.06); tf.margin_bottom = Inches(0.06)
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run(); run.text = line
        run.font.name = FONT_NAME
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
    line.line.width = Pt(1.0)
    from pptx.oxml.ns import qn
    ln = line.line._get_or_add_ln()
    tail = ln.find(qn('a:tailEnd'))
    if tail is None:
        from lxml import etree
        tail = etree.SubElement(ln, qn('a:tailEnd'))
    tail.set('type', 'triangle'); tail.set('w', 'med'); tail.set('len', 'med')


# ---------- Source + ascertainment ----------
add_box(CENTER_X, ROW_SOURCE, 5.0, BOX_H_S,
        ['Source population',
         'N = 32,969'],
        FILL_WHITE, EDGE_BLACK,
        fs_first=15, fs_rest=14, bold_first=True)
add_arrow(CENTER_X, ROW_SOURCE + BOX_H_S/2,
          CENTER_X, ROW_VACC   - BOX_H_S/2)

add_box(CENTER_X, ROW_VACC, 5.8, BOX_H_S,
        ['HPV vaccine prescription ascertained',
         'Vaccinated 2,156    \u2003   Unvaccinated 30,813'],
        FILL_WHITE, EDGE_BLACK, fs_first=14, fs_rest=13)

# Branch arrows
add_arrow(CENTER_X, ROW_VACC + BOX_H_S/2,
          LEFT_X,   ROW_HEADER - BOX_H_S/2)
add_arrow(CENTER_X, ROW_VACC + BOX_H_S/2,
          RIGHT_X,  ROW_HEADER - BOX_H_S/2)

# ---------- Cohort headers ----------
add_box(LEFT_X, ROW_HEADER, BOX_W, BOX_H_S,
        ['Cohort A \u2014 chronic-disease safety'],
        FILL_LIGHT_GRAY, EDGE_BLACK, fs_first=15, bold_first=True)
add_box(RIGHT_X, ROW_HEADER, BOX_W, BOX_H_S,
        ['Cohort B \u2014 post-surgical efficacy'],
        FILL_LIGHT_GRAY, EDGE_BLACK, fs_first=15, bold_first=True)

# ---------- Cohort A steps ----------
add_arrow(LEFT_X, ROW_HEADER + BOX_H_S/2, LEFT_X, ROW_STEP1 - BOX_H_S/2)
add_box(LEFT_X, ROW_STEP1, BOX_W, BOX_H_S,
        ['Eligibility (index date \u2264 31 Dec 2024)',
         'Pseudo-index assigned to unvaccinated controls'],
        FILL_WHITE, EDGE_BLACK, fs_first=13, fs_rest=12)
add_arrow(LEFT_X, ROW_STEP1 + BOX_H_S/2, LEFT_X, ROW_STEP2 - BOX_H_M/2)
add_box(LEFT_X, ROW_STEP2, BOX_W, BOX_H_M,
        ['1:1 propensity-score matching',
         '(caliper 0.2 \u00D7 SD of logit PS)',
         '+ \u22652 doses + 3-month landmark',
         '(matched-pair integrity preserved)'],
        FILL_WHITE, EDGE_BLACK, fs_first=13, fs_rest=12)
add_arrow(LEFT_X, ROW_STEP2 + BOX_H_M/2, LEFT_X, ROW_FINAL - BOX_H_L/2)
add_box(LEFT_X, ROW_FINAL, BOX_W, BOX_H_L,
        ['Final analytic Cohort A   n = 2,776',
         '',
         'Vaccinated 1,396',
         'Unvaccinated 1,380',
         '',
         'Outcomes: 5 chronic conditions + composites'],
        FILL_FINAL_GRAY, EDGE_BLACK, fs_first=14, fs_rest=12, bold_first=True)

# ---------- Cohort B steps ----------
add_arrow(RIGHT_X, ROW_HEADER + BOX_H_S/2, RIGHT_X, ROW_STEP1 - BOX_H_S/2)
add_box(RIGHT_X, ROW_STEP1, BOX_W, BOX_H_S,
        ['Cervical surgery (conization or hysterectomy)',
         'n = 6,890   \u2003   Eligibility index date \u2264 31 Dec 2024'],
        FILL_WHITE, EDGE_BLACK, fs_first=13, fs_rest=12)
add_arrow(RIGHT_X, ROW_STEP1 + BOX_H_S/2, RIGHT_X, ROW_STEP2 - BOX_H_M/2)
add_box(RIGHT_X, ROW_STEP2, BOX_W, BOX_H_M,
        ['Variable-ratio matching',
         '(1:up-to-5 then 1:up-to-4 fine matching)',
         '+ \u22652 doses + 3-month landmark',
         '(matched-set integrity preserved)'],
        FILL_WHITE, EDGE_BLACK, fs_first=13, fs_rest=12)
add_arrow(RIGHT_X, ROW_STEP2 + BOX_H_M/2, RIGHT_X, ROW_FINAL - BOX_H_L/2)
add_box(RIGHT_X, ROW_FINAL, BOX_W, BOX_H_L,
        ['Final analytic Cohort B   n = 912',
         '',
         'Vaccinated 203',
         'Unvaccinated 709',
         '',
         'Outcomes: lesion recurrence; hr-HPV clearance'],
        FILL_FINAL_GRAY, EDGE_BLACK, fs_first=14, fs_rest=12, bold_first=True)

# ---------- Save ----------
out_path = 'Data/Figure1_CohortSelection.pptx'
prs.save(out_path)
print(f'Saved: {out_path}')
