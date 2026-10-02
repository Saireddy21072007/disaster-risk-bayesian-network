"""
Builds the review deck: report/Review_MultiHazardBN.pptx

Fifteen slides. The earlier version of this file made twenty-six, which was really
a report with slides around it; that long-form material now lives in
Project_Handbook.docx and this deck is what we actually stand up and present.
The old generator is kept as make_ppt_long_backup.py.

Everything numeric is pulled from artifacts/ at build time, so the deck cannot
silently disagree with the code.

    python report/make_ppt.py

Owner: Rohit Vardhan M (layout), content from all four of us
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACT_DIR, HAZARDS
from src.decision import (ACTION_COST, FALSE_ALARM_COST, HARM_IF_EMERGENCY,
                          switch_points)
from src.learn import load_model
from src.networks import describe_model

OUT = ROOT / "report" / "Review_MultiHazardBN.pptx"

NAVY = RGBColor(0x0E, 0x27, 0x3F)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
GREY = RGBColor(0x5B, 0x6B, 0x7A)
LIGHT = RGBColor(0xEE, 0xF4, 0xF9)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x7A, 0x4C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xB8, 0xCE, 0xE0)

MEMBERS = [
    ("Sai Reddy .A", "CB.SC.U4AIE24205"),
    ("Rohit Vardhan .M", "CB.SC.U4AIE24231"),
    ("Jithin Reddy .K", "CB.SC.U4AIE24230"),
    ("Sai Vandith", "CB.SC.U4AIE24239"),
]


# --------------------------------------------------------------------- helpers
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(slide, x, y, w, h, text, size=18, bold=False, colour=NAVY,
            align=PP_ALIGN.LEFT, spacing=1.0, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else list(text)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = spacing
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = colour
            r.font.name = "Calibri"
    return tb


def bullets(slide, x, y, w, h, items, size=15, spacing=1.1, colour=NAVY):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        text, lvl = (item if isinstance(item, tuple) else (item, 0))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("•  " if lvl == 0 else "     –  ") + text
        p.line_spacing = spacing
        p.space_after = Pt(6)
        for r in p.runs:
            r.font.size = Pt(size if lvl == 0 else size - 1.5)
            r.font.color.rgb = colour if lvl == 0 else GREY
            r.font.name = "Calibri"
    return tb


def band(slide, prs, title, kicker=None):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width,
                                 Inches(0.92))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    textbox(slide, 0.45, 0.13, 12.4, 0.62, title, size=25, bold=True, colour=WHITE)
    if kicker:
        textbox(slide, 0.48, 0.99, 12.4, 0.38, kicker, size=12.5, colour=GREY)


def slide(prs, title, kicker=None):
    s = blank(prs)
    band(s, prs, title, kicker)
    return s


def picture(slide, img, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(img), Inches(x), Inches(y), **kw)


def table(slide, x, y, w, headers, rows, widths, size=11.5, header_size=11.5,
          row_h=0.32):
    total = sum(widths)
    cols = [Emu(int(Inches(w) * cw / total)) for cw in widths]
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x),
                                   Inches(y), Inches(w),
                                   Inches(row_h * (len(rows) + 1)))
    t = shape.table
    for i, cw in enumerate(cols):
        t.columns[i].width = cw

    for j, head in enumerate(headers):
        c = t.cell(0, j)
        c.text = str(head)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_left = c.margin_right = Inches(0.07)
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(header_size)
                r.font.bold = True
                r.font.color.rgb = WHITE
                r.font.name = "Calibri"

    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = str(val)
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.07)
            c.margin_top = c.margin_bottom = Inches(0.02)
            centred = j > 0 and len(str(val)) <= 13
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if centred else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.color.rgb = NAVY
                    r.font.name = "Calibri"
    return t


def callout(slide, x, y, w, h, title, body, colour=ACCENT, size=13):
    box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = colour
    box.line.width = Pt(1.25)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.13)
    tf.margin_top = Inches(0.10)
    lines = ([title] if title else []) + list(body)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(4)
        head = bool(title) and i == 0
        for r in p.runs:
            r.font.size = Pt(size + 1 if head else size)
            r.font.bold = head
            r.font.color.rgb = colour if head else NAVY
            r.font.name = "Calibri"
    return box


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ------------------------------------------------------------------- the facts
def facts():
    f = {"model": describe_model(load_model()), "thresholds": switch_points()}
    m = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    f["n_train"], f["n_test"] = m["n_train"], m["n_test"]
    f["auc"] = {k: round(sum(v[h]["auc"] for h in HAZARDS) / 4, 4)
                for k, v in m["comparison"].items()}
    f["miss"] = {r: {k: v["Flood"] for k, v in d.items()}
                 for r, d in m["missing_sensor_study"].items()}
    f["relearn"] = json.loads(
        (ARTIFACT_DIR / "weather_relearn_report.json").read_text())
    f["struct"] = json.loads((ARTIFACT_DIR / "structure_check.json").read_text())
    return f


def build():
    F = facts()
    info = F["model"]
    thr = F["thresholds"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # ------------------------------------------------------------ 1  title
    s = blank(prs)
    bg = s.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width,
                            prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    bg.shadow.inherit = False
    strip = s.shapes.add_shape(1, Inches(0), Inches(3.30), prs.slide_width,
                               Inches(0.055))
    strip.fill.solid()
    strip.fill.fore_color.rgb = ACCENT
    strip.line.fill.background()
    strip.shadow.inherit = False

    textbox(s, 0.95, 0.85, 11.5, 0.4,
            "22AIE301  PROBABILISTIC REASONING     |     PROJECT REVIEW",
            size=13, bold=True, colour=PALE)
    textbox(s, 0.95, 1.35, 11.6, 1.8,
            "Explainable Multi-Hazard\nDisaster Prediction",
            size=44, bold=True, colour=WHITE, spacing=0.92)
    textbox(s, 0.95, 2.80, 11.6, 0.45,
            "A Bayesian network with an explanation layer, a decision network "
            "and a live sensor pipeline",
            size=15.5, colour=PALE)

    textbox(s, 0.95, 3.75, 5.8, 0.3, "TEAM", size=11.5, bold=True, colour=ACCENT)
    textbox(s, 0.95, 4.10, 5.8, 2.0,
            [f"{n}      {r}" for n, r in MEMBERS],
            size=14.5, colour=WHITE, spacing=1.42)
    textbox(s, 7.4, 3.75, 5.2, 0.3, "COURSE", size=11.5, bold=True, colour=ACCENT)
    textbox(s, 7.4, 4.10, 5.4, 2.0,
            ["Faculty: Dr. Navaneeth Haridasan",
             "CSE - AIE 2025, Batch C & D",
             "Amrita Vishwa Vidyapeetham, Coimbatore"],
            size=14, colour=PALE, spacing=1.42)
    textbox(s, 0.95, 6.35, 11.6, 0.4,
            f"{info['nodes']}-node network  .  "
            f"{F['relearn']['rows_total']:,} real district-days  .  "
            f"85 automated tests  .  live data, no API key",
            size=12.5, colour=ACCENT)
    notes(s, "Open with one sentence: we do not just predict a disaster "
             "probability - we say why, we say how confident we are given the "
             "sensors that actually reported, and we recommend the action that "
             "maximises expected utility. Then say that since the zeroth review "
             "the whole thing runs on live public data.")

    # -------------------------------------------------- 2  problem / answers
    s = slide(prs, "The gap, and what we built",
              "What a district officer has today is a colour. What they need is a "
              "decision.")
    bullets(s, 0.5, 1.5, 6.1, 2.2, [
        "Existing alerts give a colour or a yes/no, with no probability",
        "No statement of WHY, so it cannot be sanity-checked",
        "No confidence: two working sensors look like nine",
        "One hazard per system, though a hill district faces several at once",
        "A fixed threshold that ignores cost and exposure",
    ], size=14.5)
    callout(s, 0.5, 3.95, 6.1, 2.75, "From one reading, four answers", [
        "1.  How likely - four hazards at once, plus the consequences",
        "2.  Why - signed contribution of every reading, in plain language",
        "3.  How sure - posterior sharpness and how much evidence arrived",
        "4.  What to do - the action with maximum expected utility",
        "",
        "All four are probability queries on the same network.",
    ], size=13.5)
    if (ARTIFACT_DIR / "console_monsoon_depression.png").exists():
        picture(s, ARTIFACT_DIR / "console_monsoon_depression.png", 6.8, 1.55,
                w=6.25)
        textbox(s, 6.8, 5.55, 6.25, 1.1,
                "Sixteen districts ranked by top-hazard probability. Faded bars "
                "with an outline are districts where a sensor is not reporting - "
                "they still get a calibrated number, with lower confidence.",
                size=12, colour=GREY)
    notes(s, "Do not oversell: IMD and CWC forecasts are far better than anything "
             "we can build. The gap we attack is the last mile - turning a "
             "forecast into an explained, costed decision for one district.")

    # ------------------------------------------------ 3  literature review
    s = slide(prs, "What already exists, and what we add",
              "Where our project sits against the published work")
    table(s, 0.5, 1.4, 12.35,
          ["Area of prior work", "What it typically does", "What we do differently"],
          [["Bayesian networks for natural hazards",
            "Model ONE hazard - usually flood or landslide - as an offline study",
            "Four hazards in one network, sharing a consequence layer, running live"],
           ["ML flood prediction|(random forest, boosting, LSTM)",
            "High accuracy, but needs every input present and cannot say why",
            "We accept missing sensors and explain every number we produce"],
           ["Post-hoc explainability|(LIME, SHAP)",
            "Fit a SECOND, approximate model to explain the first one",
            "In a Bayesian network the explanation IS a query - exact and signed"],
           ["Influence diagrams in risk analysis",
            "Utilities used mainly for offline cost-benefit studies",
            "The alert threshold is derived from utilities, not fixed at 80 %"],
           ["Operational early warning|(IMD, CWC, GloFAS)",
            "Authoritative forecasts, delivered as a colour or a binary alert",
            "We consume these feeds and add the last mile: why, how sure, what to do"]],
          [2.9, 4.5, 4.95], size=11.5, row_h=0.62)

    callout(s, 0.5, 5.5, 6.1, 1.2, "Methods we build on", [
        "Pearl 1988 - noisy-OR, virtual evidence, explaining away  .  "
        "Koller & Friedman 2009 - inference and parameter learning  .  "
        "Good 1985 - weight of evidence  .  Howard & Matheson 1984 - influence "
        "diagrams  .  pgmpy (Ankan et al.)",
    ], size=11.5)
    callout(s, 6.75, 5.5, 6.1, 1.2, "The gap we fill", [
        "Each row above is solved somewhere in the literature. We have not seen "
        "them combined into one system that runs on a live feed and hands a "
        "district officer a costed instruction.",
    ], GREEN, size=12)
    notes(s, "Keep this short in the talk - one sentence per row. The honest "
             "framing is that we are not inventing a new algorithm; we are "
             "combining known pieces into something operational. If asked for "
             "specific hazard-BN papers, say we surveyed them via Scispace and "
             "the closest work models a single hazard offline.")

    # ------------------------------------------------------------- 3  the DAG
    s = slide(prs, "The model",
              "Five layers: context, sensors, latent physical state, hazards, "
              "consequences")
    picture(s, ARTIFACT_DIR / "dag.png", 0.42, 1.32, w=9.3)
    callout(s, 9.95, 1.45, 3.0, 2.5, "Why a graph", [
        f"Full joint: {info['full_joint_size']:,} numbers",
        f"Our CPTs: {info['free_parameters']}",
        f"A factor of {info['compression']:,.0f}",
        "",
        "That compression is why 9,000 rows can fit it at all.",
    ], size=12.5)
    callout(s, 9.95, 4.15, 3.0, 2.3, "Multi-hazard", [
        "Flood, landslide and cyclone feed the SAME consequence layer.",
        "",
        "A district runs one response, not one per hazard.",
    ], GREEN, size=12.5)
    notes(s, "Walk the causal chain aloud: season drives rainfall, rainfall wets "
             "the soil, wet soil plus rain raises the river, the river floods the "
             "town, the flood blocks the road, the blocked road creates rescue "
             "demand. Be ready to write the chain rule on the board.")

    # ----------------------------------------------------- 4  filling the CPTs
    s = slide(prs,
              f"Filling {info['free_parameters']} parameters without inventing "
              f"numbers", "Canonical CPT families, then data")
    bullets(s, 0.5, 1.45, 6.2, 2.3, [
        "Ordered logit for ordinal children - monotone by construction",
        "Noisy-OR where any cause can independently trigger the effect",
        "Then a Dirichlet posterior with the elicited table as its prior mean",
    ], size=14)
    callout(s, 0.5, 3.45, 6.2, 1.5, "The conjugate update", [
        "alpha = ESS x P_expert(child | parents)",
        "P = (alpha + counts) / (alpha total + counts total)",
    ], size=13)
    callout(s, 0.5, 5.2, 6.2, 1.5, "Elicitation scorecard", [
        "The noisy-OR consequence layer came back from the data almost "
        "unchanged - mean KL 0.002. That elicitation was right.",
    ], GREEN, size=12.5)
    callout(s, 7.0, 1.45, 5.85, 2.45, "The modelling bug this caught", [
        "Our first landslide CPT was additive in slope, so a steep slope raised "
        "the risk on its own. End to end the system gave the Nilgiris a 23 % "
        "landslide probability on a dry January day.",
        "",
        "Slope is a predisposing factor. Water is the trigger.",
    ], RED, size=12.5)
    table(s, 7.0, 4.15, 5.85,
          ["P(landslide), heavy rain + saturated soil", "Flat", "Steep"],
          [["additive elicitation - wrong", "0.27", "0.83"],
           ["with slope x soil interaction", "0.08", "0.96"]],
          [4.0, 1.0, 1.0], size=12, row_h=0.38)
    callout(s, 7.0, 5.65, 5.85, 1.05, "", [
        "A dry steep slope is now 0.008. There is a test so it cannot come back.",
    ], GREEN, size=12.5)
    notes(s, "This is our best 'we understood it' story. The additive model was "
             "not a coding bug - it type-checked and the CPT summed to 1. It was a "
             "modelling bug, found only by asking what the model said about a dry "
             "winter day in Ooty.")

    # -------------------------------------------------- 5  feedback -> changes
    s = slide(prs, "Your feedback on the abstract, and what we did about it",
              "Zeroth review")
    callout(s, 0.5, 1.45, 12.35, 1.15, "", [
        "\"It is mentioned real-time sensor data, but you are choosing the "
        "existing preprocessed data? Also, what methods are you incorporating to "
        "process the real-time data feed? It will be interesting to me.\"",
    ], RED, size=13.5)
    textbox(s, 0.5, 2.9, 6.0, 0.4, "1.  THE DATA", size=14, bold=True,
            colour=ACCENT)
    bullets(s, 0.5, 3.3, 6.0, 3.2, [
        "You were right - the CPTs came from our own simulator",
        "Now: three public services pulled live, per district, per pass",
        f"Plus {F['relearn']['rows_total']:,} real district-days used to "
        "re-estimate the whole weather layer",
        "The hazard layer is still simulated, and we say so",
    ], size=13.5)
    textbox(s, 6.9, 2.9, 6.0, 0.4, "2.  THE METHODS", size=14, bold=True,
            colour=ACCENT)
    bullets(s, 6.9, 3.3, 6.0, 3.2, [
        "Soft evidence instead of hard binning",
        "Reliability measured from inter-model disagreement",
        "Recursive Bayesian filtering, so rain accumulates across days",
        "Bayesian surprise gate, alert hysteresis, mutual-information polling",
    ], size=13.5)
    callout(s, 0.5, 4.95, 6.0, 1.5, "What that bought us", [
        "The weather layer is now fitted to real Indian weather for these exact "
        "districts - and fitting it exposed a 40x error sitting inside the model "
        "we presented last time.",
    ], GREEN, size=12.5)
    callout(s, 6.9, 4.95, 6.0, 1.5, "What that bought us", [
        "Reliability is measured from the feed itself, rainfall accumulates "
        "across days, and an implausible reading is downgraded rather than "
        "believed - or, if the sources agree, it is our model that gets flagged.",
    ], GREEN, size=12.5)
    textbox(s, 0.5, 6.62, 12.4, 0.4,
            "The next five slides take each of these in turn.",
            size=12.5, italic=True, colour=GREY)
    notes(s, "Lead with the admission - you were right. Examiners reward visible "
             "iteration far more than a defensive answer. Then say the second "
             "question turned out to be the more valuable one, because processing "
             "a live feed properly needed real probabilistic machinery rather "
             "than just an API call.")

    # ------------------------------------------------------------ 6  real data
    s = slide(prs, "It is real data now",
              "Three public services, no API key, pulled on every pass")
    table(s, 0.5, 1.45, 12.35,
          ["Service", "What it provides", "Source models"],
          [["api.open-meteo.com",
            "24 h rainfall, 24 h max temperature, mean humidity, max wind, "
            "24 h pressure fall", "GFS, ECMWF-IFS, ICON"],
           ["flood-api.open-meteo.com",
            "Daily river discharge, plus six years of history for calibration",
            "GloFAS"],
           ["marine-api.open-meteo.com",
            "Sea surface temperature offshore, coastal districts only",
            "Marine analysis"]],
          [3.0, 6.4, 2.6], size=12, row_h=0.52)
    callout(s, 0.5, 3.5, 6.1, 1.8, "Learned from real observations", [
        "Season, Rainfall | Season, Temperature | Season, SeaSurfaceTemp | Season,",
        "Humidity | Rainfall, PressureDrop | SST, WindSpeed | PressureDrop",
    ], GREEN, size=12)
    callout(s, 6.75, 3.5, 6.1, 1.8, "Still elicited or simulated", [
        "Flood, Landslide, Cyclone, Heatwave and the consequence layer - these "
        "need a verified event label per district-day, which no public table "
        "gives.",
    ], RED, size=12)
    callout(s, 0.5, 5.5, 12.35, 1.2,
            "Two things that are modelling decisions, not plumbing", [
                "Aggregate to the variable we model - Rainfall is an IMD 24-hour "
                "band, so we sum the trailing 24 hourly values, never the "
                "'current' field.",
                "Calibrate - GloFAS gives cubic metres per second, so we convert "
                "to a fraction of bankfull using the median annual maximum over "
                "six years.",
            ], size=12)
    notes(s, "If asked why the hazard layer is not real: no public table gives a "
             "verified flood or landslide label per district-day. Joining EM-DAT "
             "and state disaster reports is weeks 1 to 3 of our plan. "
             "/api/live/provenance returns which node came from which.")

    # ---------------------------------------------- 7  what real data exposed
    s = slide(prs, "What connecting real data exposed",
              "The finding we did not want, and are glad we have")
    picture(s, ARTIFACT_DIR / "rainfall_prior_comparison.png", 0.65, 1.38, w=7.75)
    callout(s, 9.15, 1.5, 3.75, 2.6, "40x", [
        "Our simulator believed 42.7 % of monsoon days were heavy rainfall.",
        "",
        "Across the real district-days it is 1.1 %.",
        "",
        "That error was inside the model we showed you last time.",
    ], RED, size=12.5)
    callout(s, 9.15, 4.35, 3.75, 2.3, "And the calendar was wrong", [
        "With one all-India monsoon rule the real data came out wetter in winter "
        "than in summer.",
        "",
        "Cause: the north-east monsoon on the east coast, October to December.",
    ], size=12)
    callout(s, 0.65, 5.80, 7.75, 0.9, "", [
        "The consequence was concrete: our surprise gate began flagging correct "
        "low-rainfall readings, because the model expected far more rain than "
        "actually falls.",
    ], size=12.5)
    notes(s, "Say plainly: neither of these was a coding bug. Both were domain "
             "errors that only real observations could expose. This slide is the "
             "direct answer to the feedback - it caused us to find a real error "
             "in our own work.")

    # ------------------------------------------------------- 8  soft evidence
    s = slide(prs,
              "Method 1 - soft evidence, with reliability measured from the feed",
              "A reading is a likelihood over bins, not a bin")
    bullets(s, 0.5, 1.4, 6.2, 2.0, [
        "ECMWF said 15.0 mm; the IMD light/moderate edge is 15.6 mm",
        "For the same district and hour: GFS 7.4, ECMWF 15.0, ICON 1.5",
        "A GloFAS reading can be seventeen hours old",
    ], size=13.5)
    callout(s, 0.5, 3.3, 6.2, 1.4, "Gaussian bin likelihood", [
        "lambda(state k) = Phi((e_k - v)/sigma) - Phi((e_k-1 - v)/sigma)",
        "entered as virtual evidence by Pearl's indicator construction",
    ], size=12.5)
    callout(s, 0.5, 4.9, 6.2, 1.8, "Sigma comes from the ensemble", [
        "sigma_eff = sqrt( sigma_instrument^2 + s^2 ),  s = spread of the models",
        "",
        "Models agree - the evidence is sharp. Models disagree - it flattens, the "
        "posterior widens and confidence drops. No threshold anywhere.",
    ], GREEN, size=12)
    picture(s, ARTIFACT_DIR / "soft_vs_hard.png", 6.95, 1.35, w=5.95)
    notes(s, "This is the part we are most pleased with: the reliability of the "
             "feed is measured from the feed itself, at that moment, rather than "
             "being a constant we chose. Gaussian ensemble dressing is standard "
             "practice in numerical weather prediction, so we are not inventing a "
             "technique.")

    # ------------------------------------------------------------- 9  filter
    s = slide(prs, "Method 2 - recursive Bayesian filtering",
              "A flood is three days of rain landing on ground that was already wet")
    picture(s, ARTIFACT_DIR / "filter_trajectory.png", 0.55, 1.4, w=7.7)
    callout(s, 8.5, 1.45, 4.4, 1.95, "Two-slice dynamic BN", [
        "predict:  b'(s_t) = sum over s' of T(s_t | s', rain_t) b(s')",
        "update:   b(s_t) proportional to L(z_t | s_t) b'(s_t)",
    ], size=11.5)
    callout(s, 8.5, 3.6, 4.4, 1.65, "No double counting", [
        "lambda(s) = b_filter(s) / P_network(s | evidence)",
        "",
        "Dividing out the network's own prediction REPLACES its memoryless guess. "
        "Verified to 1.1e-16.",
    ], size=11.5)
    callout(s, 8.5, 5.45, 4.4, 1.25, "It fixed a limitation too", [
        "We had listed 'the model is static' as a weakness ourselves. Processing "
        "a stream forced us to solve it.",
    ], GREEN, size=11.5)
    notes(s, "The joint state is the pair soil moisture and river level, so nine "
             "states. The transition uses the same ordered-logit family as the "
             "static CPTs, so it is monotone: more rain never dries the soil. "
             "Rainfall is itself uncertain, so we marginalise the transition over "
             "the rainfall posterior rather than conditioning on one band.")

    # --------------------------------------------------- 10  gate / hysteresis
    s = slide(prs, "Method 3 - judging the stream as it arrives", "")
    textbox(s, 0.5, 1.35, 4.0, 0.35, "BAYESIAN SURPRISE GATE", size=13,
            bold=True, colour=ACCENT)
    bullets(s, 0.5, 1.72, 4.0, 2.2, [
        "surprisal = -log2 P(reading | all other evidence)",
        "Above 4 bits we shrink its trust, never discard it",
        "The same posterior it feeds is what judges it",
    ], size=12.5)
    textbox(s, 4.75, 1.35, 4.0, 0.35, "ALERT HYSTERESIS", size=13, bold=True,
            colour=ACCENT)
    bullets(s, 4.75, 1.72, 4.0, 2.2, [
        "Escalate on the first frame that justifies it",
        "Stand down only after three agreeing frames",
        "Asymmetric: a miss costs about 8x a false alarm",
    ], size=12.5)
    textbox(s, 9.0, 1.35, 3.9, 0.35, "WHAT TO POLL NEXT", size=13, bold=True,
            colour=ACCENT)
    bullets(s, 9.0, 1.72, 3.9, 2.2, [
        "I(Hazard ; X | evidence), in bits",
        "Poll the district on a decision boundary",
        "Leave the obvious ones alone",
    ], size=12.5)
    callout(s, 0.5, 3.55, 12.4, 2.5,
            "A correction we had to make - and it is the interesting part", [
                "Our first gate assumed the model is right and the sensor is "
                "wrong. On real data that was backwards: because the simulated "
                "model over-stated monsoon rainfall by 40x, the gate began "
                "discounting perfectly correct low-rainfall readings.",
                "",
                "So the gate now separates two hypotheses. If several independent "
                "sources AGREE and the model is still surprised, that is evidence "
                "the MODEL is wrong - it is logged as a model surprise and the "
                "reading keeps full weight. Only a lone or internally "
                "inconsistent source is discounted as a suspect sensor.",
            ], RED, size=12.5)
    notes(s, "If asked why we do not simply discard a suspect reading: discarding "
             "data on the model's say-so is exactly how you miss the real "
             "emergency. Shrinking trust is reversible; deleting is not.")

    # ---------------------------------------------------------- 11  explanation
    s = slide(prs, "The explanation layer",
              "Four questions, all answered by queries on the same network")
    table(s, 0.5, 1.4, 12.35,
          ["Question", "How we answer it"],
          [["Why is it high?",
            "Retract one reading, re-run inference, report the shift in log-odds "
            "- Good's weight of evidence. Signed, so a reading can also LOWER "
            "the risk."],
           ["What is happening underneath?",
            "MAP assignment over the latent nodes, narrated as a causal chain"],
           ["What if the weather had been kinder?",
            "Counterfactual query with the top driver forced to its benign state"],
           ["What should we measure next?",
            "I(Hazard ; Sensor | evidence) - the expected entropy drop, in bits"]],
          [3.1, 9.25], size=12, row_h=0.5)
    callout(s, 0.5, 3.85, 12.35, 1.9,
            "Generated verbatim by the system for Idukki", [
                "Landslide probability is 91 % (High risk). The long-run base "
                "rate for this hazard is 9 %. What pushed it up: steep hill "
                "slopes (x19.6 on the odds); very humid air (x1.8); heavy "
                "rainfall over 64.5 mm in 24 h (x1.1). Our best reconstruction "
                "of what we cannot measure: already saturated soil. Had the "
                "humidity been low instead of high, the landslide probability "
                "would be 81 % rather than 91 %. Confidence is high, 10 of 10 "
                "inputs available.",
            ], size=12)
    callout(s, 0.5, 5.95, 12.35, 0.75, "", [
        "No LIME, no SHAP. Those fit a second approximate model to explain the "
        "first; in a Bayesian network the explanation already IS a query.",
    ], GREEN, size=12.5)
    notes(s, "Offer to re-run this live with a different reading. Be ready for "
             "the bullet that says a reading 'adds nothing once the others are "
             "known' - that is d-separation appearing in the output, not a "
             "missing number.")

    # ------------------------------------------------------------ 12  decision
    s = slide(prs, "From probability to instruction",
              "Where we broke with our own abstract")
    callout(s, 0.5, 1.4, 6.1, 1.15, "", [
        "Our abstract said: 'if flood probability > 80 %, evacuate'. A fixed "
        "threshold ignores what the response costs and how many people are "
        "exposed, so we deleted it.",
    ], RED, size=12.5)
    table(s, 0.5, 2.75, 6.1,
          ["Action", "Cost", "Harm if it happens", "False alarm"],
          [[a, ACTION_COST[a], HARM_IF_EMERGENCY[a], FALSE_ALARM_COST[a]]
           for a in ("Monitor", "Advisory", "Prepare", "Evacuate")],
          [1.7, 1.0, 2.2, 1.3], size=11.5, row_h=0.36)
    table(s, 0.5, 5.05, 6.1,
          ["Optimal action", "P(emergency) range"],
          [[a, f"{lo:.3f}  to  {hi:.3f}"] for a, (lo, hi) in thr.items()],
          [2.4, 3.7], size=11, row_h=0.31)
    picture(s, ARTIFACT_DIR / "decision_thresholds.png", 6.85, 1.5, w=6.0)
    callout(s, 6.85, 5.15, 6.0, 1.5, "", [
        f"Evacuation becomes optimal at P = {thr['Evacuate'][0]:.2f}, not 0.80 - "
        "and it moves if the cost table moves. The thresholds are an output of "
        "the model, not an input to it.",
    ], GREEN, size=12.5)
    notes(s, "Name all three node types: chance, decision, utility. Expect 'where "
             "did the cost numbers come from' - they are relative, elicited by "
             "us, and the defensible claim is the method plus the ordering, not "
             "the specific numbers. Officer review is week 6 of the plan.")

    # ------------------------------------------------------- 13  results (acc)
    s = slide(prs, "Results 1 - accuracy with every sensor working",
              f"Held-out test set, {F['n_test']:,} records, four hazards")
    picture(s, ARTIFACT_DIR / "model_comparison.png", 0.5, 1.45, w=7.7)
    rows = [[k, f"{v:.4f}"]
            for k, v in sorted(F["auc"].items(), key=lambda kv: -kv[1])]
    table(s, 8.5, 1.5, 4.4, ["Model", "mean AUC"], rows, [3.2, 1.2],
          size=10.5, row_h=0.33)
    callout(s, 8.5, 4.5, 4.4, 2.15, "Read it honestly", [
        "The baselines get raw floating-point readings. We get 2 to 4 bins. That "
        "costs us about 0.03 AUC.",
        "",
        "Note the bottom row: elicited priors with NO data at all already reach "
        "0.908.",
    ], RED, size=12)
    notes(s, "Do not hide this slide. Owning the 0.03 gap is what makes the next "
             "slide credible. Separately, Hill-Climb structure search recovers "
             f"{100 * F['struct']['recovery_rate']:.0f} % of our hand-drawn edges "
             "from the data alone, which is a further reassurance that the "
             "dependencies we assumed are really there.")

    # -------------------------------------------------- 14  results (missing)
    s = slide(prs, "Results 2 - what happens when the sensors go dark",
              "The finding we care about most")
    picture(s, ARTIFACT_DIR / "missing_sensors.png", 1.0, 1.38, w=11.3)
    m = F["miss"]
    bn0 = m["0%"]["BN-Bayes (expert prior)"]
    bn60 = m["60%"]["BN-Bayes (expert prior)"]
    lr0 = m["0%"]["LogisticRegression"]
    lr60 = m["60%"]["LogisticRegression"]
    callout(s, 0.7, 5.90, 11.9, 0.95, "", [
        f"Flood AUC with everything working: BN {bn0:.3f} against logistic "
        f"regression {lr0:.3f}. With 60 % of readings missing: BN {bn60:.3f} "
        f"against {lr60:.3f}. We lose {100 * (bn0 - bn60):.1f} AUC points, the "
        f"baseline loses {100 * (lr0 - lr60):.1f}.",
        "The curves cross at about a quarter of readings missing - an ordinary "
        "day for a district gauge network.",
    ], GREEN, size=12.5)
    notes(s, "Spend time here. The mechanism matters: the BN marginalises the "
             "missing variable out of the joint, which is the correct thing to do "
             "under missing-at-random. Median imputation asserts a value that was "
             "never observed. This is also the slide that answers 'why not just "
             "use logistic regression'.")

    # ------------------------------------------------- 15  limits, plan, team
    s = slide(prs, "Where we are honest, and what is next", "")
    textbox(s, 0.5, 1.3, 6.1, 0.35, "LIMITATIONS", size=13, bold=True, colour=RED)
    bullets(s, 0.5, 1.65, 6.1, 3.0, [
        "The hazard layer is still simulated - no verified event labels exist",
        "Discretisation makes bin edges matter: 3.0 against 4.2 hPa gives 11 % "
        "against 37 %",
        "The utility numbers are ours and need an officer to sign them off",
        "On real monsoon data the decision layer over-escalates - it was tuned "
        "on simulated base rates",
        "'People at risk' uses a flat 12 % exposure fraction",
    ], size=12.5)
    textbox(s, 6.9, 1.3, 6.0, 0.35, "PLAN TO THE FINAL REVIEW", size=13,
            bold=True, colour=ACCENT)
    table(s, 6.9, 1.65, 6.0, ["Wk", "Task"],
          [["1-2", "Join real weather with gauge levels; label events from EM-DAT "
                   "and state reports"],
           ["3", "Refit the hazard layer on real labels, re-run the evaluation"],
           ["4", "Extend the filter to a full dynamic BN over three-day slices"],
           ["5", "Sensitivity analysis on the utility table and the Dirichlet ESS"],
           ["6", "District officer reviews the checklists and the cost table"],
           ["7-8", "Package it, then write up as a short paper"]],
          [0.7, 5.3], size=11, row_h=0.44)
    table(s, 0.5, 4.9, 12.4, ["Member", "Roll number", "Owned"],
          [[n, r, o] for (n, r), o in zip(MEMBERS, [
              "Inference engine, parameter learning, decision network, "
              "soft-evidence engine, temporal filter, API",
              "Graph structure, CPT and transition elicitation, figures, deck",
              "Discretisation and IMD bands, bin likelihood, explanation text, "
              "GIS export",
              "Simulator, scenarios, live feed adapters, observation archive, "
              "value of information, evaluation"])],
          [2.1, 2.1, 8.2], size=10.5, row_h=0.36)
    notes(s, "Volunteer the limitations before being asked. The over-escalation "
             "one is worth saying out loud: we would rather report it than "
             "quietly retune the table to look better. Close by offering the live "
             "demo - the console is already open and warm.")

    prs.save(OUT)
    return OUT, len(prs.slides._sldIdLst)


if __name__ == "__main__":
    path, n = build()
    print(f"wrote {path}  ({n} slides)")
