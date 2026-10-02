"""
Builds the SIMPLE-ENGLISH review deck: report/Review_Simple_English.pptx

Same project, same real numbers, same figures as make_ppt.py - but the wording is
plain and every technical term is explained the first time it appears. Fifteen
slides. Use this one if you want the audience to follow the reasoning rather than
be impressed by the vocabulary.

The professional version is still built by make_ppt.py. Both read the same
artifacts, so their numbers can never disagree.

    python report/make_ppt_simple.py

Owner: Rohit Vardhan M (layout), wording agreed by all four of us
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

OUT = ROOT / "report" / "Review_Simple_English.pptx"

NAVY = RGBColor(0x14, 0x2F, 0x4A)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
GREY = RGBColor(0x5B, 0x6B, 0x7A)
LIGHT = RGBColor(0xEF, 0xF5, 0xFA)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x7A, 0x4C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xBD, 0xD3, 0xE4)

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
            align=PP_ALIGN.LEFT, spacing=1.05, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else list(text)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(4)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = colour
            r.font.name = "Calibri"
    return tb


def bullets(slide, x, y, w, h, items, size=16, spacing=1.15, colour=NAVY):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        text, lvl = (item if isinstance(item, tuple) else (item, 0))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("•  " if lvl == 0 else "     –  ") + text
        p.line_spacing = spacing
        p.space_after = Pt(9)
        for r in p.runs:
            r.font.size = Pt(size if lvl == 0 else size - 2)
            r.font.color.rgb = colour if lvl == 0 else GREY
            r.font.name = "Calibri"
    return tb


def band(slide, prs, title, kicker=None):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width,
                                 Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    textbox(slide, 0.45, 0.14, 12.4, 0.64, title, size=26, bold=True,
            colour=WHITE)
    if kicker:
        textbox(slide, 0.48, 1.02, 12.4, 0.4, kicker, size=14, colour=GREY)


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


def table(slide, x, y, w, headers, rows, widths, size=13, header_size=13,
          row_h=0.4):
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
        c.margin_left = c.margin_right = Inches(0.08)
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
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            centred = j > 0 and len(str(val)) <= 14
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if centred else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.color.rgb = NAVY
                    r.font.name = "Calibri"
    return t


def callout(slide, x, y, w, h, title, body, colour=ACCENT, size=14):
    box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = colour
    box.line.width = Pt(1.25)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.16)
    tf.margin_right = Inches(0.14)
    tf.margin_top = Inches(0.11)
    lines = ([title] if title else []) + list(body)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(5)
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
    f["n_test"] = m["n_test"]
    f["auc"] = {k: round(sum(v[h]["auc"] for h in HAZARDS) / 4, 4)
                for k, v in m["comparison"].items()}
    f["miss"] = {r: {k: v["Flood"] for k, v in d.items()}
                 for r, d in m["missing_sensor_study"].items()}
    f["relearn"] = json.loads(
        (ARTIFACT_DIR / "weather_relearn_report.json").read_text())
    return f


def build():
    F = facts()
    info = F["model"]
    thr = F["thresholds"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # ------------------------------------------------------------- 1  title
    s = blank(prs)
    bg = s.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width,
                            prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    bg.shadow.inherit = False
    strip = s.shapes.add_shape(1, Inches(0), Inches(3.35), prs.slide_width,
                               Inches(0.06))
    strip.fill.solid()
    strip.fill.fore_color.rgb = ACCENT
    strip.line.fill.background()
    strip.shadow.inherit = False

    textbox(s, 0.95, 0.9, 11.5, 0.4,
            "22AIE301  PROBABILISTIC REASONING     |     PROJECT REVIEW",
            size=14, bold=True, colour=PALE)
    textbox(s, 0.95, 1.42, 11.6, 1.8,
            "Predicting Floods, Landslides,\nCyclones and Heat Waves",
            size=42, bold=True, colour=WHITE, spacing=0.95)
    textbox(s, 0.95, 2.85, 11.6, 0.5,
            "A system that also explains its answer and tells the officer what "
            "to do",
            size=16, colour=PALE)

    textbox(s, 0.95, 3.85, 5.8, 0.3, "OUR TEAM", size=12, bold=True,
            colour=ACCENT)
    textbox(s, 0.95, 4.2, 5.8, 2.0, [f"{n}      {r}" for n, r in MEMBERS],
            size=15, colour=WHITE, spacing=1.45)
    textbox(s, 7.4, 3.85, 5.2, 0.3, "COURSE", size=12, bold=True, colour=ACCENT)
    textbox(s, 7.4, 4.2, 5.4, 2.0,
            ["Faculty: Dr. Navaneeth Haridasan",
             "CSE - AIE 2025, Batch C & D",
             "Amrita Vishwa Vidyapeetham, Coimbatore"],
            size=14.5, colour=PALE, spacing=1.45)
    notes(s, "Say in one line what the project is: we take today's weather "
             "readings for a district and we tell the officer how likely each "
             "disaster is, why we think so, how sure we are, and what action to "
             "take.")

    # ------------------------------------------------ 2  problem + what it does
    s = slide(prs, "The problem, and what our system does",
              "Today an officer gets a colour. They need a decision.")
    textbox(s, 0.5, 1.5, 6.1, 0.35, "THE PROBLEM", size=15, bold=True,
            colour=RED)
    bullets(s, 0.5, 1.9, 6.1, 2.6, [
        "Warnings say only 'danger' or 'no danger'",
        "They do not say how likely the danger is",
        "They do not say why the computer thinks so",
        "They do not say how many sensors were working",
        "Each system warns about only one disaster",
    ], size=15.5)
    textbox(s, 0.5, 4.55, 6.1, 0.35, "WHAT WE BUILT", size=15, bold=True,
            colour=GREEN)
    bullets(s, 0.5, 4.95, 6.1, 2.2, [
        "One system for four disasters at the same time",
        "It gives a percentage, not just a colour",
        "It explains its answer in normal sentences",
        "It says which action to take, and why",
    ], size=15.5)
    if (ARTIFACT_DIR / "console_monsoon_depression.png").exists():
        picture(s, ARTIFACT_DIR / "console_monsoon_depression.png", 6.85, 1.6,
                w=6.2)
    textbox(s, 6.85, 5.5, 6.2, 1.2,
            "Our screen for 16 districts in Kerala, Tamil Nadu and Andhra "
            "Pradesh. Each bar is one district. Longer bar means higher risk.",
            size=13, colour=GREY)
    notes(s, "Be fair to the existing systems: IMD forecasts are very good. The "
             "part that is missing is the last step - turning a forecast into an "
             "explained decision for one district.")

    # ------------------------------------------------ 3  literature review
    s = slide(prs, "What other people have already done",
              "And what is new in ours")
    table(s, 0.5, 1.45, 12.35,
          ["What others have built", "The limitation", "What we add"],
          [["Bayesian networks for one disaster",
            "Usually only floods, or only landslides, and studied offline",
            "Four disasters together, in one model, running on live data"],
           ["Machine-learning weather models",
            "Very accurate, but need every reading and cannot explain",
            "Ours works with missing readings and explains its answer"],
           ["Tools that explain AI (LIME, SHAP)",
            "They build a second model just to explain the first",
            "Ours explains itself - the explanation is the same model"],
           ["Cost-benefit studies of disasters",
            "Done on paper, afterwards, not inside the system",
            "The cost table is inside ours and decides the action live"],
           ["Government warning systems",
            "Excellent forecasts, but delivered as a colour or a yes/no",
            "We use their data and add: why, how sure, and what to do"]],
          [3.2, 4.6, 4.55], size=12.5, row_h=0.66)

    callout(s, 0.5, 5.6, 12.35, 1.1, "Books and papers our methods come from", [
        "Pearl 1988  .  Koller & Friedman 2009  .  Good 1985  .  "
        "Howard & Matheson 1984  .  pgmpy library  .  IMD rainfall and heat-wave "
        "bands  .  CWC river danger levels  .  GloFAS river data",
    ], size=12.5)
    notes(s, "One line per row is enough. The point to make: we are not inventing "
             "a new algorithm, we are combining known ideas into something that "
             "actually runs and can be acted on.")

    # ------------------------------------------- 3  what is a Bayesian network
    s = slide(prs, "First, what is a Bayesian network?",
              "The main idea of our whole project, in one slide")
    callout(s, 0.5, 1.55, 6.1, 2.15, "In simple words", [
        "It is a picture of causes and effects.",
        "",
        "Every arrow means 'this thing affects that thing'.",
        "",
        "Every node has a small table saying how strongly.",
    ], size=15)
    callout(s, 0.5, 3.95, 6.1, 2.75, "An example from our project", [
        "Rain  ->  wet soil  ->  river rises  ->  flood",
        "",
        "If we know it rained heavily, the chance of wet soil goes up. If the "
        "soil is wet, the chance of the river rising goes up. And so on down "
        "the chain.",
        "",
        "The computer does this with probability, not with fixed rules.",
    ], GREEN, size=14)
    textbox(s, 7.0, 1.55, 5.85, 0.35, "WHY WE CHOSE IT", size=15, bold=True,
            colour=ACCENT)
    bullets(s, 7.0, 1.95, 5.85, 4.7, [
        "It works even when some sensors are broken. It simply uses whatever "
        "readings did arrive.",
        "It can explain itself, because we can ask it 'what changed the "
        "answer?' and it replies with numbers.",
        "We can put our own knowledge into it before we have any data. A "
        "normal machine-learning model cannot do that.",
        "One model answers many questions, not just one.",
    ], size=15)
    notes(s, "This slide is for the audience, not for the examiner. If someone "
             "in the room does not know what a Bayesian network is, everything "
             "after this will be lost on them. Take your time here.")

    # ------------------------------------------------------------ 4  our model
    s = slide(prs, "Our model",
              "18 things we track, and the arrows between them")
    picture(s, ARTIFACT_DIR / "dag.png", 0.42, 1.35, w=9.3)
    callout(s, 9.95, 1.5, 3.0, 3.2, "Read it left to right", [
        "Grey: things we already know, like the season and the slope of the "
        "land.",
        "",
        "Blue: what the sensors measure.",
        "",
        "Purple: things we cannot measure, so the model guesses them.",
        "",
        "Red: the four disasters.",
        "",
        "Orange: what happens afterwards.",
    ], size=12)
    callout(s, 9.95, 4.95, 3.0, 1.7, "Why this saves us", [
        f"A full table would need {info['full_joint_size']:,} numbers.",
        f"Our model needs only {info['free_parameters']}.",
    ], GREEN, size=13)
    notes(s, "Point at the chain while you speak: season, rain, soil, river, "
             "flood, road blocked, rescue needed. Then say the important part - "
             "flood, landslide and cyclone all point to the SAME boxes on the "
             "right, because a district runs one rescue operation, not three.")

    # -------------------------------------------------- 5  where numbers come from
    s = slide(prs, "Where the numbers inside the model come from", "")
    bullets(s, 0.5, 1.5, 6.2, 2.6, [
        "Every arrow needs numbers. For example: if the rain is heavy and the "
        "soil is already wet, how likely is a flood?",
        "We did not type these one by one. There would be too many, and we "
        "could not defend them.",
        "Instead we wrote a small formula with a few weights, and let the "
        "computer fill the full table.",
        "Then we corrected those numbers using data.",
    ], size=15)
    callout(s, 0.5, 4.35, 6.2, 2.3, "Simple version of the maths", [
        "Start with what we believe.",
        "Add what the data shows.",
        "The answer sits between the two.",
        "",
        "If we have lots of data, the data wins. If we have very little, our "
        "knowledge fills the gap.",
    ], size=14)
    callout(s, 7.0, 1.5, 5.85, 2.6, "A mistake we found ourselves", [
        "Our first version said a steep hill was dangerous even when the ground "
        "was completely dry.",
        "",
        "It gave the Nilgiris a 23 % chance of a landslide on a dry day in "
        "January. That is clearly wrong.",
    ], RED, size=14)
    callout(s, 7.0, 4.35, 5.85, 2.3, "What we learned and fixed", [
        "A steep slope alone is not dangerous. A steep slope PLUS water is "
        "dangerous.",
        "",
        "We changed the formula so the two work together. The dry hill is now "
        "0.008, and we wrote a test so this mistake cannot come back.",
    ], GREEN, size=14)
    notes(s, "This is a good story to tell because it shows we checked our own "
             "model instead of trusting it. The computer did not give an error - "
             "the numbers were valid. It was our thinking that was wrong.")

    # ------------------------------------------------------------- 6  feedback
    s = slide(prs, "Sir's feedback, and what we changed", "")
    callout(s, 0.5, 1.5, 12.35, 1.2, "", [
        "\"It is mentioned real-time sensor data, but you are choosing the "
        "existing preprocessed data? Also, what methods are you incorporating "
        "to process the real-time data feed? It will be interesting to me.\"",
    ], RED, size=14.5)
    textbox(s, 0.5, 3.0, 6.0, 0.35, "PROBLEM 1 - THE DATA", size=15, bold=True,
            colour=ACCENT)
    bullets(s, 0.5, 3.4, 6.0, 1.6, [
        "Sir was correct. We were using data we made ourselves.",
        "Now the system takes live readings from the internet, every time it "
        "runs.",
    ], size=14.5)
    textbox(s, 6.9, 3.0, 6.0, 0.35, "PROBLEM 2 - THE METHOD", size=15,
            bold=True, colour=ACCENT)
    bullets(s, 6.9, 3.4, 6.0, 1.6, [
        "Taking data from a website is easy. Using it correctly is not.",
        "We added four methods for handling live readings. They are the next "
        "five slides.",
    ], size=14.5)
    callout(s, 0.5, 5.15, 12.35, 1.5, "Why the second question was the better one", [
        "Live readings are messy. They arrive late, they disagree with each "
        "other, and sometimes a sensor is broken. If we just used the numbers "
        "as they came, our answers would be wrong. So we had to think about "
        "each of those problems separately.",
    ], GREEN, size=14)
    notes(s, "Start by agreeing with sir. Do not defend the old version. Then "
             "say the second question was harder and more useful, and that it "
             "led us to find a real mistake in our own model.")

    # ------------------------------------------------------------ 7  real data
    s = slide(prs, "Where our data comes from now",
              "All free, all public, no password needed")
    table(s, 0.5, 1.55, 12.35,
          ["Website", "What we get from it", "How often"],
          [["Open-Meteo weather",
            "Rain, temperature, humidity, wind, air pressure",
            "every time we run"],
           ["Open-Meteo flood",
            "How much water is flowing in the river",
            "every day"],
           ["Open-Meteo marine",
            "Sea temperature near the coast",
            "every time we run"]],
          [3.0, 6.5, 2.5], size=13.5, row_h=0.55)
    callout(s, 0.5, 3.85, 6.1, 1.5, "We also downloaded the past", [
        f"{F['relearn']['rows_total']:,} days of real weather for our 16 "
        "districts, from 2019 to 2024. We used it to correct the numbers "
        "inside our model.",
    ], GREEN, size=14)
    callout(s, 6.75, 3.85, 6.1, 1.5, "What is still not real", [
        "We do not have a list saying 'a flood happened here on this date'. "
        "So that part of the model is still based on our own data. We say this "
        "openly.",
    ], RED, size=14)
    callout(s, 0.5, 5.6, 12.35, 1.1, "One important detail", [
        "The weather website gives rain every hour. Our model needs the rain "
        "for a full day, because that is how the Indian Meteorological "
        "Department defines heavy rain. So we add up the last 24 hours. Small "
        "detail, but using the wrong one would make the model answer a "
        "different question.",
    ], size=13.5)
    notes(s, "If asked why the disaster part is not real data: no public list "
             "gives verified flood or landslide dates for each district. "
             "Building that list is the first job in our plan.")

    # ----------------------------------------------- 8  what real data showed
    s = slide(prs, "What we found when we used real data",
              "The most useful thing that happened to us this semester")
    picture(s, ARTIFACT_DIR / "rainfall_prior_comparison.png", 0.65, 1.45,
            w=7.75)
    callout(s, 9.15, 1.55, 3.75, 2.9, "Our model was very wrong", [
        "We believed 43 out of every 100 monsoon days had heavy rain.",
        "",
        "The real answer is about 1 in 100.",
        "",
        "That is 40 times too high, and it was inside the model we showed in "
        "the last review.",
    ], RED, size=13.5)
    callout(s, 9.15, 4.7, 3.75, 1.95, "Why this matters", [
        "Because the model expected too much rain, it started thinking correct "
        "readings were mistakes.",
        "",
        "Only real data could show us this.",
    ], size=13.5)
    callout(s, 0.65, 5.85, 7.75, 0.85, "", [
        "This was not a programming error. The code was fine. Our understanding "
        "of Indian rainfall was wrong.",
    ], size=13.5)
    notes(s, "Do not hide this slide - it is the strongest thing we have. It "
             "proves that connecting real data changed the project, which is "
             "exactly what sir asked for.")

    # ---------------------------------------------------- 9  soft evidence
    s = slide(prs, "Method 1 - a reading is not an exact fact",
              "How we deal with numbers that sit near a boundary")
    callout(s, 0.5, 1.5, 6.2, 2.0, "The problem", [
        "Rain below 15.6 mm is called 'light'. Above it is called 'moderate'.",
        "",
        "One website said 15.0 mm. Should we call that light? It is only 0.6 mm "
        "away from the other group.",
    ], RED, size=14)
    callout(s, 0.5, 3.75, 6.2, 1.5, "Our answer", [
        "We do not force it into one group. We say it is probably light, and "
        "possibly moderate, and we carry both possibilities forward.",
    ], GREEN, size=14)
    callout(s, 0.5, 5.5, 6.2, 1.2, "The second problem", [
        "Three weather services gave 7.4, 15.0 and 1.5 mm for the same place "
        "and the same hour.",
    ], size=14)
    picture(s, ARTIFACT_DIR / "soft_vs_hard.png", 6.95, 1.5, w=5.95)
    textbox(s, 6.95, 5.4, 5.95, 1.3,
            "When the three services agree, we trust the reading strongly. "
            "When they disagree, we automatically trust it less. Nobody has to "
            "decide this by hand - the disagreement itself decides.",
            size=13, colour=GREY)
    notes(s, "The clever part is the last sentence: we do not choose how much to "
             "trust a reading. The amount the three services disagree tells us "
             "how much to trust it.")

    # -------------------------------------------------------- 10  the filter
    s = slide(prs, "Method 2 - remembering the last few days",
              "One day of rain is not a flood. Three days can be.")
    picture(s, ARTIFACT_DIR / "filter_trajectory.png", 0.55, 1.45, w=7.6)
    callout(s, 8.4, 1.55, 4.5, 2.4, "The problem", [
        "Our first model looked at only one day at a time.",
        "",
        "But a flood happens when rain falls on ground that is ALREADY wet from "
        "the days before.",
    ], RED, size=13.5)
    callout(s, 8.4, 4.2, 4.5, 2.45, "What we do now", [
        "The system keeps a memory of how wet the soil is and how high the "
        "river is.",
        "",
        "Each day it updates that memory using the new rain.",
        "",
        "In the picture: risk builds up over three rainy days, then slowly "
        "comes down when the rain stops.",
    ], GREEN, size=13.5)
    notes(s, "We had already written down 'our model has no memory' as a "
             "weakness in the last review. Working with live data forced us to "
             "fix it, so this slide answers an old criticism as well as a new "
             "one.")

    # ------------------------------------------------- 11  gate + hysteresis
    s = slide(prs, "Method 3 - can we trust this reading?", "")
    textbox(s, 0.5, 1.4, 3.9, 0.35, "IS THE READING SENSIBLE?", size=14,
            bold=True, colour=ACCENT)
    bullets(s, 0.5, 1.8, 3.9, 2.4, [
        "The model checks if a new reading makes sense with everything else",
        "If it looks very strange, we use it but trust it less",
        "We never throw a reading away - that is how you miss a real disaster",
    ], size=14)
    textbox(s, 4.7, 1.4, 3.9, 0.35, "NO FLICKERING WARNINGS", size=14,
            bold=True, colour=ACCENT)
    bullets(s, 4.7, 1.8, 3.9, 2.4, [
        "A warning that turns on and off looks unreliable",
        "So we raise the alert immediately when needed",
        "But we only lower it after three calm readings in a row",
    ], size=14)
    textbox(s, 8.9, 1.4, 4.0, 0.35, "WHAT TO CHECK NEXT", size=14, bold=True,
            colour=ACCENT)
    bullets(s, 8.9, 1.8, 4.0, 2.4, [
        "The system says which missing reading would help the most",
        "For example: 'go and read the river gauge'",
        "This saves time when many districts need attention",
    ], size=14)
    callout(s, 0.5, 4.5, 12.4, 2.15, "A mistake we made, and how we fixed it", [
        "At first, if a reading looked strange, we assumed the SENSOR was "
        "wrong. But when we used real data, our model was the thing that was "
        "wrong, so it started rejecting perfectly good readings.",
        "",
        "Now we check first: if several different weather services all agree "
        "and our model is still surprised, then the problem is our model, not "
        "the sensor. We keep the reading and make a note to check the model.",
    ], RED, size=14)
    notes(s, "This follows directly from slide 8. Because the model thought "
             "monsoon days were much wetter than they are, it was surprised by "
             "normal readings. That is what taught us to separate the two "
             "cases.")

    # ---------------------------------------------------------- 12  explanation
    s = slide(prs, "How the system explains its answer",
              "This is the part an officer actually reads")
    callout(s, 0.5, 1.5, 12.35, 2.0, "A real answer from our system", [
        "\"Landslide probability is 91 %. Normally this hazard happens on 9 % "
        "of days. What pushed it up: steep hill slopes, very humid air, and "
        "heavy rainfall over 64.5 mm in 24 hours. What we think is happening "
        "underground: the soil is already saturated. If the humidity had been "
        "low instead of high, the answer would be 81 % instead of 91 %. We are "
        "confident, because all 10 inputs were available.\"",
    ], size=14)
    table(s, 0.5, 3.85, 12.35,
          ["The question", "How we answer it"],
          [["Why is the risk high?",
            "We remove one reading, ask the model again, and see how much the "
            "answer changes"],
           ["What is happening that we cannot see?",
            "The model gives its best guess for soil and river, which have no "
            "sensor"],
           ["What if the weather had been better?",
            "We change one reading to a safe value and ask again"],
           ["What should we measure next?",
            "We calculate which missing reading would reduce our doubt the "
            "most"]],
          [4.0, 8.35], size=13, row_h=0.52)
    callout(s, 0.5, 6.15, 12.35, 0.75, "", [
        "All four answers come from the same model. We did not build a second "
        "program to explain the first one.",
    ], GREEN, size=13.5)
    notes(s, "Offer to run this live for any district the examiner picks. The "
             "sentences are generated by the program, not written by us.")

    # ------------------------------------------------------------ 13  decision
    s = slide(prs, "How the system chooses what to do",
              "From a percentage to an instruction")
    callout(s, 0.5, 1.5, 6.1, 1.3, "", [
        "In our abstract we wrote: 'if flood chance is above 80 %, evacuate'. "
        "We removed that. A fixed number ignores how much an evacuation costs "
        "and how many people live there.",
    ], RED, size=13.5)
    textbox(s, 0.5, 3.0, 6.1, 0.35, "WE GIVE THE COMPUTER A COST TABLE",
            size=14, bold=True, colour=ACCENT)
    table(s, 0.5, 3.4, 6.1,
          ["Action", "Cost now", "Damage if disaster comes"],
          [[a, ACTION_COST[a], HARM_IF_EMERGENCY[a]]
           for a in ("Monitor", "Advisory", "Prepare", "Evacuate")],
          [2.0, 1.7, 2.9], size=13, row_h=0.42)
    callout(s, 0.5, 5.75, 6.1, 1.0, "", [
        "Doing nothing is free, but very expensive if a disaster comes. "
        "Evacuating is expensive, but then the damage is small.",
    ], size=13)
    picture(s, ARTIFACT_DIR / "decision_thresholds.png", 6.85, 1.55, w=6.0)
    textbox(s, 6.85, 4.95, 6.0, 0.4,
            "How to read it: for each risk level, the highest line is the best "
            "action.", size=12.5, colour=GREY)
    callout(s, 6.85, 5.35, 6.0, 1.35, "The important result", [
        f"The computer works out by itself that evacuation is worth it above "
        f"{thr['Evacuate'][0]:.0%}, not 80 %. If we change the costs, this "
        f"number changes too.",
    ], GREEN, size=13.5)
    notes(s, "The key sentence: we did not choose the threshold. We chose the "
             "costs, and the threshold came out of the calculation. If a "
             "district officer disagrees with our costs, the system gives a "
             "different answer automatically.")

    # ------------------------------------------------------------- 14  results
    s = slide(prs, "Results - how well does it work?",
              f"Tested on {F['n_test']:,} days the model had never seen")
    picture(s, ARTIFACT_DIR / "missing_sensors_simple.png", 2.65, 1.45, w=8.0)
    callout(s, 0.5, 5.75, 6.1, 1.15, "When all sensors work", [
        "Ordinary machine-learning models are slightly better than us - about "
        "3 marks out of 100. That is the price of grouping numbers into bands.",
    ], size=13.5)
    callout(s, 6.75, 5.75, 6.1, 1.15, "When sensors stop working", [
        "With 60 % of readings missing we still score 0.863. The ordinary model "
        "drops to 0.789. This is the situation in real districts.",
    ], GREEN, size=13.5)
    notes(s, "Explain the graph simply: going right means more broken sensors. "
             "The thick blue line is ours. It stays almost flat, while the "
             "dotted lines fall down. That is the main result of the project.")

    # --------------------------------------------------- 15  limits, plan, team
    s = slide(prs, "What is not finished, and who did what", "")
    textbox(s, 0.5, 1.35, 6.1, 0.35, "WHAT WE STILL NEED TO DO", size=14,
            bold=True, colour=RED)
    bullets(s, 0.5, 1.75, 6.1, 2.9, [
        "Get a real list of past floods and landslides with dates",
        "Ask a district officer to check our cost table",
        "Our warnings are a little too strong on wet monsoon days",
        "Group the numbers into more bands, so the boundaries matter less",
    ], size=14)
    textbox(s, 6.9, 1.35, 6.0, 0.35, "OUR PLAN", size=14, bold=True,
            colour=ACCENT)
    table(s, 6.9, 1.75, 6.0, ["Week", "What we will do"],
          [["1-2", "Collect real disaster dates for our districts"],
           ["3", "Train the model again on that real data"],
           ["4", "Use several days of weather together, not one"],
           ["5-6", "Check the cost table with a real officer"],
           ["7-8", "Finish the package and write a short paper"]],
          [1.2, 4.8], size=12.5, row_h=0.46)
    table(s, 0.5, 4.9, 12.4, ["Member", "Roll number", "What they did"],
          [[n, r, o] for (n, r), o in zip(MEMBERS, [
              "The main model, the learning code, the decision part, the website",
              "The picture of the network, the numbers in the tables, the slides",
              "Turning readings into bands, the explanation sentences, the map",
              "Making practice data, the live weather code, the testing"])],
          [2.1, 2.1, 8.2], size=12.5, row_h=0.4)
    notes(s, "End by saying the limitations out loud before anyone asks. Then "
             "offer the live demonstration - the website is already open on the "
             "laptop.")

    prs.save(OUT)
    return OUT, len(prs.slides._sldIdLst)


if __name__ == "__main__":
    path, n = build()
    print(f"wrote {path}  ({n} slides)")
