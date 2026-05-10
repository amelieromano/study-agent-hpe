import streamlit as st
from datetime import date
import math
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from agent import StudyAgent, PROMPTS
from ingest import ingest_documents

load_dotenv()

# ── Module accent colours ──────────────────────────────────────────────────────
MODULE_COLOURS = {
    "micro":   "#770737",
    "macro":   "#00674f",
    "history": "#daa520",
    "up1":     "#ffc5d3",
    "up2":     "#f5f5dc",
}
MODULE_TEXT_DARK = {"up1", "up2"}

MODULE_LABELS = {
    "micro":   "Microeconomics",
    "macro":   "Macroeconomics",
    "history": "History",
    "up1":     "UP1",
    "up2":     "UP2",
}

# ── Mode metadata ──────────────────────────────────────────────────────────────
MODE_INFO = {
    "qa":               "Q&A — Direct answer with source citations",
    "study_guide":      "Study Guide — Overview, key concepts, exam angles",
    "exam_questions":   "Exam Questions — Short answer + essay questions",
    "flashcards":       "Flashcards — 10 Anki-style Q&A pairs",
    "explain":          "Explain — Simple → academic → concrete example",
    "essay_plan":       "Essay Plan — Timed plan with intro, body, counter, conclusion",
    "definition_bank":  "Definition Bank — 8 concept/thinker entries",
    "priority_score":   "Priority Score — HIGH / MEDIUM / LOW exam priority",
    "equation_practice":"Equation Practice — 3 worked problems (micro/macro only)",
    "extra_practice":   "Extra Practice — 5 graded problems (micro/macro only)",
    "book_themes":      "Book Themes — Historiographical argument + essay questions (history only)",
    "chapter_summary":  "Chapter Summary — Argument, examples, position (history only)",
    "book_compare":     "Book Compare — Compare two books' arguments (history only)",
    "essay_practice":   "Essay Practice — Plan + student feedback (history only)",
    "example_bank":     "Example Bank — All examples for a theme (history only)",
    "theme_mapper":     "Theme Mapper — Top 5 themes across all books (history only)",
}

HISTORY_ONLY       = {"book_themes", "chapter_summary", "book_compare", "essay_practice", "example_bank", "theme_mapper"}
QUANT_ONLY         = {"equation_practice", "extra_practice"}
DOWNLOADABLE_MODES = {"study_guide", "exam_questions", "flashcards", "essay_plan", "book_themes"}


def render_answer(text: str):
    """Render agent response with markdown and LaTeX equation support."""
    import re
    parts = re.split(r'(\\\[.*?\\\])', text, flags=re.DOTALL)
    for part in parts:
        if part.startswith(r'\[') and part.endswith(r'\]'):
            latex = part[2:-2].strip()
            st.latex(latex)
        elif part.strip():
            st.markdown(part)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HPE Study Agent", page_icon="📖", layout="wide")

# ── Typewriter / Vintage Editorial CSS ────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Bebas+Neue&family=Special+Elite&display=swap');

/* ── CSS Variables ── */
:root {
    --paper-cream:   #f4f1e8;
    --paper-dark:    #e8e3d6;
    --paper-manila:  #d9cdb0;
    --ink-black:     #2b2b2b;
    --ink-faded:     #4a4540;
    --stamp-red:     #c1554a;
    --typewriter-blue: #4a7c8c;
    --vintage-orange:  #d97941;
    --faded-green:     #7a9b7f;
    --halftone-gray:   #6b6b6b;
    --shadow-light:    rgba(43,43,43,0.08);
    --shadow-med:      rgba(43,43,43,0.15);
}

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Courier Prime', 'Courier New', monospace !important;
    color: var(--ink-black);
}
/* Restore Material icon font — must come after the global override */
[data-testid="stIconMaterial"],
span[translate="no"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-feature-settings: 'liga' !important;
}

/* ── Paper-grain background ── */
.stApp {
    background-color: var(--paper-cream);
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E"),
        linear-gradient(160deg, #f6f3ea 0%, #ede8db 50%, #e8e3d4 100%);
    background-attachment: fixed;
}

/* Main container padding */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--ink-black) !important;
}
h1 { letter-spacing: 0.03em; }
h2 { letter-spacing: 0.02em; font-size: 1.5rem !important; }

/* ── Sidebar — filing-cabinet manila ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #d9cdb0 0%, #cfc3a8 100%) !important;
    border-right: 3px solid var(--ink-black);
    box-shadow: inset -4px 0 12px rgba(0,0,0,0.08);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}
/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    font-family: 'Courier Prime', monospace !important;
    color: var(--ink-black) !important;
    overflow-wrap: break-word;
}
/* Sidebar divider */
section[data-testid="stSidebar"] hr {
    border: none;
    border-top: 2px dashed #8b7d6b;
    margin: 0.8rem 0;
}

/* ── Sidebar selectbox ── */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #e8dcc4 !important;
    border: 2px solid var(--ink-black) !important;
    border-radius: 0 !important;
    font-family: 'Courier Prime', monospace !important;
    box-shadow: 2px 2px 0 var(--ink-black);
}

/* ── Main selectbox ── */
.main [data-baseweb="select"] > div {
    background: #fff9ef !important;
    border: none !important;
    border-bottom: 2px solid var(--ink-black) !important;
    border-radius: 0 !important;
    font-family: 'Courier Prime', monospace !important;
    box-shadow: none !important;
}

/* ── Text areas — typewriter paper look ── */
.stTextArea textarea {
    background: #fffef9 !important;
    border: none !important;
    border-bottom: 2px solid var(--ink-black) !important;
    border-radius: 0 !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 1rem !important;
    color: var(--ink-black) !important;
    box-shadow: none !important;
    resize: vertical;
    padding: 0.6rem 0.4rem !important;
    line-height: 1.8 !important;
    /* Faint ruled-paper lines */
    background-image: repeating-linear-gradient(
        transparent,
        transparent 27px,
        rgba(74,124,140,0.15) 27px,
        rgba(74,124,140,0.15) 28px
    ) !important;
}
.stTextArea textarea:focus {
    border-bottom: 2px solid var(--stamp-red) !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #a09080 !important;
    font-style: italic;
}

/* ── All buttons — typewriter key style ── */
.stButton > button {
    background: linear-gradient(180deg, #f4f1e8 0%, #d9cdb0 100%) !important;
    color: var(--ink-black) !important;
    border: 2px solid var(--ink-black) !important;
    border-radius: 3px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.12em !important;
    padding: 0.5rem 1.6rem !important;
    box-shadow: 0 4px 0 #1a1a1a, 0 6px 8px rgba(0,0,0,0.2) !important;
    transition: all 0.08s ease !important;
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(1px) !important;
    box-shadow: 0 3px 0 #1a1a1a, 0 4px 6px rgba(0,0,0,0.15) !important;
    background: linear-gradient(180deg, #ece9e0 0%, #cfc3a8 100%) !important;
}
.stButton > button:active {
    transform: translateY(4px) !important;
    box-shadow: 0 0 0 #1a1a1a, 0 1px 3px rgba(0,0,0,0.1) !important;
}
/* Primary button — red ink */
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #d06050 0%, #b04040 100%) !important;
    color: #fff !important;
    border-color: #8b2020 !important;
    box-shadow: 0 4px 0 #8b2020, 0 6px 8px rgba(0,0,0,0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #c05040 0%, #a03030 100%) !important;
    box-shadow: 0 3px 0 #8b2020, 0 4px 6px rgba(0,0,0,0.15) !important;
}
.stButton > button[kind="primary"]:active {
    box-shadow: 0 0 0 #8b2020, 0 1px 3px rgba(0,0,0,0.1) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(180deg, #e8f0e8 0%, #c8d8c8 100%) !important;
    border-color: var(--faded-green) !important;
    box-shadow: 0 4px 0 #4a6a4a, 0 6px 8px rgba(0,0,0,0.15) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
}

/* ── Tabs — folder tab style ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 3px solid var(--ink-black) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.12em !important;
    color: var(--ink-faded) !important;
    background: var(--paper-dark) !important;
    border: 2px solid var(--ink-black) !important;
    border-bottom: none !important;
    border-radius: 0 !important;
    padding: 0.4rem 1.4rem !important;
    margin-right: 4px !important;
    position: relative;
    top: 3px;
}
.stTabs [aria-selected="true"] {
    background: var(--paper-cream) !important;
    color: var(--ink-black) !important;
    border-bottom: 3px solid var(--paper-cream) !important;
    top: 3px;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: transparent !important;
}
.stTabs [data-baseweb="tab-border"] {
    background: var(--ink-black) !important;
    height: 3px !important;
}

/* ── Answer container — index card ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important;
    background: #fffef8 !important;
    box-shadow:
        2px 3px 0 rgba(0,0,0,0.12),
        4px 5px 0 rgba(0,0,0,0.05),
        0 0 0 1px rgba(43,43,43,0.12) !important;
    border-radius: 0 !important;
    border-left: 5px solid var(--typewriter-blue) !important;
    padding: 1.4rem 1.8rem !important;
    position: relative;
}
[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 0; height: 0;
    border-style: solid;
    border-width: 0 18px 18px 0;
    border-color: transparent var(--paper-dark) transparent transparent;
}

/* ── Answer text ── */
[data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stVerticalBlockBorderWrapper"] li,
[data-testid="stVerticalBlockBorderWrapper"] td,
[data-testid="stVerticalBlockBorderWrapper"] th {
    font-family: 'Courier Prime', monospace !important;
    font-size: 0.97rem !important;
    line-height: 1.8 !important;
    color: var(--ink-black) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] h1,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    font-family: 'Playfair Display', serif !important;
    border-bottom: 1px dashed var(--halftone-gray);
    padding-bottom: 0.2rem;
    margin-top: 1rem;
}
[data-testid="stVerticalBlockBorderWrapper"] code {
    font-family: 'Courier Prime', monospace !important;
    background: rgba(74,124,140,0.1) !important;
    border: 1px solid rgba(74,124,140,0.3) !important;
    border-radius: 0 !important;
    padding: 0.1em 0.3em !important;
}
[data-testid="stVerticalBlockBorderWrapper"] strong {
    color: var(--ink-black);
    font-weight: 700;
    border-bottom: 1px solid var(--vintage-orange);
}
[data-testid="stVerticalBlockBorderWrapper"] blockquote {
    border-left: 3px solid var(--stamp-red) !important;
    background: rgba(193,85,74,0.04) !important;
    margin: 0.5rem 0 0.5rem 0.5rem !important;
    padding: 0.3rem 0.8rem !important;
    font-style: italic;
}

/* ── Info / warning / error boxes ── */
.stAlert {
    border-radius: 0 !important;
    border-left: 4px solid !important;
    font-family: 'Courier Prime', monospace !important;
}

/* ── Spinner — typewriter feel ── */
.stSpinner > div {
    border-color: var(--stamp-red) transparent transparent transparent !important;
}
[data-testid="stStatusWidget"] {
    font-family: 'Special Elite', cursive !important;
    color: var(--ink-faded) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--ink-black) !important;
    border-radius: 0 !important;
    background: rgba(244,241,232,0.5) !important;
}

/* ── Caption text ── */
.stCaption p, .stCaption {
    font-family: 'Special Elite', cursive !important;
    color: var(--halftone-gray) !important;
    font-size: 0.8rem !important;
}

/* ── Toast notifications ── */
[data-testid="stToast"] {
    font-family: 'Courier Prime', monospace !important;
    border-radius: 0 !important;
    border: 2px solid var(--ink-black) !important;
    background: var(--paper-cream) !important;
}

/* ── Tables ── */
table {
    border-collapse: collapse !important;
    font-family: 'Courier Prime', monospace !important;
}
th {
    background: var(--ink-black) !important;
    color: var(--paper-cream) !important;
    padding: 4px 10px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.08em !important;
}
td {
    border-bottom: 1px dashed #bbb !important;
    padding: 4px 10px !important;
}
tr:nth-child(even) td {
    background: rgba(0,0,0,0.025) !important;
}

/* ── LaTeX containers ── */
.stLatex {
    background: rgba(74,124,140,0.06) !important;
    border-left: 3px solid var(--typewriter-blue) !important;
    padding: 0.5rem 1rem !important;
    margin: 0.5rem 0 !important;
}

/* ── Progress / metric elements ── */
[data-testid="metric-container"] {
    background: var(--paper-dark) !important;
    border: 1px solid var(--ink-black) !important;
    padding: 0.5rem 1rem !important;
    border-radius: 0 !important;
    font-family: 'Courier Prime', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--paper-dark); }
::-webkit-scrollbar-thumb {
    background: var(--ink-black);
    border-radius: 0;
}

/* ── Remove Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stExpandSidebarButton"]   { visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; }

/* ── Halftone decoration strip ── */
.halftone-strip {
    height: 6px;
    background:
        repeating-linear-gradient(
            90deg,
            var(--ink-black) 0px, var(--ink-black) 2px,
            transparent 2px, transparent 6px
        );
    opacity: 0.2;
    margin: 0.4rem 0;
}

/* ── Stamp badge ── */
.stamp-badge {
    display: inline-block;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: var(--stamp-red);
    border: 2px solid var(--stamp-red);
    padding: 3px 10px;
    transform: rotate(-1.5deg);
    opacity: 0.9;
    box-sizing: border-box;
    box-shadow: inset 0 0 0 1px #f4f1e8, inset 0 0 0 3px var(--stamp-red);
    vertical-align: middle;
}

/* ── Module pill — stamp style ── */
.module-stamp {
    display: inline-block;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border: 2px solid currentColor;
    transform: rotate(-0.5deg);
    vertical-align: middle;
    margin-right: 4px;
    max-width: 100%;
    box-sizing: border-box;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Source footnote tags ── */
.src-tag {
    display: inline-block;
    font-family: 'Courier Prime', monospace;
    font-size: 0.72rem;
    color: var(--typewriter-blue);
    border: 1px solid var(--typewriter-blue);
    padding: 1px 6px;
    margin: 2px 3px 2px 0;
    background: rgba(74,124,140,0.06);
}
.src-tag::before { content: '['; }
.src-tag::after  { content: ']'; }

/* ── Exam index card ── */
.exam-card {
    background: #fffef8;
    border: 1px solid #c8b898;
    border-left: 4px solid;
    padding: 6px 10px;
    margin-bottom: 6px;
    box-shadow: 2px 2px 0 rgba(0,0,0,0.07);
    position: relative;
}

/* ── Study time table rows ── */
.time-row {
    font-family: 'Courier Prime', monospace;
    font-size: 0.82rem;
    padding: 2px 0;
    border-bottom: 1px dotted #bbb;
}

/* ── Weak spot row ── */
.ws-row {
    background: rgba(255,254,248,0.8);
    border-left: 3px solid;
    padding: 4px 10px;
    margin-bottom: 4px;
    font-family: 'Courier Prime', monospace;
    font-size: 0.87rem;
}

/* ── Sidebar section header ── */
.sidebar-section {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.12em;
    color: var(--ink-black);
    border-bottom: 2px solid var(--ink-black);
    padding-bottom: 2px;
    margin: 1rem 0 0.4rem 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Confidence buttons row ── */
.conf-label {
    font-family: 'Special Elite', cursive;
    font-size: 0.88rem;
    color: var(--ink-faded);
    margin-bottom: 0.3rem;
}

/* ── Typewriter cursor blink ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.tw-cursor {
    display: inline-block;
    width: 9px;
    height: 1.1em;
    background: var(--ink-black);
    vertical-align: text-bottom;
    animation: blink 1.1s step-end infinite;
    margin-left: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.2rem 0 0.6rem 0; position:relative;">
  <div style="
    font-family:'Playfair Display',Georgia,serif;
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 900;
    color: #2b2b2b;
    letter-spacing: 0.04em;
    transform: rotate(-0.4deg);
    display: inline-block;
    text-shadow: 3px 3px 0 rgba(0,0,0,0.04);
    line-height: 1.1;
  ">Year 1 HPE Study Agent</div>
  <br>
  <span class="stamp-badge">UCL · GPT-4o-mini · Course Materials Indexed</span>
  <div class="halftone-strip" style="margin-top:1rem;"></div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-section'>Module &amp; Mode</div>", unsafe_allow_html=True)

    module = st.selectbox("Module", options=list(MODULE_LABELS.keys()), format_func=lambda m: MODULE_LABELS[m])

    accent  = MODULE_COLOURS[module]
    txt_col = "#2b2b2b" if module in MODULE_TEXT_DARK else "#ffffff"
    st.markdown(
        f"<span class='module-stamp' style='color:{accent}'>{MODULE_LABELS[module].upper()}</span>",
        unsafe_allow_html=True,
    )

    available_modes = [
        m for m in MODE_INFO
        if not (m in HISTORY_ONLY and module != "history")
        and not (m in QUANT_ONLY and module not in ("micro", "macro"))
    ]

    mode = st.selectbox("Mode", options=available_modes, format_func=lambda m: MODE_INFO[m])
    st.caption("essay_practice: submit for plan, then resubmit as `question ||| your essay` for feedback.")

    # ── Exam Countdown ─────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Exam Countdown</div>", unsafe_allow_html=True)

    EXAMS = [
        {"code": "SESS0007", "short": "Micro",   "date": date(2026, 5, 5),  "time": "1PM",  "accent": MODULE_COLOURS["micro"]},
        {"code": "SESS0008", "short": "Macro",   "date": date(2026, 5, 7),  "time": "9AM",  "accent": MODULE_COLOURS["macro"]},
        {"code": "SEHI0003", "short": "History", "date": date(2026, 5, 12), "time": "9AM",  "accent": MODULE_COLOURS["history"]},
        {"code": "SESS0017", "short": "UP2",     "date": date(2026, 5, 15), "time": "10AM", "accent": MODULE_COLOURS["up2"]},
        {"code": "SESS0016", "short": "UP1",     "date": date(2026, 5, 29), "time": "10AM", "accent": MODULE_COLOURS["up1"]},
    ]

    today = date.today()
    for exam in EXAMS:
        days = (exam["date"] - today).days
        if days < 0:
            urgency_col = "#aaaaaa"; days_label = "DONE"
        elif days < 14:
            urgency_col = "#c0392b"; days_label = f"{days}d"
        elif days <= 21:
            urgency_col = "#d97941"; days_label = f"{days}d"
        else:
            urgency_col = "#7a9b7f"; days_label = f"{days}d"

        st.markdown(f"""
        <div class='exam-card' style='border-left-color:{exam["accent"]}'>
          <span style='font-family:"Bebas Neue",sans-serif;font-size:1.1rem;color:{urgency_col};
                       letter-spacing:0.06em'>{days_label}</span>
          <span style='font-family:"Courier Prime",monospace;font-size:0.82rem;
                       font-weight:700;color:#333'> · {exam["short"]}</span>
          <span style='font-size:0.7rem;color:#999'> ({exam["code"]})</span><br>
          <span style='font-family:"Special Elite",cursive;font-size:0.7rem;
                       color:#888'>{exam["date"].strftime("%d %b")} · {exam["time"]}</span>
        </div>""", unsafe_allow_html=True)

    # ── Study Time Estimator ───────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Study Time</div>", unsafe_allow_html=True)

    MODULES_DATA = [
        {"label": "Micro",   "exam_date": date(2026, 5, 5),  "exam_weight": 35,  "coursework_done": True},
        {"label": "Macro",   "exam_date": date(2026, 5, 7),  "exam_weight": 100, "coursework_done": False},
        {"label": "History", "exam_date": date(2026, 5, 12), "exam_weight": 27,  "coursework_done": True},
        {"label": "UP2",     "exam_date": date(2026, 5, 15), "exam_weight": 100, "coursework_done": False},
        {"label": "UP1",     "exam_date": date(2026, 5, 29), "exam_weight": 35,  "coursework_done": True},
    ]

    today_dt = date.today()
    rows = []
    for m in MODULES_DATA:
        days = max((m["exam_date"] - today_dt).days, 1)
        urgency = m["exam_weight"] / days * (1.4 if not m["coursework_done"] else 1.0)
        rows.append({**m, "days": days, "urgency": urgency})

    max_urgency   = max(r["urgency"] for r in rows)
    total_urgency = sum(r["urgency"] for r in rows)
    TOTAL_DAILY   = 8.0

    for r in rows:
        r["hours"] = round((r["urgency"] / total_urgency) * TOTAL_DAILY * 2) / 2
        ratio = r["urgency"] / max_urgency
        if   ratio >= 0.7: r["priority"] = "CRITICAL"; r["p_col"] = "#c0392b"
        elif ratio >= 0.4: r["priority"] = "HIGH";     r["p_col"] = "#d97941"
        else:              r["priority"] = "MEDIUM";   r["p_col"] = "#7a9b7f"

    hcols = st.columns([2, 1, 1.4, 2])
    for col, hdr in zip(hcols, ["Module", "Days", "Hrs/d", "Priority"]):
        col.markdown(f"<span style='font-family:\"Bebas Neue\",sans-serif;font-size:0.72rem;"
                     f"letter-spacing:0.1em;color:#5a4a3a'>{hdr}</span>", unsafe_allow_html=True)

    for r in rows:
        cols = st.columns([2, 1, 1.4, 2])
        cols[0].markdown(f"<span style='font-size:0.8rem;font-family:\"Courier Prime\",monospace'>{r['label']}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"<span style='font-size:0.8rem;font-family:\"Courier Prime\",monospace'>{r['days']}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span style='font-size:0.8rem;font-family:\"Courier Prime\",monospace'>{r['hours']:.1f}h</span>", unsafe_allow_html=True)
        cols[3].markdown(
            f"<span style='color:{r['p_col']};font-family:\"Bebas Neue\",sans-serif;"
            f"font-size:0.75rem;letter-spacing:0.08em'>{r['priority']}</span>",
            unsafe_allow_html=True,
        )

    st.caption(f"Based on {TOTAL_DAILY:.0f}h/day. No-coursework modules weighted higher.")

    # ── Upload Notes ───────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>Upload Notes</div>", unsafe_allow_html=True)

    upload_module  = st.selectbox("Add to module", options=list(MODULE_LABELS.keys()), key="upload_module")
    uploaded_files = st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True, key="uploaded_files")

    if st.button("Embed & Add", key="embed_btn") and uploaded_files:
        all_docs = []
        for uf in uploaded_files:
            suffix = os.path.splitext(uf.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            try:
                if suffix == ".pdf":
                    pages = PyPDFLoader(tmp_path).load()
                else:
                    text  = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
                    pages = [Document(page_content=text, metadata={"source": uf.name})]
                for page in pages:
                    page.metadata.setdefault("source", uf.name)
                    page.metadata["doc_type"] = "user_upload"
                all_docs.extend(pages)
            finally:
                os.unlink(tmp_path)

        if all_docs:
            with st.spinner("Embedding…"):
                n = ingest_documents(all_docs, upload_module)
                st.session_state.pop(f"agent_{upload_module}", None)
            st.success(f"✓ {n} chunks added to {upload_module}.")
        else:
            st.warning("No content extracted.")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [("weak_spots", []), ("last_result", None), ("weak_spot_result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Agent cache ────────────────────────────────────────────────────────────────
agent_key = f"agent_{module}"
if agent_key not in st.session_state:
    with st.spinner(f"Loading {MODULE_LABELS[module]} collection…"):
        try:
            st.session_state[agent_key] = StudyAgent(module)
        except Exception as e:
            st.error(f"Failed to load collection for '{module}': {e}")
            st.stop()

agent = st.session_state[agent_key]

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_main, tab_weak = st.tabs(["  Study  ", "  Weak Spots  "])

# ── Study tab ──────────────────────────────────────────────────────────────────
with tab_main:
    # Current mode indicator
    st.markdown(
        f"<div style='margin-bottom:0.5rem'>"
        f"<span class='module-stamp' style='color:{MODULE_COLOURS[module]}'>"
        f"{MODULE_LABELS[module].upper()}</span>&nbsp;"
        f"<span style='font-family:\"Special Elite\",cursive;font-size:0.82rem;"
        f"color:#7a6a5a'>{MODE_INFO.get(mode, mode)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Question or topic",
        height=110,
        placeholder="e.g.  economies of scale…",
        label_visibility="collapsed",
    )

    col_submit, col_clear, _ = st.columns([1.2, 1.2, 7])
    submitted = col_submit.button("Submit ↵", type="primary")
    if col_clear.button("Clear ✕", key="clear_chat"):
        st.session_state["last_result"] = None
        st.rerun()

    if submitted:
        if not question.strip():
            st.warning("Please enter a question or topic.")
        else:
            with st.spinner("Thinking…"):
                try:
                    result = agent.query(question, mode)
                    st.session_state["last_result"] = {
                        "result": result, "module": module, "topic": question
                    }
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

    if st.session_state["last_result"]:
        result      = st.session_state["last_result"]["result"]
        last_module = st.session_state["last_result"]["module"]
        last_topic  = st.session_state["last_result"]["topic"]
        m_accent    = MODULE_COLOURS[last_module]

        # Mode label strip
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:0.6rem 0 0.4rem 0'>"
            f"<span class='module-stamp' style='color:{m_accent}'>{MODULE_LABELS[last_module].upper()}</span>"
            f"<span style='font-family:\"Special Elite\",cursive;font-size:0.8rem;color:#7a6a5a'>"
            f"{MODE_INFO.get(result['mode'], result['mode'])}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Answer — index card container
        with st.container(border=True):
            render_answer(result["answer"])

        # Source footnotes
        all_sources = result.get("sources", [])
        if all_sources:
            past = set(result.get("past_paper_sources", []))
            tags = ""
            for src in all_sources:
                fname  = os.path.basename(src)
                badge  = " ★" if src in past else ""
                tags  += f"<span class='src-tag'>{fname}{badge}</span>"
            st.markdown(
                f"<div style='margin-top:0.7rem;font-size:0.7rem;"
                f"font-family:\"Special Elite\",cursive;color:#888'>"
                f"Sources: {tags}</div>",
                unsafe_allow_html=True,
            )

        # Download
        if result["mode"] in DOWNLOADABLE_MODES:
            slug     = last_topic.strip().lower().replace(" ", "_")[:40]
            filename = f"{last_module}_{result['mode']}_{slug}.md"
            md_body  = f"# {result['mode'].replace('_', ' ').title()}: {last_topic}\n\n{result['answer']}"
            st.download_button("↓ Download Markdown", data=md_body, file_name=filename, mime="text/markdown")

        # Confidence rating
        st.markdown("<div class='halftone-strip'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='conf-label'>— How confident are you on this topic?</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, _ = st.columns([1.3, 1.2, 1.4, 5])

        def _add_weak_spot(rating):
            existing = [(w["module"], w["topic"]) for w in st.session_state["weak_spots"]]
            if (last_module, last_topic) not in existing:
                st.session_state["weak_spots"].append({
                    "module": last_module, "topic": last_topic, "rating": rating
                })

        if c1.button("Confident", key="conf"):
            st.toast("✓ Marked as confident.")
        if c2.button("Unsure", key="unsure"):
            _add_weak_spot("Unsure")
            st.toast("Added to Weak Spots.")
        if c3.button("Struggled", key="struggled"):
            _add_weak_spot("Struggled")
            st.toast("Added to Weak Spots.")

# ── Weak Spots tab ─────────────────────────────────────────────────────────────
with tab_weak:
    st.markdown("""
    <div style='font-family:"Playfair Display",serif;font-size:1.6rem;
                font-weight:700;margin-bottom:0.2rem;transform:rotate(-0.3deg);
                display:inline-block'>Weak Spots</div>
    <div class='halftone-strip'></div>
    """, unsafe_allow_html=True)

    flagged = [w for w in st.session_state["weak_spots"] if w["rating"] in ("Unsure", "Struggled")]

    if not flagged:
        st.markdown("""
        <div style='font-family:"Special Elite",cursive;font-size:0.92rem;
                    color:#7a6a5a;border:1px dashed #bbb;padding:1rem;
                    background:rgba(255,254,248,0.7);margin-top:0.5rem'>
            No weak spots yet — rate your confidence after each query to track topics here.
        </div>""", unsafe_allow_html=True)
    else:
        if st.session_state["weak_spot_result"]:
            ws = st.session_state["weak_spot_result"]
            st.markdown(
                f"<div style='font-family:\"Playfair Display\",serif;font-size:1.1rem;"
                f"font-weight:700;margin-bottom:0.3rem'>Study Guide: {ws['topic']}</div>",
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                render_answer(ws["answer"])
            if ws.get("sources"):
                tags = "".join(f"<span class='src-tag'>{os.path.basename(s)}</span>" for s in ws["sources"])
                st.markdown(f"<div style='margin-top:0.5rem'>Sources: {tags}</div>", unsafe_allow_html=True)
            st.markdown("<div class='halftone-strip' style='margin:1rem 0'></div>", unsafe_allow_html=True)

        rating_colour = {"Struggled": "#c0392b", "Unsure": "#d97941"}

        for i, ws in enumerate(flagged):
            col_label, col_btn, col_remove = st.columns([5, 1.8, 1.4])
            rc = rating_colour.get(ws["rating"], "#888")
            ma = MODULE_COLOURS.get(ws["module"], "#888")

            col_label.markdown(
                f"<div class='ws-row' style='border-left-color:{rc}'>"
                f"<span style='font-family:\"Bebas Neue\",sans-serif;font-size:0.8rem;"
                f"color:{rc};letter-spacing:0.08em'>{ws['rating']}</span>&nbsp;"
                f"<span class='module-stamp' style='color:{ma};font-size:0.65rem'>"
                f"{ws['module'].upper()}</span>&nbsp;"
                f"<span>{ws['topic']}</span></div>",
                unsafe_allow_html=True,
            )
            if col_btn.button("Study Guide", key=f"ws_sg_{i}"):
                ws_agent_key = f"agent_{ws['module']}"
                if ws_agent_key not in st.session_state:
                    st.session_state[ws_agent_key] = StudyAgent(ws["module"])
                with st.spinner(f"Generating study guide for '{ws['topic']}'…"):
                    ws_res = st.session_state[ws_agent_key].query(ws["topic"], "study_guide")
                st.session_state["weak_spot_result"] = {
                    "topic": ws["topic"], "answer": ws_res["answer"], "sources": ws_res["sources"],
                }
                st.rerun()
            if col_remove.button("Remove", key=f"ws_rm_{i}"):
                st.session_state["weak_spots"].remove(ws)
                st.rerun()
