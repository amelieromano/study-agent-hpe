import streamlit as st
from datetime import date
import math
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from agent import StudyAgent, PROMPTS
from ingest import ingest_documents

load_dotenv()

# ── Module accent colours ──────────────────────────────────────────────────────
MODULE_COLOURS = {
    "micro":   "#770737",  # deep burgundy
    "macro":   "#00674f",  # forest green
    "history": "#daa520",  # goldenrod
    "up1":     "#ffc5d3",  # blush pink  (dark text)
    "up2":     "#f5f5dc",  # cream       (dark text)
}
MODULE_TEXT_DARK = {"up1", "up2"}  # these light accents need dark text

MODULE_LABELS = {
    "micro": "Microeconomics",
    "macro": "Macroeconomics",
    "history": "History",
    "up1": "UP1",
    "up2": "UP2",
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
    "extra_practice":   "Extra Practice — 5 graded problems mirroring tutorial structure (micro/macro only)",
    "book_themes":      "Book Themes — Historiographical argument + essay questions (history only)",
    "chapter_summary":  "Chapter Summary — Argument, examples, historiographical position (history only)",
    "book_compare":     "Book Compare — Compare two books' arguments and methods (history only)",
    "essay_practice":   "Essay Practice — Plan + student feedback (history only)",
    "example_bank":     "Example Bank — All examples for a theme across books (history only)",
    "theme_mapper":     "Theme Mapper — Top 5 themes across all books (history only)",
}

HISTORY_ONLY     = {"book_themes", "chapter_summary", "book_compare", "essay_practice", "example_bank", "theme_mapper"}
QUANT_ONLY       = {"equation_practice", "extra_practice"}
DOWNLOADABLE_MODES = {"study_guide", "exam_questions", "flashcards", "essay_plan", "book_themes"}

# ── Page config & global CSS ───────────────────────────────────────────────────
st.set_page_config(page_title="Year 1 HPE Study Agent", page_icon="📖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #faf9f6;
    color: #1a1a1a;
}

/* Main headings in serif */
h1, h2, h3 {
    font-family: 'Lora', Georgia, serif !important;
    color: #1a1a1a;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #f3ede3;
}

/* Cards / answer area */
.answer-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-top: 0.8rem;
}

/* Source tag */
.src-tag {
    display: inline-block;
    background: #ede8e0;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #555;
    margin: 2px 3px 2px 0;
    font-family: 'Inter', monospace;
}

/* Module pill */
.module-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 0.4rem;
}

/* Confidence buttons row */
.stButton button {
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
}

/* Sidebar section headers */
.sidebar-section {
    font-family: 'Lora', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #3a2a1a;
    margin: 0.6rem 0 0.3rem 0;
}

/* Exam card */
.exam-card {
    border-radius: 6px;
    padding: 7px 10px;
    margin-bottom: 7px;
    background: #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom:0.1rem'>Year 1 HPE Study Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#777; font-size:0.9rem; margin-top:0'>University College London · Powered by GPT-4o-mini</p>", unsafe_allow_html=True)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-section'>Module & Mode</div>", unsafe_allow_html=True)

    module = st.selectbox("Module", options=list(MODULE_LABELS.keys()), format_func=lambda m: MODULE_LABELS[m])

    accent  = MODULE_COLOURS[module]
    txt_col = "#1a1a1a" if module in MODULE_TEXT_DARK else "#ffffff"
    st.markdown(
        f"<span class='module-pill' style='background:{accent}; color:{txt_col}'>{MODULE_LABELS[module]}</span>",
        unsafe_allow_html=True,
    )

    available_modes = [
        m for m in MODE_INFO
        if not (m in HISTORY_ONLY and module != "history")
        and not (m in QUANT_ONLY and module not in ("micro", "macro"))
    ]

    mode = st.selectbox("Mode", options=available_modes, format_func=lambda m: MODE_INFO[m])

    st.caption("essay_practice: submit the question for a plan, then resubmit as `question ||| your essay` for feedback.")

    # ── Exam Countdown ─────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section' style='margin-top:1rem'>Exam Countdown</div>", unsafe_allow_html=True)

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
            urgency_col = "#aaaaaa"
            days_label  = "Done"
        elif days < 14:
            urgency_col = "#c0392b"
            days_label  = f"{days}d"
        elif days <= 21:
            urgency_col = "#e67e22"
            days_label  = f"{days}d"
        else:
            urgency_col = "#27ae60"
            days_label  = f"{days}d"

        st.markdown(f"""
        <div class='exam-card' style='border-left: 4px solid {exam["accent"]}'>
            <span style='font-weight:700; font-size:1rem; color:{urgency_col}'>{days_label}</span>
            <span style='font-size:0.8rem; font-weight:600; color:#333'> · {exam["short"]}</span>
            <span style='font-size:0.72rem; color:#999'> ({exam["code"]})</span><br>
            <span style='font-size:0.72rem; color:#aaa'>{exam["date"].strftime("%d %b")} · {exam["time"]}</span>
        </div>""", unsafe_allow_html=True)

    # ── Study Time Estimator ───────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section' style='margin-top:1rem'>Study Time Estimator</div>", unsafe_allow_html=True)

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
        if ratio >= 0.7:
            r["priority"] = "Critical"; r["p_col"] = "#c0392b"
        elif ratio >= 0.4:
            r["priority"] = "High";     r["p_col"] = "#e67e22"
        else:
            r["priority"] = "Medium";   r["p_col"] = "#27ae60"

    hcols = st.columns([2, 1, 1.4, 1.8])
    for col, hdr in zip(hcols, ["**Module**", "**Days**", "**Hrs/d**", "**Priority**"]):
        col.markdown(f"<span style='font-size:0.75rem; color:#888'>{hdr}</span>", unsafe_allow_html=True)

    for r in rows:
        cols = st.columns([2, 1, 1.4, 1.8])
        cols[0].markdown(f"<span style='font-size:0.8rem'>{r['label']}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"<span style='font-size:0.8rem'>{r['days']}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span style='font-size:0.8rem'>{r['hours']:.1f}h</span>", unsafe_allow_html=True)
        cols[3].markdown(
            f"<span style='color:{r['p_col']}; font-weight:700; font-size:0.8rem'>{r['priority']}</span>",
            unsafe_allow_html=True,
        )

    st.caption(f"Based on {TOTAL_DAILY:.0f}h/day. Modules with no coursework marks weighted higher.")

    # ── Upload Notes ───────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section' style='margin-top:1rem'>Upload Notes</div>", unsafe_allow_html=True)

    upload_module = st.selectbox("Add to module", options=list(MODULE_LABELS.keys()), key="upload_module")
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
            st.success(f"{n} chunks added to {upload_module}.")
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
tab_main, tab_weak = st.tabs(["Study", "Weak Spots"])

# ── Study tab ──────────────────────────────────────────────────────────────────
with tab_main:
    question = st.text_area("Your question or topic", height=100, placeholder="e.g. economies of scale")

    col_submit, col_clear, _ = st.columns([1.2, 1.2, 7])
    submitted = col_submit.button("Submit", type="primary")
    if col_clear.button("Clear", key="clear_chat"):
        st.session_state["last_result"] = None
        st.rerun()

    if submitted:
        if not question.strip():
            st.warning("Please enter a question or topic.")
        else:
            with st.spinner("Thinking…"):
                try:
                    result = agent.query(question, mode)
                    st.session_state["last_result"] = {"result": result, "module": module, "topic": question}
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

    if st.session_state["last_result"]:
        result      = st.session_state["last_result"]["result"]
        last_module = st.session_state["last_result"]["module"]
        last_topic  = st.session_state["last_result"]["topic"]
        m_accent    = MODULE_COLOURS[last_module]
        m_txt       = "#1a1a1a" if last_module in MODULE_TEXT_DARK else "#ffffff"

        # Module pill + mode label
        st.markdown(
            f"<span class='module-pill' style='background:{m_accent}; color:{m_txt}'>{MODULE_LABELS[last_module]}</span> "
            f"<span style='font-size:0.8rem; color:#888'>{MODE_INFO.get(result['mode'], result['mode'])}</span>",
            unsafe_allow_html=True,
        )

        # Answer card
        st.markdown(f"<div class='answer-card'>{result['answer'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

        # Source tags
        all_sources = result.get("sources", [])
        if all_sources:
            past = set(result.get("past_paper_sources", []))
            tags = ""
            for src in all_sources:
                fname = os.path.basename(src)
                badge = " 📄" if src in past else ""
                tags += f"<span class='src-tag'>{fname}{badge}</span>"
            st.markdown(f"<div style='margin-top:0.6rem'>{tags}</div>", unsafe_allow_html=True)

        # Download button
        if result["mode"] in DOWNLOADABLE_MODES:
            slug     = last_topic.strip().lower().replace(" ", "_")[:40]
            filename = f"{last_module}_{result['mode']}_{slug}.md"
            md_body  = f"# {result['mode'].replace('_', ' ').title()}: {last_topic}\n\n{result['answer']}"
            st.download_button("Download as Markdown", data=md_body, file_name=filename, mime="text/markdown")

        # Confidence rating
        st.divider()
        st.markdown("<span style='font-size:0.88rem; font-weight:600'>How confident are you on this topic?</span>", unsafe_allow_html=True)
        c1, c2, c3, _ = st.columns([1.3, 1.2, 1.4, 5])

        def _add_weak_spot(rating):
            existing = [(w["module"], w["topic"]) for w in st.session_state["weak_spots"]]
            if (last_module, last_topic) not in existing:
                st.session_state["weak_spots"].append({"module": last_module, "topic": last_topic, "rating": rating})

        if c1.button("Confident", key="conf"):
            st.toast("Great — marked as confident.")
        if c2.button("Unsure", key="unsure"):
            _add_weak_spot("Unsure")
            st.toast("Added to Weak Spots.")
        if c3.button("Struggled", key="struggled"):
            _add_weak_spot("Struggled")
            st.toast("Added to Weak Spots.")

# ── Weak Spots tab ─────────────────────────────────────────────────────────────
with tab_weak:
    st.markdown("<h2 style='margin-bottom:0.3rem'>Weak Spots</h2>", unsafe_allow_html=True)

    flagged = [w for w in st.session_state["weak_spots"] if w["rating"] in ("Unsure", "Struggled")]

    if not flagged:
        st.info("No weak spots yet. Rate your confidence after each query to track topics here.")
    else:
        if st.session_state["weak_spot_result"]:
            ws = st.session_state["weak_spot_result"]
            st.markdown(f"### Study Guide: {ws['topic']}")
            st.markdown(f"<div class='answer-card'>{ws['answer'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            if ws.get("sources"):
                tags = "".join(f"<span class='src-tag'>{os.path.basename(s)}</span>" for s in ws["sources"])
                st.markdown(tags, unsafe_allow_html=True)
            st.divider()

        rating_colour = {"Struggled": "#c0392b", "Unsure": "#e67e22"}

        for i, ws in enumerate(flagged):
            col_label, col_btn, col_remove = st.columns([5, 1.8, 1.4])
            rc = rating_colour.get(ws["rating"], "#888")
            ma = MODULE_COLOURS.get(ws["module"], "#888")
            mt = "#1a1a1a" if ws["module"] in MODULE_TEXT_DARK else "#ffffff"
            col_label.markdown(
                f"<span style='color:{rc}; font-weight:700; font-size:0.85rem'>{ws['rating']}</span> "
                f"<span class='module-pill' style='background:{ma}; color:{mt}; font-size:0.7rem'>{ws['module'].upper()}</span> "
                f"<span style='font-size:0.85rem'>{ws['topic']}</span>",
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
