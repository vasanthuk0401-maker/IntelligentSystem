"""8. User Dashboard (Streamlit).

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import uuid

import pandas as pd
import streamlit as st

from core import config
from core.llm import get_llm
from core.memory import Memory
from core.schemas import UserInput
from orchestrator import Orchestrator
from tools.cv_parser import parse_cv_bytes
from tools.job_sources import available_datasets, default_sources
from ui import theme as ui
from ui.theme import AMBER, EMERALD, INDIGO, PINK, ROSE, SKY, VIOLET

st.set_page_config(page_title="Intelligent Job Search", page_icon="🎯", layout="wide")
ui.inject_css()


@st.cache_resource
def shared_memory() -> Memory:
    return Memory()


@st.cache_resource(show_spinner="Loading and indexing job dataset...")
def load_sources(dataset_path: str):
    sources = default_sources(dataset_path)
    _ = sources[0].jobs  # load + build index once
    return sources


ss = st.session_state
ss.setdefault("profile", None)
ss.setdefault("cv_text", "")
ss.setdefault("result", None)
ss.setdefault("tailored", {})
ss.setdefault("packs", {})
ss.setdefault("feedback", {})

# ---------------- Sidebar ----------------
llm = get_llm()
with st.sidebar:
    ui.sidebar_brand(llm.enabled, llm.mode if llm.enabled else "Offline mode (no API key)")
    if config.PUBLIC_DEMO:
        user_id = ss.setdefault("uid", f"guest-{uuid.uuid4().hex[:10]}")
        st.caption("🔒 Public demo: your data stays in this browser session and is not kept long-term.")
    else:
        user_id = st.text_input("👤 User ID", value="default", help="Memory is stored per user ID.")
    if llm.last_error:
        st.caption(f"Last LLM note: {llm.last_error}")
    datasets = available_datasets()
    names = [p.name for p in datasets]
    default_ds = next((names.index(n) for n in config.DEFAULT_DATASETS if n in names), 0)
    if os.getenv("JOBS_DATASET") and config.JOBS_DATASET.name in names:
        default_ds = names.index(config.JOBS_DATASET.name)
    dataset = st.selectbox("🗂️ Job dataset", datasets, index=default_ds, format_func=lambda p: p.name,
                           help="Add more with: python import_jobs.py data/raw/postings.csv")
    threshold = st.slider("🎚️ Match threshold", 30, 90, int(config.MATCH_THRESHOLD), 5,
                          help="Below this, the Decision Agent broadens the search.")
    max_iter = st.slider("🔁 Max search iterations", 1, 5, config.MAX_SEARCH_ITERATIONS)
    st.divider()
    ui.sidebar_flow()

sources = load_sources(str(dataset))
orch = Orchestrator(user_id=user_id, llm=llm, memory=shared_memory(), sources=sources)
orch.decision_agent.threshold = threshold
orch.decision_agent.max_iterations = max_iter

# ---------------- Header ----------------
res = ss.result
n_jobs = len(sources[0].jobs)
best = f"{res.matches[0].score:.0f}" if res and res.matches else "—"
ui.hero(
    "Find the job that fits you ✨",
    "Seven AI agents read your CV, search real job postings, score every match, and help you tailor "
    "your CV and prepare for the interview.",
    [(f"{n_jobs:,}", "jobs indexed"), (best, "best match"),
     (str(len(ss.profile.skills)) if ss.profile else "0", "skills found")])
ui.pipeline([
    ("👤", "Profile", ss.profile is not None),
    ("🔍", "Search", res is not None),
    ("📄", "Analysis", res is not None),
    ("🎯", "Matching", res is not None),
    ("🧠", "Decision", res is not None),
    ("✍️", "CV & Interview", bool(ss.tailored or ss.packs)),
    ("💾", "Memory", res is not None),
])

tab_profile, tab_jobs, tab_cv, tab_iv, tab_mem = st.tabs(
    ["👤 Profile", "🎯 Job Matches", "✍️ CV Tailoring", "🎤 Interview Prep", "💾 Memory"])

# ---------------- 1. Profile ----------------
with tab_profile:
    ui.section("👤", "Your profile", INDIGO, "Upload a CV or paste it, then tell us what you're looking for")
    if ss.profile is None:
        saved = orch.memory.load_profile(user_id)
        if saved:
            st.info("💾 A saved profile was found in memory for this user.")
            if st.button("Load saved profile"):
                ss.profile = saved
                st.rerun()

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        up = st.file_uploader("📎 Upload CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        if up is not None:
            try:
                ss.cv_text = parse_cv_bytes(up.getvalue(), up.name)
                st.caption(f"✅ Extracted {len(ss.cv_text.split())} words from {up.name}.")
            except Exception as e:
                st.error(f"Could not read CV: {e}")
        if st.button("✨ Use sample CV"):
            ss.cv_text = (config.DATA_DIR / "sample_cv.txt").read_text(encoding="utf-8")
        ss.cv_text = st.text_area("📝 CV text", ss.cv_text, height=320)
    with c2:
        with st.form("profile_form"):
            st.markdown("**🎯 Your preferences**")
            name = st.text_input("Name")
            skills = st.text_input("Extra skills (comma separated)")
            exp = st.number_input("Years of experience (leave 0 to auto-detect)", 0.0, 40.0, 0.0, 0.5)
            role = st.text_input("Preferred role(s)", placeholder="e.g. Machine Learning Engineer, Data Scientist")
            loc = st.text_input("Preferred location(s)", placeholder="e.g. Bengaluru, Hyderabad")
            remote = st.checkbox("🌐 Open to remote", value=True)
            goal = st.text_area("Career goal", placeholder="e.g. Build production AI systems", height=80)
            submitted = st.form_submit_button("🚀 Build profile", type="primary", use_container_width=True)
        if submitted:
            if not ss.cv_text.strip() and not skills.strip():
                st.error("Upload a CV or enter some skills first.")
            else:
                with st.spinner("Profile Agent is working..."):
                    ss.profile = orch.build_profile(UserInput(
                        cv_text=ss.cv_text, name=name,
                        skills=[s.strip() for s in skills.split(",") if s.strip()],
                        experience_years=exp or None, preferred_role=role, location=loc,
                        remote_ok=remote, career_goal=goal))
                ss.result, ss.tailored, ss.packs = None, {}, {}
                st.rerun()

    p = ss.profile
    if p:
        ui.section("🪪", "Structured career profile", VIOLET)
        if p.missing_fields:
            st.warning("Profile incomplete - please add: " + ", ".join(p.missing_fields))
        ui.tiles([
            (str(len(p.skills)), "Skills detected", "🧩", INDIGO),
            (f"{p.experience_years:g}", "Years of experience", "⏳", PINK),
            (str(len(p.preferred_roles)), "Target roles", "🎯", EMERALD),
            (str(len(p.achievements)), "Achievements", "🏆", AMBER),
        ])
        ui.profile_card(p)
        with st.expander("🔧 Full profile JSON"):
            st.json(p.model_dump(exclude={"source_text"}))

# ---------------- 2. Job matches ----------------
with tab_jobs:
    ui.section("🎯", "Recommended jobs", PINK, "Ranked by skills, semantic fit and experience")
    if ss.profile is None:
        ui.empty_state("👤", "Build your profile first", "Head to the Profile tab to get started.")
    else:
        b1, b2 = st.columns([1, 2])
        wants_prep = b2.checkbox("🎤 I also want interview preparation")
        if b1.button("🔍 Find matching jobs", type="primary", use_container_width=True):
            with st.status("🤖 Agents at work...", expanded=True) as status:
                for a in (orch.search_agent, orch.analysis_agent, orch.matching_agent, orch.decision_agent):
                    a.on_log = st.write
                ss.result = orch.find_jobs(ss.profile, wants_interview_prep=wants_prep)
                status.update(label=f"Done in {ss.result.iterations} search iteration(s)", state="complete")
            st.rerun()

        res = ss.result
        if res:
            d = res.decision
            ui.decision_banner(d)
            good = sum(m.score >= threshold for m in res.matches)
            ui.tiles([
                (str(len(res.matches)), "Jobs analysed", "📄", INDIGO),
                (str(good), f"Matches ≥ {threshold}", "⭐", EMERALD),
                (f"{res.matches[0].score:.0f}" if res.matches else "—", "Best score", "🏆", AMBER),
                (str(res.iterations), "Search rounds", "🔁", SKY),
            ])

            view = st.radio("View", ["🃏 Cards", "📊 Table"], horizontal=True, label_visibility="collapsed")
            if view == "📊 Table":
                rows = []
                for m in res.matches:
                    j = res.job(m.job_id)
                    rows.append({"Score": m.score, "Title": j.title, "Company": j.company,
                                 "Location": j.location + (" / Remote" if j.remote else ""),
                                 "Salary": j.salary, "Skills": m.skill_score, "Semantic": m.semantic_score,
                                 "Experience": m.experience_score, "Top pick": m.job_id in d.recommended_job_ids})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                             column_config={"Score": st.column_config.ProgressColumn(
                                 "Match", min_value=0, max_value=100, format="%.0f")})
            else:
                for rank, m in enumerate(res.matches[:10], 1):
                    j, r = res.job(m.job_id), res.req(m.job_id)
                    ui.job_card(j, m, m.job_id in d.recommended_job_ids, rank)
                    with st.expander("📖 Full description & actions"):
                        if r:
                            st.caption(f"Seniority: {r.seniority or '—'} · Min experience: "
                                       f"{r.min_experience_years:g} yrs"
                                       + (f" · Qualifications: {'; '.join(r.qualifications)}"
                                          if r.qualifications else ""))
                        st.write(j.description)
                        c1, c2, c3, c4 = st.columns(4)
                        if j.url:
                            c4.link_button("🔗 Open posting", j.url, use_container_width=True)
                        if c1.button("💾 Save / approve", key=f"save_{j.id}", use_container_width=True):
                            orch.memory.save_job(user_id, j.id, j.title, j.company, m.score, "approved")
                            st.toast("Saved to memory", icon="💾")
                        if c2.button("👍 Good fit", key=f"up_{j.id}", use_container_width=True):
                            orch.memory.add_feedback(user_id, j.id, 1)
                            st.toast("Feedback saved", icon="👍")
                        if c3.button("👎 Not for me", key=f"down_{j.id}", use_container_width=True):
                            orch.memory.add_feedback(user_id, j.id, -1)
                            st.toast("Will be excluded from future searches", icon="👎")

            with st.expander("🛠️ Agent trace"):
                st.code("\n".join(res.trace))


def job_picker(key: str):
    res = ss.result
    ids = (res.decision.prepare_job_ids or []) + [m.job_id for m in res.matches]
    ids = list(dict.fromkeys(ids))
    score = {m.job_id: m.score for m in res.matches}
    return st.selectbox("Choose a job", ids, key=key,
                        format_func=lambda i: f"{score[i]:.0f} · {res.job(i).title} — {res.job(i).company}")


# ---------------- 3. CV tailoring ----------------
with tab_cv:
    ui.section("✍️", "CV Tailoring Agent", VIOLET, "Honest, job-specific improvements to your CV")
    if not ss.result:
        ui.empty_state("🔍", "Find jobs first", "Run a search in the Job Matches tab, then pick a job here.")
    else:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        with c1:
            jid = job_picker("cv_job")
        if c2.button("✍️ Tailor my CV", type="primary", use_container_width=True):
            with st.spinner("CV Tailoring Agent is working..."):
                ss.tailored[jid] = orch.tailor_cv(ss.result, jid, ss.cv_text)
            st.rerun()
        t = ss.tailored.get(jid)
        if t:
            left, right = st.columns([1, 1], gap="large")
            with left:
                ui.section("💡", "Suggested improvements", AMBER)
                ui.numbered(t.improvements)
                ui.section("🌟", "Highlight these skills", EMERALD)
                ui.chips(t.highlighted_skills)
                ui.section("📝", "Tailored summary", PINK)
                ui.render(f'<div class="card accent" style="--c:{PINK}">{ui.esc(t.tailored_summary)}</div>')
            with right:
                ui.section("📄", "Tailored CV", INDIGO)
                with st.container(border=True):
                    st.markdown(t.tailored_cv_markdown)
                st.download_button("⬇️ Download tailored CV (.md)", t.tailored_cv_markdown,
                                   file_name=f"tailored_cv_{jid}.md", use_container_width=True)

# ---------------- 4. Interview prep ----------------
with tab_iv:
    ui.section("🎤", "Interview Agent", EMERALD, "Role-specific questions, model answers and feedback")
    if not ss.result:
        ui.empty_state("🔍", "Find jobs first", "Run a search in the Job Matches tab, then pick a job here.")
    else:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        with c1:
            jid = job_picker("iv_job")
        if c2.button("🎤 Generate questions", type="primary", use_container_width=True):
            with st.spinner("Interview Agent is working..."):
                ss.packs[jid] = orch.interview_pack(ss.result, jid)
            st.rerun()
        pack = ss.packs.get(jid)
        if pack:
            cats = pd.Series([q.category for q in pack.questions]).value_counts()
            ui.chips([f"{c}: {n}" for c, n in cats.items()], label="Question mix")
            for i, q in enumerate(pack.questions):
                with st.container(border=True):
                    ui.question_header(i + 1, q.category, q.question)
                    ans = st.text_area("Your practice answer", key=f"ans_{jid}_{i}", height=90,
                                       placeholder="Type your answer, then get instant feedback...")
                    b1, b2 = st.columns([1, 3])
                    if b1.button("🧪 Get feedback", key=f"fb_{jid}_{i}", use_container_width=True) and ans.strip():
                        with st.spinner("Evaluating..."):
                            ss.feedback[(jid, i)] = orch.interview_agent.feedback(ss.result.job(jid), q.question, ans)
                    show = b2.toggle("Show model answer & tips", key=f"show_{jid}_{i}")
                    fb = ss.feedback.get((jid, i))
                    if fb:
                        color = EMERALD if fb.score >= 7 else (AMBER if fb.score >= 5 else ROSE)
                        ui.render(f'<div class="card accent" style="--c:{color}"><b style="font-size:1.4rem;color:{color}">'
                                  f'{fb.score}/10</b><div class="chips"><span class="legend">Strengths</span>'
                                  f'{ui.chips_html(fb.strengths, "ok")}</div><div class="chips"><span class="legend">'
                                  f'Improve</span>{ui.chips_html(fb.improvements, "rel")}</div>'
                                  + (f'<div class="why">✨ {ui.esc(fb.improved_answer)}</div>' if fb.improved_answer else "")
                                  + "</div>")
                    if show:
                        st.info(q.sample_answer)
                        ui.chips(q.tips, label="Tips")

# ---------------- 5. Memory ----------------
with tab_mem:
    ui.section("💾", "Memory / Knowledge Base", SKY, "What the system remembers to improve future searches")
    mem = orch.memory
    sj, searches, fb = mem.saved_jobs(user_id), mem.recent_searches(user_id), mem.all_feedback(user_id)
    ui.tiles([
        (str(len(sj)), "Saved jobs", "💾", INDIGO),
        (str(len(searches)), "Recent searches", "🔍", VIOLET),
        (str(sum(f["rating"] > 0 for f in fb)), "Liked", "👍", EMERALD),
        (str(sum(f["rating"] < 0 for f in fb)), "Excluded", "👎", ROSE),
    ])
    ui.section("⭐", "Saved & approved jobs", AMBER)
    if sj:
        st.dataframe(pd.DataFrame(sj)[["title", "company", "score", "status"]], hide_index=True,
                     use_container_width=True)
    else:
        ui.empty_state("💾", "No saved jobs yet", "Use “Save / approve” on a job card.")
    ui.section("🕘", "Previous searches", VIOLET)
    if searches:
        for s in searches:
            ui.render(f'<div class="tip"><div class="n" style="background:{VIOLET}">🔍</div><div>'
                      f'<b>{pd.to_datetime(s["created"], unit="s"):%Y-%m-%d %H:%M}</b> · '
                      f'{ui.esc(s["decision"]["action"].replace("_", " "))} · top: '
                      + ui.esc(", ".join(f'{t["title"]} ({t["score"]:.0f})' for t in s["top_jobs"][:3]))
                      + "</div></div>")
    else:
        st.caption("No searches yet.")
    ui.section("💬", "Feedback", PINK)
    if fb:
        st.dataframe(pd.DataFrame(fb)[["job_id", "rating", "note"]], hide_index=True, use_container_width=True)
    else:
        st.caption("No feedback yet.")
