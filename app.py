"""8. User Dashboard (Streamlit).

Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config
from core.llm import get_llm
from core.memory import Memory
from core.schemas import UserInput
from orchestrator import Orchestrator
from tools.cv_parser import parse_cv_bytes

st.set_page_config(page_title="Intelligent Job Search", page_icon="🎯", layout="wide")


@st.cache_resource
def shared_memory() -> Memory:
    return Memory()


ss = st.session_state
ss.setdefault("profile", None)
ss.setdefault("cv_text", "")
ss.setdefault("result", None)
ss.setdefault("tailored", {})
ss.setdefault("packs", {})
ss.setdefault("feedback", {})

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎯 Intelligent Job Search")
    user_id = st.text_input("User ID", value="default", help="Memory is stored per user ID.")
    llm = get_llm()
    if llm.enabled:
        st.success(f"LLM: {llm.mode}")
    else:
        st.warning("LLM: offline heuristics\n\nSet `ANTHROPIC_API_KEY` in `.env` to enable Claude.")
    if llm.last_error:
        st.caption(f"Last LLM note: {llm.last_error}")
    threshold = st.slider("Match threshold", 30, 90, int(config.MATCH_THRESHOLD), 5,
                          help="Below this, the Decision Agent broadens the search.")
    max_iter = st.slider("Max search iterations", 1, 5, config.MAX_SEARCH_ITERATIONS)
    st.divider()
    st.caption("Workflow: Profile → Search → Analysis → Matching → Decision → CV / Interview → Dashboard → Memory")

orch = Orchestrator(user_id=user_id, llm=llm, memory=shared_memory())
orch.decision_agent.threshold = threshold
orch.decision_agent.max_iterations = max_iter

tab_profile, tab_jobs, tab_cv, tab_iv, tab_mem = st.tabs(
    ["1 · Profile", "2 · Job Matches", "3 · CV Tailoring", "4 · Interview Prep", "5 · Memory"])

# ---------------- 1. Profile ----------------
with tab_profile:
    st.header("Your profile")
    if ss.profile is None:
        saved = orch.memory.load_profile(user_id)
        if saved:
            st.info("A saved profile was found in memory for this user.")
            if st.button("Load saved profile"):
                ss.profile = saved
                st.rerun()

    c1, c2 = st.columns([1, 1])
    with c1:
        up = st.file_uploader("Upload CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        if up is not None:
            try:
                ss.cv_text = parse_cv_bytes(up.getvalue(), up.name)
                st.caption(f"Extracted {len(ss.cv_text.split())} words from {up.name}.")
            except Exception as e:
                st.error(f"Could not read CV: {e}")
        if st.button("Use sample CV"):
            ss.cv_text = (config.DATA_DIR / "sample_cv.txt").read_text(encoding="utf-8")
        ss.cv_text = st.text_area("CV text", ss.cv_text, height=300)
    with c2:
        with st.form("profile_form"):
            name = st.text_input("Name")
            skills = st.text_input("Extra skills (comma separated)")
            exp = st.number_input("Years of experience (leave 0 to auto-detect)", 0.0, 40.0, 0.0, 0.5)
            role = st.text_input("Preferred role(s)", placeholder="e.g. Machine Learning Engineer, Data Scientist")
            loc = st.text_input("Preferred location(s)", placeholder="e.g. Bengaluru, Hyderabad")
            remote = st.checkbox("Open to remote", value=True)
            goal = st.text_area("Career goal", placeholder="e.g. Build production AI systems")
            submitted = st.form_submit_button("Build profile", type="primary")
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

    p = ss.profile
    if p:
        st.subheader(p.name or "Structured career profile")
        if p.missing_fields:
            st.warning("Profile incomplete - please add: " + ", ".join(p.missing_fields))
        m1, m2, m3 = st.columns(3)
        m1.metric("Skills", len(p.skills))
        m2.metric("Experience (yrs)", f"{p.experience_years:g}")
        m3.metric("Target roles", len(p.preferred_roles))
        st.write(f"**Headline:** {p.headline}")
        st.write(f"**Summary:** {p.summary}")
        st.write("**Skills:** " + " · ".join(p.skills))
        st.write("**Preferred roles:** " + ", ".join(p.preferred_roles))
        st.write("**Locations:** " + (", ".join(p.preferred_locations) or "any")
                 + (" (+ remote)" if p.remote_ok else ""))
        with st.expander("Full profile JSON"):
            st.json(p.model_dump())

# ---------------- 2. Job matches ----------------
with tab_jobs:
    st.header("Recommended jobs")
    if ss.profile is None:
        st.info("Build your profile first (tab 1).")
    else:
        wants_prep = st.checkbox("I also want interview preparation")
        if st.button("🔍 Find matching jobs", type="primary"):
            with st.status("Running agents...", expanded=True) as status:
                orch.search_agent.on_log = orch.analysis_agent.on_log = st.write
                for a in (orch.matching_agent, orch.decision_agent):
                    a.on_log = st.write
                ss.result = orch.find_jobs(ss.profile, wants_interview_prep=wants_prep)
                status.update(label=f"Done in {ss.result.iterations} search iteration(s)", state="complete")

        res = ss.result
        if res:
            d = res.decision
            icon = {"recommend": "✅", "prepare": "🧭", "search_more": "🔁"}[d.action]
            st.info(f"{icon} **Decision Agent: {d.action.replace('_', ' ').title()}** — {d.reason}")

            rows = []
            for m in res.matches:
                j = res.job(m.job_id)
                rows.append({"Score": m.score, "Title": j.title, "Company": j.company,
                             "Location": j.location + (" / Remote" if j.remote else ""),
                             "Salary": j.salary, "Skills": m.skill_score, "Semantic": m.semantic_score,
                             "Experience": m.experience_score, "Recommended": m.job_id in d.recommended_job_ids})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"Score": st.column_config.ProgressColumn("Match", min_value=0, max_value=100, format="%.0f")})

            st.subheader("Details")
            for m in res.matches[:10]:
                j, r = res.job(m.job_id), res.req(m.job_id)
                with st.expander(f"{m.score:.0f}/100 · {j.title} — {j.company}"
                                 + ("  ⭐" if m.job_id in d.recommended_job_ids else "")):
                    st.write(m.explanation)
                    a, b, c = st.columns(3)
                    a.write("**Matched:** " + (", ".join(m.matched_skills) or "—"))
                    b.write("**Related:** " + (", ".join(m.related_skills) or "—"))
                    c.write("**Missing:** " + (", ".join(m.missing_skills) or "—"))
                    if r:
                        st.caption(f"Seniority: {r.seniority or '—'} · Min experience: {r.min_experience_years:g} yrs"
                                   + (f" · Qualifications: {'; '.join(r.qualifications)}" if r.qualifications else ""))
                    st.write(j.description)
                    if j.url:
                        st.link_button("Open posting", j.url)
                    c1, c2, c3 = st.columns(3)
                    if c1.button("💾 Save / approve", key=f"save_{j.id}"):
                        orch.memory.save_job(user_id, j.id, j.title, j.company, m.score, "approved")
                        st.toast("Saved to memory")
                    if c2.button("👍 Good fit", key=f"up_{j.id}"):
                        orch.memory.add_feedback(user_id, j.id, 1)
                        st.toast("Feedback saved")
                    if c3.button("👎 Not for me", key=f"down_{j.id}"):
                        orch.memory.add_feedback(user_id, j.id, -1)
                        st.toast("Will be excluded from future searches")

            with st.expander("Agent trace"):
                st.code("\n".join(res.trace))


def job_picker(key: str):
    res = ss.result
    ids = (res.decision.prepare_job_ids or []) + [m.job_id for m in res.matches]
    ids = list(dict.fromkeys(ids))
    score = {m.job_id: m.score for m in res.matches}
    return st.selectbox("Choose a job", ids, key=key,
                        format_func=lambda i: f"{res.job(i).title} — {res.job(i).company} ({score[i]:.0f})")


# ---------------- 3. CV tailoring ----------------
with tab_cv:
    st.header("CV Tailoring Agent")
    if not ss.result:
        st.info("Find jobs first (tab 2).")
    else:
        jid = job_picker("cv_job")
        if st.button("✍️ Tailor my CV for this job", type="primary"):
            with st.spinner("CV Tailoring Agent is working..."):
                ss.tailored[jid] = orch.tailor_cv(ss.result, jid, ss.cv_text)
        t = ss.tailored.get(jid)
        if t:
            st.subheader("Suggested improvements")
            for imp in t.improvements:
                st.markdown(f"- {imp}")
            st.write("**Highlight these skills:** " + ", ".join(t.highlighted_skills))
            st.write("**Tailored summary:** " + t.tailored_summary)
            st.subheader("Tailored CV")
            st.markdown(t.tailored_cv_markdown)
            st.download_button("Download tailored CV (.md)", t.tailored_cv_markdown,
                               file_name=f"tailored_cv_{jid}.md")

# ---------------- 4. Interview prep ----------------
with tab_iv:
    st.header("Interview Agent")
    if not ss.result:
        st.info("Find jobs first (tab 2).")
    else:
        jid = job_picker("iv_job")
        if st.button("🎤 Generate interview questions", type="primary"):
            with st.spinner("Interview Agent is working..."):
                ss.packs[jid] = orch.interview_pack(ss.result, jid)
        pack = ss.packs.get(jid)
        if pack:
            for i, q in enumerate(pack.questions):
                with st.expander(f"Q{i + 1} [{q.category}] {q.question}"):
                    ans = st.text_area("Your practice answer", key=f"ans_{jid}_{i}")
                    if st.button("Get feedback", key=f"fb_{jid}_{i}") and ans.strip():
                        with st.spinner("Evaluating..."):
                            ss.feedback[(jid, i)] = orch.interview_agent.feedback(ss.result.job(jid), q.question, ans)
                    fb = ss.feedback.get((jid, i))
                    if fb:
                        st.metric("Score", f"{fb.score}/10")
                        st.write("**Strengths:** " + "; ".join(fb.strengths))
                        st.write("**Improve:** " + "; ".join(fb.improvements))
                        if fb.improved_answer:
                            st.write("**Improved answer:** " + fb.improved_answer)
                    if st.toggle("Show model answer & tips", key=f"show_{jid}_{i}"):
                        st.write(q.sample_answer)
                        for tip in q.tips:
                            st.markdown(f"- {tip}")

# ---------------- 5. Memory ----------------
with tab_mem:
    st.header("Memory / Knowledge Base")
    mem = orch.memory
    st.subheader("Saved & approved jobs")
    sj = mem.saved_jobs(user_id)
    if sj:
        st.dataframe(pd.DataFrame(sj), hide_index=True)
    else:
        st.caption("None yet.")
    st.subheader("Previous searches")
    for s in mem.recent_searches(user_id):
        st.write(f"**{pd.to_datetime(s['created'], unit='s'):%Y-%m-%d %H:%M}** — "
                 f"{s['decision']['action']} — top: "
                 + ", ".join(f"{t['title']} ({t['score']:.0f})" for t in s["top_jobs"][:3]))
    st.subheader("Feedback")
    fb = mem.all_feedback(user_id)
    if fb:
        st.dataframe(pd.DataFrame(fb), hide_index=True)
    else:
        st.caption("None yet.")
