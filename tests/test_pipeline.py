"""Offline tests (no API key needed): run the whole workflow with heuristics."""
from pathlib import Path

import pytest

from core import config
from core.llm import LLM
from core.memory import Memory
from core.schemas import UserInput
from orchestrator import Orchestrator


@pytest.fixture
def orch(tmp_path: Path):
    return Orchestrator(user_id="test", llm=LLM(enabled=False), memory=Memory(tmp_path / "m.db"))


@pytest.fixture
def cv():
    return (config.DATA_DIR / "sample_cv.txt").read_text(encoding="utf-8")


def test_profile_extraction(orch, cv):
    p = orch.build_profile(UserInput(cv_text=cv, preferred_role="Data Scientist",
                                     location="Bengaluru", career_goal="Work on NLP"))
    assert {"Python", "SQL", "PyTorch"} <= set(p.skills)
    assert p.experience_years == pytest.approx(0.5)
    assert p.missing_fields == []
    assert orch.memory.load_profile("test").skills == p.skills


def test_find_jobs_ranks_relevant_first(orch, cv):
    p = orch.build_profile(UserInput(cv_text=cv, preferred_role="Machine Learning Engineer",
                                     location="Bengaluru", career_goal="AI"))
    res = orch.find_jobs(p)
    assert res.decision.action in {"recommend", "prepare"}
    top_titles = [res.job(m.job_id).title for m in res.matches[:3]]
    assert any(k in t for t in top_titles for k in ("Data Scientist", "AI", "NLP", "Machine Learning"))
    assert orch.memory.recent_searches("test")


def test_feedback_loop_broadens_search(orch):
    # A weak profile triggers 'search_more' and the Update Search Criteria loop.
    p = orch.build_profile(UserInput(skills=["Excel", "Communication", "SEO"],
                                     preferred_role="Marketing", location="Kolkata", career_goal="grow"))
    orch.decision_agent.threshold = 95
    res = orch.find_jobs(p)
    assert res.iterations > 1
    assert any("Updated search criteria" in t for t in res.trace)


def test_cv_and_interview(orch, cv):
    p = orch.build_profile(UserInput(cv_text=cv, preferred_role="Data Scientist", career_goal="NLP"))
    res = orch.find_jobs(p, wants_interview_prep=True)
    assert res.decision.action == "prepare"
    jid = res.matches[0].job_id
    t = orch.tailor_cv(res, jid, cv)
    assert t.improvements and t.tailored_cv_markdown
    pack = orch.interview_pack(res, jid)
    assert len(pack.questions) >= 5
    fb = orch.interview_agent.feedback(res.job(jid), pack.questions[0].question,
                                       "In my internship I built a PyTorch classifier that improved accuracy by 18%.")
    assert 1 <= fb.score <= 10


def test_related_skills_get_partial_credit():
    from tools.skills import skills_overlap

    matched, related, missing = skills_overlap(["PyTorch", "Tableau"], ["TensorFlow", "Power BI", "Kafka", "PyTorch"])
    assert matched == ["PyTorch"]
    assert related == ["TensorFlow ~ PyTorch", "Power BI ~ Tableau"]
    assert missing == ["Kafka"]


def test_tfidf_fallback_without_embeddings(orch, cv):
    from agents.skill_matching_agent import SkillMatchingAgent

    p = orch.build_profile(UserInput(cv_text=cv, preferred_role="Data Scientist", career_goal="NLP"))
    jobs = orch.search_agent.sources[0].jobs[:5]
    m = SkillMatchingAgent(llm=orch.llm, use_embeddings=False)
    scores = m.semantic_scores(p, jobs)
    assert m.last_method == "tfidf" and all(0 <= s <= 1 for s in scores)


def test_disliked_jobs_excluded(orch, cv):
    p = orch.build_profile(UserInput(cv_text=cv, preferred_role="Data Scientist", career_goal="x"))
    first = orch.find_jobs(p).matches[0].job_id
    orch.memory.add_feedback("test", first, -1)
    assert first not in {m.job_id for m in orch.find_jobs(p).matches}
