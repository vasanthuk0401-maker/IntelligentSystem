"""5. Skill Matching Agent - compares user skills with job requirements and
calculates a match score with an explanation.

Score = 50% skill coverage + 30% semantic similarity + 20% experience fit
        (+ small location bonus/penalty).

- Skill coverage: exact/alias matches get full credit, related skills
  (e.g. TensorFlow when the job asks for PyTorch) get partial credit;
  required skills weigh 2x preferred ones.
- Semantic similarity: local sentence embeddings (bge-small) of the profile +
  CV vs. the job posting; falls back to TF-IDF cosine if embeddings are
  unavailable.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agents.base import BaseAgent
from core.schemas import CareerProfile, JobListing, JobRequirements, MatchResult
from tools.embeddings import get_embedder
from tools.skills import RELATED_CREDIT, skills_overlap

W_SKILL, W_SEMANTIC, W_EXP = 0.5, 0.3, 0.2
# Map raw bge-small cosine (unrelated ~0.55, near-identical ~0.85) onto 0-1.
EMB_LO, EMB_HI = 0.55, 0.85
TFIDF_SCALE = 0.35


def profile_text(p: CareerProfile) -> str:
    return " ".join([
        p.headline, p.summary, " ".join(p.skills) * 2, " ".join(p.past_roles),
        " ".join(p.preferred_roles) * 2, " ".join(p.achievements), p.career_goal,
    ])


def profile_embed_text(p: CareerProfile) -> str:
    return (f"{p.headline}. Target roles: {', '.join(p.preferred_roles)}. "
            f"Skills: {', '.join(p.skills)}. Goal: {p.career_goal}. "
            f"{p.source_text[:1500] or p.summary}")


def job_text(j: JobListing) -> str:
    return f"{j.title}. {j.description}"


class SkillMatchingAgent(BaseAgent):
    name = "Skill Matching Agent"

    def __init__(self, use_embeddings: bool = True, **kw):
        super().__init__(**kw)
        self.use_embeddings = use_embeddings
        self.last_method = "tfidf"

    def run(self, profile: CareerProfile, jobs: List[JobListing],
            reqs: List[JobRequirements]) -> List[MatchResult]:
        if not jobs:
            return []
        req_by_id = {r.job_id: r for r in reqs}
        semantic = self.semantic_scores(profile, jobs)
        self.log(f"Matching profile against {len(jobs)} jobs (semantic: {self.last_method})...")
        results = []
        for job, sem in zip(jobs, semantic):
            req = req_by_id.get(job.id) or JobRequirements(job_id=job.id)
            results.append(self.score(profile, job, req, sem))
        results.sort(key=lambda r: r.score, reverse=True)
        self.log(f"Best match: {results[0].score:.0f}/100 ({results[0].job_id}).")
        return results

    def semantic_scores(self, profile: CareerProfile, jobs: List[JobListing]) -> List[float]:
        emb = get_embedder() if self.use_embeddings else None
        if emb is not None:
            try:
                q = emb.embed([profile_embed_text(profile)])[0]
                sims = emb.embed([job_text(j) for j in jobs]) @ q
                self.last_method = "embeddings"
                return [float(np.clip((s - EMB_LO) / (EMB_HI - EMB_LO), 0, 1)) for s in sims]
            except Exception as e:
                self.log(f"Embedding failed ({e}); using TF-IDF.")
        self.last_method = "tfidf"
        return self._tfidf_scores(profile, jobs)

    @staticmethod
    def _tfidf_scores(profile: CareerProfile, jobs: List[JobListing]) -> List[float]:
        docs = [profile_text(profile) + " " + profile.source_text] + [f"{j.title} {job_text(j)}" for j in jobs]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        m = vec.fit_transform(docs)
        sims = cosine_similarity(m[0:1], m[1:]).flatten()
        return [min(1.0, float(s) / TFIDF_SCALE) for s in sims]

    @staticmethod
    def score(profile: CareerProfile, job: JobListing, req: JobRequirements,
              semantic: float) -> MatchResult:
        m_req, r_req, x_req = skills_overlap(profile.skills, req.required_skills)
        m_pref, r_pref, x_pref = skills_overlap(profile.skills, req.preferred_skills)
        n_req, n_pref = len(req.required_skills), len(req.preferred_skills)
        if n_req or n_pref:
            got_req = len(m_req) + RELATED_CREDIT * len(r_req)
            got_pref = len(m_pref) + RELATED_CREDIT * len(r_pref)
            skill = (2 * got_req + got_pref) / (2 * n_req + n_pref)  # required weigh 2x
        else:
            skill = semantic

        gap = req.min_experience_years - profile.experience_years
        exp = 1.0 if gap <= 0 else max(0.0, 1.0 - gap / 4.0)

        total = 100 * (W_SKILL * skill + W_SEMANTIC * semantic + W_EXP * exp)
        loc_note = ""
        if profile.preferred_locations:
            in_loc = any(l.lower() in job.location.lower() for l in profile.preferred_locations)
            if in_loc or (profile.remote_ok and job.remote):
                total += 3
            else:
                total -= 5
                loc_note = f" Location ({job.location}) is outside your preferences."
        total = max(0.0, min(100.0, total))

        parts = [f"Matches {len(m_req)}/{n_req} required skills"
                 + (f" and {len(m_pref)}/{n_pref} preferred" if n_pref else "") + "."]
        if r_req or r_pref:
            parts.append("Related (partial credit): " + ", ".join((r_req + r_pref)[:4]) + ".")
        if x_req:
            parts.append("Missing required: " + ", ".join(x_req[:5]) + ".")
        if gap > 0:
            parts.append(f"Asks for {req.min_experience_years:g}+ yrs; you have {profile.experience_years:g}.")
        else:
            parts.append("Experience requirement met.")
        parts.append(f"Profile/description similarity {semantic:.0%}.{loc_note}")

        return MatchResult(
            job_id=job.id,
            score=round(total, 1),
            skill_score=round(skill * 100, 1),
            semantic_score=round(semantic * 100, 1),
            experience_score=round(exp * 100, 1),
            matched_skills=m_req + m_pref,
            related_skills=r_req + r_pref,
            missing_skills=x_req + x_pref,
            explanation=" ".join(parts),
        )
