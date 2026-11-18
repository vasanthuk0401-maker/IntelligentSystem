"""Shared data models passed between agents."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- 1/2. User input & profile ----------
class UserInput(BaseModel):
    cv_text: str = ""
    name: str = ""
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    preferred_role: str = ""
    location: str = ""
    remote_ok: bool = True
    career_goal: str = ""


class ExtractedProfile(BaseModel):
    """Fields the Profile Agent extracts (this is the schema sent to Claude)."""
    name: str = ""
    headline: str = ""
    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    experience_years: float = 0.0
    past_roles: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    preferred_roles: List[str] = Field(default_factory=list)
    preferred_locations: List[str] = Field(default_factory=list)
    remote_ok: bool = True
    career_goal: str = ""
    missing_fields: List[str] = Field(default_factory=list)


class CareerProfile(ExtractedProfile):
    # Raw CV text (trimmed), used for semantic matching; not asked of the LLM.
    source_text: str = ""


# ---------- 3. Job search ----------
class SearchCriteria(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    remote_ok: bool = True
    exclude_job_ids: List[str] = Field(default_factory=list)
    limit: int = 10
    broaden_level: int = 0  # increased by the feedback loop
    query_text: str = ""  # profile summary used to semantically re-rank candidates


class JobListing(BaseModel):
    id: str
    title: str
    company: str
    location: str = ""
    remote: bool = False
    description: str
    salary: str = ""
    url: str = ""
    source: str = "dataset"


# ---------- 4. Job analysis ----------
class JobRequirements(BaseModel):
    job_id: str
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience_years: float = 0.0
    qualifications: List[str] = Field(default_factory=list)
    seniority: str = ""
    salary: str = ""
    location: str = ""
    summary: str = ""


# ---------- 5. Skill matching ----------
class MatchResult(BaseModel):
    job_id: str
    score: float
    skill_score: float
    semantic_score: float
    experience_score: float
    matched_skills: List[str] = Field(default_factory=list)
    related_skills: List[str] = Field(default_factory=list)  # "job skill ~ your skill"
    missing_skills: List[str] = Field(default_factory=list)
    explanation: str = ""


# ---------- 6. Decision ----------
class Decision(BaseModel):
    action: Literal["search_more", "recommend", "prepare"]
    reason: str
    recommended_job_ids: List[str] = Field(default_factory=list)
    prepare_job_ids: List[str] = Field(default_factory=list)


# ---------- 7A. CV tailoring ----------
class TailoredCV(BaseModel):
    job_id: str = ""
    improvements: List[str] = Field(default_factory=list)
    highlighted_skills: List[str] = Field(default_factory=list)
    tailored_summary: str = ""
    tailored_cv_markdown: str = ""


# ---------- 7B. Interview ----------
class InterviewQuestion(BaseModel):
    question: str
    category: str = "technical"
    sample_answer: str = ""
    tips: List[str] = Field(default_factory=list)


class InterviewPack(BaseModel):
    job_id: str = ""
    questions: List[InterviewQuestion] = Field(default_factory=list)


class AnswerFeedback(BaseModel):
    score: int = 0  # 1-10
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    improved_answer: str = ""


# ---------- Pipeline result ----------
class PipelineResult(BaseModel):
    profile: CareerProfile
    criteria: SearchCriteria
    jobs: List[JobListing] = Field(default_factory=list)
    requirements: List[JobRequirements] = Field(default_factory=list)
    matches: List[MatchResult] = Field(default_factory=list)
    decision: Optional[Decision] = None
    iterations: int = 0
    trace: List[str] = Field(default_factory=list)

    def job(self, job_id: str) -> Optional[JobListing]:
        return next((j for j in self.jobs if j.id == job_id), None)

    def req(self, job_id: str) -> Optional[JobRequirements]:
        return next((r for r in self.requirements if r.job_id == job_id), None)
