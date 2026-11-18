"""7A. CV Tailoring Agent - suggests job-specific improvements, highlights
relevant skills & experience, and provides a tailored CV version."""
from __future__ import annotations

from agents.base import BaseAgent
from core.schemas import CareerProfile, JobListing, JobRequirements, MatchResult, TailoredCV

SYSTEM = """You are the CV Tailoring Agent of a job-search assistant.
Tailor the candidate's CV to one specific job.
Hard rules:
- Never invent experience, employers, degrees, metrics or skills the candidate does not have.
  You may rephrase, reorder and emphasise what is already there.
- For missing skills, suggest honest actions (a project, a course, or how to mention related experience) in `improvements`.
- improvements: 5-8 concrete, specific edits ("Move the RAG chatbot project to the top and mention FastAPI").
- highlighted_skills: the candidate's skills most relevant to this job, most important first.
- tailored_summary: a 3-4 sentence professional summary aimed at this role.
- tailored_cv_markdown: the full tailored CV in Markdown (headings: Summary, Skills, Experience, Projects, Education),
  using ATS-friendly keywords from the job posting only where they truthfully apply."""


class CVTailoringAgent(BaseAgent):
    name = "CV Tailoring Agent"

    def run(self, profile: CareerProfile, cv_text: str, job: JobListing,
            req: JobRequirements, match: MatchResult) -> TailoredCV:
        self.log(f"Tailoring CV for {job.title} at {job.company}...")
        prompt = (
            f"<job>\nTitle: {job.title}\nCompany: {job.company}\n{job.description}\n</job>\n\n"
            f"<requirements>\n{req.model_dump_json(indent=1)}\n</requirements>\n\n"
            f"<match>\nscore: {match.score}\nmatched: {match.matched_skills}\n"
            f"missing: {match.missing_skills}\n</match>\n\n"
            f"<profile>\n{profile.model_dump_json(indent=1)}\n</profile>\n\n"
            f"<original_cv>\n{cv_text or '(no CV uploaded - build from profile)'}\n</original_cv>"
        )
        out = self.llm.parse(SYSTEM, prompt, TailoredCV, effort="medium", max_tokens=32000)
        if out is None:
            out = self._heuristic(profile, job, req, match)
            self.log("Tailored CV generated with offline heuristics.")
        else:
            self.log("Tailored CV generated with Claude.")
        out.job_id = job.id
        return out

    @staticmethod
    def _heuristic(profile: CareerProfile, job: JobListing, req: JobRequirements,
                   match: MatchResult) -> TailoredCV:
        relevant = [s for s in match.matched_skills] + [s for s in profile.skills if s not in match.matched_skills]
        improvements = [
            f"Open your summary with the target title: '{job.title}'.",
            f"List these matching skills first: {', '.join(match.matched_skills[:6]) or 'your strongest skills'}.",
            "Quantify results in each experience/project bullet (%, time saved, users, accuracy).",
            "Mirror the job's wording for skills you genuinely have (helps ATS keyword matching).",
        ]
        for s in match.missing_skills[:3]:
            improvements.append(f"Gap: {s} - add a small project or course, or mention related experience if you have it.")
        summary = (f"{profile.headline or 'Candidate'} with {profile.experience_years:g} years of experience, "
                   f"skilled in {', '.join(relevant[:5])}. Seeking a {job.title} role at {job.company}"
                   + (f" to {profile.career_goal.lower()}" if profile.career_goal else "") + ".")
        md = [f"# {profile.name or 'Your Name'}", f"**Target role:** {job.title}", "", "## Summary", summary,
              "", "## Skills", ", ".join(relevant), "", "## Experience"]
        md += [f"- {r}" for r in profile.past_roles] or ["- (add your roles)"]
        if profile.achievements:
            md += ["", "## Key Achievements"] + [f"- {a}" for a in profile.achievements]
        md += ["", "## Education"] + ([f"- {e}" for e in profile.education] or ["- (add education)"])
        return TailoredCV(job_id=job.id, improvements=improvements, highlighted_skills=relevant[:8],
                          tailored_summary=summary, tailored_cv_markdown="\n".join(md))
