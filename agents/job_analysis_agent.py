"""4. Job Analysis Agent - reads job descriptions and extracts skills,
experience, qualifications, salary, location and other requirements."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from agents.base import BaseAgent
from core.memory import Memory
from core.schemas import JobListing, JobRequirements
from tools.skills import extract_skills, normalize_skills

SYSTEM = """You are the Job Analysis Agent of a job-search assistant.
Read one job posting and extract its requirements.
- required_skills: skills the posting says are required / must-have.
- preferred_skills: nice-to-have / preferred / bonus skills.
- Use short canonical skill names ("Python", "SQL", "Docker", "Machine Learning").
- min_experience_years: lowest number of years asked for (0 if entry level / not stated).
- qualifications: degrees, certifications or other formal requirements.
- seniority: one of intern, entry, mid, senior, lead.
- summary: one sentence describing the role.
Only use information in the posting."""


class JobAnalysisAgent(BaseAgent):
    name = "Job Analysis Agent"

    def __init__(self, memory: Optional[Memory] = None, max_workers: int = 4, **kw):
        super().__init__(**kw)
        self.memory = memory
        self.max_workers = max_workers

    def run(self, jobs: List[JobListing]) -> List[JobRequirements]:
        self.log(f"Analysing {len(jobs)} job descriptions...")
        cached, todo = {}, []
        for j in jobs:
            hit = self.memory.get_analysis(j.id) if self.memory else None
            if hit:
                cached[j.id] = hit
            else:
                todo.append(j)
        if cached:
            self.log(f"{len(cached)} analyses loaded from memory.")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            fresh = list(pool.map(self.analyze_one, todo))
        for r in fresh:
            cached[r.job_id] = r
            if self.memory:
                self.memory.put_analysis(r)
        return [cached[j.id] for j in jobs]

    def analyze_one(self, job: JobListing) -> JobRequirements:
        prompt = (
            f"Title: {job.title}\nCompany: {job.company}\nLocation: {job.location}"
            f"{' (remote available)' if job.remote else ''}\nSalary: {job.salary or 'not stated'}\n\n"
            f"<description>\n{job.description}\n</description>"
        )
        req = self.llm.parse(SYSTEM, prompt, JobRequirements, effort="low")
        if req is None:
            req = self._heuristic(job)
        req.job_id = job.id
        req.required_skills = normalize_skills(req.required_skills)
        req.preferred_skills = [s for s in normalize_skills(req.preferred_skills)
                                if s.lower() not in {r.lower() for r in req.required_skills}]
        req.salary = req.salary or job.salary
        req.location = req.location or job.location
        return req

    @staticmethod
    def _heuristic(job: JobListing) -> JobRequirements:
        text = job.description
        parts = re.split(r"(?i)(preferred:|nice to have:|bonus:)", text, maxsplit=1)
        required_text = parts[0]
        preferred_text = "".join(parts[1:]) if len(parts) > 1 else ""
        years = re.findall(r"(\d+)\s*(?:-\s*\d+)?\+?\s*years?", text, re.I)
        min_years = float(min(int(y) for y in years)) if years else 0.0
        quals = re.findall(r"(?i)((?:bachelor|master|ph\.?d)[^.]*\.)", text)
        title = job.title.lower()
        seniority = ("intern" if "intern" in title else "senior" if "senior" in title or "lead" in title
                     else "entry" if any(w in title for w in ("junior", "graduate")) or min_years < 1
                     else "mid")
        return JobRequirements(
            job_id=job.id,
            required_skills=extract_skills(required_text),
            preferred_skills=extract_skills(preferred_text),
            min_experience_years=min_years,
            qualifications=[q.strip() for q in quals],
            seniority=seniority,
            salary=job.salary,
            location=job.location,
            summary=text.split(".")[0] + ".",
        )
