"""3. Job Search Agent - searches job APIs / boards / dataset, applies user
preferences and filters, returns relevant listings."""
from __future__ import annotations

from typing import List, Optional

from agents.base import BaseAgent
from core.schemas import CareerProfile, JobListing, SearchCriteria
from tools.job_sources import JobSource, default_sources


class JobSearchAgent(BaseAgent):
    name = "Job Search Agent"

    def __init__(self, sources: Optional[List[JobSource]] = None, **kw):
        super().__init__(**kw)
        self.sources = sources if sources is not None else default_sources()

    def build_criteria(self, profile: CareerProfile, exclude: Optional[List[str]] = None,
                       limit: int = 10) -> SearchCriteria:
        keywords = list(profile.preferred_roles) + profile.skills[:6]
        return SearchCriteria(
            keywords=keywords,
            locations=profile.preferred_locations,
            remote_ok=profile.remote_ok,
            exclude_job_ids=exclude or [],
            limit=limit,
        )

    def run(self, criteria: SearchCriteria) -> List[JobListing]:
        self.log(
            f"Searching {', '.join(s.name for s in self.sources)} | keywords={criteria.keywords[:8]} "
            f"| locations={criteria.locations or 'any'} | remote_ok={criteria.remote_ok} "
            f"| broaden_level={criteria.broaden_level}"
        )
        seen, jobs = set(), []
        for src in self.sources:
            try:
                for j in src.search(criteria):
                    key = (j.title.lower(), j.company.lower())
                    if j.id not in seen and key not in seen:
                        seen.update({j.id, key})
                        jobs.append(j)
            except Exception as e:
                self.log(f"Source {src.name} failed: {e}")
        self.log(f"Found {len(jobs)} job listings.")
        return jobs[: criteria.limit * max(1, len(self.sources))]
