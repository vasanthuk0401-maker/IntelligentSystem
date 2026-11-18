"""3. Job Search Agent - searches job APIs / boards / dataset, applies user
preferences and filters, returns relevant listings.

Two stages, so it scales to large datasets:
  1. keyword retrieval (IDF-weighted inverted index) -> a wide candidate pool
  2. semantic re-rank of that pool against the profile (local embeddings)
Only the top `limit` go on to the (more expensive) Job Analysis Agent.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from agents.base import BaseAgent
from agents.skill_matching_agent import job_text, profile_embed_text
from core.schemas import CareerProfile, JobListing, SearchCriteria
from tools.embeddings import get_embedder
from tools.job_sources import JobSource, default_sources

POOL_FACTOR = 5  # candidates retrieved per final slot
MAX_POOL = 60


class JobSearchAgent(BaseAgent):
    name = "Job Search Agent"

    def __init__(self, sources: Optional[List[JobSource]] = None, rerank: bool = True, **kw):
        super().__init__(**kw)
        self.sources = sources if sources is not None else default_sources()
        self.rerank = rerank

    def build_criteria(self, profile: CareerProfile, exclude: Optional[List[str]] = None,
                       limit: int = 10) -> SearchCriteria:
        keywords = list(profile.preferred_roles) + profile.skills[:6]
        return SearchCriteria(
            keywords=keywords,
            locations=profile.preferred_locations,
            remote_ok=profile.remote_ok,
            exclude_job_ids=exclude or [],
            limit=limit,
            query_text=profile_embed_text(profile),
        )

    def run(self, criteria: SearchCriteria) -> List[JobListing]:
        self.log(
            f"Searching {', '.join(s.name for s in self.sources)} | keywords={criteria.keywords[:8]} "
            f"| locations={criteria.locations or 'any'} | remote_ok={criteria.remote_ok} "
            f"| broaden_level={criteria.broaden_level}"
        )
        pool_criteria = criteria.model_copy(update={"limit": min(MAX_POOL, criteria.limit * POOL_FACTOR)})
        seen, jobs = set(), []
        for src in self.sources:
            try:
                for j in src.search(pool_criteria):
                    key = (j.title.lower(), j.company.lower())
                    if j.id not in seen and key not in seen:
                        seen.update({j.id, key})
                        jobs.append(j)
            except Exception as e:
                self.log(f"Source {src.name} failed: {e}")
        self.log(f"Retrieved {len(jobs)} keyword candidates.")
        jobs = self._rerank(jobs, criteria)
        return jobs[: criteria.limit]

    def _rerank(self, jobs: List[JobListing], criteria: SearchCriteria) -> List[JobListing]:
        if not (self.rerank and criteria.query_text and len(jobs) > criteria.limit):
            return jobs
        emb = get_embedder()
        if emb is None:
            return jobs
        try:
            sims = emb.embed([job_text(j) for j in jobs]) @ emb.embed([criteria.query_text])[0]
        except Exception as e:
            self.log(f"Re-rank skipped ({e}).")
            return jobs
        order = np.argsort(-sims)
        self.log(f"Re-ranked candidates semantically; keeping top {criteria.limit}.")
        return [jobs[i] for i in order]
