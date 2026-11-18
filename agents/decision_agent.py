"""6. Decision Agent - evaluates match results and decides the next step:

- search_more : best match below threshold -> Update Search Criteria & re-run
- prepare     : good matches, but skill gaps or user asked for interview prep
                -> CV Tailoring + Interview Agent
- recommend   : good matches -> recommend jobs (CV tailoring available)

The policy is deterministic (transparent and testable); Claude is used by the
downstream agents, not to make this routing decision.
"""
from __future__ import annotations

from typing import List

from agents.base import BaseAgent
from core import config
from core.schemas import CareerProfile, Decision, MatchResult, SearchCriteria

# Related role keywords used to broaden searches
RELATED_ROLES = {
    "data scientist": ["machine learning", "data analyst", "ml engineer"],
    "data analyst": ["business intelligence", "data scientist", "analyst"],
    "machine learning": ["data scientist", "ai engineer", "mlops", "nlp"],
    "ml engineer": ["machine learning", "ai engineer", "data scientist"],
    "ai engineer": ["machine learning", "generative ai", "nlp", "llm"],
    "software engineer": ["backend developer", "full stack", "developer", "graduate"],
    "backend": ["software engineer", "python", "java", "api"],
    "frontend": ["react", "full stack", "ui"],
    "full stack": ["frontend", "backend", "software engineer"],
    "devops": ["cloud", "mlops", "sre", "platform"],
    "data engineer": ["etl", "spark", "cloud data", "analytics engineer"],
}


class DecisionAgent(BaseAgent):
    name = "Decision Agent"

    def __init__(self, threshold: float = config.MATCH_THRESHOLD,
                 max_iterations: int = config.MAX_SEARCH_ITERATIONS, top_n: int = 5, **kw):
        super().__init__(**kw)
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.top_n = top_n

    def run(self, matches: List[MatchResult], iteration: int,
            wants_interview_prep: bool = False) -> Decision:
        good = [m for m in matches if m.score >= self.threshold]
        best = matches[0].score if matches else 0.0

        if len(good) < 2 and iteration < self.max_iterations:
            d = Decision(action="search_more",
                         reason=f"Only {len(good)} job(s) scored >= {self.threshold:g} "
                                f"(best {best:.0f}). Broadening the search (iteration {iteration + 1}).")
        else:
            pool = good or matches
            recommended = [m.job_id for m in pool[: self.top_n]]
            gaps = [m.job_id for m in pool[: self.top_n] if m.missing_skills]
            if wants_interview_prep or (good and gaps):
                d = Decision(action="prepare",
                             reason=("Good matches found. " if good else "Best available matches shown (below threshold). ")
                                    + ("Interview prep requested. " if wants_interview_prep else "")
                                    + ("Some skill gaps detected - CV tailoring and interview prep recommended."
                                       if gaps else ""),
                             recommended_job_ids=recommended,
                             prepare_job_ids=gaps or recommended)
            else:
                d = Decision(action="recommend",
                             reason=(f"{len(good)} job(s) scored >= {self.threshold:g}." if good else
                                     f"Search limit reached; showing best available (top score {best:.0f})."),
                             recommended_job_ids=recommended)
        self.log(f"Decision: {d.action.upper()} - {d.reason}")
        return d

    def update_criteria(self, criteria: SearchCriteria, profile: CareerProfile,
                        matches: List[MatchResult]) -> SearchCriteria:
        """'Update Search Criteria' box: adjust filters / keywords for the re-run."""
        c = criteria.model_copy(deep=True)
        c.broaden_level += 1
        c.limit = criteria.limit + 5
        # Exclude jobs that were clearly poor fits so new ones surface.
        c.exclude_job_ids = list(set(c.exclude_job_ids) | {m.job_id for m in matches if m.score < 35})
        new_kw = []
        for role in profile.preferred_roles + profile.past_roles:
            for key, rel in RELATED_ROLES.items():
                if key in role.lower():
                    new_kw += rel
        if c.broaden_level >= 2:
            new_kw += profile.skills  # search on all skills, location filter relaxed
        c.keywords = list(dict.fromkeys(c.keywords + new_kw))
        changes = [f"+keywords {new_kw[:6]}" if new_kw else "keywords unchanged",
                   f"limit {c.limit}"]
        if c.broaden_level >= 2:
            changes.append("location filter relaxed")
        self.log("Updated search criteria: " + ", ".join(changes))
        return c
