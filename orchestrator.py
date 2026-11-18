"""Orchestrator - runs the workflow:

User Input -> Profile Agent -> Job Search -> Job Analysis -> Skill Matching
-> Decision Agent --(match low)--> Update Search Criteria -> Job Search (loop)
                 --(match good)--> recommend   (CV Tailoring on demand)
                 --(prep needed)--> prepare    (CV Tailoring + Interview on demand)
-> Dashboard; results saved to Memory.

CV tailoring and interview prep run on demand from the dashboard, after the
user reviews and approves which job to act on.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from agents import (CVTailoringAgent, DecisionAgent, InterviewAgent, JobAnalysisAgent,
                    JobSearchAgent, ProfileAgent, SkillMatchingAgent)
from core import config
from core.llm import LLM, get_llm
from core.memory import Memory
from core.schemas import (CareerProfile, InterviewPack, JobListing, PipelineResult,
                          TailoredCV, UserInput)


class Orchestrator:
    def __init__(self, user_id: str = "default", llm: Optional[LLM] = None,
                 memory: Optional[Memory] = None, sources=None,
                 on_log: Optional[Callable[[str], None]] = None):
        self.user_id = user_id
        self.llm = llm or get_llm()
        self.memory = memory or Memory()
        self.trace: List[str] = []
        kw = dict(llm=self.llm, trace=self.trace, on_log=on_log)
        self.profile_agent = ProfileAgent(**kw)
        self.search_agent = JobSearchAgent(sources=sources, **kw)
        self.analysis_agent = JobAnalysisAgent(memory=self.memory, **kw)
        self.matching_agent = SkillMatchingAgent(**kw)
        self.decision_agent = DecisionAgent(**kw)
        self.cv_agent = CVTailoringAgent(**kw)
        self.interview_agent = InterviewAgent(**kw)

    # 1 + 2
    def build_profile(self, user: UserInput) -> CareerProfile:
        profile = self.profile_agent.run(user)
        self.memory.save_profile(self.user_id, profile)
        return profile

    # 3 -> 6 with feedback loop
    def find_jobs(self, profile: CareerProfile, wants_interview_prep: bool = False) -> PipelineResult:
        self.trace.clear()
        excluded = self.memory.disliked_job_ids(self.user_id)
        if excluded:
            self.search_agent.log(f"Memory: excluding {len(excluded)} job(s) you disliked before.")
        criteria = self.search_agent.build_criteria(profile, exclude=excluded, limit=config.TOP_K_ANALYZE)

        all_jobs: Dict[str, JobListing] = {}
        reqs, matches, decision = [], [], None
        iteration = 0
        while True:
            found = self.search_agent.run(criteria)
            new = [j for j in found if j.id not in all_jobs]
            for j in new:
                all_jobs[j.id] = j
            if new:
                reqs += self.analysis_agent.run(new)
            jobs = list(all_jobs.values())
            matches = self.matching_agent.run(profile, jobs, reqs)
            decision = self.decision_agent.run(matches, iteration, wants_interview_prep)
            if decision.action != "search_more":
                break
            if not new and iteration > 0 and criteria.broaden_level >= 2:
                self.decision_agent.log("No new jobs found after broadening; stopping the loop.")
                decision = self.decision_agent.run(matches, self.decision_agent.max_iterations,
                                                   wants_interview_prep)
                break
            criteria = self.decision_agent.update_criteria(criteria, profile, matches)
            iteration += 1

        result = PipelineResult(profile=profile, criteria=criteria, jobs=list(all_jobs.values()),
                                requirements=reqs, matches=matches, decision=decision,
                                iterations=iteration + 1, trace=list(self.trace))
        top = [{"job_id": m.job_id, "title": result.job(m.job_id).title, "score": m.score}
               for m in matches[:5]]
        self.memory.log_search(self.user_id, criteria.model_dump(), decision.model_dump(), top)
        return result

    # 7A
    def tailor_cv(self, result: PipelineResult, job_id: str, cv_text: str) -> TailoredCV:
        job, req = result.job(job_id), result.req(job_id)
        match = next(m for m in result.matches if m.job_id == job_id)
        out = self.cv_agent.run(result.profile, cv_text, job, req, match)
        self.memory.save_artifact(self.user_id, job_id, "tailored_cv", out.model_dump_json())
        return out

    # 7B
    def interview_pack(self, result: PipelineResult, job_id: str) -> InterviewPack:
        job, req = result.job(job_id), result.req(job_id)
        match = next(m for m in result.matches if m.job_id == job_id)
        out = self.interview_agent.run(result.profile, job, req, match)
        self.memory.save_artifact(self.user_id, job_id, "interview_pack", out.model_dump_json())
        return out


if __name__ == "__main__":  # quick CLI demo
    import sys

    from tools.cv_parser import parse_cv_file

    cv = parse_cv_file(sys.argv[1]) if len(sys.argv) > 1 else (config.DATA_DIR / "sample_cv.txt").read_text()
    orch = Orchestrator(user_id="cli", on_log=print)
    print("LLM mode:", orch.llm.mode)
    prof = orch.build_profile(UserInput(cv_text=cv, preferred_role="Machine Learning Engineer",
                                        location="Bengaluru", career_goal="Build production AI systems"))
    res = orch.find_jobs(prof)
    print("\nTop matches:")
    for m in res.matches[:5]:
        j = res.job(m.job_id)
        print(f"  {m.score:5.1f}  {j.title} @ {j.company} ({j.location})")
        print(f"         {m.explanation}")
