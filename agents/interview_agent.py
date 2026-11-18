"""7B. Interview Agent - generates role-specific questions, provides practice
answers, and gives feedback and tips on the user's answers."""
from __future__ import annotations

from agents.base import BaseAgent
from core.schemas import (AnswerFeedback, CareerProfile, InterviewPack, InterviewQuestion,
                          JobListing, JobRequirements, MatchResult)

GEN_SYSTEM = """You are the Interview Agent of a job-search assistant.
Create an interview preparation pack for one specific job and candidate.
- 8 questions: 4 technical (on the job's required skills), 2 behavioural (STAR format),
  1 on the candidate's skill gaps, 1 role/company-motivation question.
- category: one of technical, behavioural, gap, motivation.
- sample_answer: a strong model answer grounded in the candidate's real profile (do not invent experience;
  where the candidate lacks experience, show how to answer honestly).
- tips: 2-3 short, specific tips per question."""

FEEDBACK_SYSTEM = """You are an interview coach. Evaluate the candidate's answer to the interview question
for the given role. score: 1-10. strengths and improvements: 2-4 short bullet points each.
improved_answer: a better version of THEIR answer, keeping their facts and not inventing new experience."""


class InterviewAgent(BaseAgent):
    name = "Interview Agent"

    def run(self, profile: CareerProfile, job: JobListing, req: JobRequirements,
            match: MatchResult) -> InterviewPack:
        self.log(f"Generating interview questions for {job.title} at {job.company}...")
        prompt = (
            f"<job>\nTitle: {job.title}\nCompany: {job.company}\n{job.description}\n</job>\n\n"
            f"<requirements>\n{req.model_dump_json(indent=1)}\n</requirements>\n\n"
            f"<candidate>\n{profile.model_dump_json(indent=1)}\nmissing skills: {match.missing_skills}\n</candidate>"
        )
        pack = self.llm.parse(GEN_SYSTEM, prompt, InterviewPack, effort="medium", max_tokens=32000)
        if pack is None:
            pack = self._heuristic(profile, job, req, match)
            self.log("Interview pack generated with offline heuristics.")
        else:
            self.log("Interview pack generated with Claude.")
        pack.job_id = job.id
        return pack

    def feedback(self, job: JobListing, question: str, answer: str) -> AnswerFeedback:
        prompt = (f"Role: {job.title} at {job.company}\n\n<question>\n{question}\n</question>\n\n"
                  f"<answer>\n{answer}\n</answer>")
        fb = self.llm.parse(FEEDBACK_SYSTEM, prompt, AnswerFeedback, effort="low")
        return fb or self._heuristic_feedback(answer)

    @staticmethod
    def _heuristic(profile, job, req, match) -> InterviewPack:
        qs = []
        for s in (req.required_skills or match.matched_skills)[:4]:
            qs.append(InterviewQuestion(
                question=f"Walk me through a project where you used {s}. What problem did it solve and what was the outcome?",
                category="technical",
                sample_answer=f"Describe the context, your specific use of {s}, a technical decision you made, and a measurable result.",
                tips=["Be specific about YOUR contribution", "Mention one trade-off you considered", "End with a metric"]))
        qs += [
            InterviewQuestion(question="Tell me about a time you faced a difficult problem with a deadline.",
                              category="behavioural",
                              sample_answer="Situation, Task, Action, Result - keep the Action part the longest.",
                              tips=["Use the STAR structure", "Keep it under 2 minutes"]),
            InterviewQuestion(question="Describe a time you disagreed with a teammate. How did you resolve it?",
                              category="behavioural",
                              sample_answer="Show listening, data-driven reasoning and a positive outcome.",
                              tips=["Don't blame others", "Highlight what you learned"]),
        ]
        if match.missing_skills:
            gap = match.missing_skills[0]
            qs.append(InterviewQuestion(
                question=f"This role uses {gap}. What is your experience with it?",
                category="gap",
                sample_answer=f"Be honest, connect to related skills you have, and describe how you're learning {gap}.",
                tips=["Never bluff", "Show a concrete learning plan"]))
        qs.append(InterviewQuestion(
            question=f"Why do you want to work as a {job.title} at {job.company}?",
            category="motivation",
            sample_answer="Connect the company's work to your career goal and your strongest matching skills.",
            tips=["Research the company beforehand", "Link to your career goal"]))
        return InterviewPack(job_id=job.id, questions=qs)

    @staticmethod
    def _heuristic_feedback(answer: str) -> AnswerFeedback:
        words = len(answer.split())
        has_metric = any(ch.isdigit() for ch in answer)
        star = sum(w in answer.lower() for w in ("situation", "task", "action", "result", "i did", "outcome"))
        score = min(10, 3 + (words > 60) * 2 + (words > 120) + has_metric * 2 + min(star, 2))
        strengths, improvements = [], []
        (strengths if words > 60 else improvements).append(
            "Good level of detail." if words > 60 else "Answer is short - add context and specifics.")
        (strengths if has_metric else improvements).append(
            "Includes measurable results." if has_metric else "Add a measurable outcome (numbers, %).")
        if star < 2:
            improvements.append("Structure it with STAR: Situation, Task, Action, Result.")
        return AnswerFeedback(score=score, strengths=strengths, improvements=improvements,
                              improved_answer="")
