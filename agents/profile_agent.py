"""2. Profile Agent - extracts and structures user information, validates
completeness, and creates a structured career profile."""
from __future__ import annotations

import re

from agents.base import BaseAgent
from core.schemas import CareerProfile, ExtractedProfile, UserInput
from tools.skills import extract_skills, normalize_skills

SYSTEM = """You are the Profile Agent of a job-search assistant.
Extract a structured career profile from the candidate's CV text and form answers.
Rules:
- Only use facts present in the input; never invent employers, degrees or numbers.
- skills: concrete technical and professional skills (canonical names, e.g. "Python", "SQL", "React").
- experience_years: total professional experience (internships count as fractions); 0 for students.
- headline: one line such as "Final-year CS student focused on ML".
- summary: 2-3 sentences written in the third person.
- achievements: quantified accomplishments or notable projects, if present.
- Leave lists empty when unknown."""


class ProfileAgent(BaseAgent):
    name = "Profile Agent"

    def run(self, user: UserInput) -> CareerProfile:
        self.log("Extracting and structuring user information...")
        profile = None
        if user.cv_text.strip():
            prompt = (
                f"<cv>\n{user.cv_text}\n</cv>\n\n"
                f"<form>\nname: {user.name}\nskills: {', '.join(user.skills)}\n"
                f"experience_years: {user.experience_years}\npreferred_role: {user.preferred_role}\n"
                f"location: {user.location}\nremote_ok: {user.remote_ok}\ncareer_goal: {user.career_goal}\n</form>"
            )
            extracted = self.llm.parse(SYSTEM, prompt, ExtractedProfile, effort="low")
            if extracted:
                profile = CareerProfile(**extracted.model_dump())
                self.log("Profile extracted with Claude.")
        if profile is None:
            profile = self._heuristic(user)
            self.log("Profile extracted with offline heuristics.")

        # Explicit form input always wins over extraction.
        if user.name:
            profile.name = user.name
        profile.skills = normalize_skills(list(user.skills) + list(profile.skills))
        if user.experience_years is not None:
            profile.experience_years = float(user.experience_years)
        if user.preferred_role:
            roles = [r.strip() for r in re.split(r"[,/;]", user.preferred_role) if r.strip()]
            profile.preferred_roles = roles + [r for r in profile.preferred_roles if r not in roles]
        if user.location:
            profile.preferred_locations = [l.strip() for l in user.location.split(",") if l.strip()]
        profile.remote_ok = user.remote_ok
        profile.source_text = user.cv_text.strip()[:4000]
        if user.career_goal:
            profile.career_goal = user.career_goal

        profile.missing_fields = self.validate(profile)
        if profile.missing_fields:
            self.log("Profile incomplete - missing: " + ", ".join(profile.missing_fields))
        else:
            self.log("Profile complete.")
        return profile

    @staticmethod
    def validate(p: CareerProfile) -> list:
        missing = []
        if len(p.skills) < 3:
            missing.append("skills (at least 3)")
        if not p.preferred_roles:
            missing.append("preferred role")
        if not p.preferred_locations and not p.remote_ok:
            missing.append("location")
        if not p.career_goal:
            missing.append("career goal")
        return missing

    @staticmethod
    def _heuristic(user: UserInput) -> CareerProfile:
        text = user.cv_text or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        name = user.name or (lines[0] if lines and len(lines[0].split()) <= 4 else "")
        years = 0.0
        m = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text, re.I)
        m = [float(x) for x in m if float(x) < 40]
        if m:
            years = max(m)
        else:
            months = re.findall(r"(\d+)\s*months?", text, re.I)
            if months:
                years = round(sum(int(x) for x in months) / 12, 1)
        edu_kw = re.compile(r"\b(b\.?tech|b\.?e\.?|bachelor|master|m\.?tech|m\.?sc|b\.?sc|mba|ph\.?d|degree|university|college)\b", re.I)
        education = [l for l in lines if edu_kw.search(l)][:4]
        role_kw = re.compile(r"\b(engineer|developer|analyst|scientist|intern|manager|designer|consultant)\b", re.I)
        past_roles = [l for l in lines if role_kw.search(l) and len(l) < 80][:5]
        achievements = [l.lstrip("-•* ") for l in lines if re.search(r"\d+%|\bimproved\b|\breduced\b|\bbuilt\b|\bled\b", l, re.I)][:5]
        skills = extract_skills(text)
        role = user.preferred_role or (past_roles[0] if past_roles else "")
        return CareerProfile(
            name=name,
            headline=role,
            summary=" ".join(lines[1:3])[:300] if len(lines) > 1 else "",
            skills=skills,
            experience_years=years,
            past_roles=past_roles,
            education=education,
            achievements=achievements,
            preferred_roles=[role] if role else [],
            career_goal=user.career_goal,
        )
