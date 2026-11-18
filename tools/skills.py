"""Skill vocabulary, normalisation and keyword extraction (used by heuristics
and to canonicalise skills returned by Claude)."""
from __future__ import annotations

import re
from typing import Iterable, List

# canonical name -> aliases (lower-case)
SKILL_ALIASES = {
    "Python": ["python", "python3"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript", "ts"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", ".net", "dotnet"],
    "Go": ["golang", "go lang"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql"],
    "NoSQL": ["nosql", "mongodb", "mongo", "cassandra", "dynamodb"],
    "R": ["r programming", "r language"],
    "React": ["react", "reactjs", "react.js"],
    "Angular": ["angular"],
    "Vue": ["vue", "vuejs", "vue.js"],
    "Node.js": ["node", "nodejs", "node.js", "express"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring Boot": ["spring", "spring boot"],
    "HTML/CSS": ["html", "css", "html5", "css3", "tailwind"],
    "REST APIs": ["rest", "restful", "rest api", "rest apis", "api design"],
    "GraphQL": ["graphql"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "neural networks", "dl"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "opencv", "image processing"],
    "LLMs": ["llm", "llms", "large language models", "generative ai", "genai", "prompt engineering", "rag"],
    "TensorFlow": ["tensorflow", "keras"],
    "PyTorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "matplotlib", "seaborn", "plotly", "dashboards", "dashboard"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "spreadsheets", "pivot tables"],
    "Statistics": ["statistics", "statistical analysis", "hypothesis testing", "a/b testing"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Hadoop": ["hadoop"],
    "Airflow": ["airflow"],
    "ETL": ["etl", "data pipelines", "data pipeline"],
    "Kafka": ["kafka"],
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker", "containers", "containerised", "containerized", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "cicd", "jenkins", "github actions", "gitlab ci", "deployment pipelines", "deployment pipeline"],
    "Terraform": ["terraform", "infrastructure as code"],
    "Linux": ["linux", "unix", "bash", "shell scripting", "shell scripts"],
    "Git": ["git", "github", "gitlab", "version control"],
    "Microservices": ["microservices", "microservice"],
    "System Design": ["system design", "distributed systems", "scalability"],
    "Testing": ["unit testing", "pytest", "junit", "selenium", "test automation", "testing"],
    "Agile": ["agile", "scrum", "kanban", "jira"],
    "Cybersecurity": ["cybersecurity", "security", "penetration testing", "siem", "network security"],
    "Networking": ["networking", "tcp/ip", "dns", "firewalls"],
    "Android": ["android", "kotlin"],
    "iOS": ["ios", "swift"],
    "Flutter": ["flutter", "dart"],
    "Figma": ["figma"],
    "UI/UX Design": ["ui/ux", "ux design", "ui design", "user research", "wireframing", "prototyping"],
    "Product Management": ["product management", "roadmap", "product strategy"],
    "Project Management": ["project management", "pmp", "stakeholder management"],
    "Digital Marketing": ["digital marketing", "seo", "sem", "google ads", "social media marketing"],
    "Communication": ["communication", "presentation skills", "written communication"],
    "Leadership": ["leadership", "team lead", "mentoring", "people management"],
    "Problem Solving": ["problem solving", "analytical thinking", "critical thinking"],
}

_ALIAS_TO_CANON = {}
for canon, aliases in SKILL_ALIASES.items():
    _ALIAS_TO_CANON[canon.lower()] = canon
    for a in aliases:
        _ALIAS_TO_CANON[a] = canon

# Short / ambiguous aliases that need exact word boundaries and context
_AMBIGUOUS = {"go", "r", "ts", "js", "ml", "dl", "rest", "security", "testing", "spring", "node", "communication"}


def normalize_skill(skill: str) -> str:
    s = skill.strip()
    return _ALIAS_TO_CANON.get(s.lower(), s.strip().title() if s.islower() else s)


def normalize_skills(skills: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in skills:
        if not s or not s.strip():
            continue
        n = normalize_skill(s)
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def extract_skills(text: str) -> List[str]:
    """Find known skills mentioned in free text."""
    low = " " + text.lower() + " "
    found = []
    for alias, canon in _ALIAS_TO_CANON.items():
        if alias in _AMBIGUOUS and len(alias) <= 2:
            # require explicit boundaries and a comma/slash/space context
            pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9+#])"
            if alias in {"go", "r"}:
                continue  # too ambiguous in prose; matched via longer aliases
        else:
            pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, low):
            found.append(canon)
    return normalize_skills(found)


# Interchangeable / closely related skills. Having one earns partial credit for
# another in the same group. Curated rather than embedding-based: on short
# skill names the embedding model was unreliable (e.g. it rated Java~Python
# higher than PyTorch~Deep Learning).
RELATED_GROUPS = [
    {"PyTorch", "TensorFlow", "Deep Learning"},
    {"Machine Learning", "Deep Learning", "scikit-learn"},
    {"NLP", "LLMs"},
    {"Power BI", "Tableau", "Data Visualization"},
    {"Data Analysis", "Statistics", "Pandas"},
    {"AWS", "Azure", "GCP"},
    {"React", "Angular", "Vue"},
    {"JavaScript", "TypeScript"},
    {"Django", "Flask", "FastAPI"},
    {"Docker", "Kubernetes"},
    {"Spark", "Hadoop"},
    {"Airflow", "ETL"},
    {"SQL", "NoSQL"},
    {"Android", "iOS", "Flutter"},
    {"Java", "C#"},
    {"CI/CD", "Terraform"},
    {"Figma", "UI/UX Design"},
]
RELATED_CREDIT = 0.5

_RELATED = {}
for _g in RELATED_GROUPS:
    for _s in _g:
        _RELATED.setdefault(_s.lower(), set()).update(x for x in _g if x != _s)


def related_to(skill: str, user_skills: Iterable[str]) -> str:
    """Return a user skill related to `skill`, or ''."""
    rel = {r.lower() for r in _RELATED.get(normalize_skill(skill).lower(), set())}
    return next((u for u in user_skills if u.lower() in rel), "")


def skills_overlap(user_skills: Iterable[str], job_skills: Iterable[str]):
    """Split job skills into (matched, related, missing).

    related entries are formatted "JobSkill ~ YourSkill"."""
    user = normalize_skills(user_skills)
    u = {s.lower() for s in user}
    matched, related, missing = [], [], []
    for s in normalize_skills(job_skills):
        if s.lower() in u:
            matched.append(s)
        elif (r := related_to(s, user)):
            related.append(f"{s} ~ {r}")
        else:
            missing.append(s)
    return matched, related, missing
