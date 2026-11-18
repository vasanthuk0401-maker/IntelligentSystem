# Intelligent Job Search – Multi-Agent System

A multi-agent career assistant that turns a CV into a structured profile, finds and analyses jobs,
scores how well you match, decides the next step (search more, recommend, or prepare), and then
tailors your CV and prepares you for interviews. Everything is saved to memory for future runs.

## Architecture (maps 1:1 to the workflow diagram)

| # | Diagram box | Code |
|---|---|---|
| 1 | User Input | `core/schemas.py → UserInput`, Profile tab in `app.py` |
| 2 | Profile Agent | `agents/profile_agent.py` |
| 3 | Job Search Agent | `agents/job_search_agent.py` + `tools/job_sources.py` |
| 4 | Job Analysis Agent | `agents/job_analysis_agent.py` |
| 5 | Skill Matching Agent | `agents/skill_matching_agent.py` |
| 6 | Decision Agent | `agents/decision_agent.py` |
| – | Update Search Criteria (feedback loop) | `DecisionAgent.update_criteria` + loop in `orchestrator.py` |
| 7A | CV Tailoring Agent | `agents/cv_tailoring_agent.py` |
| 7B | Interview Agent | `agents/interview_agent.py` |
| 8 | User Dashboard | `app.py` (Streamlit) |
| – | Memory / Knowledge Base | `core/memory.py` (SQLite: profiles, searches, feedback, saved jobs, cached analyses) |
| – | Tools / External Sources | `tools/cv_parser.py`, `tools/job_sources.py`, `tools/skills.py` |

`orchestrator.py` runs the flow. The LLM layer (`core/llm.py`) calls Claude (`claude-opus-5-5`) with
**structured outputs** (Pydantic models), so every agent returns validated data. If no API key is
set or a call fails, each agent falls back to an offline heuristic, so the system always runs.

### How matching works
`score = 50% skill coverage + 30% semantic similarity + 20% experience fit`, ±location adjustment.

- **Skill coverage:** exact matches get full credit; *related* skills (e.g. you know TensorFlow, the job
  asks for PyTorch) get 50% credit via a curated map in `tools/skills.py`; required skills weigh 2× preferred.
- **Semantic similarity:** local embeddings (`BAAI/bge-small-en-v1.5` via `fastembed`, ~130 MB downloaded once
  to `data/models/`, no API key) comparing your profile + CV with the job posting. Falls back to TF-IDF
  automatically if unavailable; set `EMBEDDING_MODEL=tfidf` to force it.
- Each match lists matched / related / missing skills and a plain-English explanation.

### Matching quality
`python evaluate.py` scores the final ranking on labelled cases in `data/eval_cases.json`:

| Semantic method | Mean AUC | Mean P@k |
|---|---|---|
| TF-IDF | 0.903 | 0.771 |
| bge-small embeddings | **0.957** | **0.854** |

The biggest gain is on CVs that describe skills in their own words rather than using keywords.
Only 4 labelled profiles so far, so add more cases before reading much into small differences.

### Decision policy
- Fewer than 2 jobs ≥ threshold (default 60) → **search_more**: broaden keywords with related roles,
  raise the limit, drop poor matches, and (level 2) relax the location filter. Repeats up to `MAX_SEARCH_ITERATIONS`.
- Good matches with skill gaps, or interview prep requested → **prepare**.
- Otherwise → **recommend**.

## Setup

```bash
pip install -r requirements.txt
```

```bash
copy .env.example .env
```

Put your Claude API key in `.env` (`ANTHROPIC_API_KEY=...`). Optional: Adzuna keys for live job listings.

## Run

```bash
streamlit run app.py
```

CLI demo (prints the agent trace and top matches):

```bash
python orchestrator.py data/sample_cv.txt
```

Tests (offline, no API key needed):

```bash
python -m pytest -q
```

## Extending
- **More job sources:** add a class with `name` and `search(criteria) -> list[JobListing]` in
  `tools/job_sources.py` and register it in `default_sources()`.
- **Bigger dataset:** point `JOBS_DATASET` at any JSON list with `id, title, company, location, remote, description, salary, url`.
- **Different embedding model:** set `EMBEDDING_MODEL` to any fastembed-supported model, then re-run
  `python evaluate.py` and retune `EMB_LO`/`EMB_HI` in `agents/skill_matching_agent.py`.
