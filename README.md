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
`python evaluate.py` scores the final ranking for 17 labelled CVs (`data/eval_cases.json`, 7 of them
paraphrased) against the 24-job sample set:

| Semantic method | Mean AUC | Mean P@k |
|---|---|---|
| TF-IDF | 0.947 | 0.799 |
| bge-small embeddings (default) | **0.957** | **0.819** |
| hybrid (50/50) | 0.950 | 0.819 |

Differences are small: on 24 jobs, skill overlap dominates the score. Embeddings help most on
paraphrased CVs (DevOps P@k 0.33 → 0.67) and lose slightly on a few keyword-heavy ones.
`python evaluate.py --dataset data/jobs_linkedin.jsonl` measures precision@10 of the real search
pipeline on a large dataset, using each case's `title_pattern` as an approximate relevance label.

## Real job data (LinkedIn 2023-24, Kaggle)
1. Download https://www.kaggle.com/datasets/arshkon/linkedin-job-postings and place `postings.csv`
   in `data/raw/`.
2. Import (random 20k sample; `--limit 0` for all ~124k):
   ```bash
   python import_jobs.py data/raw/postings.csv
   ```
3. Pick `jobs_linkedin.jsonl` in the dashboard's **Job dataset** dropdown (or set `JOBS_DATASET` in `.env`).

Search is two-stage so it scales: an IDF-weighted inverted index retrieves ~50 keyword candidates, then
local embeddings re-rank them against your profile, and only the top 10 go to the Job Analysis Agent
(which matters for Claude cost). Note this dataset is US-only; with a non-US location the Decision
Agent relaxes the location filter on the second search iteration.

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
