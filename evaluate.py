"""Evaluate matching quality on labelled CVs (data/eval_cases.json). Offline,
no API key needed.

1) Curated mode (default) - the 24-job sample dataset, hand-labelled relevant
   job ids per CV. Scores every job and reports ranking metrics of the FINAL
   match score for each semantic method:
     AUC  - probability a relevant job outranks an irrelevant one
     P@k  - share of the top-k (k = #relevant) that are relevant
     MRR  - 1 / rank of the first relevant job

       python evaluate.py

2) Real-data mode - a large imported dataset, where relevance is approximated
   by each case's `title_pattern` (regex on the job title; weak labels).
   Runs the real search + matching pipeline and reports precision@10 of:
     search (keywords only) | search + semantic re-rank | final match ranking

       python evaluate.py --dataset data/jobs_linkedin.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

from agents.job_analysis_agent import JobAnalysisAgent  # noqa: E402
from agents.job_search_agent import JobSearchAgent  # noqa: E402
from agents.profile_agent import ProfileAgent  # noqa: E402
from agents.skill_matching_agent import SkillMatchingAgent  # noqa: E402
from core import config  # noqa: E402
from core.llm import LLM  # noqa: E402
from core.schemas import CareerProfile, UserInput  # noqa: E402
from tools import embeddings  # noqa: E402
from tools.job_sources import LocalDatasetSource  # noqa: E402

LLM_OFF = LLM(enabled=False)


def load_cases() -> List[dict]:
    return json.loads((config.DATA_DIR / "eval_cases.json").read_text(encoding="utf-8"))


def build_profile(case: dict) -> CareerProfile:
    cv = case.get("cv_text") or (config.DATA_DIR / case["cv_file"]).read_text(encoding="utf-8")
    return ProfileAgent(llm=LLM_OFF).run(UserInput(
        cv_text=cv, preferred_role=case["preferred_role"], career_goal=case["career_goal"]))


def metrics(scores: List[float], relevant: List[bool]) -> Dict[str, float]:
    s, y = np.asarray(scores), np.asarray(relevant)
    pos, neg = s[y], s[~y]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    order = np.argsort(-s)
    k = int(y.sum())
    return {"AUC": auc, "P@k": float(y[order[:k]].mean()), "MRR": 1.0 / (int(np.argmax(y[order])) + 1)}


def curated(method: str) -> Dict[str, Dict[str, float]]:
    jobs = LocalDatasetSource(config.DATA_DIR / "jobs.json").jobs
    reqs = [JobAnalysisAgent._heuristic(j) for j in jobs]
    matcher = SkillMatchingAgent(llm=LLM_OFF, mode=method)
    out = {}
    for case in load_cases():
        results = {m.job_id: m.score for m in matcher.run(build_profile(case), jobs, reqs)}
        out[case["name"]] = metrics([results[j.id] for j in jobs], [j.id in case["relevant"] for j in jobs])
    return out


def print_table(title: str, res: Dict[str, Dict[str, float]]) -> None:
    print(f"\n== {title} ==")
    keys = list(next(iter(res.values())).keys())
    for name, m in res.items():
        print(f"  {name:20s} " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()))
    print(f"  {'MEAN':20s} " + "  ".join(f"{k} {np.mean([m[k] for m in res.values()]):.3f}" for k in keys))


def real_data(dataset: Path, k: int = 10) -> Dict[str, Dict[str, float]]:
    src = LocalDatasetSource(dataset)
    print(f"Loaded {len(src.jobs):,} jobs from {dataset}")
    analysis = JobAnalysisAgent(llm=LLM_OFF)
    matcher = SkillMatchingAgent(llm=LLM_OFF)
    out = {}
    for case in load_cases():
        pat = re.compile(case["title_pattern"], re.I)
        base_rate = np.mean([bool(pat.search(j.title)) for j in src.jobs])
        profile = build_profile(case)
        row = {"base": float(base_rate)}
        for label, rerank in (("kw", False), ("rerank", True)):
            agent = JobSearchAgent(sources=[src], rerank=rerank, llm=LLM_OFF)
            crit = agent.build_criteria(profile, limit=k * 3 if label == "rerank" else k)
            found = agent.run(crit)
            top = found[:k]
            row[f"P@{k} {label}"] = float(np.mean([bool(pat.search(j.title)) for j in top])) if top else 0.0
            if label == "rerank":
                reqs = analysis.run(found)
                ranked = matcher.run(profile, found, reqs)[:k]
                by_id = {j.id: j for j in found}
                row[f"P@{k} final"] = float(np.mean([bool(pat.search(by_id[m.job_id].title)) for m in ranked]))
        out[case["name"]] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, help="large dataset for real-data mode (.jsonl/.json)")
    a = ap.parse_args()
    if a.dataset:
        print_table(f"real data: {a.dataset.name} (weak title labels; base = share of all jobs matching)",
                    real_data(a.dataset))
        return
    for method in ["tfidf"] + (["embeddings", "hybrid"] if embeddings.get_embedder() else []):
        print_table(f"curated 24-job set, semantic={method}", curated(method))


if __name__ == "__main__":
    main()
