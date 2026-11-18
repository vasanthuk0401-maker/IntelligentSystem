"""Evaluate matching quality on labelled profile -> relevant-job cases.

Scores every job in the dataset for each case and reports, per similarity
method, ranking metrics of the FINAL match score:
  AUC   - probability a relevant job outranks an irrelevant one
  P@k   - share of the top-k (k = #relevant) that are relevant
  MRR   - 1 / rank of the first relevant job

Usage:  python evaluate.py            (offline; no API key needed)
Add cases in data/eval_cases.json.
"""
from __future__ import annotations

import json
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

from agents.job_analysis_agent import JobAnalysisAgent  # noqa: E402
from agents.profile_agent import ProfileAgent  # noqa: E402
from agents.skill_matching_agent import SkillMatchingAgent  # noqa: E402
from core import config  # noqa: E402
from core.llm import LLM  # noqa: E402
from core.schemas import UserInput  # noqa: E402
from tools import embeddings  # noqa: E402
from tools.job_sources import LocalDatasetSource  # noqa: E402


def metrics(scores: List[float], relevant: List[bool]) -> Dict[str, float]:
    s, y = np.asarray(scores), np.asarray(relevant)
    pos, neg = s[y], s[~y]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    order = np.argsort(-s)
    k = int(y.sum())
    p_at_k = float(y[order[:k]].mean())
    mrr = 1.0 / (int(np.argmax(y[order])) + 1)
    return {"AUC": auc, "P@k": p_at_k, "MRR": mrr}


def run(method: str) -> Dict[str, Dict[str, float]]:
    llm = LLM(enabled=False)
    jobs = LocalDatasetSource().jobs
    reqs = [JobAnalysisAgent._heuristic(j) for j in jobs]
    matcher = SkillMatchingAgent(llm=llm, use_embeddings=(method == "embeddings"))
    out = {}
    for case in json.loads((config.DATA_DIR / "eval_cases.json").read_text(encoding="utf-8")):
        cv = case.get("cv_text") or (config.DATA_DIR / case["cv_file"]).read_text(encoding="utf-8")
        profile = ProfileAgent(llm=llm).run(UserInput(
            cv_text=cv, preferred_role=case["preferred_role"], career_goal=case["career_goal"]))
        results = {m.job_id: m.score for m in matcher.run(profile, jobs, reqs)}
        out[case["name"]] = metrics([results[j.id] for j in jobs],
                                    [j.id in case["relevant"] for j in jobs])
    return out


if __name__ == "__main__":
    methods = ["tfidf"] + (["embeddings"] if embeddings.get_embedder() else [])
    for method in methods:
        res = run(method)
        print(f"\n== {method} ==")
        for name, m in res.items():
            print(f"  {name:20s} " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()))
        mean = {k: np.mean([m[k] for m in res.values()]) for k in ("AUC", "P@k", "MRR")}
        print(f"  {'MEAN':20s} " + "  ".join(f"{k} {v:.3f}" for k, v in mean.items()))
