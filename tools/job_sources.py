"""Job sources used by the Job Search Agent.

- LocalDatasetSource: bundled JSON dataset (always available, offline).
- AdzunaSource: real job API (used when ADZUNA_APP_ID / ADZUNA_APP_KEY are set).

Add new sources by implementing `search(criteria) -> list[JobListing]`.
LinkedIn / Indeed do not offer open public job-search APIs, so they are not
included; Adzuna (or JSearch on RapidAPI) are the practical options.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Protocol

from core import config
from core.schemas import JobListing, SearchCriteria

log = logging.getLogger(__name__)


class JobSource(Protocol):
    name: str

    def search(self, criteria: SearchCriteria) -> List[JobListing]: ...


_STOP = {"and", "or", "the", "of", "in", "for", "with", "to", "a", "an", "on", "at", "is", "are",
         "we", "you", "our", "be", "as", "by", "will", "this", "that", "your", "from"}


def _tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9+#]+(?:\.[a-z0-9]+)*", text.lower())
    return {t for t in toks if len(t) > 1 and t not in _STOP}


def available_datasets() -> List[Path]:
    """Job dataset files in data/ (the bundled sample plus imported ones)."""
    return sorted(p for p in config.DATA_DIR.glob("jobs*.json*") if p.is_file())


class LocalDatasetSource:
    """Keyword search over a local dataset (.json list or .jsonl).

    Builds an inverted index on first use and ranks with IDF weighting, so
    title words like "engineer" that appear everywhere count less than
    distinctive ones like "nlp". Title matches weigh 3x description matches.
    """
    name = "dataset"

    def __init__(self, path: Path | str = config.JOBS_DATASET):
        self.path = Path(path)
        self.name = f"dataset:{self.path.name}"
        self._jobs: List[JobListing] | None = None
        self._title_idx: Dict[str, List[int]] = {}
        self._body_idx: Dict[str, List[int]] = {}

    @property
    def jobs(self) -> List[JobListing]:
        if self._jobs is None:
            text = self.path.read_text(encoding="utf-8")
            if self.path.suffix == ".jsonl":
                raw = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                raw = json.loads(text)
            self._jobs = [JobListing(**{**j, "source": j.get("source") or "dataset"}) for j in raw]
            self._build_index()
        return self._jobs

    def _build_index(self) -> None:
        title, body = defaultdict(list), defaultdict(list)
        for i, j in enumerate(self._jobs):
            for t in _tokens(j.title):
                title[t].append(i)
            for t in _tokens(j.description):
                body[t].append(i)
        self._title_idx, self._body_idx = dict(title), dict(body)

    def search(self, criteria: SearchCriteria) -> List[JobListing]:
        jobs = self.jobs
        n = len(jobs)
        kw = set()
        for k in criteria.keywords:
            kw |= _tokens(k)
        scores: Dict[int, float] = defaultdict(float)
        for t in kw:
            t_hits, b_hits = self._title_idx.get(t, []), self._body_idx.get(t, [])
            idf = math.log((n + 1) / (len(b_hits) + len(t_hits) + 1)) + 1
            for i in t_hits:
                scores[i] += 3 * idf
            for i in b_hits:
                scores[i] += idf
        if not kw:
            scores = {i: 0.0 for i in range(n)}

        locs = [l.lower() for l in criteria.locations if l.strip()]
        excluded = set(criteria.exclude_job_ids)
        out = []
        for i, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            job = jobs[i]
            if job.id in excluded:
                continue
            # location filter (relaxed by the feedback loop at broaden_level >= 2)
            if locs and criteria.broaden_level < 2:
                if not any(l in job.location.lower() for l in locs) and not (criteria.remote_ok and job.remote):
                    continue
            out.append(job)
            if len(out) >= criteria.limit:
                break
        return out


class AdzunaSource:
    name = "adzuna"
    BASE = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"

    def __init__(self, app_id: str = config.ADZUNA_APP_ID, app_key: str = config.ADZUNA_APP_KEY,
                 country: str = config.ADZUNA_COUNTRY):
        self.app_id, self.app_key, self.country = app_id, app_key, country

    @property
    def available(self) -> bool:
        return bool(self.app_id and self.app_key)

    def search(self, criteria: SearchCriteria) -> List[JobListing]:
        if not self.available:
            return []
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": criteria.limit,
            "what_or": " ".join(criteria.keywords[:6]),
            "content-type": "application/json",
        }
        if criteria.locations and criteria.broaden_level < 2:
            params["where"] = criteria.locations[0]
        url = self.BASE.format(country=self.country) + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            log.warning("Adzuna search failed: %s", e)
            return []
        out = []
        for item in data.get("results", []):
            jid = f"adzuna-{item.get('id')}"
            if jid in criteria.exclude_job_ids:
                continue
            smin, smax = item.get("salary_min"), item.get("salary_max")
            out.append(JobListing(
                id=jid,
                title=item.get("title", ""),
                company=(item.get("company") or {}).get("display_name", ""),
                location=(item.get("location") or {}).get("display_name", ""),
                remote="remote" in (item.get("title", "") + item.get("description", "")).lower(),
                description=item.get("description", ""),
                salary=f"{smin:,.0f} - {smax:,.0f}" if smin and smax else "",
                url=item.get("redirect_url", ""),
                source="adzuna",
            ))
        return out


_SOURCE_CACHE: Dict[str, LocalDatasetSource] = {}


def dataset_source(path: Path | str = config.JOBS_DATASET) -> LocalDatasetSource:
    """Shared (cached) source per file, so big datasets load and index once."""
    key = str(Path(path).resolve())
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = LocalDatasetSource(path)
    return _SOURCE_CACHE[key]


def default_sources(dataset: Path | str | None = None) -> List[JobSource]:
    sources: List[JobSource] = [dataset_source(dataset or config.JOBS_DATASET)]
    adz = AdzunaSource()
    if adz.available:
        sources.append(adz)
    return sources
