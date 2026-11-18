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
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Protocol

from core import config
from core.schemas import JobListing, SearchCriteria

log = logging.getLogger(__name__)


class JobSource(Protocol):
    name: str

    def search(self, criteria: SearchCriteria) -> List[JobListing]: ...


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if len(t) > 1}


class LocalDatasetSource:
    name = "dataset"

    def __init__(self, path: Path | str = config.JOBS_DATASET):
        self.path = Path(path)
        self._jobs: List[JobListing] | None = None

    @property
    def jobs(self) -> List[JobListing]:
        if self._jobs is None:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._jobs = [JobListing(**{**j, "source": "dataset"}) for j in raw]
        return self._jobs

    def search(self, criteria: SearchCriteria) -> List[JobListing]:
        kw = set()
        for k in criteria.keywords:
            kw |= _tokens(k)
        locs = [l.lower() for l in criteria.locations if l.strip()]
        scored = []
        for job in self.jobs:
            if job.id in criteria.exclude_job_ids:
                continue
            # location filter (relaxed by the feedback loop at broaden_level >= 2)
            if locs and criteria.broaden_level < 2:
                loc_ok = any(l in job.location.lower() for l in locs)
                if not loc_ok and not (criteria.remote_ok and job.remote):
                    continue
            title_t = _tokens(job.title)
            body_t = _tokens(job.description)
            score = 3 * len(kw & title_t) + len(kw & body_t)
            if score == 0 and kw:
                continue
            scored.append((score, job))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [j for _, j in scored[: criteria.limit]]


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


def default_sources() -> List[JobSource]:
    sources: List[JobSource] = [LocalDatasetSource()]
    adz = AdzunaSource()
    if adz.available:
        sources.append(adz)
    return sources
