"""Import a Kaggle job-postings CSV into the app's dataset format (JSONL).

Built for the LinkedIn Job Postings 2023-24 dataset
(kaggle.com/datasets/arshkon/linkedin-job-postings, file postings.csv), but
column names are matched by alias so most job CSVs work.

Usage:
  python import_jobs.py data/raw/postings.csv                  # 20k jobs -> data/jobs_linkedin.jsonl
  python import_jobs.py data/raw/postings.csv --limit 0        # all jobs
  python import_jobs.py data/raw/postings.csv --out data/jobs_us.jsonl --limit 50000

Then pick the dataset in the dashboard sidebar, or set JOBS_DATASET in .env.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

# our field -> candidate CSV column names (first match wins, case-insensitive)
ALIASES = {
    "id": ["job_id", "id", "uniq_id", "jobid"],
    "title": ["title", "job_title", "jobtitle", "position"],
    "company": ["company_name", "company", "employer"],
    "location": ["location", "job_location", "joblocation_address", "locationnormalized", "city"],
    "description": ["description", "job_description", "jobdescription", "fulldescription"],
    "skills": ["skills_desc", "skills", "key_skills"],
    "remote": ["remote_allowed", "remote", "is_remote"],
    "level": ["formatted_experience_level", "experience_level", "experience", "seniority"],
    "work_type": ["formatted_work_type", "work_type", "contracttime", "employment_type"],
    "url": ["job_posting_url", "url", "application_url", "pageurl"],
    "min_salary": ["min_salary", "salary_min"],
    "med_salary": ["med_salary", "salarynormalized", "salary"],
    "max_salary": ["max_salary", "salary_max"],
    "pay_period": ["pay_period"],
    "currency": ["currency"],
}
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\r\f\v]+")


def map_columns(columns) -> dict:
    lower = {c.lower(): c for c in columns}
    found = {}
    for field, cands in ALIASES.items():
        for c in cands:
            if c in lower:
                found[field] = lower[c]
                break
    missing = [f for f in ("title", "description") if f not in found]
    if missing:
        sys.exit(f"CSV is missing required columns for: {missing}. Columns: {list(columns)[:30]}")
    return found


def clean_text(s, max_len: int) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(TAG_RE.sub(" ", s))
    s = "\n".join(WS_RE.sub(" ", line).strip() for line in s.splitlines())
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s[:max_len]


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def fmt_salary(row, cols) -> str:
    lo, mid, hi = (num(row.get(cols.get(k))) for k in ("min_salary", "med_salary", "max_salary"))
    period = str(row.get(cols.get("pay_period"), "") or "").lower()
    period = "" if period == "nan" else period
    cur = str(row.get(cols.get("currency"), "") or "USD")
    cur = "USD" if cur == "nan" else cur
    if lo and hi:
        s = f"{cur} {lo:,.0f} - {hi:,.0f}"
    elif mid:
        s = f"{cur} {mid:,.0f}"
    else:
        return ""
    return f"{s} / {period}" if period else s


def is_remote(row, cols, title: str, location: str) -> bool:
    v = row.get(cols.get("remote"))
    if num(v):
        return True
    if isinstance(v, str) and v.strip().lower() in {"true", "yes", "remote"}:
        return True
    return "remote" in f"{title} {location}".lower()


def convert(csv_path: Path, out: Path, limit: int, min_desc: int, max_desc: int, seed: int) -> int:
    head = pd.read_csv(csv_path, nrows=5)
    cols = map_columns(head.columns)
    print("Column mapping:", {k: v for k, v in cols.items()})
    df = pd.read_csv(csv_path, usecols=sorted(set(cols.values())), dtype=str, keep_default_na=True)
    print(f"Loaded {len(df):,} rows")

    df = df[df[cols["title"]].notna() & df[cols["description"]].notna()]
    df = df[df[cols["description"]].str.len() >= min_desc]
    key = df[cols["title"]].str.lower().str.strip() + "|" + (
        df[cols["company"]].fillna("").str.lower().str.strip() if "company" in cols else "")
    df = df.loc[~key.duplicated()]
    print(f"{len(df):,} rows after dropping empty/short descriptions and duplicates")
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(df.to_dict("records")):
            title = clean_text(row[cols["title"]], 200)
            location = clean_text(row.get(cols.get("location"), ""), 120)
            desc = clean_text(row[cols["description"]], max_desc)
            skills = clean_text(row.get(cols.get("skills"), ""), 1000)
            level = clean_text(row.get(cols.get("level"), ""), 60)
            wtype = clean_text(row.get(cols.get("work_type"), ""), 60)
            extra = " | ".join(x for x in (
                f"Experience level: {level}" if level else "",
                f"Work type: {wtype}" if wtype else "",
                f"Skills: {skills}" if skills else "") if x)
            raw_id = str(row.get(cols.get("id"), "") or "").split(".")[0] or str(i)
            job = {
                "id": f"li-{raw_id}",
                "title": title,
                "company": clean_text(row.get(cols.get("company"), ""), 120),
                "location": location,
                "remote": is_remote(row, cols, title, location),
                "description": (desc + ("\n\n" + extra if extra else "")),
                "salary": fmt_salary(row, cols),
                "url": str(row.get(cols.get("url"), "") or "") if cols.get("url") else "",
            }
            if job["url"] == "nan":
                job["url"] = ""
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/jobs_linkedin.jsonl"))
    ap.add_argument("--limit", type=int, default=20000, help="random sample size (0 = all)")
    ap.add_argument("--min-desc", type=int, default=200, help="drop descriptions shorter than this")
    ap.add_argument("--max-desc", type=int, default=5000, help="truncate descriptions (controls LLM cost)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    if not a.csv.exists():
        sys.exit(f"Not found: {a.csv}")
    n = convert(a.csv, a.out, a.limit, a.min_desc, a.max_desc, a.seed)
    print(f"Wrote {n:,} jobs to {a.out}")


if __name__ == "__main__":
    main()
