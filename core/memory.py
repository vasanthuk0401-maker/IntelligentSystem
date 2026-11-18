"""Memory / Knowledge Base (SQLite).

Stores user profile & preferences, previous searches & feedback, saved jobs &
interactions, and cached job analyses (long-term memory).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from core import config
from core.schemas import CareerProfile, JobRequirements

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, criteria TEXT,
    decision TEXT, top_jobs TEXT, created REAL);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, job_id TEXT,
    rating INTEGER, note TEXT, created REAL);
CREATE TABLE IF NOT EXISTS saved_jobs (
    user_id TEXT, job_id TEXT, title TEXT, company TEXT, score REAL,
    status TEXT, created REAL, PRIMARY KEY (user_id, job_id));
CREATE TABLE IF NOT EXISTS job_analysis (
    job_id TEXT PRIMARY KEY, data TEXT NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, job_id TEXT,
    kind TEXT, data TEXT, created REAL);
"""


class Memory:
    def __init__(self, path: Path | str = config.DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- profile ----
    def save_profile(self, user_id: str, profile: CareerProfile) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO profiles VALUES (?,?,?)",
            (user_id, profile.model_dump_json(), time.time()),
        )
        self.conn.commit()

    def load_profile(self, user_id: str) -> Optional[CareerProfile]:
        row = self.conn.execute(
            "SELECT data FROM profiles WHERE user_id=?", (user_id,)
        ).fetchone()
        return CareerProfile.model_validate_json(row["data"]) if row else None

    # ---- searches ----
    def log_search(self, user_id: str, criteria: dict, decision: dict, top_jobs: list) -> None:
        self.conn.execute(
            "INSERT INTO searches (user_id, criteria, decision, top_jobs, created) VALUES (?,?,?,?,?)",
            (user_id, json.dumps(criteria), json.dumps(decision), json.dumps(top_jobs), time.time()),
        )
        self.conn.commit()

    def recent_searches(self, user_id: str, limit: int = 10) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM searches WHERE user_id=? ORDER BY created DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {**dict(r), "criteria": json.loads(r["criteria"]), "decision": json.loads(r["decision"]),
             "top_jobs": json.loads(r["top_jobs"])}
            for r in rows
        ]

    # ---- feedback ----
    def add_feedback(self, user_id: str, job_id: str, rating: int, note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO feedback (user_id, job_id, rating, note, created) VALUES (?,?,?,?,?)",
            (user_id, job_id, rating, note, time.time()),
        )
        self.conn.commit()

    def disliked_job_ids(self, user_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT job_id FROM feedback WHERE user_id=? AND rating < 0", (user_id,)
        ).fetchall()
        return [r["job_id"] for r in rows]

    def all_feedback(self, user_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM feedback WHERE user_id=? ORDER BY created DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- saved jobs ----
    def save_job(self, user_id: str, job_id: str, title: str, company: str,
                 score: float, status: str = "saved") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO saved_jobs VALUES (?,?,?,?,?,?,?)",
            (user_id, job_id, title, company, score, status, time.time()),
        )
        self.conn.commit()

    def saved_jobs(self, user_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM saved_jobs WHERE user_id=? ORDER BY created DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- job analysis cache ----
    def get_analysis(self, job_id: str) -> Optional[JobRequirements]:
        row = self.conn.execute(
            "SELECT data FROM job_analysis WHERE job_id=?", (job_id,)
        ).fetchone()
        return JobRequirements.model_validate_json(row["data"]) if row else None

    def put_analysis(self, req: JobRequirements) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO job_analysis VALUES (?,?,?)",
            (req.job_id, req.model_dump_json(), time.time()),
        )
        self.conn.commit()

    # ---- generated artifacts (tailored CVs, interview packs) ----
    def save_artifact(self, user_id: str, job_id: str, kind: str, data: str) -> None:
        self.conn.execute(
            "INSERT INTO artifacts (user_id, job_id, kind, data, created) VALUES (?,?,?,?,?)",
            (user_id, job_id, kind, data, time.time()),
        )
        self.conn.commit()
