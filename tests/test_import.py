"""Importer + large-dataset search, on a synthetic CSV with the LinkedIn columns."""
import json

import pandas as pd

import import_jobs
from core.schemas import SearchCriteria
from tools.job_sources import LocalDatasetSource

LONG = " We value collaboration, ownership and continuous learning across teams." * 4


def make_csv(path):
    rows = [
        dict(job_id=3904095574.0, company_name="Acme AI", title="Machine Learning Engineer",
             description="<p>Build NLP models with <b>Python</b> &amp; PyTorch.</p>" + LONG,
             max_salary=150000, med_salary=None, min_salary=120000, pay_period="YEARLY", currency="USD",
             location="San Francisco, CA", remote_allowed=1.0, formatted_experience_level="Mid-Senior level",
             formatted_work_type="Full-time", skills_desc="Python, PyTorch, NLP",
             job_posting_url="https://www.linkedin.com/jobs/view/3904095574/"),
        dict(job_id=2, company_name="Acme AI", title="Machine Learning Engineer",  # duplicate title+company
             description="dup" + LONG, location="Remote"),
        dict(job_id=3, company_name="ShortCo", title="Barista", description="Make coffee."),  # too short
        dict(job_id=4, company_name="Shop", title="Retail Associate", description="Help customers." + LONG,
             med_salary=18.5, pay_period="HOURLY", location="Austin, TX"),
        dict(job_id=5, company_name="DataCo", title="Data Analyst", description="SQL and Tableau dashboards." + LONG,
             location="New York, NY"),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_import_linkedin_schema(tmp_path):
    csv, out = tmp_path / "postings.csv", tmp_path / "jobs_test.jsonl"
    make_csv(csv)
    n = import_jobs.convert(csv, out, limit=0, min_desc=200, max_desc=5000, seed=1)
    jobs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert n == len(jobs) == 3  # duplicate and short one dropped
    ml = next(j for j in jobs if j["title"] == "Machine Learning Engineer")
    assert ml["id"] == "li-3904095574"
    assert "<p>" not in ml["description"] and "Python & PyTorch" in ml["description"]
    assert "Skills: Python, PyTorch, NLP" in ml["description"]
    assert ml["salary"] == "USD 120,000 - 150,000 / yearly"
    assert ml["remote"] is True
    retail = next(j for j in jobs if j["title"] == "Retail Associate")
    assert retail["salary"] == "USD 18 / hourly" or retail["salary"].startswith("USD 18")
    assert retail["remote"] is False and retail["url"] == ""


def test_jsonl_source_idf_search(tmp_path):
    csv, out = tmp_path / "postings.csv", tmp_path / "jobs_test.jsonl"
    make_csv(csv)
    import_jobs.convert(csv, out, limit=0, min_desc=200, max_desc=5000, seed=1)
    src = LocalDatasetSource(out)
    res = src.search(SearchCriteria(keywords=["NLP Engineer", "PyTorch"], limit=2))
    assert res[0].title == "Machine Learning Engineer"
    # location filter: New York only (no remote) -> Data Analyst; remote ML job also allowed
    res = src.search(SearchCriteria(keywords=["data", "machine"], locations=["New York"], remote_ok=False))
    assert [j.title for j in res] == ["Data Analyst"]
