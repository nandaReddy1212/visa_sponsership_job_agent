import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from jobs.common import CONFIG, BOARDS_PATH, get_json

FETCH = CONFIG["fetch"]
RAW_DIR = Path("data/raw/jobs")

URLS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true",
    "lever": "https://api.lever.co/v0/postings/{slug}?mode=json",
}

def _jobs_in(ats: str, data) -> list:
    """Greenhouse wraps jobs in {'jobs': [...]}; Lever returns the list itself."""
    return data.get("jobs",[]) if ats == 'greenhouse' else data 

def fetch_board(ats: str, slug: str, run_dir: Path) -> dict:
    log = {"ats": ats, "slug": slug, "n_jobs": None, "path": None, "error": None,
           "fetched_at": pd.Timestamp.now(tz="UTC")}
    try:
        data = get_json(URLS[ats].format(slug=slug), timeout=FETCH["timeout_s"])
        if data is None:
            return {**log, "error": "not_found"}        # board removed since discovery
        path = run_dir / ats / f"{slug}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="UTF-8")
        return {**log, "n_jobs": len(_jobs_in(ats, data)), "path": str(path)}
    except requests.RequestException as e:
        return {**log, "error": type(e).__name__}

def fetch_all(boards: pd.DataFrame, run_date: str | None = None) -> pd.DataFrame:
    run_dir = RAW_DIR / (run_date or date.today().isoformat())
    unique = boards[["ats", "slug"]].drop_duplicates()
    print(f"fetching {len(unique):,} boards -> {run_dir}")

    logs = []
    with ThreadPoolExecutor(max_workers=FETCH['workers']) as pool:
        futures = [pool.submit(fetch_board, r.ats,r.slug, run_dir)
                    for r in unique.itertuples(index=False)]
        for i, f in enumerate(as_completed(futures), 1):
            logs.append(f.result())
            if i % 25 == 0:
                print(f"  fetched {i:,}/{len(futures):,}")
    log = pd.DataFrame(logs)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.to_parquet(run_dir / "_fetch_log.parquet", index=False)
    return log

if __name__ == "__main__":
    boards = pd.read_parquet(BOARDS_PATH)
    log = fetch_all(boards)

    ok = log[log["error"].isna()]
    print(f"\nboards fetched: {len(ok):,} of {len(log):,}")
    print("errors:\n", log["error"].value_counts() if log["error"].notna().any() else "none")
    print("total jobs:", int(ok["n_jobs"].sum()))
    print("\nlargest boards:\n", ok.sort_values("n_jobs", ascending=False)
          [["ats", "slug", "n_jobs"]].head(10).to_string(index=False))

    # show the feilds of one job from ATS
    for ats in ["greenhouse", "lever"]:
        row = ok[(ok['ats'] == ats) & (ok['n_jobs'] > 0)].head(1)
        if len(row):
            data = json.loads(Path(row["path"].iat[0]).read_text(encoding="utf-8"))
            job = _jobs_in(ats, data)[0]
            print(f"\n{ats} job fields: {sorted(job.keys())}")

