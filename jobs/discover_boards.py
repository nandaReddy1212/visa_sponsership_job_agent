import os
import pandas as pd
from google.cloud import bigquery
from ingest.clean_lca import normalize_name
from pathlib import Path
import yaml

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT = os.environ["GCP_PROJECT"]

CONFIG_PATH = Path("config/ats_boards.yml")
VALID_ATS = {"greenhouse", "lever"}

GREENHOUSE_BOARD = "https://boards-api.greenhouse.io/v1/boards/{slug}"
GREENHOUSE_JOBS = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER_POSTINGS = "https://api.lever.co/v0/postings/{slug}?mode=json"
PROBE_CACHE = Path("data/interim/ats_probe.parquet")

_local = threading.local()


CANDIDATE_SQL = f"""
SELECT company_key, company_name, company_type, location_pattern, sponsor_rank, sponsor_score
FROM `{PROJECT}.h1b_jobs.sponsors`
WHERE sponsor_rank is not null
AND (
    sponsor_rank <= @top_n
    OR (location_pattern = 'Distributed' AND company_type = 'Direct employer'))
ORDER BY sponsor_rank
"""



def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["discovery"]["generic_words"] = set(cfg["discovery"]["generic_words"])

    for o in cfg.get("overrides") or []:
        if o["ats"] not in VALID_ATS:
            raise ValueError(f"unknown ats '{o['ats']}' for {o['company_key']}")
    cfg["overrides"] = cfg.get("overrides") or []
    return cfg

CONFIG = load_config()
DISCOVERY = CONFIG["discovery"]

def _session() -> requests.Session:
    """ One HTTP session Per thread, with retries on 429/5xx. """
    if not hasattr(_local, "session"):
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1,
                       status_forcelist=[429,500,502,503,504],
                       allowed_methods=["GET"])
        s.mount("https://",HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "h1b-job-agent/0.1 (portfolio project)"
        _local.session  = s
    return _local.session 

def _get_json(url: str):
    """JSON body, or None if the board doesn't exist (404)."""
    r = _session().get(url, timeout=DISCOVERY["timeout_s"])
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()



def load_candidates(client: bigquery.Client) -> pd.DataFrame:
    """Load the top N sponsors from BigQuery."""
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("top_n", "INT64", DISCOVERY['top_n'])
        ]
    )
    return client.query(CANDIDATE_SQL, job_config=cfg).to_dataframe()

def _core(norm_name: str) -> list[str]:
    """Meaningful words of a normalized name: 'WAL MART ASSOCIATES' -> ['wal', 'mart', 'associates']"""
    words = norm_name.lower().split()
    return [w for w in words
            if w.upper() not in DISCOVERY["generic_words"] and len(w) > 1] or words

def slug_candidates(norm_name: str) -> list[str]:
    """'SOCIAL FINANCE' -> ['socialfinance', 'social-finance', 'social']"""
    core = _core(norm_name)
    guesses = ["".join(core), "-".join(core), "".join(core[:2]), core[0]]
    return list(dict.fromkeys(g for g in guesses if len(g) >= DISCOVERY['min_slug_len']))  # deduplicate, keep order

def build_slug_guesses(cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.copy()
    cands["name_norm"] = normalize_name(cands["company_name"])
    cands["slug_guesses"] = [slug_candidates(n) for n in cands["name_norm"]]
    return cands

def _names_match(board_name: str, company_norm: str) -> bool:
    a = "".join(_core(normalize_name(pd.Series([board_name])).iat[0]))
    b = "".join(_core(company_norm))
    if not a or not b:
        return False
    return a in b or b in a or SequenceMatcher(None, a, b).ratio() >= DISCOVERY["name_match_ratio"]

def probe_company(row) -> dict:
    result = {"company_key": row.company_key, "company_name": row.company_name,
              "ats": None, "slug": None, "board_name": None, "n_jobs": None,
              "match_source": None, "error": None,
              "checked_at": pd.Timestamp.now(tz="UTC")}
    core = _core(row.name_norm)
    full_core = {"".join(core), "-".join(core)}

    try:
        for slug in row.slug_guesses:
            gh = _get_json(GREENHOUSE_BOARD.format(slug=slug))
            if gh and _names_match(gh.get("name",""), row.name_norm):
                jobs = _get_json(GREENHOUSE_JOBS.format(slug=slug)) or {}
                return {**result, "ats": "greenhouse", "slug": slug,
                        "board_name": gh.get("name"), "n_jobs": len(jobs.get("jobs", [])),
                        "match_source": "guess"}

            if slug in full_core:
                postings = _get_json(LEVER_POSTINGS.format(slug=slug))
                if postings:                        # a non-empty list means the board has jobs
                    return {**result, "ats": "lever", "slug": slug,
                            "n_jobs": len(postings), "match_source": "guess"}
    except requests.RequestException as e:
        result['error'] = type(e).__name__
    return result

def discover_boards(cands: pd.DataFrame) -> pd.DataFrame:
    cache = pd.read_parquet(PROBE_CACHE) if PROBE_CACHE.exists() else pd.DataFrame()
    done = set(cache.loc[cache["error"].isna(), "company_key"]) if len(cache) else set()
    todo = cands[~cands["company_key"].isin(done)]
    print(f"to probe: {len(todo):,}  (cached: {len(done):,})")

    results = []
    with ThreadPoolExecutor(max_workers=DISCOVERY["workers"]) as pool:
        futures = [pool.submit(probe_company, r) for r in todo.itertuples(index=False)]
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            if i % 100 == 0:
                print(f"  probed {i:,}/{len(futures):,}")
    
    fresh = pd.DataFrame(results)
    if len(cache):
        cache = cache[~cache["company_key"].isin(fresh["company_key"])]
    boards = pd.concat([cache, fresh], ignore_index=True)

    PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    boards.to_parquet(PROBE_CACHE, index=False)
    return apply_overrides(boards)

def apply_overrides(boards: pd.DataFrame) -> pd.DataFrame:
    boards = boards.copy()
    for o in CONFIG["overrides"]:
        hit = boards["company_key"] == o["company_key"]
        boards.loc[hit, ["ats", "slug", "match_source"]] = [o["ats"], o["slug"], "override"]
    return boards

if __name__ == "__main__":
    client = bigquery.Client(project=PROJECT)
    cands = build_slug_guesses(load_candidates(client))
    boards = discover_boards(cands)

    found = boards[boards["ats"].notna()]
    print(f"\nfound boards: {len(found):,} of {len(boards):,} "
          f"({len(found) / len(boards):.1%})")
    print("errors (will retry next run):", boards["error"].notna().sum())
    print("\nby ats:\n", found["ats"].value_counts())
    print("total open jobs:", int(found["n_jobs"].sum()))

    hit_rate = (cands.merge(boards[["company_key", "ats"]], on="company_key")
                     .groupby("company_type")["ats"].apply(lambda s: s.notna().mean()).round(3))
    print("\nhit rate by type:\n", hit_rate.sort_values(ascending=False))

    print("\nsample found:\n", found.sort_values("n_jobs", ascending=False)
          [["company_name", "ats", "slug", "board_name", "n_jobs"]].head(20).to_string(index=False))

    missed = cands[cands["company_key"].isin(boards.loc[boards["ats"].isna(), "company_key"])]
    print("\ntop-ranked not found (override candidates):\n",
          missed[["sponsor_rank", "company_name", "slug_guesses"]].head(20).to_string(index=False))

 