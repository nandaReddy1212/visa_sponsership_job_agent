import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
from google.cloud import bigquery

from ingest.clean_lca import normalize_name
from jobs.common import CONFIG, BOARDS_PATH, get_json

PROJECT = os.environ["GCP_PROJECT"]

DISCOVERY = CONFIG["discovery"]

GREENHOUSE_BOARD = "https://boards-api.greenhouse.io/v1/boards/{slug}"
GREENHOUSE_JOBS = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER_POSTINGS = "https://api.lever.co/v0/postings/{slug}?mode=json"
PROBE_CACHE = Path("data/interim/ats_probe.parquet")

CANDIDATE_SQL = f"""
SELECT company_key, company_name, company_type, location_pattern, sponsor_rank, sponsor_score
FROM `{PROJECT}.h1b_jobs.sponsors`
WHERE sponsor_rank IS NOT NULL
  AND (sponsor_rank <= @top_n
       OR (location_pattern = 'Distributed' AND company_type = 'Direct employer'))
ORDER BY sponsor_rank
"""


# ---------------------------------------------------------------- candidates and slug guesses

def load_candidates(client: bigquery.Client) -> pd.DataFrame:
    """Top N sponsors plus all distributed direct employers, from BigQuery."""
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("top_n", "INT64", DISCOVERY["top_n"])])
    return client.query(CANDIDATE_SQL, job_config=cfg).to_dataframe()


def _core(norm_name: str) -> list[str]:
    """Meaningful words of a normalized name: 'WAL MART ASSOCIATES' -> ['wal', 'mart', 'associates']"""
    words = norm_name.lower().split()
    return [w for w in words
            if w.upper() not in DISCOVERY["generic_words"] and len(w) > 1] or words


def _board_core(board_name) -> str:
    """Core name of a Greenhouse board name, joined: 'Simpson Gumpertz & Heger' -> 'simpsongumpertzheger'"""
    if not isinstance(board_name, str) or not board_name:
        return ""
    return "".join(_core(normalize_name(pd.Series([board_name])).iat[0]))


def slug_candidates(norm_name: str) -> list[str]:
    """'SOCIAL FINANCE' -> ['socialfinance', 'social-finance', 'social']"""
    core = _core(norm_name)
    guesses = ["".join(core), "-".join(core), "".join(core[:2]), core[0]]
    return list(dict.fromkeys(g for g in guesses if len(g) >= DISCOVERY["min_slug_len"]))


def build_slug_guesses(cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.copy()
    cands["name_norm"] = normalize_name(cands["company_name"])
    cands["slug_guesses"] = [slug_candidates(n) for n in cands["name_norm"]]
    return cands


# ---------------------------------------------------------------- probing

def _names_match(board_name: str, company_norm: str) -> bool:
    """Loose check used while probing; classify_matches applies the strict tiers afterwards."""
    a = _board_core(board_name)
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
    full_core = {"".join(core), "-".join(core)}      # only these are trusted on Lever

    try:
        for slug in row.slug_guesses:
            gh = get_json(GREENHOUSE_BOARD.format(slug=slug), timeout=DISCOVERY["timeout_s"])
            if gh and _names_match(gh.get("name", ""), row.name_norm):
                jobs = get_json(GREENHOUSE_JOBS.format(slug=slug), timeout=DISCOVERY["timeout_s"]) or {}
                return {**result, "ats": "greenhouse", "slug": slug,
                        "board_name": gh.get("name"), "n_jobs": len(jobs.get("jobs", [])),
                        "match_source": "guess"}

            if slug in full_core:
                postings = get_json(LEVER_POSTINGS.format(slug=slug), timeout=DISCOVERY["timeout_s"])
                if postings:                        # a non-empty list means the board has jobs
                    return {**result, "ats": "lever", "slug": slug,
                            "n_jobs": len(postings), "match_source": "guess"}
    except requests.RequestException as e:
        result["error"] = type(e).__name__          # not cached as "not found": retried next run
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

    if results:
        fresh = pd.DataFrame(results)
        if len(cache):                               # replace old rows that were just retried
            cache = cache[~cache["company_key"].isin(fresh["company_key"])]
        boards = pd.concat([cache, fresh], ignore_index=True)
        PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        boards.to_parquet(PROBE_CACHE, index=False)
    else:
        boards = cache

    return apply_overrides(boards)


# ---------------------------------------------------------------- overrides and classification

def apply_overrides(boards: pd.DataFrame) -> pd.DataFrame:
    boards = boards.copy()
    for o in CONFIG["overrides"]:
        hit = boards["company_key"] == o["company_key"]
        if not hit.any():
            print(f"WARNING: override company_key {o['company_key']} is not a candidate - skipped")
            continue
        boards.loc[hit, ["ats", "slug", "match_source"]] = [o["ats"], o["slug"], "override"]
        # board_name / n_jobs came from the old guess, so they no longer apply
        boards.loc[hit, ["board_name", "n_jobs"]] = None
    return boards


def _match_tier(row) -> str:
    if pd.isna(row.ats):
        return "none"
    if row.match_source == "override":
        return "override"
    if row.ats == "greenhouse" and _board_core(row.board_name) == "".join(_core(row.name_norm)):
        return "high"
    return "medium"       # partial Greenhouse name match, or Lever (which returns no name)


def classify_matches(boards: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    b = boards.merge(cands[["company_key", "name_norm"]], on="company_key", how="left")
    b["match_tier"] = [_match_tier(r) for r in b.itertuples(index=False)]

    exact = b["match_tier"].isin(["high", "override"])
    medium = b["match_tier"] == "medium"
    shared = b.groupby("slug")["company_key"].transform("nunique").fillna(0) > 1
    small = b["n_jobs"].fillna(0) < DISCOVERY["min_jobs_medium"]

    rejected = {(r["company_key"], r["slug"]) for r in CONFIG["rejects"]}
    manual_reject = pd.Series([(k, s) in rejected for k, s in zip(b["company_key"], b["slug"])],
                              index=b.index)

    b["accepted"] = (exact | (medium & ~shared & ~small)) & ~manual_reject
    return b


# ---------------------------------------------------------------- run

if __name__ == "__main__":
    client = bigquery.Client(project=PROJECT)
    cands = build_slug_guesses(load_candidates(client))
    boards = classify_matches(discover_boards(cands), cands)

    found = boards[boards["ats"].notna()]
    acc = boards[boards["accepted"]]

    print(f"\nfound boards: {len(found):,} of {len(boards):,} ({len(found) / len(boards):.1%})")
    print("errors (will retry next run):", boards["error"].notna().sum())
    print("\nmatch tiers:\n", found["match_tier"].value_counts())
    print(f"\naccepted boards: {len(acc):,}  |  open jobs: {int(acc['n_jobs'].fillna(0).sum()):,}")
    print("\naccepted by ats:\n", acc["ats"].value_counts())

    hit_rate = (cands.merge(boards[["company_key", "accepted"]], on="company_key")
                     .groupby("company_type")["accepted"].mean().round(3))
    print("\naccepted rate by type:\n", hit_rate.sort_values(ascending=False))

    # Columns saved for the fetch step
    keep = ["company_key", "company_name", "ats", "slug", "board_name", "match_tier", "match_source"]
    BOARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    acc[keep].to_parquet(BOARDS_PATH, index=False)
    print(f"\nsaved {len(acc):,} accepted boards -> {BOARDS_PATH}")

    # Columns shown in the review tables (include n_jobs so you can judge each match)
    cols = ["company_name", "ats", "slug", "board_name", "n_jobs", "match_tier"]
    print("\nREVIEW - found but not accepted (add real ones to overrides):\n",
          found[~found["accepted"]].sort_values("n_jobs", ascending=False)[cols].to_string(index=False))
    print("\nREVIEW - accepted medium matches (add wrong ones to rejects):\n",
          acc[acc["match_tier"] == "medium"].sort_values("n_jobs", ascending=False)[cols].to_string(index=False))

    missed = cands[cands["company_key"].isin(boards.loc[boards["ats"].isna(), "company_key"])]
    print("\ntop-ranked not found (override candidates):\n",
          missed[["sponsor_rank", "company_name", "slug_guesses"]].head(20).to_string(index=False))