import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

from jobs.common import CONFIG
from jobs.parse_postings import POSTINGS_PATH, US_STATE_NAMES, US_STATES
from functools import lru_cache

PROJECT = os.environ["GCP_PROJECT"]
EVIDENCE = CONFIG["evidence"]
ENRICHED_PATH = Path("data/interim/job_postings_enriched.parquet")

# US_STATES lists the 50 states in the same (alphabetical-by-name) order as US_STATE_NAMES, then DC

assert len(US_STATE_NAMES) == 50 and US_STATES[-1] == 'DC'
STATE_NAME_TO_CODE = dict(zip(US_STATE_NAMES, US_STATES[:50]))
DC_RE = re.compile(r"\bWashington,?\s*D\.?C\.?(?!\w)|\bD\.C\.")
STATE_CODE_RE = re.compile(r"(?:^|[,;\-]\s*)(" + "|".join(US_STATES) + r")\b")
STATE_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(US_STATE_NAMES, key=len, reverse=True)) + r")\b")

INTERN_RE = re.compile(r"\b(intern|internship|co-?op)\b", re.I)

DROP_WORDS = set(EVIDENCE['drop_words'])
SYNONYMS = EVIDENCE["synonyms"]


FILINGS_SQL = f"""
    SELECT company_key, job_title, soc_code, worksite_state,
       wage_annual_from, wage_is_suspect, decision_date
       FROM `{PROJECT}.h1b_jobs.lca_filings`
       WHERE company_key IN UNNEST(@keys)
""" 


SPONSER_SQL = f"""
    SELECT company_key, sponsor_rank, sponsor_score, company_type, location_pattern,
       filings AS company_filings, median_wage_to_pw, n_worksite_states
    FROM `{PROJECT}.h1b_jobs.sponsors`
    WHERE company_key IN UNNEST(@keys)
"""

TITLE_SOC_SQL = f"""

Select job_title, soc_code, COUNT(*) AS n
FROM `{PROJECT}.h1b_jobs.lca_filings`
WHERE job_title is not null and soc_code is not null
GROUP BY job_title, soc_code 
"""
TITLE_SOC_CACHE = Path("data/interim/lca_title_soc.parquet")
# ---------------------------------------------------------------- loading

def _query(client: bigquery.Client, sql: str, keys: list[str]) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("keys", "STRING", keys)])
    return client.query(sql, job_config=cfg).to_dataframe(create_bqstorage_client=False)

def load_lca(client: bigquery.Client, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only the filings and sponsor rows for companies that have postings."""
    filings = _query(client, FILINGS_SQL, keys )
    filings['decision_date'] = pd.to_datetime(filings['decision_date'])
    sponsers = _query(client, SPONSER_SQL, keys)
    return filings, sponsers



def title_tokens(title: str | None) -> frozenset[str]:
    """'Sr. Software Developer II (Payments)' -> {'software', 'engineer', 'payments'}"""
    if not title:
        return frozenset()
    text = re.sub(r"[^a-z0-9+#]", " ", title.lower())
    words = [part for w in text.split() for part in SYNONYMS.get(w,w).split()]
    return frozenset(w for w in words if w not in DROP_WORDS and not w.isdigit())



def title_score(job: frozenset[str], lca: frozenset[str]) -> float:
    """Share of the shorter title's words found in the longer one.
    Needs at least `min_shared_words` in common, unless the titles are identical,
    so 'Data Engineer' never matches 'Software Engineer' on 'engineer' alone."""
    if not job or not lca:
        return 0.0
    if job == lca:
        return 1.0
    shared = len(job & lca)
    if shared < EVIDENCE['min_shared_words']:
        return 0.0
    return shared/min(len(job), len(lca))

def load_title_soc(client: bigquery.Client, refresh: bool = False) -> pd.DataFrame:
    """National (title, SOC) counts from all LCA filings, cached locally."""
    if TITLE_SOC_CACHE.exists() and not refresh:
        return pd.read_parquet(TITLE_SOC_CACHE)
    df = client.query(TITLE_SOC_SQL).to_dataframe()
    TITLE_SOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TITLE_SOC_CACHE, index=False)
    return df

def build_soc_index(title_soc: pd.DataFrame) -> dict:
    """National title -> SOC lookup, plus an inverted index word -> titles."""
    t = title_soc.copy()
    t["soc6"] = t["soc_code"].str[:7]
    t["token_key"] = t["job_title"].map(lambda x: " ".join(sorted(title_tokens(x))))
    t = t[t["token_key"] != ""]
    g = t.groupby(['soc6', 'token_key'], as_index=False)['n'].sum()
    g["total"] = g.groupby("token_key")["n"].transform("sum")
    g = g[g["total"] >= EVIDENCE["soc_min_filings"]]

    titles = {}
    for key, grp in g.groupby('token_key', sort=False):
        titles[key] = {
            "tokens": frozenset(key.split()),
            "total": int(grp['total'].iat[0]),
            "socs": dict(zip(grp["soc6"], grp["n"].astype(int))),
        }
    inverted: dict[str, set[str]] = {}
    for key, entry in titles.items():
        for word in entry['tokens']:
            inverted.setdefault(word, set()).add(key)

    return {'titles':titles, "inverted":inverted}

def make_soc_lookup(soc_index: dict):
    """Returns job_socs(tokens) -> {soc: share}, with the national index built in and results cached."""
    titles, inverted = soc_index['titles'], soc_index['inverted']

    @lru_cache(maxsize=None)
    def job_socs(job_tokens: frozenset[str]) -> dict[str, float]:
        if not job_tokens:
            return None
        candidates = set().union(*(inverted.get(w, set()) for w in job_tokens))
        scored = [(title_score(job_tokens, titles[k]["tokens"]),k) for k in candidates]
        scored = [(s,k) for s, k in scored if s >= EVIDENCE["title_match_min"]]
        if not scored:
            return {}

        best = max(s for s, _ in scored)
        counts: dict[str, int] = {}
        for s, k in scored:
            if s == best:
                for soc, n in titles[k]["socs"].items():
                    counts[soc] = counts.get(soc,0) + n
        total = sum(counts.values())
        return {
            soc: round(n/total,3)
            for soc, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                if n / total >= EVIDENCE["soc_min_share"]}
    return job_socs    

def build_title_index(filings: pd.DataFrame) -> dict[str, list[dict]]:
    """Per company: one entry per distinct normalized LCA title, with its filing stats."""
    df = filings.copy()
    df["tokens"] = df["job_title"].map(title_tokens)
    df["token_key"] = df["tokens"].map(lambda t: " ".join(sorted(t)))

    index: dict[str, list[dict]] = {}
    for (key, token_key), g  in df.groupby(["company_key", "token_key"], sort=False):
        if not token_key:
            continue
        index.setdefault(key, []).append(
            {
                "tokens": frozenset(token_key.split()),
                "example_title": g["job_title"].mode().iat[0],
                "n_filings": len(g),
                "wages": g.loc[~g["wage_is_suspect"], "wage_annual_from"].dropna().to_numpy(),
                "last_filed": g["decision_date"].max(),
            }
        )
    return index

def job_states(location_raw: str | None, all_locations) -> set[str]:
    """US state codes named in a posting's locations: 'Austin, TX' -> {'TX'}"""
    texts = [location_raw, *(list(all_locations) if all_locations is not None else [])]
    found = set()
    for t in texts:
        if not t:
            continue
        if DC_RE.search(t):
            found.add("DC")
            t = DC_RE.sub(" ", t)          # so "Washington" isn't then read as WA
        found.update(m.group(1) for m in STATE_CODE_RE.finditer(t))
        found.update(STATE_NAME_TO_CODE[m.group(1)] for m in STATE_NAME_RE.finditer(t))
    return found

def evidence_for_job(row, titles: list[dict], company_states: set[str],
                     company_socs: dict[str, int], job_socs) -> dict:
    jt = title_tokens(row.title)
    scored = [(title_score(jt, t["tokens"]), t) for t in titles]
    matched = [t for s, t in scored if s >= EVIDENCE["title_match_min"]]
    best_score, best = max(scored, key=lambda x: x[0], default=(0.0, None))

    n_matched = sum(t["n_filings"] for t in matched)
    wages = np.concatenate([t["wages"] for t in matched]) if matched else np.array([])

    socs = job_socs(jt)
    occupation_filings = sum(company_socs.get(s, 0) for s in socs)

    states = job_states(row.location_raw, row.all_locations)
    state_match = bool(states & company_states) if states else None

    if n_matched >= EVIDENCE["strong_min_filings"]:
        level = "strong"
    elif n_matched > 0:
        level = "some"
    elif occupation_filings > 0:
        level = "occupation"
    else:
        level = "company_only"

    return {
        "job_uid": row.job_uid,
        "title_match_filings": n_matched,
        "title_match_titles": len(matched),
        "best_lca_title": best["example_title"] if best and best_score > 0 else None,
        "best_title_score": round(best_score, 3),
        "matched_median_wage": float(np.median(wages)) if wages.size else None,
        "matched_last_filed": max((t["last_filed"] for t in matched), default=None),
        "job_socs": list(socs),
        "occupation_filings": occupation_filings,
        "job_states": sorted(states),
        "state_match": state_match,
        "evidence_level": level,
        "is_internship": bool(INTERN_RE.search(row.title or "")),
    }

def add_evidence(postings: pd.DataFrame, filings: pd.DataFrame,
                 sponsors: pd.DataFrame, job_socs) -> pd.DataFrame:
    index = build_title_index(filings)
    states_by_company = (filings.groupby("company_key")["worksite_state"]
                                .agg(lambda s: set(s.dropna())).to_dict())

    soc_counts = (filings.assign(soc6=filings["soc_code"].str[:7])
                         .groupby(["company_key", "soc6"]).size())
    socs_by_company = {k: g.droplevel(0).to_dict() for k, g in soc_counts.groupby(level=0)}

    rows = [evidence_for_job(r, index.get(r.company_key, []),
                             states_by_company.get(r.company_key, set()),
                             socs_by_company.get(r.company_key, {}),
                             job_socs)
            for r in postings.itertuples(index=False)]
    ev = pd.DataFrame(rows)
    ev["state_match"] = ev["state_match"].astype("boolean")

    return (postings.merge(ev, on="job_uid", how="left")
                    .merge(sponsors, on="company_key", how="left"))


if __name__ == "__main__":
    postings = pd.read_parquet(POSTINGS_PATH)
    keys = sorted(postings["company_key"].dropna().unique())

    client = bigquery.Client(project=PROJECT)
    filings, sponsors = load_lca(client, keys)
    job_socs = make_soc_lookup(build_soc_index(load_title_soc(client)))
    print(f"companies: {len(keys):,}  |  LCA filings loaded: {len(filings):,}")

    df = add_evidence(postings, filings, sponsors, job_socs)
    assert len(df) == len(postings), "merge changed the row count"

    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ENRICHED_PATH, index=False)
    print(f"saved {len(df):,} postings -> {ENRICHED_PATH}")
    print("title-to-SOC cache:", job_socs.cache_info())

    target = df[df["is_us"].fillna(False) & ~df["is_stale"] & ~df["is_internship"]]
    print(f"\nTARGET SET (US, not stale, not intern): {len(target):,} postings")
    print(target["evidence_level"].value_counts())

    de = target[target["title"].str.contains(r"\bdata engineer|\banalytics engineer", case=False, na=False)]
    print(f"\ndata/analytics engineer jobs: {len(de)}")
    print(de["evidence_level"].value_counts())

    cols = ["company_name", "title", "job_socs", "occupation_filings"]
    occ = target[target["evidence_level"] == "occupation"]
    print("\nsample OCCUPATION (check the occupations fit the titles):\n",
          occ.sample(min(15, len(occ)), random_state=1)[cols].to_string(index=False))