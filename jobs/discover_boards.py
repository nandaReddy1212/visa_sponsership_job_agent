import os
import pandas as pd
from google.cloud import bigquery
from ingest.clean_lca import normalize_name
from pathlib import Path
import yaml

PROJECT = os.environ["GCP_PROJECT"]

CONFIG_PATH = Path("config/ats_boards.yml")
VALID_ATS = {"greenhouse", "lever"}

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

def load_candidates(client: bigquery.Client) -> pd.DataFrame:
    """Load the top N sponsors from BigQuery."""
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("top_n", "INT64", DISCOVERY['top_n'])
        ]
    )
    return client.query(CANDIDATE_SQL, job_config=cfg).to_dataframe()

def slug_candidates(norm_name: str) -> list[str]:
    """'SOCIAL FINANCE' -> ['socialfinance', 'social-finance', 'social']"""
    words = norm_name.lower().split()
    core = [w for w in words if w.upper() not in DISCOVERY['generic_words'] and len(w) > 1] or words
    guesses = ["".join(words), "".join(core), "-".join(core), "".join(core[:2]), core[0]]
    return list(dict.fromkeys([g for g in guesses if len(g) >= DISCOVERY['min_slug_len']]))  # deduplicate, keep order

def build_slug_guesses(cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.copy()
    cands["name_norm"] = normalize_name(cands["company_name"])
    cands["slug_guesses"] = [slug_candidates(n) for n in cands["name_norm"]]
    return cands


if __name__ == "__main__":
    client = bigquery.Client(project=PROJECT)
    cands = build_slug_guesses(load_candidates(client))
    n_guesses = cands["slug_guesses"].map(len)
    print("\nguesses per company:", round(n_guesses.mean(), 2))
    print("total guesses:", n_guesses.sum())
    print(f"candidates: {len(cands):,}")
    print("\nby type:\n", cands["company_type"].value_counts())
    print("\nguesses per company:", round(cands["slug_guesses"].str.len().mean(), 2))
    print("total guesses:", cands["slug_guesses"].str.len().sum())
    print("\nsample:\n", cands[["sponsor_rank", "company_name", "slug_guesses"]]
          .head(20).to_string(index=False))

 