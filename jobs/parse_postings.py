import html
import json
import re
from pathlib import Path

import pandas as pd
import pycountry
from bs4 import BeautifulSoup

from jobs.common import BOARDS_PATH, CONFIG

RAW_DIR = Path("data/raw/jobs")
POSTINGS_PATH = Path("data/interim/job_postings.parquet")
PARSE = CONFIG["parse"]

# ---------------------------------------------------------------- location rules

US_STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
             "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()

US_STATE_NAMES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming",
]

US_RE = re.compile(
    r"\b(United States|USA|U\.S\.A?\.?|US|D\.C\.)"
    r"|[,;\-]\s*(" + "|".join(US_STATES) + r")\b"       # ", CA"  "- AZ"  "; TX"
    r"|\b(" + "|".join(US_STATE_NAMES) + r")\b"
)

CA_PROVINCES = "AB BC MB NB NL NS NT NU ON PE QC SK YT".split()
CA_PROVINCE_RE = re.compile(r"[,;\-]\s*(" + "|".join(CA_PROVINCES) + r")\b")

REMOTE_RE = re.compile(r"\b(remote|anywhere|distributed|work from home|wfh)\b", re.I)
HYBRID_RE = re.compile(r"\bhybrid\b", re.I)
LEVER_WORKPLACE = {"remote": "remote", "hybrid": "hybrid", "on-site": "onsite", "onsite": "onsite"}
GENERIC_LOCATIONS = {"remote", "hybrid", "in-office", "distributed", "anywhere"}


def _word_list_re(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)


# Every ISO country except the US (and US territories that contain "United States"),
# plus common names where pycountry has one (e.g. "Taiwan" for "Taiwan, Province of China")
COUNTRY_NAMES = sorted({name
                        for c in pycountry.countries if c.alpha_2 not in {"US", "UM"}
                        for name in (c.name, getattr(c, "common_name", None)) if name})

US_CITIES_RE = _word_list_re(PARSE["us_cities"])
NON_US_RE = _word_list_re(COUNTRY_NAMES + PARSE["non_us_markers"])

def _clean(text: str | None) -> str | None:
    """'United  States ' -> 'United States'"""
    return re.sub(r"\s+", " ", text).strip() if text else None


def _location_display(location: str | None, all_locations: list[str]) -> str | None:
    """Use the office list when the main location is only a work arrangement like 'Hybrid'."""
    if location and location.strip().lower() not in GENERIC_LOCATIONS:
        return location
    offices = ", ".join(all_locations[:3]) if all_locations else None
    if location and offices and offices.lower() != location.lower():
        return f"{location} · {offices}"
    return location or offices


def infer_workplace(*texts: str | None) -> str:
    joined = " ".join(t for t in texts if t)
    if HYBRID_RE.search(joined):
        return "hybrid"
    if REMOTE_RE.search(joined):
        return "remote"
    return "unknown"


def us_status(*texts: str | None, country: str | None = None) -> bool | None:
    """True = US, False = clearly outside the US, None = no country given."""
    if country:                                   # Lever sometimes gives an ISO country code
        return country.upper() in {"US", "USA"}
    joined = " ".join(t for t in texts if t)
    if not joined:
        return None
    if US_RE.search(joined) or US_CITIES_RE.search(joined):
        return True                               # "New York / London" counts as US
    if NON_US_RE.search(joined) or CA_PROVINCE_RE.search(joined):
        return False
    return None


# ---------------------------------------------------------------- text

def html_to_text(raw: str | None) -> str | None:
    """Greenhouse 'content' is escaped HTML: unescape, then strip tags."""
    if not raw:
        return None
    soup = BeautifulSoup(html.unescape(raw), "html.parser")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


# ---------------------------------------------------------------- parsers

def parse_greenhouse(job: dict, slug: str) -> dict:
    location = _clean((job.get("location") or {}).get("name"))
    offices = [_clean(o.get("name")) for o in job.get("offices") or [] if o.get("name")]
    all_locations = offices or ([location] if location else [])
    depts = [d.get("name") for d in job.get("departments") or [] if d.get("name")]
    return {
        "job_uid": f"greenhouse:{slug}:{job['id']}",
        "ats": "greenhouse", "slug": slug, "source_job_id": str(job["id"]),
        "title": job.get("title"),
        "location_raw": location,
        "country": None,
        "all_locations": all_locations,
        "location_display": _location_display(location, all_locations),
        "workplace_type": infer_workplace(location, job.get("title"), *offices),
        "is_us": us_status(location, *offices),
        "department": depts[0] if depts else None,
        "employment_type": None,
        "posted_at": job.get("first_published"),
        "updated_at": job.get("updated_at"),
        "url": job.get("absolute_url"),
        "description_text": html_to_text(job.get("content")),
    }


def parse_lever(job: dict, slug: str) -> dict:
    cats = job.get("categories") or {}
    location = _clean(cats.get("location"))
    all_locations = ([_clean(x) for x in cats.get("allLocations") or [] if x]
                    or ([location] if location else []))
    lists_text = "\n\n".join(
        f"{lst.get('text', '')}\n{html_to_text(lst.get('content')) or ''}"
        for lst in job.get("lists") or [])
    description = "\n\n".join(p for p in [job.get("descriptionPlain"), lists_text,
                                          job.get("additionalPlain")] if p)
    workplace = (LEVER_WORKPLACE.get((job.get("workplaceType") or "").lower())
                 or infer_workplace(location, job.get("text")))
    return {
        "job_uid": f"lever:{slug}:{job['id']}",
        "ats": "lever", "slug": slug, "source_job_id": job["id"],
        "title": job.get("text"),
        "location_raw": location,
        "country": job.get("country"),
        "all_locations": all_locations,
        "location_display": _location_display(location, all_locations),
        "workplace_type": workplace,
        "is_us": us_status(location, *all_locations, country=job.get("country")),
        "department": cats.get("department") or cats.get("team"),
        "employment_type": cats.get("commitment"),
        "posted_at": job.get("createdAt"),
        "updated_at": None,
        "url": job.get("hostedUrl"),
        "description_text": description or None,
    }


PARSERS = {"greenhouse": (parse_greenhouse, lambda d: d.get("jobs", [])),
           "lever": (parse_lever, lambda d: d)}


# ---------------------------------------------------------------- run

def latest_run_dir() -> Path:
    runs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no fetch runs in {RAW_DIR}")
    return runs[-1]


def parse_run(run_dir: Path) -> pd.DataFrame:
    rows = []
    for ats, (parse, jobs_in) in PARSERS.items():
        for path in sorted((run_dir / ats).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(parse(job, path.stem) for job in jobs_in(data))
    df = pd.DataFrame(rows)

    # Timestamps: Greenhouse sends ISO strings with an offset, Lever sends epoch milliseconds
    gh = df["ats"] == "greenhouse"
    df["posted_at"] = pd.concat([
        pd.to_datetime(df.loc[gh, "posted_at"], utc=True, errors="coerce"),
        pd.to_datetime(df.loc[~gh, "posted_at"], unit="ms", utc=True, errors="coerce"),
    ]).sort_index()
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")

    df["is_us"] = df["is_us"].astype("boolean")
    df["run_date"] = pd.to_datetime(run_dir.name).date()
    df["description_chars"] = df["description_text"].str.len()

    run_ts = pd.Timestamp(run_dir.name, tz="UTC")
    df["posted_age_days"] = (run_ts - df["posted_at"]).dt.days
    df["is_stale"] = df["posted_age_days"] > PARSE["stale_after_days"]
    return df


def attach_companies(df: pd.DataFrame) -> pd.DataFrame:
    boards = (pd.read_parquet(BOARDS_PATH)[["ats", "slug", "company_key", "company_name"]]
                .drop_duplicates(["ats", "slug"]))
    return df.merge(boards, on=["ats", "slug"], how="left")


if __name__ == "__main__":
    run_dir = latest_run_dir()
    df = attach_companies(parse_run(run_dir))

    dupes = df["job_uid"].duplicated().sum()
    df = df.drop_duplicates("job_uid")

    POSTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(POSTINGS_PATH, index=False)

    print(f"run: {run_dir.name}  |  postings: {len(df):,}  |  duplicate job_uids dropped: {dupes}")
    print("\nby ats:\n", df["ats"].value_counts())
    print("\nworkplace_type:\n", df["workplace_type"].value_counts())
    print("\nis_us:\n", df["is_us"].value_counts(dropna=False))

    us_remote = df["is_us"].fillna(False) & (df["workplace_type"] == "remote")
    maybe = df["is_us"].isna() & (df["workplace_type"] == "remote")
    print(f"US remote jobs: {us_remote.sum():,}  |  remote with unknown country: {maybe.sum():,}")

    print("\nmissing:\n", df[["title", "location_raw", "posted_at", "description_text",
                              "company_key"]].isna().sum())
    print("\nposted_at range:", df["posted_at"].min(), "->", df["posted_at"].max())
    print("median description length:", int(df["description_chars"].median()))
    print("\nis_stale:\n", df["is_stale"].value_counts(dropna=False))

    print("\nsample US remote jobs:\n",
          df[us_remote][["company_name", "title", "location_display", "posted_at"]]
          .head(15).to_string(index=False))
    print("\nlocations marked NOT US (should all be foreign):\n",
          df.loc[df["is_us"].eq(False), "location_display"].value_counts().head(20))
    print("\nlocations with UNKNOWN country (check for US cities to add):\n",
          df.loc[df["is_us"].isna(), "location_display"].value_counts().head(20))