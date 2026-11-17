import pandas as pd
import numpy as np
import re


KEEP_VISA = {"H-1B"}
KEEP_STATUS = {"Certified", "Certified - Withdrawn"}

SUFFIXES = r"(LLC|L L C|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LP|LLP|PLLC|PC|PA|NA|USA|US)"
SUFFIX_RE = re.compile(rf"\s+{SUFFIXES}$")
BAD_FEINS = {d * 9 for d in "0123456789"}          # 000000000, 111111111, ...

def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only rows with H-1B certified filings. 

    Parameters:
    df (pd.DataFrame): The input DataFrame containing LCA data.
    """

    counts = {"Raw": len(df)}

    # Filter by visa class
    df = df[df["VISA_CLASS"].str.strip().isin(KEEP_VISA)]
    counts["h1b_only"] = len(df)

    df = df[df["CASE_STATUS"].str.strip().isin(KEEP_STATUS)]
    counts["certified"] = len(df)

    #DECISION_DATE is an ISO string, so sorting it as text gives the right order.

    df = (df.sort_values("DECISION_DATE", ascending=False)
                .drop_duplicates(subset=["CASE_NUMBER"], keep="first"))
    counts["deduped"] = len(df)

    for step, n in counts.items():
        print(f"{step:>10}: {n:,}")

    return df.reset_index(drop=True)

BOOL_COLS = ["FULL_TIME_POSITION", "H_1B_DEPENDENT", "SECONDARY_ENTITY", "WILLFUL_VIOLATOR"]
DATE_COLS = ["RECEIVED_DATE", "DECISION_DATE", "BEGIN_DATE", "END_DATE"]
INT_COLS = ["TOTAL_WORKER_POSITIONS", "NEW_EMPLOYMENT", "CONTINUED_EMPLOYMENT",
            "CHANGE_EMPLOYER", "TOTAL_WORKSITE_LOCATIONS"]
FLOAT_COLS = ["WAGE_RATE_OF_PAY_FROM", "WAGE_RATE_OF_PAY_TO", "PREVAILING_WAGE"]
STATE_COLS = ["EMPLOYER_STATE", "WORKSITE_STATE"]

BOOL_MAP = {"Y": True, "YES": True, "N": False, "NO": False}


def _digits_only(s: pd.Series) -> pd.Series:
    """Keep only the digits in a string column."""
    return s.str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)

def _clean_zip(s: pd.Series) -> pd.Series:
    """'02139-1234' -> '02139', '2139' -> '02139'."""
    base = s.str.split("-").str[0]
    return _digits_only(base).str.zfill(5).str[:5]

def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trim every text column and turn empty strings into null
    obj_cols = df.select_dtypes(include=["object", "str"]).columns

    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip().replace("", pd.NA))

    before_nulls = df.isna().sum() # used for the conversion-failure report below

    # 2. Booleans
    for col in BOOL_COLS:
        df[col] = df[col].str.upper().map(BOOL_MAP).astype("boolean")

    # 3. Dates
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        df[col] = df[col].where(pd.notnull(df[col]), None)

    # 4. Numbers
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c].str.replace(r"[$,]", "", regex=True), errors="coerce")

    # 5. ID codes stay as text

    df["EMPLOYER_FEIN"] = _digits_only(df["EMPLOYER_FEIN"]).str.zfill(9)
    df["NAICS_CODE"] = _digits_only(df["NAICS_CODE"])
    for c in ["EMPLOYER_POSTAL_CODE", "WORKSITE_POSTAL_CODE"]:
        if c in df.columns:
            df[c] = _clean_zip(df[c])
    for c in STATE_COLS:
        df[c] = df[c].str.upper()

    # 6 Report values that were present but failed to convert

    failed = (df.isna().sum() - before_nulls)
    failed = failed[failed > 0]
    print("\nvalues that failed to convert:\n", failed if len(failed) else "none")

    return df

ANNUAL_MULTIPLIER = {"Year": 1, "Month": 12, "Bi-Weekly": 26, "Week": 52, "Hour": 2080}
WAGE_MIN, WAGE_MAX = 30_000, 1_000_000   # outside this range = suspicious


def annualize_wages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    wage_unit = df["WAGE_UNIT_OF_PAY"].mask(df["WAGE_RATE_OF_PAY_FROM"] >= WAGE_MIN, "Year")
    pw_unit = df["PW_UNIT_OF_PAY"].mask(df["PREVAILING_WAGE"] >= WAGE_MIN, "Year")
    df["wage_unit_corrected"] = wage_unit != df["WAGE_UNIT_OF_PAY"]

    wage_mult = wage_unit.map(ANNUAL_MULTIPLIER)
    pw_mult = pw_unit.map(ANNUAL_MULTIPLIER)

    unknown = set(df["WAGE_UNIT_OF_PAY"].dropna()) - ANNUAL_MULTIPLIER.keys()
    unknown |= set(df["PW_UNIT_OF_PAY"].dropna()) - ANNUAL_MULTIPLIER.keys()
    if unknown:
        print("unmapped pay units:", unknown)

    df["wage_annual_from"] = (df["WAGE_RATE_OF_PAY_FROM"] * wage_mult).round(2)
    df["wage_annual_to"] = (df["WAGE_RATE_OF_PAY_TO"] * wage_mult).round(2)
    df["pw_annual"] = (df["PREVAILING_WAGE"] * pw_mult).round(2)
    df["wage_to_pw_ratio"] = (df["wage_annual_from"] / df["pw_annual"]).round(3)

    df["wage_is_suspect"] = ~df["wage_annual_from"].between(WAGE_MIN, WAGE_MAX)

    print(f"\nsuspect wages: {df['wage_is_suspect'].sum():,} "
          f"({df['wage_is_suspect'].mean():.2%})")
    return df

def normalize_name(s: pd.Series) -> pd.Series:
    """Remove common suffixes and punctuation."""
    out = (
        s.str.upper()
        .str.replace(r"^THE\s+", "", regex=True)
        .str.replace("&", " AND ", regex=False)
        .str.replace(r"[^A-Z0-9]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    out = out.str.replace(r"\s+(F K A|FKA|FORMERLY|D B A|DBA|A K A|AKA)\b.*$", "", regex=True).str.strip()
    for _ in range(3):  # handles stacked suffixes like "... CO INC
        out = out.str.replace(SUFFIX_RE, "", regex=True).str.strip()
    return out

def build_company_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a normalized company name and a composite key for grouping."""
    df = df.copy()
    df["employer_name_norm"] = normalize_name(df["EMPLOYER_NAME"])

    bad_fein = df["EMPLOYER_FEIN"].isin(BAD_FEINS)
    df["company_key"] = df["EMPLOYER_FEIN"].where(~bad_fein, "Name:" + df["employer_name_norm"])
    UMBRELLA_MIN_NAMES = 5
    names_per_fein = df.groupby("EMPLOYER_FEIN")["employer_name_norm"].transform("nunique")
    umbrella = names_per_fein > UMBRELLA_MIN_NAMES
    df["company_key"] = df["company_key"].mask(umbrella, df["EMPLOYER_FEIN"] + "|" + df["employer_name_norm"])
    print(f"umbrella FEINs split by name: {df.loc[umbrella, 'EMPLOYER_FEIN'].nunique():,}")
    print(f"\nplaceholder FEINs -> name key: {bad_fein.sum():,}")

    # Display name = most common original spelling for each key
    display = (df.groupby(["company_key","EMPLOYER_NAME"]).size()
                            .reset_index(name="n")
                            .sort_values( "n", ascending=False)
                            .drop_duplicates("company_key")
                            .set_index("company_key")["EMPLOYER_NAME"])
    df["company_name"] = df["company_key"].map(display)

    return df

LABEL_RULES = {
    "consulting_naics": r"^(541511|541512|541519|5416)",
    "staffing_naics":   r"^5613",
    "edu_naics":        r"^6113",
    "research_naics":   r"^5417",
    "professional_naics": r"^(5416|5412)"   # management consulting, accounting
}
MIN_SECONDARY_CONSULT = 0.30
MIN_DEPENDENT_CONSULT = 0.50
MIN_SECONDARY_PLACEMENT = 0.50


def _mode_by_key(df: pd.DataFrame, col: str) -> pd.Series:
    """Most common value of col per company_key (faster than groupby + lambda)."""
    return (df.dropna(subset=[col])
              .groupby(["company_key", col]).size()
              .reset_index(name="n")
              .sort_values("n", ascending=False)
              .drop_duplicates("company_key")
              .set_index("company_key")[col])


def build_company_profile(df: pd.DataFrame) -> pd.DataFrame:
    prof = df.groupby("company_key").agg(
        company_name=("company_name", "first"),
        filings=("CASE_NUMBER", "size"),
        workers = ("TOTAL_WORKER_POSITIONS", "sum"),
        pct_secondary = ("SECONDARY_ENTITY", "mean"),
        pct_h1b_dependent = ("H_1B_DEPENDENT", "mean"),
        willful_violator = ("WILLFUL_VIOLATOR", "any"),
    )
    prof["naics_code"] = _mode_by_key(df, "NAICS_CODE")

    naics = prof["naics_code"].fillna("")
    is_consult_naics = naics.str.match(LABEL_RULES["consulting_naics"])
    sec = prof["pct_secondary"].fillna(0)
    dep = prof["pct_h1b_dependent"].fillna(0)

    prof["company_type"] = "Direct employer"
    # Assign in reverse priority so the higher-priority rule overwrites
    prof.loc[naics.str.match(LABEL_RULES["research_naics"]), "company_type"] = "Research"
    prof.loc[naics.str.match(LABEL_RULES["edu_naics"]), "company_type"] = "University (likely cap-exempt)"
    prof.loc[naics.str.match(LABEL_RULES["professional_naics"]), "company_type"] = "Professional services"
    prof.loc[sec >= MIN_SECONDARY_PLACEMENT, "company_type"] = "Client placement" 
    prof.loc[is_consult_naics & ((sec >= MIN_SECONDARY_CONSULT) | (dep >= MIN_DEPENDENT_CONSULT)), "company_type"] = "IT Consulting"
    prof.loc[naics.str.match(LABEL_RULES["staffing_naics"]), "company_type"] = "Staffing"

    return prof.reset_index()

MIN_FILINGS_TO_RANK = 5
DISTRIBUTED_MIN_STATES = 5
DISTRIBUTED_MIN_OUTSIDE = 0.40
SCORE_WEIGHTS = {"volume": 0.6, "wage": 0.4}
PLACEMENT_TYPES = {"IT Consulting", "Client placement", "Staffing"}

def add_location_and_wage_stats(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    key = df["company_key"]
    state_counts = df.groupby(["company_key", "WORKSITE_STATE"]).size()
    per_company = state_counts.groupby(level=0)
    ok = df[~df["wage_is_suspect"]]

    stats = pd.DataFrame({
        "n_worksite_states": per_company.size(),
        "top_state": per_company.idxmax().map(lambda t: t[1]),
        "pct_outside_top_state": 1 - per_company.max() / per_company.sum(),
        "median_wage_annual": ok.groupby("company_key")["wage_annual_from"].median(),
        "median_wage_to_pw": ok.groupby("company_key")["wage_to_pw_ratio"].median(),
        "pct_level_3_4": df["PW_WAGE_LEVEL"].isin(["III", "IV"]).groupby(key).mean(),
        "last_decision_date": df.groupby("company_key")["DECISION_DATE"].max(),
    }).round(3)

    prof = prof.merge(stats, left_on="company_key", right_index=True, how="left")

    distributed = ((prof["n_worksite_states"] >= DISTRIBUTED_MIN_STATES)
                   & (prof["pct_outside_top_state"] >= DISTRIBUTED_MIN_OUTSIDE))
    prof["location_pattern"] = np.select(
        [distributed & prof["company_type"].isin(PLACEMENT_TYPES), distributed],
        ["Distributed (client sites)", "Distributed"],
        default="Concentrated",
    )
    return prof

def rank_sponsors(prof: pd.DataFrame) -> pd.DataFrame:
    r = prof[(prof["filings"] >= MIN_FILINGS_TO_RANK)
             & ~prof["willful_violator"].fillna(False)].copy()
    # Log scale keeps big sponsors apart: 54 filings -> 0.42, Amazon -> 1.0
    r["volume_score"] = np.log1p(r["filings"])/np.log1p(r["filings"].max())
    r["wage_score"] = r["median_wage_to_pw"].rank(pct=True).fillna(0)
    r["sponsor_score"] = (100 * (SCORE_WEIGHTS["volume"] * r["volume_score"]
                               + SCORE_WEIGHTS["wage"] * r["wage_score"])).round(1)

    r["sponsor_rank"] = r["sponsor_score"].rank(ascending=False, method="min").astype(int)
    # Separate attribute for remote filtering, not part of sponsor_score
    r["distribution_score"] = (100 * r["pct_outside_top_state"].rank(pct=True)).round(1)
    return r.sort_values("sponsor_rank")

def run(parquet_path: str = "data/interim/lca_raw.parquet"):
    df = filter_rows(pd.read_parquet(parquet_path))
    df = standardize_types(df)
    df = annualize_wages(df)
    df = build_company_key(df)
    prof = build_company_profile(df)
    prof = add_location_and_wage_stats(df, prof)
    ranked = rank_sponsors(prof)
    return df, prof, ranked



if __name__ == "__main__":
    df, prof, ranked = run()
    print("names with edge spaces:", (df["EMPLOYER_NAME"] != df["EMPLOYER_NAME"].str.strip()).sum())
    print("empty strings left:", (df[obj_cols_check := [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]] == "").sum().sum())

    print("\ndistinct raw names:", df["EMPLOYER_NAME"].nunique(),
          "| normalized:", df["employer_name_norm"].nunique(),
          "| company keys:", df["company_key"].nunique())

    # One FEIN, several normalized names: either missed spelling variants or a placeholder FEIN
    names_per_key = df.groupby("company_key")["employer_name_norm"].nunique()
    print("\nkeys with most name variants:\n", names_per_key.sort_values(ascending=False).head(10))
    top = names_per_key.idxmax()
    print(f"\nnames under {top}:\n", df.loc[df.company_key == top, "employer_name_norm"].value_counts().head(8))

    # One name, several FEINs: subsidiaries, or separate legal entities with the same name
    keys_per_name = df.groupby("employer_name_norm")["company_key"].nunique()
    print("\nnames with most FEINs:\n", keys_per_name.sort_values(ascending=False).head(10))

    print("\ntop 15 sponsors by filings:\n",
          df.groupby("company_name").size().sort_values(ascending=False).head(15))

    print("\ncompanies by type:\n", prof["company_type"].value_counts())
    print("\nfilings by type:\n", prof.groupby("company_type")["filings"].sum().sort_values(ascending=False))
    print("\ntop 25 sponsors with labels:\n",
            prof.sort_values("filings", ascending=False).head(25)[["company_name", "company_type", "filings"]].to_string(index=False))
    print(f"\nranked companies: {len(ranked):,} of {len(prof):,}")
    print("\nlocation pattern:\n", ranked["location_pattern"].value_counts())

    cols = ["sponsor_rank", "company_name", "company_type", "filings", "n_worksite_states",
            "pct_outside_top_state", "median_wage_to_pw", "sponsor_score", "distribution_score"]
    print("\ntop 20 overall:\n", ranked.head(20)[cols].to_string(index=False))
    print("\ntop 15 distributed direct employers (remote-friendly candidates):\n",
          ranked[(ranked["location_pattern"] == "Distributed")
                 & (ranked["company_type"] == "Direct employer")].head(15)[cols].to_string(index=False))
    


