from pathlib import Path
import pandas as pd

RAW = Path("data/raw/LCA_Disclosure_Data_FY2026_Q3.xlsx")
OUT = Path("data/interim/lca_raw.parquet")

KEEP_COLUMNS = [
    "CASE_NUMBER", "CASE_STATUS", "VISA_CLASS", "RECEIVED_DATE", "DECISION_DATE",
    "JOB_TITLE", "SOC_CODE", "SOC_TITLE", "FULL_TIME_POSITION", "BEGIN_DATE", "END_DATE",
    "TOTAL_WORKER_POSITIONS", "NEW_EMPLOYMENT", "CONTINUED_EMPLOYMENT", "CHANGE_EMPLOYER",
    "EMPLOYER_NAME", "TRADE_NAME_DBA", "EMPLOYER_FEIN", "EMPLOYER_CITY", "EMPLOYER_STATE",
    "NAICS_CODE", "H_1B_DEPENDENT", "SECONDARY_ENTITY", "SECONDARY_ENTITY_BUSINESS_NAME",
    "WILLFUL_VIOLATOR",
    "WORKSITE_CITY", "WORKSITE_STATE", "WORKSITE_POSTAL_CODE", "TOTAL_WORKSITE_LOCATIONS",
    "WAGE_RATE_OF_PAY_FROM", "WAGE_RATE_OF_PAY_TO", "WAGE_UNIT_OF_PAY",
    "PREVAILING_WAGE", "PW_UNIT_OF_PAY", "PW_WAGE_LEVEL",
]

if not OUT.exists():
    df = pd.read_excel(RAW, usecols=KEEP_COLUMNS, dtype=str)  # Specify dtype to avoid issues with Excel formatting
    OUT.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    df.to_parquet(OUT, index=False)
df = pd.read_parquet(OUT)

print("rows:", len(df))
print("\nnull %:\n", (df.isna().mean() * 100).round(1).sort_values(ascending=False).head(15))

for col in ["CASE_STATUS", "VISA_CLASS", "WAGE_UNIT_OF_PAY", "PW_WAGE_LEVEL",
            "FULL_TIME_POSITION", "H_1B_DEPENDENT", "SECONDARY_ENTITY"]:
    print(f"\n--- {col} ---\n", df[col].value_counts(dropna=False).head(10))

print("\ndistinct EMPLOYER_NAME:", df["EMPLOYER_NAME"].nunique())
print("distinct EMPLOYER_FEIN:", df["EMPLOYER_FEIN"].nunique())
print("\nsample wages:\n", df[["WAGE_RATE_OF_PAY_FROM", "WAGE_UNIT_OF_PAY"]].sample(5, random_state=1))
print("\nsample dates:\n", df["DECISION_DATE"].dropna().head(3).tolist())
