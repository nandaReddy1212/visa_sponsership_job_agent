import os
import pandas as pd
from google.cloud import bigquery
from ingest.clean_lca import run

PROJECT = os.environ["GCP_PROJECT"]
DATASET = "h1b_jobs"
SCORE_COLS = ["company_key", "volume_score", "wage_score", "sponsor_score",
              "sponsor_rank", "distribution_score"]

def _prepare(df:pd.DataFrame) -> pd.DataFrame:
    """Prepare the DataFrame for BigQuery upload."""
    # Ensure all columns are present
    df = df.copy()
    df.columns = df.columns.str.lower()
    df['load_ts'] = pd.Timestamp.now(tz='UTC')
    return df

def load(client: bigquery.Client, df: pd.DataFrame, table: str,
         partition_field: str | None = None, clustering: list[str] | None = None) -> None:
    """Load a DataFrame into BigQuery."""
    cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        clustering_fields=clustering)
    if partition_field:
        cfg.time_partitioning = bigquery.TimePartitioning(
            type_ = bigquery.TimePartitioningType.MONTH,
            field = partition_field
        )
    table_id = f"{PROJECT}.{DATASET}.{table}"
    client.load_table_from_dataframe(_prepare(df), table_id, job_config=cfg).result()
    print(f"{table}: {client.get_table(table_id).num_rows:,} rows loaded")

if __name__ == "__main__":
    df, prof, ranked = run()
    client = bigquery.Client(project=PROJECT)
    sponsors = prof.merge(ranked[SCORE_COLS], on="company_key", how="left")
    sponsors["sponsor_rank"] = sponsors["sponsor_rank"].astype("Int64")
    load(client, df, "lca_filings", partition_field="decision_date",
         clustering=["company_key", "worksite_state", "soc_code"])
    load(client, sponsors, "sponsors", clustering=["company_type", "location_pattern"])