import os

import pandas as pd
from google.cloud import bigquery

from jobs.sponsor_evidence import ENRICHED_PATH

PROJECT = os.environ["GCP_PROJECT"]
TABLE = f"{PROJECT}.h1b_jobs.job_postings"
CLUSTERING = ["company_key", "evidence_level", "workplace_type"]


def _table_exists(client: bigquery.Client, table_id: str) -> bool:
    try:
        client.get_table(table_id)
        return True
    except Exception:          # NotFound
        return False

def load_postings(client: bigquery.Client, df: pd.DataFrame) -> None:
    if df["run_date"].nunique() != 1:
        raise ValueError("expected exactly one run_date per load")
    run_date = df["run_date"].iloc[0]

    df = df.copy()
    df["load_ts"] = pd.Timestamp.now(tz="UTC")

    # Idempotent: remove this day's snapshot if it was already loaded
    if _table_exists(client, TABLE):
        job = client.query(
            f"DELETE FROM `{TABLE}` where run_date = @d",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("d", "DATE", run_date)]))
        job.result()
        print(f"removed {job.num_dml_affected_rows or 0:,} existing rows for {run_date}")

    cfg = bigquery.LoadJobConfig(
        write_disposition = "WRITE_APPEND",
        time_partitioning = bigquery.TimePartitioning(
            type_ = bigquery.TimePartitioningType.DAY, field="run_date"
        ),
        clustering_fields=CLUSTERING,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    client.load_table_from_dataframe(df, TABLE, job_config=cfg).result()

    print(f"loaded {len(df):,} postings for {run_date} -> {TABLE}")


if __name__ == "__main__":
    client = bigquery.Client(project=PROJECT)
    load_postings(client, pd.read_parquet(ENRICHED_PATH))
