-- Partitions created (expect ~3 monthly partitions for one quarter)
SELECT partition_id, total_rows
FROM `h1b_jobs.INFORMATION_SCHEMA.PARTITIONS`
WHERE table_name = 'lca_filings'
ORDER BY partition_id;

-- The agent's remote query: distributed direct employers, best sponsors first
SELECT company_name, filings, n_worksite_states, pct_outside_top_state,
       median_wage_to_pw, sponsor_score
FROM `h1b_jobs.sponsors`
WHERE location_pattern = 'Distributed'
  AND company_type = 'Direct employer'
ORDER BY sponsor_rank
LIMIT 15;