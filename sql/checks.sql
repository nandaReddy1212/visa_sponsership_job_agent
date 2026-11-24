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

-- Snapshot sizes per day
SELECT run_date, count(*) AS postings, COUNTIF(evidence_level = "strong") As strong
from `h1b_jobs.job_postings`
GROUP BY run_date ORDER BY run_date DESC;

-- Preview of the daily email: best US remote jobs with strong evidence
SELECT company_name, title, location_display, matched_median_wage,
       title_match_filings, sponser_rank, posted_at, url
FROM `h1b_jobs.job_postings`
WHERE run_date = (SELECT MAX(run_date) from `h1b_jobs.job_postings`)
AND is_us AND NOT is_stale AND NOT is_intership
AND evidence_level IN ('strong', 'some', 'occupation')
ORDER BY 
CASE workplace_type WHEN 'remote' THEN 1 WHEN 'hybrid' THEN 2 ELSE 3 END,
CASE evidence_level  WHEN 'Strong' THEN 1 WHEN 'some' THEN 2 ELSE 3 END,
sponsor_score DESC, posted_at DESC
LIMIT 10;