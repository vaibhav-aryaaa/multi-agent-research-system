# Analysis Results and Confound Checks Summary (N = 20 Questions)

This directory contains statistical evaluations, confound checks, and visual figures across the benchmark evaluation:

## Core Comparison Files
1. **per_system.csv**: Evaluates N=20 questions per system (S1 Baseline, S2 Fixed, S3 Adaptive LLM, S4 Adaptive Classifier). Reports mean, median, standard deviation, and null counts for RAGAS Faithfulness and Answer Relevancy, alongside mean critic scores, mean retry counts, and total token usage.
2. **paired.csv**: Contains paired comparisons (S2->S3, S2->S4, S3->S4) across both metrics. Reports mean differences, 10,000-resample bootstrap 95% confidence intervals (seed 0), two-sided Wilcoxon signed-rank test p-values (dropping zero-difference pairs), step-down Holm-Bonferroni adjusted p-values across all 6 comparisons, and win/tie/loss counts.
3. **retry_effect.csv**: Analyzes questions where adaptive systems triggered retries (retry_count > 0). Shows pre-retry and post-retry critic scores alongside RAGAS Faithfulness compared against the shared attempt-0 S2 baseline.
4. **judge_disagreement.csv**: Documents per-question retry counts between S3 (LLM judge) and S4 (Cross-Encoder judge), identifying decision differences and which system achieved higher final Faithfulness.
5. **cost.csv**: Summarizes total token consumption and marginal token expenditure per +0.1 gain in RAGAS Faithfulness relative to S2 Fixed.

## Confound Checks (Task I)
6. **dose_response.csv**: Records per-question metrics for S2, S3, and S4 (N=60 rows total), capturing retry_count, number of context passages (n_contexts), total context characters, RAGAS Faithfulness, and Answer Relevancy.
7. **dose_response_correlations.csv**: Spearman rank correlation between Faithfulness and retry_count, n_contexts, and context characters. Computed for pooled systems (N=60) and per-system subsets (N=20 each). Note: S2 retry_count and n_contexts are constant (rho=N/A).
8. **matched_s3_s4.csv**: Isolates questions where S3 and S4 underwent identical retry counts (N=6 questions: Q05, Q07, Q11, Q15, Q16, Q18).
9. **matched_s3_s4_diff_retries.csv**: Isolates questions where retry counts differed between S3 and S4 (N=14 questions).
10. **matched_s3_s4_summary.csv**: Aggregated comparison for matched retries vs differing retries. Reports mean faithfulness differences, bootstrap 95% CI (seed 0), win/tie/loss counts, and notes where subset sizes are n < 5 (specifically, the S3 > S4 split has n=3).
11. **judge_agreement.csv**: Documents attempt-0 judge behavior for all N=20 questions, recording first-pass critic score, first-pass faithfulness, individual judge decisions (accept vs retry), and categorizing agreement.
12. **judge_agreement_summary.csv**: Group summary of the four agreement categories (both_accept, both_retry, only_llm_retries, only_classifier_retries) with counts, mean first-pass faithfulness, and mean critic score. Notes where group size is n < 5 (both_accept has n=1, only_llm_retries has n=1).

## Publication Figures (`figures/`)
- **fig_faithfulness_by_system.png**: Mean RAGAS Faithfulness per system with 95% bootstrap CI error bars and overlaid individual question points.
- **fig_relevancy_by_system.png**: Mean RAGAS Answer Relevancy per system with 95% bootstrap CI error bars and overlaid individual question points.
- **fig_paired_faithfulness.png**: Per-question slope plot tracing individual trajectories for S2 -> S3 and S2 -> S4 alongside the mean trend.
- **fig_cost_vs_faithfulness.png**: Total token consumption vs. mean RAGAS Faithfulness for all four systems with 95% bootstrap CI.
- **fig_retries_vs_gain.png**: Number of retries vs. faithfulness gain over S2, colored by system (S3 vs S4).
