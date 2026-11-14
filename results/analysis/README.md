# Analysis Results Summary (N = 20 Questions)

This directory contains statistical evaluations and comparisons across the benchmark questions:

1. **per_system.csv**: Evaluates N=20 questions per system (S1 Baseline, S2 Fixed, S3 Adaptive LLM, S4 Adaptive Classifier). Reports mean, median, standard deviation, and null counts for RAGAS Faithfulness and Answer Relevancy, alongside mean critic scores, mean retry counts, and total token usage.
2. **paired.csv**: Contains paired comparisons (S2->S3, S2->S4, S3->S4) across both metrics. Reports mean differences, 10,000-resample bootstrap 95% confidence intervals (seed 0), two-sided Wilcoxon signed-rank test p-values (dropping zero-difference pairs), step-down Holm-Bonferroni adjusted p-values across all 6 comparisons, and win/tie/loss counts.
3. **retry_effect.csv**: Analyzes questions where adaptive systems triggered retries (retry_count > 0). Shows pre-retry and post-retry critic scores alongside RAGAS Faithfulness compared against the shared attempt-0 S2 baseline.
4. **judge_disagreement.csv**: Documents per-question retry counts between S3 (LLM judge) and S4 (Cross-Encoder judge), identifying decision differences and which system achieved higher final Faithfulness.
5. **cost.csv**: Summarizes total token consumption and marginal token expenditure per +0.1 gain in RAGAS Faithfulness relative to S2 Fixed.
