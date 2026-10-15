# Human Annotation Protocol & Labeling Guide

This document defines the strict labeling instructions for evaluating claim faithfulness in the research pipeline.

---

### Annotation Rules

1. **Look only at the two evidence passages shown (`evidence_top1`, `evidence_top2`).**
   - Do **NOT** use outside knowledge, prior domain expertise, or the web.
   - The goal is to measure whether the report's statements are grounded in the specific evidence retrieved by the pipeline, not whether they happen to be true in the real world.

2. **Label `1` (Supported) only if EVERY factual part of the claim is supported.**
   - Every number, unit, proper name, date, and causal assertion must be explicitly supported by the shown evidence passages.
   - If a claim has multiple factual clauses and even one clause is ungrounded or contradicted by the evidence, mark it **`0`**.

3. **Label `0` (Unsupported) if any part is unsupported, contradicted, or the evidence is unrelated.**
   - Hallucinated details or invented statistics receive **`0`**.
   - A claim whose evidence is completely irrelevant or topical but non-supportive is marked **`0`**, **not skipped**.

4. **Ambiguity / Unresolvable Cases (`?`).**
   - If you truly cannot decide due to severe ambiguity or linguistic truncation in the passage, write **`?`**.
   - Note: Rows labeled with `?` are automatically excluded from the final evaluation metrics.

5. **Class Balance.**
   - Aim for a rough balance between supported (`1`) and unsupported (`0`) claims across the dataset; do **not** force artificial balance if the evidence is clear.

6. **Blind Annotation.**
   - Do **NOT** look at any classifier scores, model predictions, or pipeline critic scores while labeling.

7. **File Saving.**
   - Save the finished, annotated file as:
     `data/validation_claims_labeled.csv`
   - Do **NOT** overwrite the original unlabeled `data/validation_claims.csv`.
