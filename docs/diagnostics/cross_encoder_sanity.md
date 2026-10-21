# Cross-Encoder Inference Diagnostics & Parity Verification

The DistilBERT cross-encoder (`vaibhavarya/distilbert-faithfulness-nli`) inference path was verified against direct PyTorch transformers inference on a balanced sample of 500 GLUE MNLI `validation_matched` sentence pairs (seed 42). Both `score_claims` and direct transformer inference achieve identical accuracy of **0.8380** (macro-F1: **0.8380**) with a maximum absolute probability difference of $\approx 1.19 \times 10^{-6}$. This confirms zero implementation discrepancy between training preprocessing and runtime inference (matching token order, truncation, sequence length, and classification head logits).

On canonical test pairs, the model demonstrates high sensitivity to strict logical entailment:
- A clear hypernym entailment (*"guitar on stage"* $\to$ *"instrument"*) scores **0.9714**.
- A clear contradiction (*"sleeping on the couch"* $\to$ *"running outside"*) scores **0.0644**.
- Domain claims with plausible lexical substitutions (e.g., *"conducting plates"* $\to$ *"metal plates"* in the Casimir effect) score low (**0.2189**).

This behavior stems directly from the standard 3-class NLI definition: MNLI categorizes plausible world-knowledge assertions that lack explicit premise deduction as `neutral`, and binary training maps both `neutral` and `contradiction` to `0` (*not supported*). Whether this strict semantic entailment boundary aids or hurts in-domain report verification is evaluated on human-labeled research pipeline claims, rather than assumed.
