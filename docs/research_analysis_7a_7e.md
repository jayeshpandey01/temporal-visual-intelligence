# Research Analysis: Assessment Questions 7A–7E

**Evidence scope:** Current saved-checkpoint diagnostics and the five recomputed paired comparisons in `results/final/paired_statistics.json`. This is not fresh training replication. Dataset identity, label provenance, and clean-clone reproduction remain unresolved.

## 7A. When did temporal information help?

On the current held-out prediction records, the 8-frame GRU has higher accuracy than the single-frame baseline by 26.00 pp (H1; paired 95% interval [+21.67, +30.44] pp) and higher than mean pooling by 26.56 pp (H2; [+22.11, +31.11] pp). The parameter-matched CompactGRU exceeds mean pooling by 15.33 pp (H14; [+12.33, +18.33] pp). These intervals condition on saved checkpoints and omit training-seed uncertainty. They describe this dataset and do not establish which semantic action categories benefit; stored cohort counts conflict.

## 7B. When did it not help?

In aggregate, mean pooling scores 0.56 pp below the single-frame model (8.22% vs 8.78%). That does not show that pooling hurts every sequence. The per-cohort rescue/regression tables are unreconciled, so this review cannot reliably say which examples benefit, remain unchanged, or regress. No validated class-level or real-world failure taxonomy is available.

## 7C. How can we separate order use from more visual input?

Use three controls: a single-frame model, an order-invariant model receiving the same frames, and a recurrent model. Pair predictions on the same sequences; compare canonical and shuffled/reversed order while preserving the frames. Match parameter counts where practical. The current H2–H4 and H14 results provide checkpoint-conditional evidence that outputs differ under these comparisons, but do not prove a causal mechanism or general temporal reasoning. Training provenance, dataset identity, and independent replication remain open.

## 7D. What would convince us that temporal modeling is not useful?

Before a new evaluation, define a practically meaningful minimum benefit (SESOI) and an equivalence/non-inferiority procedure. On a rights-cleared, independently retrained benchmark, evidence against usefulness would be a confidence interval for the paired temporal-minus-control gain whose upper bound is below that threshold across the prespecified test population. A nonsignificant difference alone is not evidence of equivalence. Similar canonical and shuffled performance would be diagnostic, but should be interpreted with the paired interval and power analysis.

## 7E. What would strengthen the experiment?

1. Resolve whether the data are synthetic or from a named public dataset; document source URLs, license terms, file hashes, and label mapping.
2. Regenerate or acquire the exact authorized inputs and publish the preprocessing/split manifest; audit event/identity isolation beyond sequence IDs.
3. Retrain the baseline, mean-pooling control, and recurrent models under a frozen protocol, preserving per-sequence predictions for all conditions and seeds.
4. Predefine the primary paired comparisons, minimum effect of interest, uncertainty method, and multiplicity family.
5. Recompute sampling, corruption, alternate-encoder, and failure-cohort analyses from complete paired inputs.
6. If semantic claims are important, use a fixed sample, independent blinded annotators or a documented live VLM run, retain provenance, and report disagreements.

These are proposed follow-up steps, not completed experiments. See `LIMITATIONS.md` and `ASSET_PROVENANCE.md`.
