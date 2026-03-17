# Official Experiment Summary (2026-03-17)

## Latest official transductive rerun

- Bundle version: `official_bundle_20260317T115548Z`
- Selected model: `stacked_transductive`
- Primary validation protocol: `label_mask_transductive_cv`
- Primary selected threshold: `0.1492`
- Calibrator: `isotonic`

## Primary metrics

- Precision: `0.29827315541601257`
- Recall: `0.4634146341463415`
- F1: `0.3629417382999045`
- FPR: `0.0362111914454098`
- Average precision: `0.28418408775482373`

## Secondary group stress metrics

- Precision: `0.30241758241758243`
- Recall: `0.4195121951219512`
- F1: `0.3514687100893997`
- FPR: `0.03214047025943253`
- Average precision: `0.2560239991748314`

## Threshold reference points

- Submission / max-F1 threshold: `0.1492`
- Higher-precision review threshold candidate: `0.2788`
  - Precision: `0.4491017964071856`
  - Recall: `0.22865853658536586`
  - F1: `0.30303030303030304`

## Key outputs

- Validation report:
  - `bitoguard_core/artifacts/reports/official_validation_report.json`
- Predict-label scores:
  - `bitoguard_core/artifacts/predictions/official_predict_scores.csv`
- OOF predictions:
  - `bitoguard_core/artifacts/official_features/official_oof_predictions.parquet`
- Secondary OOF predictions:
  - `bitoguard_core/artifacts/official_features/official_secondary_oof_predictions_full.parquet`
- Bundle:
  - `bitoguard_core/artifacts/official_bundle.json`

## Notes

- This rerun replaces the older strict labeled-only baseline as the best competition-aligned offline result in this repo.
- The old strict baseline remains useful as a conservative stress reference, but not as the main selection target.
