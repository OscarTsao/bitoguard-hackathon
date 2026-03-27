# No Final Tuned-XGB Hybrid Run

Decision: **do not continue** with `hybrid_primary_locked_eval_with_graphsage_and_tuned_xgb`.

## Why it was not allowed to proceed

The completed 5-fold XGB grid was used only as a pair-selection gate.

Best pair from the full 25 pairs × 5 folds matrix:

- `(max_depth=5, learning_rate=0.10)`

But relative to the baseline XGB pair `(7, 0.05)`:

- mean uplift vs baseline: only `+0.0007153320`
- folds beating or tying baseline: `3 / 5`
- the score landscape remained effectively flat
- the selected pair did not show a strong enough, stable enough advantage

That fails the intended conservative gate for launching one more final apples-to-apples hybrid run.

## Operational note

A tentative tuned-XGB hybrid process was started before the gate was finalized.
That process was treated as invalid for decision-making and should not be used.

## Mainline remains

- mainline: [hybrid_primary_locked_eval_with_graphsage_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/hybrid_primary_locked_eval_with_graphsage/hybrid_primary_locked_eval_with_graphsage_report.json)
- final mainline F1: `0.3764769066`
- AP: `0.3194102732`
- threshold: `0.1735`

## Submission fallback

The ready provisional fallback remains:

- [submission_provisional_graphsage_mainline.csv](/Users/oscartsao/Developer/bitoguard-hackathon/submission_provisional_graphsage_mainline.csv)

## Final recommendation

**STOP_HERE_USE_0.3765_MAINLINE**
