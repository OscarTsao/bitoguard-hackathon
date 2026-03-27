# XGB Pair Selection Gate

## Repo state

- repo path: `/Users/oscartsao/Developer/bitoguard-hackathon`
- branch: `kiro`
- commit: `cc09242`
- mainline report source: [hybrid_primary_locked_eval_with_graphsage_report.json](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/official_features/hybrid_primary_locked_eval_with_graphsage/hybrid_primary_locked_eval_with_graphsage_report.json)
- XGB grid artifact root: [xgb_grid_nested_ablation](/Users/oscartsao/Developer/bitoguard-hackathon/bitoguard_core/artifacts/xgb_grid_nested_ablation)

## Baseline pair

- baseline pair: `(max_depth=7, learning_rate=0.05)`
- 5-fold scores:
  - fold 0: `0.3707077937`
  - fold 1: `0.3787459449`
  - fold 2: `0.3666373884`
  - fold 3: `0.3739266198`
  - fold 4: `0.3689948569`
- 5-fold mean F1: `0.3718025207`
- 5-fold std: `0.0042067311`

## Selected pair

Selection rule:

1. choose the single complete pair with the best 5-fold mean score
2. if tied, prefer lower std
3. then prefer the more conservative pair

Selected pair:

- `(max_depth=5, learning_rate=0.10)`

Selected pair metrics:

- 5-fold scores:
  - fold 0: `0.3717714498`
  - fold 1: `0.3804181929`
  - fold 2: `0.3657628921`
  - fold 3: `0.3726650022`
  - fold 4: `0.3719717265`
- 5-fold mean F1: `0.3725178527`
- 5-fold std: `0.0046673738`

Relative to baseline:

- mean uplift vs baseline: `+0.0007153320`
- per-fold uplift:
  - fold 0: `+0.0010636562`
  - fold 1: `+0.0016722480`
  - fold 2: `-0.0008744963`
  - fold 3: `-0.0012616176`
  - fold 4: `+0.0029768696`
- folds beating or tying baseline: `3 / 5`

## Top pairs by 5-fold mean

| rank | pair | mean F1 | std | mean uplift vs baseline | folds >= baseline |
|---|---|---:|---:|---:|---:|
| 1 | `(5, 0.10)` | `0.3725178527` | `0.0046673738` | `+0.0007153320` | `3` |
| 2 | `(6, 0.05)` | `0.3720164649` | `0.0042188013` | `+0.0002139442` | `4` |
| 3 | `(8, 0.02)` | `0.3718375913` | `0.0042067730` | `+0.0000350705` | `2` |
| 4 | `(6, 0.03)` | `0.3718299825` | `0.0039513317` | `+0.0000274617` | `2` |
| 5 | baseline `(7, 0.05)` | `0.3718025207` | `0.0042067311` | `0.0000000000` | `5` |

## Gate decision

Gate rule required:

- mean uplift vs baseline `> 0`
- at least `3 / 5` folds not worse than baseline
- uplift not driven by one anomalous fold
- overall signal not just a flat-landscape noise difference

Observed:

- mean uplift is positive, but only `+0.0007`
- `3 / 5` folds are not worse than baseline
- two folds are worse than baseline
- the overall landscape is very flat:
  - top 5 pairs are separated by only about `0.0007`
  - the selected pair has **higher std** than baseline
  - `learning_rate` does not stabilize across fold winners:
    - fold 0 winner: `(8, 0.05)`
    - fold 1 winner: `(5, 0.07)`
    - fold 2 winner: `(6, 0.07)`
    - fold 3 winner: `(6, 0.03)`
    - fold 4 winner: `(6, 0.02)`

## Conclusion

**Gate result: FAIL**

Reason:

- the uplift over baseline is too small to treat as robust
- the winning region is still effectively a flat landscape
- `max_depth` trends toward `6`, but the full pair does not stabilize
- this is not strong enough evidence to justify one more full apples-to-apples hybrid run

## Final recommendation

**STOP_HERE_USE_0.3765_MAINLINE**
