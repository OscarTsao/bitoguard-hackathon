# Final Decision

## Decision table

| route | key metric | status | decision weight |
|---|---:|---|---|
| nested-CV final aggregate | F1 `0.3636`, AP `0.3222` | complete | strict backing number |
| hybrid primary locked eval | F1 `0.3729`, AP `0.3192` | complete | strongest current mainline candidate |
| hybrid + Base C(GraphSAGE) locked eval | F1 `0.3765`, AP `0.3194` | complete | strongest current mainline candidate after final precise check |
| GraphSAGE 5-fold (`157`) | mean F1 `0.3645`, mean AP `0.3131` | complete | does not beat hybrid |
| GraphSAGE 5-fold (`102.5`) | mean F1 `0.3722`, mean AP `0.3170` | complete but not reproducible vs `157` | interesting, not decisive |
| XGB-only nested ablation (`157`) | mean best inner-objective F1 `0.3687` | partial / not directly comparable | weak evidence |
| XGB-only nested ablation (`102.5`) | mean best inner-objective F1 `0.3771` | partial / not directly comparable | weak evidence |
| XGB grid nested ablation (`102.5`, 5-fold) | mean best inner-objective F1 `0.3742` | complete (5-fold targeted check) / not directly comparable | depth trends toward `6`, but pair does not stabilize |
| provisional mainline submission | threshold `0.1735`, predict rows `12753` | complete | ready-to-upload fallback based on current mainline |

## Answers

### 1. 主線結果應該仍然是什麼？

**主線應更新為 `hybrid + Base C(GraphSAGE) locked eval`。**

Best currently confirmed mainline metric:

- **F1 = `0.3765`**
- Precision = `0.3364`
- Recall = `0.4274`
- AP = `0.3194`
- threshold = `0.1735`

Why:

- it is the highest final metric produced under the current honest, locked-eval framework
- it directly compares against the old hybrid mainline under the same primary 5-fold OOF protocol
- it uses frozen honestly-selected tabular params and adds Base C without opening new large HPO

### 2. nested-CV backing number 是什麼？

The strict backing number is:

- **pooled F1 = `0.3636`**
- pooled AP = `0.3222`
- pooled threshold = `0.18`
- mean fold F1 = `0.3620`

Use this as the conservative, most honest reference.

### 3. GraphSAGE 是否值得納入主線？

**Yes, in this specific form.**

Current evidence says:

- standalone GraphSAGE 5-fold follow-ups were inconsistent across machines
- but the decisive apples-to-apples test is now complete:
  - `hybrid primary locked eval`: `0.3729`
  - `hybrid + Base C(GraphSAGE) locked eval`: `0.3765`
- AP is effectively flat to slightly improved (`0.3192 -> 0.3194`) while F1 improved by about `+0.0036`

So GraphSAGE is worth including **as Base C inside the hybrid locked mainline**, but the separate GraphSAGE-only follow-up line does not justify more standalone training.

### 4. XGB-only 是否值得繼續？

**Not as a mainline continuation.**

Why:

- the bias measurement already showed baseline-XGB honest F1 `0.3640`
- XGB-only Optuna ablations produced only inner-objective numbers, not clean final outer metrics
- the completed 5-fold XGB grid check is still not stable enough:
  - fold 0 selected `max_depth=8, lr=0.05`
  - fold 1 selected `max_depth=5, lr=0.07`
  - fold 2 selected `max_depth=6, lr=0.07`
  - fold 3 selected `max_depth=6, lr=0.03`
  - fold 4 selected `max_depth=6, lr=0.02`
- `max_depth` shows some convergence toward `6`, but `learning_rate` remains inconsistent, so the pair-selection signal is not strong enough to justify one more full hybrid run

So XGB is not the best next large investment.

## Next-step recommendation

**STOP_HERE**

### Why this choice

Because the final precise checks have already answered the remaining questions:

- `hybrid + Base C(GraphSAGE)` did beat the previous hybrid mainline (`0.3765` vs `0.3729`)
- the gain is real enough to adopt, but modest
- the completed XGB grid check did **not** produce a stable enough parameter pair across folds
- further honest XGB tuning would likely consume more compute without a reliable expected gain

### Expected benefit

Expected additional benefit from more honest tuning is now low. The mainline already improved to `0.3765`, and the remaining unexplained upside from XGB appears too unstable to justify another round before finalizing.

## Practical stop line

The stop line has been reached:

- keep `hybrid + Base C(GraphSAGE)` as the mainline result
- keep nested-CV `0.3636` as the strict backing number
- treat XGB grid as informative but non-decisive
- keep the provisional submission generated from the current mainline as the ready fallback
