# transductive_v1

`transductive_v1` 是一條和 `official/` 完全分開的競賽導向 MVP 管線。

## 目標

- 以競賽最終 F1 為優先
- primary validation 採 `label-mask transductive CV`
- secondary validation 僅做 strict group-aware stress test
- artifacts 全部寫到 `bitoguard_core/artifacts/transductive_v1/`

## 目前包含

- user universe 建構
- label-free features
- graph store
- fold-safe label-aware graph features
- Base A / Base B tabular models
- logistic stacker
- calibration and decision-rule selection
- final predict scoring

## 尚未包含

- `base_c_graphsage`
- 更重的 graph embeddings
- production-oriented online / streaming constraints

## 輸出

- `features/`
- `models/`
- `reports/`
- `predictions/`
- `bundle.json`
