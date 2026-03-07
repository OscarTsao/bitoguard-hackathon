# BitoGuard Transaction Simulator

這個 bundle 提供兩個東西：

1. `bitoguard_schema.json`：目前建議的 canonical schema。
2. `bitoguard_transaction_simulator.py`：可直接產生 pseudo data 的 transaction simulator。

## 產出表

- `users.csv`
- `devices.csv`
- `user_device_links.csv`
- `bank_accounts.csv`
- `user_bank_links.csv`
- `crypto_wallets.csv`
- `login_events.csv`
- `fiat_transactions.csv`
- `trade_orders.csv`
- `crypto_transactions.csv`
- `scenarios.csv`
- `scenario_members.csv`
- `entity_edges.csv`
- `manifest.csv`

## 模擬的重點情境

- `mule_quick_out`
- `fan_in_hub`
- `shared_device_ring`
- `blacklist_2hop_chain`

## 執行方式

建議使用 Python 3.8 以上。

若你在專案根目錄 `/mnt/c/Users/a0210/remote/sideProject` 執行，先安裝依賴：

```bash
python -m pip install -r bitoguard_simulator/requirements.txt
```

再執行模擬器：

```bash
python bitoguard_simulator/bitoguard_transaction_simulator.py --n-users 1200 --days 30 --start-date 2026-01-01 --seed 42 --output-dir bitoguard_sim_output
```

## 調整方向

- 想更像交易所真實資料：拿正式欄位說明後，優先改 table 欄位與 enum。
- 想更像公開 dataset：調高 `suspicious_user_ratio`，並增加 scenario 權重。
- 想做 time-series validation：把 `start_date` 固定，然後改 seed 產多份月份資料。
- 想接 graph model：直接吃 `entity_edges.csv`。
