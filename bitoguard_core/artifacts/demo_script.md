# BitoGuard Demo Script

## 1. 啟動服務

1. 啟動 `bitoguard_mock_api`：展示這是外部 Swagger source simulator
2. 啟動 `bitoguard_core` internal API（port 8001）
3. 啟動 Next.js frontend（port 3000）

## 2. 同步資料

1. 呼叫 `POST /pipeline/sync`
2. 展示 `ops.sync_runs` 有成功紀錄
3. 說明 source API 可替換，internal schema 不需重寫

## 3. 重建特徵

1. 呼叫 `POST /features/rebuild`
2. 展示 `feature_snapshots_user_day` 與 `feature_snapshots_user_30d`
3. 說明特徵分成速度型、KYC mismatch、行為統計、登入異常、圖關聯、KYC profile

## 4. 訓練與驗證

1. 呼叫 `POST /model/train`
2. 展示 Precision / Recall / F1 / FPR 與 threshold sensitivity
3. 說明使用 time-based split 與不平衡處理

## 5. 風險評分與告警

1. 呼叫 `POST /model/score`
2. 展示 alerts 自動產生
3. 說明 risk score 由 rule + supervised model + anomaly + graph risk 聚合

## 6. Alert Center

1. 在前端的 Alert Center 篩選 `high` / `critical`
2. 點進特定 alert
3. 說明風險分數、alert 狀態、case 狀態

## 7. Risk Diagnosis Report

1. 展示 SHAP Top Factors
2. 展示 rule hits 與 graph evidence
3. 展示 timeline summary 與 recommended action

## 8. User 360 與 Decision

1. 展示 User 360 的 KYC、近期交易、關聯摘要
2. 針對 alert 做 `confirm_suspicious` 或 `request_monitoring`
3. 展示 case action 已落庫

## 9. 切換來源驗證

1. 修改 `BITOGUARD_SOURCE_URL`
2. 重跑 `POST /pipeline/sync`
3. 強調 canonical、feature、model、frontend 均未修改
