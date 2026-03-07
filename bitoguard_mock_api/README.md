# BitoGuard Mock API

這個專案把 `bitoguard_sim_output` 的 pseudo data 投影成一份 source-facing、read-only 的 FastAPI mock server。

## 功能

- 實作 `/v1/...` list endpoints
- 將 oracle / generator metadata 從 public API 排除
- 統一時間格式為 `ISO 8601 +08:00`
- 支援 `start_time` / `end_time`、分頁、基本欄位過濾
- 將 `users.csv` 內的 `observed_blacklist_label` 投影成 `/v1/known-blacklist-users`

## 安裝

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_mock_api
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## 啟動

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_mock_api
. .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

若資料不在預設位置，可設定：

```bash
export BITOGUARD_DATA_DIR=/home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_sim_output
```

## 測試

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_mock_api
. .venv/bin/activate
pytest
```

## 匯出 OpenAPI

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_mock_api
. .venv/bin/activate
PYTHONPATH=. python scripts/export_openapi.py
```
