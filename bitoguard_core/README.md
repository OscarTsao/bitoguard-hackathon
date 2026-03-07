# BitoGuard Core

`bitoguard_core` 是 BitoGuard hackathon demo 的內部產品層，負責：

- 從 `bitoguard_mock_api` 同步資料進 DuckDB
- 建 canonical tables、graph edges、feature snapshots
- 訓練 LightGBM 與 Isolation Forest
- 產出 risk score、alerts、cases、risk diagnosis
- 提供 internal FastAPI，供 Next.js 前端讀取

## 安裝

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_core
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## 核心流程

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_core
. .venv/bin/activate

PYTHONPATH=. python pipeline/sync.py --full
PYTHONPATH=. python features/graph_features.py
PYTHONPATH=. python features/build_features.py
PYTHONPATH=. python models/train.py
PYTHONPATH=. python models/anomaly.py
PYTHONPATH=. python models/score.py
PYTHONPATH=. python models/validate.py
```

## Internal API

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_core
. .venv/bin/activate
PYTHONPATH=. uvicorn api.main:app --reload --port 8001
```
