# BitoGuard Project Bundle

這個 bundle 目前包含三個主要可執行模組：

- `bitoguard_mock_api`：外部資料來源模擬 API，預設 `http://127.0.0.1:8000`
- `bitoguard_core`：內部風控 API，預設 `http://127.0.0.1:8001`
- `bitoguard_frontend`：Next.js 前端，預設 `http://127.0.0.1:3000`

舊版 `Streamlit` dashboard 已移除，現在只保留 Next.js 前端。

## 最快啟動方式

如果你想先把部署流程固定下來，現在也可以直接走 Docker：

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle
cp deploy/.env.compose.example .env
docker compose up --build
```

如果你要重跑 sync pipeline，再改用：

```bash
docker compose --profile sync up --build
```

AWS EC2 的 Docker + Nginx 部署指引在 [deploy/README.md](/home/a0210/projects/sideProject/bitoguard_project_bundle/deploy/README.md)。

### 1. 啟動 `bitoguard_core`

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_core
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=. uvicorn api.main:app --reload --port 8001
```

### 2. 啟動 `bitoguard_frontend`

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_frontend
npm install
cp .env.example .env.local
npm run dev
```

然後開啟 <http://127.0.0.1:3000>。

## 何時需要 `bitoguard_mock_api`

如果你只是要查看 bundle 內已附帶的 demo 結果，不需要先啟動 `bitoguard_mock_api`。

只有在要重跑資料同步流程時，才需要額外啟動它：

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle/bitoguard_mock_api
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
