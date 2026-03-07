# Docker + AWS EC2 Deployment

這份專案已補上本地 Docker 與 EC2 Docker Compose 部署資產，目標是讓本地與 AWS 使用同一套執行模型。

## 本地 Docker 啟動

在專案根目錄執行：

```bash
cd /home/a0210/projects/sideProject/bitoguard_project_bundle
cp deploy/.env.compose.example .env
docker compose up --build
```

開啟：

- <http://127.0.0.1:3000>
- <http://127.0.0.1:8001/healthz>

若你要重跑 sync pipeline，再額外啟動 mock API：

```bash
docker compose --profile sync up --build
```

## 容器拓樸

- `frontend`: Next.js production server，對外映射 `127.0.0.1:3000`
- `backend`: FastAPI internal API，對外映射 `127.0.0.1:8001`
- `mock-api`: 只有重跑 sync 流程時才需要，使用 `sync` profile 啟動

`bitoguard_core/artifacts` 會以 bind mount 方式掛進 backend，保留 DuckDB 與模型 artifacts。

## AWS EC2 建議做法

### 1. 建立主機

- Ubuntu 24.04 LTS
- 開 `22`, `80`, `443`
- 不要對外開 `3000`, `8000`, `8001`
- 綁 `Elastic IP`

### 2. 安裝 Docker 與 Compose

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nginx
sudo usermod -aG docker $USER
```

重新登入一次 shell。

### 3. 上傳部署包

將整個專案放到 EC2，例如：

```bash
mkdir -p ~/apps
tar -xzf bitoguard-project-bundle.tar.gz -C ~/apps
cd ~/apps/bitoguard_project_bundle
cp deploy/.env.compose.example .env
```

### 4. 啟動容器

```bash
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8001/healthz
```

### 5. 設定 Nginx

將 [bitoguard.conf](/home/a0210/projects/sideProject/bitoguard_project_bundle/deploy/ec2/nginx/bitoguard.conf) 複製到：

```bash
sudo cp deploy/ec2/nginx/bitoguard.conf /etc/nginx/sites-available/bitoguard.conf
sudo ln -s /etc/nginx/sites-available/bitoguard.conf /etc/nginx/sites-enabled/bitoguard.conf
sudo nginx -t
sudo systemctl reload nginx
```

### 6. 設定 HTTPS

兩種做法擇一：

- 省成本：`certbot + nginx`
- AWS 標準：前面加 `ALB + ACM`

若你是單機 demo，先用 `certbot` 最直接。

### 7. 設成開機自啟

將 [bitoguard-compose.service](/home/a0210/projects/sideProject/bitoguard_project_bundle/deploy/ec2/systemd/bitoguard-compose.service) 放到：

```bash
sudo cp deploy/ec2/systemd/bitoguard-compose.service /etc/systemd/system/bitoguard-compose.service
sudo systemctl daemon-reload
sudo systemctl enable --now bitoguard-compose.service
```

## 備份建議

- 定期把 `bitoguard_core/artifacts/` 同步到 S3
- 至少保留：
  - `bitoguard.duckdb`
  - 模型 artifacts
  - 匯出報表

