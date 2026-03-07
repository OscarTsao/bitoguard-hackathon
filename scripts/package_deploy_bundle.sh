#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ARG="${1:-bitoguard-project-bundle.tar.gz}"
if [[ "${OUTPUT_ARG}" = */* ]]; then
  OUTPUT_PATH="${OUTPUT_ARG}"
else
  OUTPUT_PATH="$(dirname "${ROOT_DIR}")/${OUTPUT_ARG}"
fi

cd "${ROOT_DIR}"

tar \
  --exclude='.git' \
  --exclude='.env' \
  --exclude='bitoguard_frontend/node_modules' \
  --exclude='bitoguard_frontend/.next' \
  --exclude='bitoguard_frontend/.env.local' \
  --exclude='bitoguard_frontend/.env.production' \
  --exclude='bitoguard_core/.venv' \
  --exclude='**/__pycache__' \
  -czf "${OUTPUT_PATH}" \
  .

echo "Created ${OUTPUT_PATH}"
