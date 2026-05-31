#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/deployment/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "MEDAI Edge Backend Launcher"
echo "Project: $PROJECT_DIR"
echo "========================================"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed or not available."
    echo "Hospital IT must install Docker/NVIDIA runtime for Jetson first."
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "medai_inference_core"; then
    echo "[INFO] Starting existing container: medai_inference_core"
    sudo docker start medai_inference_core >/dev/null || true
else
    echo "[INFO] Container medai_inference_core not found."

    if docker compose version >/dev/null 2>&1; then
        echo "[INFO] Starting backend using docker compose..."
        sudo docker compose up -d
    elif command -v docker-compose >/dev/null 2>&1; then
        echo "[INFO] Starting backend using docker-compose..."
        sudo docker-compose up -d
    else
        echo "[ERROR] No existing container and Docker Compose not available."
        echo "Cannot start backend automatically."
        exit 1
    fi
fi

echo "[INFO] Waiting for backend health..."
for i in $(seq 1 90); do
    if curl -fsS http://localhost:5000/health >/dev/null 2>&1; then
        echo "[OK] MEDAI backend is online."
        curl -fsS http://localhost:5000/health
        echo ""
        exit 0
    fi
    sleep 2
done

echo "[ERROR] Backend did not become healthy within timeout."
echo "Recent logs:"
sudo docker logs --tail=80 medai_inference_core || true
exit 1
