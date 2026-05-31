#!/usr/bin/env bash
set -e

echo "Stopping MEDAI backend..."

if docker ps -a --format '{{.Names}}' | grep -qx "medai_inference_core"; then
    sudo docker stop medai_inference_core
    echo "[OK] medai_inference_core stopped."
else
    echo "[INFO] medai_inference_core container not found."
fi
