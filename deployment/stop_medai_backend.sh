#!/usr/bin/env bash
set -e

echo "Stopping MEDAI backend..."

DOCKER="docker"
if ! docker ps >/dev/null 2>&1; then
    DOCKER="sudo docker"
fi

if $DOCKER ps -a --format '{{.Names}}' | grep -qx "medai_inference_core"; then
    $DOCKER stop medai_inference_core
    echo "[OK] medai_inference_core stopped."
else
    echo "[INFO] medai_inference_core container not found."
fi
