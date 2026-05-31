#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

JETSON_API="${JETSON_API:-http://10.42.0.51:5000}"
HEALTH_URL="$JETSON_API/health"

echo "========================================"
echo "MEDAI Edge GUI Launcher"
echo "Project: $PROJECT_DIR"
echo "Jetson API: $JETSON_API"
echo "========================================"

if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "[ERROR] Python virtual environment not found: $PROJECT_DIR/venv"
    echo "Run deployment/install_medai_workstation.sh first."
    exit 1
fi

echo "[INFO] Checking Jetson backend health..."
if command -v curl >/dev/null 2>&1; then
    if ! curl -fsS "$HEALTH_URL" >/tmp/medai_health.json 2>/tmp/medai_health.err; then
        echo "[WARNING] Jetson backend is not reachable."
        echo "Tried: $HEALTH_URL"
        echo ""
        cat /tmp/medai_health.err || true
        echo ""
        echo "GUI will still open, but analysis will be disabled until Jetson is online."
        sleep 2
    else
        echo "[OK] Jetson backend reachable:"
        cat /tmp/medai_health.json
        echo ""
    fi
else
    echo "[WARNING] curl not found. Skipping health check."
fi

echo "[INFO] Starting MEDAI PyQt GUI..."
source "$PROJECT_DIR/venv/bin/activate"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

python3 "$PROJECT_DIR/src/gui/gui_main.py"
