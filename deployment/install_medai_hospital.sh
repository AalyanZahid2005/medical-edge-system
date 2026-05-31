#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "MEDAI Hospital Deployment Installer"
echo "Project: $PROJECT_DIR"
echo "========================================"

echo ""
echo "[1/6] Hardware / OS check"
if [ -f /etc/nv_tegra_release ]; then
    echo "Jetson L4T:"
    cat /etc/nv_tegra_release
else
    echo "[WARNING] /etc/nv_tegra_release not found. This may not be a Jetson device."
fi

if [ -f /proc/device-tree/model ]; then
    echo "Device model:"
    tr -d '\0' < /proc/device-tree/model
    echo ""
fi

echo ""
echo "[2/6] Docker check"
if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker not found."
    echo "Install Docker/NVIDIA runtime on Jetson before running MEDAI."
    exit 1
fi
docker --version

echo ""
echo "[3/6] Compose check"
if docker compose version >/dev/null 2>&1; then
    docker compose version
elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose --version
else
    echo "[WARNING] Docker Compose not found. Existing-container start will still work."
fi

echo ""
echo "[4/6] Engine/cache check"
ENGINE_COUNT=$(find jetson_deployment_package -name "*.engine" 2>/dev/null | wc -l || true)
ONNX_COUNT=$(find jetson_deployment_package -name "*.onnx" 2>/dev/null | wc -l || true)
echo "TensorRT engines found: $ENGINE_COUNT"
echo "ONNX models found: $ONNX_COUNT"

if [ "$ONNX_COUNT" -gt 0 ] && [ "$ENGINE_COUNT" -eq 0 ]; then
    echo "[WARNING] No TensorRT engines found. Engines may need to be built on this Jetson."
fi

echo ""
echo "[5/6] Creating desktop launchers"
mkdir -p "$HOME/.local/share/applications"

cat > "$HOME/.local/share/applications/medai-backend.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MEDAI Backend
Comment=Start MEDAI Jetson backend
Exec=gnome-terminal -- bash -lc "$PROJECT_DIR/deployment/start_medai_backend.sh; echo; read -p 'Press Enter to close...'"
Icon=utilities-terminal
Terminal=false
Categories=Utility;
EOF

cat > "$HOME/.local/share/applications/medai-backend-stop.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MEDAI Stop Backend
Comment=Stop MEDAI Jetson backend
Exec=gnome-terminal -- bash -lc "$PROJECT_DIR/deployment/stop_medai_backend.sh; echo; read -p 'Press Enter to close...'"
Icon=process-stop
Terminal=false
Categories=Utility;
EOF

chmod +x "$HOME/.local/share/applications/medai-backend.desktop"
chmod +x "$HOME/.local/share/applications/medai-backend-stop.desktop"

if [ -d "$HOME/Desktop" ]; then
    cp "$HOME/.local/share/applications/medai-backend.desktop" "$HOME/Desktop/MEDAI Backend.desktop" || true
    cp "$HOME/.local/share/applications/medai-backend-stop.desktop" "$HOME/Desktop/MEDAI Stop Backend.desktop" || true
    chmod +x "$HOME/Desktop/MEDAI Backend.desktop" "$HOME/Desktop/MEDAI Stop Backend.desktop" || true
fi

echo ""
echo "[6/6] Final status"
echo "[OK] Installer completed."
echo "Start backend with:"
echo "  $PROJECT_DIR/deployment/start_medai_backend.sh"
echo ""
echo "Check health with:"
echo "  $PROJECT_DIR/deployment/check_medai_health.sh"
