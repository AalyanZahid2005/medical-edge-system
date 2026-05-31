#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "MEDAI Workstation GUI Installer"
echo "Project: $PROJECT_DIR"
echo "========================================"

echo ""
echo "[1/5] Python check"
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found."
    exit 1
fi
python3 --version

echo ""
echo "[2/5] Virtual environment check"
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "[INFO] Creating venv..."
    python3 -m venv venv
else
    echo "[OK] Existing venv found."
fi

echo ""
echo "[3/5] Installing/checking GUI dependencies"
source "$PROJECT_DIR/venv/bin/activate"

python3 - <<'PY'
mods = {
    "PyQt5": "PyQt5",
    "requests": "requests",
    "PIL": "pillow",
    "reportlab": "reportlab",
    "matplotlib": "matplotlib",
    "pyqtgraph": "pyqtgraph",
}
missing = []
for mod, pkg in mods.items():
    try:
        __import__(mod)
        print("[OK]", mod)
    except Exception:
        print("[MISSING]", mod)
        missing.append(pkg)

if missing:
    print("")
    print("[INFO] Missing packages:", " ".join(missing))
    print("[INFO] Installing missing packages with pip...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
PY

echo ""
echo "[4/5] Creating desktop launcher"
mkdir -p "$HOME/.local/share/applications"

cat > "$HOME/.local/share/applications/medai-edge-gui.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MEDAI Edge GUI
Comment=Open MEDAI Medical AI Dashboard
Exec=gnome-terminal -- bash -lc "$PROJECT_DIR/deployment/start_medai_gui.sh; echo; read -p 'Press Enter to close...'"
Icon=applications-science
Terminal=false
Categories=Science;MedicalSoftware;Utility;
EOF

chmod +x "$HOME/.local/share/applications/medai-edge-gui.desktop"

if [ -d "$HOME/Desktop" ]; then
    cp "$HOME/.local/share/applications/medai-edge-gui.desktop" "$HOME/Desktop/MEDAI Edge GUI.desktop" || true
    chmod +x "$HOME/Desktop/MEDAI Edge GUI.desktop" || true
fi

echo ""
echo "[5/5] Final status"
echo "[OK] MEDAI GUI installer completed."
echo ""
echo "Start GUI with:"
echo "  $PROJECT_DIR/deployment/start_medai_gui.sh"
echo ""
echo "Or click desktop icon:"
echo "  MEDAI Edge GUI"
