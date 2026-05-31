#!/usr/bin/env bash
set -e

API_URL="${MEDAI_API_URL:-http://localhost:5000/health}"

echo "Checking MEDAI backend health: $API_URL"
if command -v curl >/dev/null 2>&1; then
    curl -fsS "$API_URL"
    echo ""
    echo "[OK] MEDAI backend is reachable."
else
    python3 - <<PY
import json, urllib.request
url = "$API_URL"
with urllib.request.urlopen(url, timeout=10) as r:
    print(r.read().decode())
print("[OK] MEDAI backend is reachable.")
PY
fi
