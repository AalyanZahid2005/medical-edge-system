import os
import requests
import sqlite3

API = "http://127.0.0.1:5000/health"
ORTHANC = "http://127.0.0.1:8042/system"
DB = "fyp_medical.db"

def check_api():
    try:
        r = requests.get(API, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_orthanc():
    try:
        r = requests.get(ORTHANC, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_db():
    try:
        if not os.path.exists(DB):
            return False
        con = sqlite3.connect(DB)
        con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        con.close()
        return True
    except Exception:
        return False

checks = {
    "API": check_api(),
    "ORTHANC": check_orthanc(),
    "SQLITE_DB": check_db(),
}

for name, ok in checks.items():
    print(f"{name}: {'OK' if ok else 'FAIL'}")

print("SYSTEM HEALTH:", "OK" if all(checks.values()) else "FAIL")
