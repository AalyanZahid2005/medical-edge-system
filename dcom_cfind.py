"""
dicom_cfind.py  —  FYP: Accelerated Medical Imaging using ML Architectures
Ahmed Rasheed (M3) — Day 18 Deliverable

Autonomous DICOM C-FIND poller:
    - Queries Orthanc REST API every 60 seconds for new studies
    - Compares against DB to find studies not yet processed
    - Downloads new .dcm files via Orthanc /instances/{id}/file
    - Routes each file to the correct modality subfolder
    - Forwards to flask_api.py /analyze (same as dicom_listener.py)
    - Logs all activity to DB system_logs

This replaces manual push from PACS — the system autonomously hunts for
new scans without waiting for Orthanc to push them.

Usage:
    python dicom_cfind.py --run          # autonomous daemon
    python dicom_cfind.py --once         # single poll and exit
    python dicom_cfind.py --test         # self-test (no Orthanc needed)
    python dicom_cfind.py --list         # list all studies in Orthanc

Install:
    pip install requests pydicom
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import requests
from datetime import datetime

# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from db_manager import DatabaseManager
except ImportError:
    DatabaseManager = None
    print("[WARNING] db_manager.py not found.")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ORTHANC_URL         = "http://127.0.0.1:8042"
ORTHANC_USER        = "orthanc"
ORTHANC_PASS        = "orthanc"
FLASK_ANALYZE_URL   = "http://127.0.0.1:5000/analyze"
POLL_INTERVAL_S     = 60          # Day 18 spec: every 60 seconds
OUTPUT_DIR          = "./received_dicoms"
DOWNLOAD_TIMEOUT_S  = 30

# DICOM modality tag value → pipeline modality string
# (mirrors dicom_listener.py DICOM_MODALITY_MAP)
DICOM_MODALITY_MAP = {
    "MR":  "brain_mri",
    "OP":  "retinal",
    "CF":  "retinal",
    "XC":  "skin",
    "DX":  "chest_xray",
    "CR":  "chest_xray",
    "DR":  "bone",
    "ECG": "ecg",
    "HD":  "ecg",
}

BONE_BODY_PARTS = {
    "KNEE", "HIP", "WRIST", "SPINE", "SHOULDER",
    "ANKLE", "ELBOW", "FOOT", "HAND", "FEMUR", "TIBIA", "RADIUS"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("dicom_cfind.log", mode="a"),
    ]
)
log = logging.getLogger("dicom_cfind")


# ══════════════════════════════════════════════════════════════════════════════
#  ORTHANC REST CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class OrthancClient:
    """Thin wrapper around Orthanc REST API for C-FIND-style queries."""

    def __init__(self, base_url: str = ORTHANC_URL,
                 user: str = ORTHANC_USER,
                 password: str = ORTHANC_PASS):
        self.base_url = base_url.rstrip("/")
        self.auth     = (user, password)
        self.session  = requests.Session()
        self.session.auth = self.auth

    def ping(self) -> bool:
        """Return True if Orthanc is reachable."""
        try:
            r = self.session.get(f"{self.base_url}/system", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_studies(self) -> list:
        """
        GET /studies → list of Orthanc study IDs.
        Each ID is a random Orthanc UUID (not the DICOM StudyInstanceUID).
        """
        try:
            r = self.session.get(f"{self.base_url}/studies", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"[ORTHANC] list_studies failed: {e}")
            return []

    def get_study_info(self, orthanc_study_id: str) -> dict:
        """
        GET /studies/{id} → full study metadata including PatientID,
        StudyInstanceUID, Modality, etc.
        """
        try:
            r = self.session.get(f"{self.base_url}/studies/{orthanc_study_id}",
                                  timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"[ORTHANC] get_study_info({orthanc_study_id}) failed: {e}")
            return {}

    def list_instances_for_study(self, orthanc_study_id: str) -> list:
        """
        GET /studies/{id}/instances → list of instance metadata dicts.
        """
        try:
            r = self.session.get(
                f"{self.base_url}/studies/{orthanc_study_id}/instances",
                timeout=10
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"[ORTHANC] list_instances failed for {orthanc_study_id}: {e}")
            return []

    def download_instance(self, orthanc_instance_id: str, save_path: str) -> bool:
        """
        GET /instances/{id}/file → download raw .dcm bytes to save_path.
        Returns True on success.
        """
        try:
            r = self.session.get(
                f"{self.base_url}/instances/{orthanc_instance_id}/file",
                timeout=DOWNLOAD_TIMEOUT_S,
                stream=True
            )
            r.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            log.info(f"[ORTHANC] Downloaded instance {orthanc_instance_id} → {save_path}")
            return True
        except Exception as e:
            log.error(f"[ORTHANC] download_instance failed: {e}")
            return False

    def get_instance_tags(self, orthanc_instance_id: str) -> dict:
        """GET /instances/{id}/simplified-tags → dict of DICOM tags."""
        try:
            r = self.session.get(
                f"{self.base_url}/instances/{orthanc_instance_id}/simplified-tags",
                timeout=10
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"[ORTHANC] get_instance_tags failed: {e}")
            return {}


# ══════════════════════════════════════════════════════════════════════════════
#  MODALITY DETECTION  (same logic as dicom_listener.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_modality_from_tags(tags: dict) -> str:
    """
    Map Orthanc simplified-tags to pipeline modality string.
    Mirrors dicom_listener.py detect_modality() exactly.
    """
    dicom_mod = tags.get("Modality", "").strip().upper()
    body_part = tags.get("BodyPartExamined", "").strip().upper()

    if dicom_mod in ("DX", "CR"):
        return "bone" if body_part in BONE_BODY_PARTS else "chest_xray"

    mapped = DICOM_MODALITY_MAP.get(dicom_mod)
    if mapped is None:
        log.warning(f"[CFIND] Unknown modality code '{dicom_mod}' — "
                    f"defaulting to brain_mri")
        return "brain_mri"
    return mapped


# ══════════════════════════════════════════════════════════════════════════════
#  C-FIND POLLER
# ══════════════════════════════════════════════════════════════════════════════

class DicomCFindPoller:
    """
    Autonomous Orthanc poll loop.
    Compares Orthanc study list against DB and processes new arrivals.
    """

    def __init__(self, db=None, orthanc_url: str = ORTHANC_URL,
                 flask_url: str = FLASK_ANALYZE_URL,
                 output_dir: str = OUTPUT_DIR):
        self.db          = db or (DatabaseManager() if DatabaseManager else None)
        self.client      = OrthancClient(base_url=orthanc_url)
        self.flask_url   = flask_url
        self.output_dir  = output_dir
        self.running     = False
        self._thread     = None
        self._seen_uids  = set()   # In-memory cache of processed StudyInstanceUIDs
        self.stats = {
            "polls":     0,
            "new_found": 0,
            "routed":    0,
            "errors":    0,
        }

    # ── Poll once ─────────────────────────────────────────────────────────────

    def poll_once(self) -> dict:
        """
        Single poll cycle:
          1. List all Orthanc studies
          2. For each, get StudyInstanceUID
          3. Skip if already in DB or seen this session
          4. Download first instance, detect modality, forward to Flask
        """
        self.stats["polls"] += 1
        log.info(f"[CFIND] Poll #{self.stats['polls']} — querying Orthanc...")

        if not self.client.ping():
            log.warning("[CFIND] Orthanc not reachable. Skipping poll.")
            return {"new_found": 0, "routed": 0, "error": "orthanc_unreachable"}

        orthanc_ids = self.client.list_studies()
        if not orthanc_ids:
            log.info("[CFIND] No studies in Orthanc.")
            return {"new_found": 0, "routed": 0}

        new_found = 0
        routed    = 0

        for orthanc_study_id in orthanc_ids:
            info = self.client.get_study_info(orthanc_study_id)
            if not info:
                continue

            study_uid = (info.get("MainDicomTags", {})
                             .get("StudyInstanceUID", ""))
            if not study_uid:
                continue

            # Skip if already processed this session
            if study_uid in self._seen_uids:
                continue

            # Skip if already in DB
            if self._already_in_db(study_uid):
                self._seen_uids.add(study_uid)
                continue

            new_found += 1
            self.stats["new_found"] += 1
            log.info(f"[CFIND] New study found: {study_uid}")

            ok = self._process_study(orthanc_study_id, study_uid, info)
            if ok:
                routed += 1
                self.stats["routed"] += 1
            else:
                self.stats["errors"] += 1

            self._seen_uids.add(study_uid)

        log.info(f"[CFIND] Poll complete — {new_found} new, {routed} routed.")
        return {"new_found": new_found, "routed": routed}

    def _already_in_db(self, study_uid: str) -> bool:
        """Return True if study_uid already exists in dicom_studies table."""
        if self.db is None:
            return False
        try:
            conn = self.db._get_connection()
            row  = conn.execute(
                "SELECT 1 FROM dicom_studies WHERE study_uid=? LIMIT 1",
                (study_uid,)
            ).fetchone()
            return row is not None
        except Exception as e:
            log.error(f"[CFIND] DB check failed for {study_uid}: {e}")
            return False

    def _process_study(self, orthanc_study_id: str,
                       study_uid: str,
                       study_info: dict) -> bool:
        """Download first instance, detect modality, forward to Flask /analyze."""
        # Get first instance
        instances = self.client.list_instances_for_study(orthanc_study_id)
        if not instances:
            log.warning(f"[CFIND] No instances for study {study_uid}")
            return False

        first_instance_id = instances[0]["ID"]

        # Get tags for modality detection
        tags     = self.client.get_instance_tags(first_instance_id)
        modality = detect_modality_from_tags(tags)

        patient_id = (study_info.get("PatientMainDicomTags", {})
                                 .get("PatientID", "PT-UNKNOWN"))

        # Build save path (mirrors dicom_listener.py build_output_path)
        safe_uid   = study_uid.replace(".", "_")
        save_path  = os.path.join(self.output_dir, modality, patient_id,
                                  f"{safe_uid}.dcm")

        # Download
        ok = self.client.download_instance(first_instance_id, save_path)
        if not ok:
            log.error(f"[CFIND] Download failed for {study_uid}")
            return False

        # Insert into DB (upsert patient + study)
        if self.db:
            try:
                self.db.upsert_patient(patient_id)
                study_date = tags.get("StudyDate")
                study_time = tags.get("StudyTime")
                desc       = tags.get("StudyDescription", "")
                self.db.insert_study(
                    study_uid       = study_uid,
                    patient_id      = patient_id,
                    modality        = modality,
                    dicom_file_path = save_path,
                    description     = desc,
                    study_date      = study_date,
                    study_time      = study_time,
                )
                self.db.log_event(
                    level     = "INFO",
                    component = "dicom_cfind",
                    message   = (f"C-FIND auto-routed study {study_uid} "
                                 f"modality={modality} patient={patient_id}")
                )
            except Exception as e:
                log.error(f"[CFIND] DB insert failed for {study_uid}: {e}")

        # Forward to Flask /analyze
        self._forward_to_flask(save_path, modality, patient_id, study_uid)
        return True

    def _forward_to_flask(self, dcm_path: str, modality: str,
                          patient_id: str, study_uid: str):
        """
        POST the .dcm file to /analyze.
        Non-blocking — runs in a daemon thread.
        Mirrors dicom_listener.py forward_to_flask().
        """
        def _post():
            try:
                with open(dcm_path, "rb") as f:
                    files = {"file": (os.path.basename(dcm_path), f,
                                      "application/dicom")}
                    data  = {
                        "patient_id": patient_id,
                        "modality":   modality,
                        "study_uid":  study_uid,
                    }
                    resp = requests.post(self.flask_url, files=files,
                                         data=data, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    log.info(f"[CFIND] /analyze → "
                             f"class={result.get('prediction', {}).get('class')} "
                             f"tier={result.get('routing_tier')}")
                else:
                    log.warning(f"[CFIND] /analyze returned {resp.status_code}")
            except requests.exceptions.ConnectionError:
                log.warning("[CFIND] Flask /analyze unreachable.")
            except Exception as e:
                log.error(f"[CFIND] Forward to Flask failed: {e}")

        threading.Thread(target=_post, daemon=True).start()

    # ── Daemon loop ───────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"[CFIND] Autonomous polling started — "
                 f"interval={POLL_INTERVAL_S}s, Orthanc={self.client.base_url}")

    def stop(self):
        self.running = False
        log.info("[CFIND] Polling stopped.")

    def _loop(self):
        while self.running:
            try:
                self.poll_once()
            except Exception as e:
                log.error(f"[CFIND] Unhandled error: {e}")
                self.stats["errors"] += 1
            time.sleep(POLL_INTERVAL_S)

    def get_stats(self) -> dict:
        return dict(self.stats)


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _run_self_test():
    print("\n" + "="*55)
    print("  dicom_cfind.py — M3 Day 18 Self-Test")
    print("="*55)

    # Test modality detection
    cases = [
        ({"Modality": "MR",  "BodyPartExamined": ""},      "brain_mri"),
        ({"Modality": "DX",  "BodyPartExamined": "CHEST"},  "chest_xray"),
        ({"Modality": "DX",  "BodyPartExamined": "KNEE"},   "bone"),
        ({"Modality": "ECG", "BodyPartExamined": ""},       "ecg"),
        ({"Modality": "XC",  "BodyPartExamined": ""},       "skin"),
        ({"Modality": "OP",  "BodyPartExamined": ""},       "retinal"),
        ({"Modality": "???", "BodyPartExamined": ""},       "brain_mri"),  # fallback
    ]
    for tags, expected in cases:
        result = detect_modality_from_tags(tags)
        assert result == expected, f"Expected {expected}, got {result} for {tags}"
    print("  ✓ detect_modality_from_tags() — all 7 cases correct")

    # Test OrthancClient.ping() graceful failure
    client = OrthancClient(base_url="http://127.0.0.1:9999")
    assert not client.ping(), "ping() should return False for unreachable Orthanc"
    print("  ✓ OrthancClient.ping() handles connection refused gracefully")

    # Test poller without DB/Orthanc
    poller = DicomCFindPoller(db=None, orthanc_url="http://127.0.0.1:9999")
    result = poller.poll_once()
    assert "error" in result
    print("  ✓ poll_once() returns error dict when Orthanc unreachable")

    # Test start/stop
    poller.start()
    time.sleep(0.2)
    poller.stop()
    print("  ✓ start()/stop() lifecycle OK")

    stats = poller.get_stats()
    assert "polls" in stats
    print("  ✓ get_stats() returns expected keys")

    print("\n  All tests passed.")
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DICOM C-FIND Autonomous Poller — FYP (M3 Day 18)"
    )
    parser.add_argument("--run",       action="store_true",
                        help="Start autonomous polling daemon")
    parser.add_argument("--once",      action="store_true",
                        help="Single poll pass and exit")
    parser.add_argument("--list",      action="store_true",
                        help="List all studies in Orthanc and exit")
    parser.add_argument("--test",      action="store_true",
                        help="Self-test (no Orthanc needed)")
    parser.add_argument("--orthanc",   default=ORTHANC_URL,
                        help=f"Orthanc base URL (default: {ORTHANC_URL})")
    parser.add_argument("--output",    default=OUTPUT_DIR)
    parser.add_argument("--interval",  type=int, default=POLL_INTERVAL_S,
                        help="Poll interval in seconds (default: 60)")
    args = parser.parse_args()

    POLL_INTERVAL_S = args.interval   # Allow CLI override

    if args.test:
        _run_self_test()
    elif args.list:
        client = OrthancClient(base_url=args.orthanc)
        if not client.ping():
            print(f"[ERROR] Orthanc not reachable at {args.orthanc}")
            sys.exit(1)
        ids = client.list_studies()
        print(f"Studies in Orthanc ({len(ids)}):")
        for sid in ids:
            info = client.get_study_info(sid)
            tags = info.get("MainDicomTags", {})
            print(f"  {sid}  UID={tags.get('StudyInstanceUID','?')}  "
                  f"Date={tags.get('StudyDate','?')}")
    elif args.once:
        db     = DatabaseManager() if DatabaseManager else None
        poller = DicomCFindPoller(db=db, orthanc_url=args.orthanc,
                                  output_dir=args.output)
        result = poller.poll_once()
        print(json.dumps(result, indent=2))
    elif args.run:
        db     = DatabaseManager() if DatabaseManager else None
        poller = DicomCFindPoller(db=db, orthanc_url=args.orthanc,
                                  output_dir=args.output)
        poller.start()
        log.info("[CFIND] Running autonomously. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(5)
                stats = poller.get_stats()
                log.debug(f"[CFIND] Stats: {stats}")
        except KeyboardInterrupt:
            poller.stop()
    else:
        parser.print_help()