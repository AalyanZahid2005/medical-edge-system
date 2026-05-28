"""
dicom_listener.py — FYP: Accelerated Medical Imaging using ML Architectures


Starts a DICOM SCP (Service Class Provider) that:
  1. Listens for incoming C-STORE requests from Orthanc PACS
  2. Saves each .dcm file to a local folder
  3. Auto-detects modality from DICOM tag (0008,0060)
  4. Routes the file to the correct modality subfolder
  5. Inserts a record into dicom_studies via db_manager.py
  6. Forwards the file path to flask_api.py /analyze via HTTP POST

Modality mapping aligns with:
  - pipeline_skeleton.py  → payload["modality"] values
  - db_schema.sql         → dicom_studies.modality CHECK constraint
  - gpio_alerts.py        → CRITICAL_FINDINGS per modality

Install dependency:
    pip install pynetdicom pydicom requests

Run:
    python dicom_listener.py
    python dicom_listener.py --host 0.0.0.0 --port 11112 --output ./received_dicoms

Test (send a dummy DICOM to this listener):
    python -m pynetdicom storescu 127.0.0.1 11112 your_file.dcm
"""

import os
import sys
import time
import argparse
import threading
import requests
import logging

# ── pydicom + pynetdicom ───────────────────────────────────────────────────────
try:
    import pydicom
    from pydicom.dataset import Dataset
    from pynetdicom import AE, evt, AllStoragePresentationContexts
    from pynetdicom.sop_class import VerificationSOPClass
except Exception as e:
    print(f"REAL ERROR: {e}")
    sys.exit(1)
# ── Local project import ───────────────────────────────────────────────────────
try:
    from db_manager import DatabaseManager
except ImportError:
    print("[WARNING] db_manager.py not found in path. DB insertion will be skipped.")
    DatabaseManager = None

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_HOST        = "0.0.0.0"
DEFAULT_PORT        = 11112          # Standard DICOM port
DEFAULT_OUTPUT_DIR  = "./received_dicoms"
FLASK_ANALYZE_URL   = "http://127.0.0.1:5000/analyze"
AE_TITLE            = "FYP_LISTENER"  # This SCP's Application Entity title
ORTHANC_AE_TITLE    = "ORTHANC"       # Orthanc's AE title (set in Orthanc config)

# ── Modality tag (0008,0060) value → pipeline_skeleton.py payload["modality"] ─
# Maps standard DICOM modality codes to your project's internal modality strings
# Aligns with db_schema.sql CHECK constraint and gpio_alerts.py CRITICAL_FINDINGS
DICOM_MODALITY_MAP = {
    # Brain MRI
    "MR":  "brain_mri",
    # Retinal Fundus (stored as color photography in DICOM)
    "OP":  "retinal",       # Ophthalmic Photography
    "CF":  "retinal",       # Color Fundus (some vendors use this)
    # Skin Lesion (dermoscopy)
    "XC":  "skin",          # External Camera / Dermoscopy
    "DX":  "chest_xray",    # Digital X-Ray → Chest X-Ray
    "CR":  "chest_xray",    # Computed Radiography → also Chest or Bone
    # Bone Fracture X-Ray — disambiguated by body part tag if needed
    "DR":  "bone",          # Digital Radiography (bone)
    # ECG / Waveform
    "ECG": "ecg",
    "HD":  "ecg",           # Hemodynamic Waveform (some ECG vendors)
}

# Fallback: if body part examined tag (0018,0015) contains these strings → bone
BONE_BODY_PARTS = {"KNEE", "HIP", "WRIST", "SPINE", "SHOULDER", "ANKLE",
                   "ELBOW", "FOOT", "HAND", "FEMUR", "TIBIA", "RADIUS"}

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("dicom_listener.log", mode="a")
    ]
)
log = logging.getLogger("dicom_listener")

# ══════════════════════════════════════════════════════════════════════════════
#  MODALITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_modality(ds: Dataset) -> str:
    """
    Extract modality from DICOM tag (0008,0060) and map to pipeline string.

    Falls back to body part tag (0018,0015) to distinguish chest_xray vs bone
    when both are DX/CR modality code.

    Returns one of: brain_mri | retinal | skin | chest_xray | bone | ecg
    Defaults to 'brain_mri' if unrecognised (safe fallback for demo).
    """
    dicom_modality = getattr(ds, "Modality", "").strip().upper()
    mapped = DICOM_MODALITY_MAP.get(dicom_modality)

    # Disambiguate DX/CR → chest_xray vs bone using body part tag
    if dicom_modality in ("DX", "CR"):
        body_part = getattr(ds, "BodyPartExamined", "").strip().upper()
        if body_part in BONE_BODY_PARTS:
            mapped = "bone"
        else:
            mapped = "chest_xray"   # default DX/CR to chest

    if mapped is None:
        log.warning(f"[MODALITY] Unknown DICOM modality code '{dicom_modality}'. "
                    f"Defaulting to 'brain_mri'.")
        mapped = "brain_mri"

    log.info(f"[MODALITY] DICOM tag '{dicom_modality}' → pipeline modality '{mapped}'")
    return mapped


# ══════════════════════════════════════════════════════════════════════════════
#  FILE ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def build_output_path(output_dir: str, modality: str,
                      patient_id: str, study_uid: str) -> str:
    """
    Build the local save path, replicating a folder structure:
        output_dir/
          brain_mri/
            PT-001/
              1.2.840.xxxxx.dcm

    Mirrors the subfolder structure in batch_resize.py for consistency.
    """
    folder = os.path.join(output_dir, modality, patient_id)
    os.makedirs(folder, exist_ok=True)
    # Use StudyInstanceUID as filename — guaranteed unique per DICOM standard
    safe_uid = study_uid.replace(".", "_")
    return os.path.join(folder, f"{safe_uid}.dcm")


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK FORWARD  (non-blocking)
# ══════════════════════════════════════════════════════════════════════════════

def forward_to_flask(dcm_path: str, modality: str, patient_id: str):
    """
    POST the saved .dcm file to flask_api.py /analyze endpoint.
    Runs in a background thread so it never blocks the DICOM handler.

    flask_api.py /analyze contract:
        POST multipart/form-data: file=<.dcm bytes>, patient_id=<str>
    Response contract (from flask_api.py):
        { status, modality, prediction: {class, mean_confidence, uncertainty},
          ood_flag, routing_tier, llm_report, gradcam_base64 }
    """
    def _post():
        try:
            with open(dcm_path, "rb") as f:
                files   = {"file": (os.path.basename(dcm_path), f, "application/dicom")}
                data    = {"patient_id": patient_id, "modality": modality}
                resp    = requests.post(FLASK_ANALYZE_URL, files=files,
                                        data=data, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                log.info(f"[FLASK] /analyze response → "
                         f"class={result.get('prediction', {}).get('class')} "
                         f"tier={result.get('routing_tier')}")
            else:
                log.warning(f"[FLASK] /analyze returned HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            log.warning("[FLASK] Flask API not running — skipping forward. "
                        "Start flask_api.py on port 5000.")
        except Exception as e:
            log.error(f"[FLASK] Forward failed: {e}")

    t = threading.Thread(target=_post, daemon=True)
    t.start()


# ══════════════════════════════════════════════════════════════════════════════
#  CORE EVENT HANDLER — called by pynetdicom on every C-STORE
# ══════════════════════════════════════════════════════════════════════════════

def build_handle_store(output_dir: str, db: "DatabaseManager | None"):
    """
    Factory that returns a pynetdicom EVT_C_STORE handler closure.
    Captures output_dir and db instance from the outer scope.
    """
    def handle_store(event):
        """
        Called once per incoming DICOM file (C-STORE request from Orthanc).

        Steps:
          1. Parse DICOM dataset from the event
          2. Extract patient_id, study_uid, modality, study_date, study_time
          3. Build output path and save .dcm file to disk
          4. Insert dicom_studies record via db_manager.py
          5. Forward file path to flask_api.py /analyze (background thread)
          6. Return 0x0000 (Success) to Orthanc

        Returns:
            0x0000  — Success (Orthanc marks transfer complete)
            0xA700  — Out of resources (on disk/DB error)
        """
        receive_time = time.time()

        # ── 1. Parse dataset ─────────────────────────────────────────────────
        try:
            ds = event.dataset
            ds.file_meta = event.file_meta
        except Exception as e:
            log.error(f"[HANDLER] Failed to parse DICOM dataset: {e}")
            return 0xA700  # Failure — out of resources

        # ── 2. Extract key DICOM tags ────────────────────────────────────────
        # PatientID tag (0010,0020) — map to pipeline "PT-{n}" format if absent
        raw_patient_id = getattr(ds, "PatientID", "").strip()
        patient_id     = raw_patient_id if raw_patient_id else f"PT-UNKNOWN-{int(receive_time)}"

        # StudyInstanceUID tag (0020,000D) — primary key in dicom_studies
        study_uid      = getattr(ds, "StudyInstanceUID", "").strip()
        if not study_uid:
            study_uid  = f"GENERATED.{int(receive_time * 1000)}"
            log.warning(f"[HANDLER] Missing StudyInstanceUID — generated: {study_uid}")

        # StudyDate (0008,0020) and StudyTime (0008,0030)
        study_date     = getattr(ds, "StudyDate", None)
        study_time_tag = getattr(ds, "StudyTime", None)

        # StudyDescription (0008,1030)
        description    = getattr(ds, "StudyDescription", None)

        # ── 3. Detect modality ───────────────────────────────────────────────
        modality = detect_modality(ds)

        # ── 4. Build path and save .dcm file ─────────────────────────────────
        try:
            dcm_path = build_output_path(output_dir, modality, patient_id, study_uid)
            pydicom.dcmwrite(dcm_path, ds)
            elapsed_ms = int((time.time() - receive_time) * 1000)
            log.info(f"[SAVED] {dcm_path} | modality={modality} | "
                     f"patient={patient_id} | {elapsed_ms}ms")
        except Exception as e:
            log.error(f"[HANDLER] Failed to save DICOM file: {e}")
            return 0xA700

        # ── 5. Insert into dicom_studies via db_manager.py ───────────────────
        if db is not None:
            try:
                # Ensure patient row exists (upsert with minimal info)
                # Full patient demographics come from the GUI / PACS later
                db.upsert_patient(patient_id)

                inserted = db.insert_study(
                    study_uid       = study_uid,
                    patient_id      = patient_id,
                    modality        = modality,
                    dicom_file_path = dcm_path,
                    description     = description,
                    study_date      = study_date,
                    study_time      = str(study_time_tag) if study_time_tag else None
                )
                if inserted:
                    log.info(f"[DB] Study inserted: {study_uid} → {modality}")
                else:
                    log.warning(f"[DB] insert_study returned False for {study_uid} "
                                f"(may already exist — INSERT OR IGNORE)")
            except Exception as e:
                log.error(f"[DB] insert_study failed: {e}")
                # Don't return failure — file is saved, DB miss is recoverable

        # ── 6. Forward to Flask API (background, non-blocking) ───────────────
        forward_to_flask(dcm_path, modality, patient_id)

        # ── 7. Return DICOM Success status to Orthanc ────────────────────────
        log.info(f"[HANDLER] C-STORE complete → returning 0x0000 (Success) "
                 f"to Orthanc for {study_uid}")
        return 0x0000  # DICOM Success

    return handle_store


# ══════════════════════════════════════════════════════════════════════════════
#  LISTENER STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def start_listener(host: str, port: int, output_dir: str):
    """
    Initialise pynetdicom Application Entity and start blocking SCP server.

    Accepts all standard DICOM Storage SOP classes (CT, MR, DX, ECG, etc.)
    plus C-ECHO (Verification) so Orthanc can ping this listener.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Initialise DB manager ─────────────────────────────────────────────────
    db = None
    if DatabaseManager is not None:
        try:
            db = DatabaseManager()
            log.info("[DB] DatabaseManager initialised successfully.")
        except Exception as e:
            log.warning(f"[DB] DatabaseManager init failed: {e}. "
                        f"Continuing without DB insertion.")

    # ── Build Application Entity ──────────────────────────────────────────────
    ae = AE(ae_title=AE_TITLE)

    # Accept ALL standard storage SOP classes (MR, CT, DX, ECG, OP, XC...)
    # This covers all 6 modalities: Brain, Retinal, Skin, Chest, Bone, ECG
    ae.supported_contexts = AllStoragePresentationContexts

    # Also accept C-ECHO (Verification) — lets Orthanc do a "ping" test
    ae.add_supported_context(VerificationSOPClass)

    # ── Register event handler ────────────────────────────────────────────────
    handlers = [
        (evt.EVT_C_STORE, build_handle_store(output_dir, db)),
    ]

    # ── Start SCP server ──────────────────────────────────────────────────────
    log.info(f"[SYSTEM] DICOM Listener starting on {host}:{port}")
    log.info(f"[SYSTEM] AE Title   : {AE_TITLE}")
    log.info(f"[SYSTEM] Output dir : {os.path.abspath(output_dir)}")
    log.info(f"[SYSTEM] Flask API  : {FLASK_ANALYZE_URL}")
    log.info(f"[SYSTEM] Accepting all Storage SOP classes + C-ECHO Verification")
    log.info(f"[SYSTEM] Waiting for C-STORE from Orthanc...")

    try:
        ae.start_server(
            (host, port),
            block=True,          # blocking — runs until KeyboardInterrupt
            evt_handlers=handlers
        )
    except KeyboardInterrupt:
        log.info("[SYSTEM] Listener stopped by user (KeyboardInterrupt).")
    except OSError as e:
        log.error(f"[SYSTEM] Failed to bind {host}:{port} — {e}")
        log.error("[SYSTEM] Is port 11112 already in use? Try --port 11113")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION MODE  (Windows testing without real Orthanc / DICOM files)
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(output_dir: str):
    """
    Simulates receiving DICOM files from Orthanc without a real PACS.
    Creates minimal pydicom datasets for each modality and runs them
    through the full handler pipeline (file save + DB insert + Flask forward).

    Usage:
        python dicom_listener.py --simulate
    """
    log.info("[SIMULATE] Running Windows simulation mode (no real Orthanc needed)")

    db = None
    if DatabaseManager is not None:
        db = DatabaseManager()

    handler = build_handle_store(output_dir, db)

    # Test cases mirroring gpio_alerts.py test_cases
    test_cases = [
        # (PatientID, StudyUID,           Modality, BodyPart,  Description)
        ("PT-1",  "1.2.3.4.5.6.1",  "MR",  "",       "Brain MRI — Glioma suspect"),
        ("PT-2",  "1.2.3.4.5.6.2",  "DX",  "CHEST",  "Chest X-Ray — Pneumonia screen"),
        ("PT-3",  "1.2.3.4.5.6.3",  "DX",  "KNEE",   "Bone X-Ray — Fracture screen"),
        ("PT-4",  "1.2.3.4.5.6.4",  "ECG", "",       "ECG — Arrhythmia screen"),
        ("PT-5",  "1.2.3.4.5.6.5",  "XC",  "",       "Skin Dermoscopy — MEL screen"),
        ("PT-6",  "1.2.3.4.5.6.6",  "OP",  "",       "Retinal Fundus — APTOS screen"),
    ]

    os.makedirs(output_dir, exist_ok=True)

    for patient_id, study_uid, modality_code, body_part, desc in test_cases:
        # Build a minimal pydicom Dataset (simulates what Orthanc would send)
        ds = pydicom.Dataset()
        ds.file_meta                = pydicom.Dataset()
        ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # ExplicitVRLittleEndian
        ds.PatientID                = patient_id
        ds.StudyInstanceUID         = study_uid
        ds.Modality                 = modality_code
        ds.BodyPartExamined         = body_part
        ds.StudyDescription         = desc
        ds.StudyDate                = "20260502"
        ds.StudyTime                = "120000"
        # Minimal pixel data so pydicom doesn't complain on save
        ds.Rows                     = 4
        ds.Columns                  = 4
        ds.BitsAllocated            = 8
        ds.BitsStored               = 8
        ds.HighBit                  = 7
        ds.PixelRepresentation      = 0
        ds.SamplesPerPixel          = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData                = bytes(16)

        log.info(f"\n[SIMULATE] ─── Sending: {patient_id} | {modality_code} | {desc}")

        # Simulate the EVT_C_STORE event object
        class FakeEvent:
            dataset   = ds
            file_meta = ds.file_meta

        result = handler(FakeEvent())
        status = "SUCCESS (0x0000)" if result == 0x0000 else f"FAILED (0x{result:04X})"
        log.info(f"[SIMULATE] Handler returned: {status}")
        time.sleep(0.5)  # small gap between cases

    log.info("\n[SIMULATE] All test cases complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DICOM SCP Listener — FYP Medical Imaging Pipeline (M3 Day 3)"
    )
    parser.add_argument("--host",     default=DEFAULT_HOST,
                        help=f"Bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT,
                        help=f"DICOM port (default: {DEFAULT_PORT})")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT_DIR,
                        help=f"Root folder to save received .dcm files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--simulate", action="store_true",
                        help="Run Windows simulation mode (no real Orthanc needed)")
    args = parser.parse_args()

    if args.simulate:
        run_simulation(args.output)
    else:
        start_listener(args.host, args.port, args.output)
