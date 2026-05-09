"""
send_dicom_results.py — FYP: Accelerated Medical Imaging using ML Architectures
Ahmed Rasheed (M3) — Day 6 Deliverable

Builds a DICOM Structured Report (SR) from an inference_record and sends it
back to Orthanc PACS via HTTP REST (C-STORE over multipart/form-data).

Flow:
    flask_api.py /analyze
        → inference result written to DB (db_manager.insert_record)
        → send_sr(record_id)                  ← called from flask_api Day 8
            → build_sr_dataset()              ← constructs pydicom SR
            → send_to_orthanc()               ← POSTs to Orthanc REST API
            → db.log_sr_send()                ← logs receive_to_send_ms

Transfer syntax: ExplicitVRLittleEndian
    Matches db_schema.sql dicom_sr_log.transfer_syntax DEFAULT value.
    Matches Day 16 end-to-end timing target (<5000ms receive_to_send_ms).

Install:
    pip install pydicom pynetdicom requests

Run standalone test:
    python send_dicom_results.py --test
    python send_dicom_results.py --record-id 1 --orthanc http://127.0.0.1:8042
"""

import os
import sys
import time
import uuid
import argparse
import logging
import requests
from datetime import datetime
from io import BytesIO

# ── pydicom ───────────────────────────────────────────────────────────────────
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import (
        generate_uid,
        ExplicitVRLittleEndian,
        UID,
    )
    import pydicom.uid as dicom_uid
except ImportError:
    print("[ERROR] Missing pydicom. Run: pip install pydicom")
    sys.exit(1)

# ── Local project import ───────────────────────────────────────────────────────
try:
    from db_manager import DatabaseManager
except ImportError:
    print("[WARNING] db_manager.py not found. DB logging will be skipped.")
    DatabaseManager = None

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ORTHANC_URL        = "http://127.0.0.1:8042"          # Orthanc REST API base URL
ORTHANC_INSTANCES  = f"{ORTHANC_URL}/instances"       # POST endpoint for DICOM upload
ORTHANC_USER       = "orthanc"                         # Orthanc default credentials
ORTHANC_PASS       = "orthanc"
SR_OUTPUT_DIR      = "./sr_outputs"                    # Local folder for generated SRs

# DICOM SR SOP Class — Basic Text SR Storage
SR_SOP_CLASS_UID   = "1.2.840.10008.5.1.4.1.1.88.11"

# Implementation UID — identifies this FYP system in DICOM metadata
IMPLEMENTATION_UID = "1.2.826.0.1.3680043.8.498.1"
IMPLEMENTATION_VER = "FYP_MEDIMG_01"

# Transfer syntax — matches db_schema.sql dicom_sr_log.transfer_syntax DEFAULT
TRANSFER_SYNTAX    = ExplicitVRLittleEndian   # "1.2.840.10008.1.2.1"

# Routing tier → human-readable label for SR text
TIER_LABELS = {1: "AUTO-APPROVED", 2: "RADIOLOGIST REVIEW", 3: "URGENT"}

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("send_dicom_results.log", mode="a"),
    ]
)
log = logging.getLogger("send_dicom_results")


# ══════════════════════════════════════════════════════════════════════════════
#  SR DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_sr_dataset(record: dict) -> FileDataset:
    """
    Construct a DICOM Basic Text Structured Report from an inference_record dict.

    The SR encodes:
        - Patient demographics (from dicom_studies JOIN)
        - Modality + predicted class + confidence
        - MC Dropout uncertainty percent
        - Routing tier (1=Auto, 2=Review, 3=Urgent)
        - OOD flag
        - GPIO alert type (RED/GREEN)
        - LLM clinical report (Phi-2 text, Day 9)

    Transfer syntax: ExplicitVRLittleEndian
        → matches db_schema.sql dicom_sr_log.transfer_syntax DEFAULT

    Args:
        record: dict from db_manager.query_history() row — contains all
                inference_records + dicom_studies JOIN fields.

    Returns:
        pydicom FileDataset ready to serialise and POST to Orthanc.
    """
    now         = datetime.utcnow()
    date_str    = now.strftime("%Y%m%d")
    time_str    = now.strftime("%H%M%S.%f")[:13]
    sr_uid      = generate_uid()                     # Unique SOP Instance UID for this SR
    study_uid   = record.get("study_uid", generate_uid())
    patient_id  = record.get("patient_id", "PT-UNKNOWN")

    # ── File Meta (required header block) ─────────────────────────────────────
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = SR_SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = sr_uid
    file_meta.TransferSyntaxUID          = TRANSFER_SYNTAX
    file_meta.ImplementationClassUID     = IMPLEMENTATION_UID
    file_meta.ImplementationVersionName  = IMPLEMENTATION_VER
    file_meta.SourceApplicationEntityTitle = "FYP_SR_SENDER"

    # ── Root dataset ───────────────────────────────────────────────────────────
    ds = FileDataset(
        filename_or_obj = None,
        dataset         = {},
        file_meta       = file_meta,
        is_implicit_VR  = False,    # ExplicitVR
        is_little_endian= True,
        preamble        = b"\x00" * 128,
    )

    # ── Patient Module ─────────────────────────────────────────────────────────
    ds.PatientID   = patient_id
    ds.PatientName = record.get("patient_name") or patient_id
    ds.PatientBirthDate = record.get("date_of_birth", "")
    ds.PatientSex  = record.get("sex", "U")

    # ── General Study Module ───────────────────────────────────────────────────
    ds.StudyInstanceUID   = study_uid
    ds.StudyDate          = record.get("study_date") or date_str
    ds.StudyTime          = record.get("study_time") or time_str
    ds.StudyDescription   = f"FYP ML Inference SR - {record.get('modality','unknown').upper()}"
    ds.AccessionNumber    = ""
    ds.ReferringPhysicianName = "FYP_SYSTEM"

    # ── SR Document Series Module ──────────────────────────────────────────────
    ds.Modality              = "SR"
    ds.SeriesInstanceUID     = generate_uid()
    ds.SeriesNumber          = "1"
    ds.SeriesDescription     = "AI Inference Structured Report"

    # ── General Equipment Module ───────────────────────────────────────────────
    ds.Manufacturer          = "FYP_Medical_Imaging"
    ds.ManufacturerModelName = "Accelerated ML Pipeline v1.0"
    ds.SoftwareVersions      = "Day6"

    # ── SOP Common Module ──────────────────────────────────────────────────────
    ds.SOPClassUID    = SR_SOP_CLASS_UID
    ds.SOPInstanceUID = sr_uid
    ds.InstanceNumber = "1"
    ds.ContentDate    = date_str
    ds.ContentTime    = time_str

    # ── SR Document General Module ─────────────────────────────────────────────
    ds.ValueType             = "CONTAINER"
    ds.ContinuityOfContent   = "SEPARATE"
    ds.CompletionFlag        = "COMPLETE"
    ds.VerificationFlag      = "UNVERIFIED"

    # ── Concept Name (document title) ─────────────────────────────────────────
    concept_name             = Dataset()
    concept_name.CodeValue   = "18748-4"
    concept_name.CodingSchemeDesignator = "LN"
    concept_name.CodeMeaning = "Diagnostic Imaging Report"
    ds.ConceptNameCodeSequence = Sequence([concept_name])

    # ── Content Sequence — the actual SR payload ───────────────────────────────
    ds.ContentSequence = Sequence(_build_content_items(record, date_str, time_str))

    log.info(f"[SR] Built SR dataset: SOPInstanceUID={sr_uid} "
             f"patient={patient_id} study={study_uid}")
    return ds


def _build_content_items(record: dict, date_str: str, time_str: str) -> list:
    """
    Build the ContentSequence items encoding all inference result fields.
    Each item is a DICOM dataset with ValueType TEXT or NUM or CODE.

    Fields encoded (matching flask_api.py /analyze response contract):
        - Modality
        - Predicted class
        - Mean confidence (%)
        - Uncertainty (%)
        - Routing tier
        - OOD flag
        - GPIO alert type
        - LLM clinical report (Phi-2 text)
        - Record ID (for DB cross-reference)
        - Inference timestamp
    """
    items = []

    def _text_item(concept_meaning: str, text_value: str) -> Dataset:
        """Helper: create a TEXT content item."""
        item = Dataset()
        item.RelationshipType     = "CONTAINS"
        item.ValueType            = "TEXT"
        cn = Dataset()
        cn.CodeValue              = "FINDING"
        cn.CodingSchemeDesignator = "FYP"
        cn.CodeMeaning            = concept_meaning
        item.ConceptNameCodeSequence = Sequence([cn])
        item.TextValue            = str(text_value)
        return item

    def _num_item(concept_meaning: str, numeric_value: float, unit: str = "%") -> Dataset:
        """Helper: create a NUM content item."""
        item = Dataset()
        item.RelationshipType     = "CONTAINS"
        item.ValueType            = "NUM"
        cn = Dataset()
        cn.CodeValue              = "MEASUREMENT"
        cn.CodingSchemeDesignator = "FYP"
        cn.CodeMeaning            = concept_meaning
        item.ConceptNameCodeSequence = Sequence([cn])
        # Measured Value Sequence
        mv = Dataset()
        mv.NumericValue           = str(round(numeric_value, 4))
        unit_ds = Dataset()
        unit_ds.CodeValue         = unit
        unit_ds.CodingSchemeDesignator = "UCUM"
        unit_ds.CodeMeaning       = unit
        mv.MeasurementUnitsCodeSequence = Sequence([unit_ds])
        item.MeasuredValueSequence = Sequence([mv])
        return item

    # ── 1. Modality ─────────────────────────────────────────────────────────
    items.append(_text_item(
        "Imaging Modality",
        record.get("modality", "unknown").upper()
    ))

    # ── 2. Predicted Class ──────────────────────────────────────────────────
    items.append(_text_item(
        "AI Predicted Class",
        record.get("predicted_class", "Unknown")
    ))

    # ── 3. Mean Confidence ──────────────────────────────────────────────────
    conf = record.get("mean_confidence")
    if conf is not None:
        items.append(_num_item("AI Mean Confidence", float(conf), "%"))

    # ── 4. MC Dropout Uncertainty ───────────────────────────────────────────
    unc = record.get("uncertainty_percent")
    if unc is not None:
        items.append(_num_item("MC Dropout Uncertainty", float(unc), "%"))

    # ── 5. Routing Tier ─────────────────────────────────────────────────────
    tier = record.get("routing_tier", 1)
    tier_label = TIER_LABELS.get(tier, f"TIER-{tier}")
    items.append(_text_item(
        "Clinical Routing Tier",
        f"Tier {tier} - {tier_label}"
    ))

    # ── 6. OOD Flag ─────────────────────────────────────────────────────────
    ood = record.get("ood_flag", 0)
    items.append(_text_item(
        "Out-of-Distribution Flag",
        "OOD DETECTED - result unreliable" if ood else "In-distribution - result reliable"
    ))

    # ── 7. GPIO Alert Type ──────────────────────────────────────────────────
    gpio = record.get("gpio_alert_type", "NONE")
    items.append(_text_item(
        "GPIO Hardware Alert",
        f"{gpio} - {'CRITICAL FINDING: Physical alert triggered' if gpio == 'RED' else 'Normal finding'}"
    ))

    # ── 8. LLM Clinical Report (Phi-2) ──────────────────────────────────────
    llm = record.get("llm_report") or "[LLM report pending - Phi-2 not yet integrated]"
    items.append(_text_item("AI Clinical Report (LLM)", llm))

    # ── 9. DB Record ID (cross-reference) ───────────────────────────────────
    items.append(_text_item(
        "DB Record ID",
        str(record.get("record_id", "N/A"))
    ))

    # ── 10. Inference Timestamp ──────────────────────────────────────────────
    items.append(_text_item(
        "Inference Timestamp",
        record.get("inferred_at") or datetime.utcnow().isoformat()
    ))

    return items


# ══════════════════════════════════════════════════════════════════════════════
#  SR SERIALISER
# ══════════════════════════════════════════════════════════════════════════════

def serialise_sr(ds: FileDataset, output_dir: str = SR_OUTPUT_DIR) -> tuple:
    """
    Write the SR dataset to a local .dcm file and also return bytes for HTTP POST.

    Returns:
        (sr_file_path: str, sr_bytes: bytes)
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_uid     = ds.SOPInstanceUID.replace(".", "_")
    sr_file_path = os.path.join(output_dir, f"SR_{safe_uid}.dcm")

    # Write to disk
    pydicom.dcmwrite(sr_file_path, ds, write_like_original=False)
    log.info(f"[SR] Written to disk: {sr_file_path}")

    # Also capture bytes for HTTP upload (avoid re-reading from disk)
    buf = BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    sr_bytes = buf.getvalue()

    return sr_file_path, sr_bytes


# ══════════════════════════════════════════════════════════════════════════════
#  ORTHANC REST SENDER
# ══════════════════════════════════════════════════════════════════════════════

def send_to_orthanc(sr_bytes: bytes,
                    orthanc_url: str = ORTHANC_URL) -> tuple:
    """
    POST DICOM SR bytes to Orthanc REST API /instances endpoint.

    Orthanc accepts raw DICOM bytes as application/dicom.
    Returns (orthanc_instance_id: str | None, success: bool).

    Orthanc REST API reference:
        POST /instances  →  { "ID": "<instance-uuid>", "Status": "Success" }
    """
    url = f"{orthanc_url}/instances"
    try:
        resp = requests.post(
            url,
            data    = sr_bytes,
            headers = {"Content-Type": "application/dicom"},
            auth    = (ORTHANC_USER, ORTHANC_PASS),
            timeout = 10,
        )
        if resp.status_code in (200, 201):
            body = resp.json()
            instance_id = body.get("ID", "unknown")
            log.info(f"[ORTHANC] SR accepted → instance_id={instance_id}")
            return instance_id, True
        else:
            log.warning(f"[ORTHANC] POST failed: HTTP {resp.status_code} — {resp.text[:200]}")
            return None, False

    except requests.exceptions.ConnectionError:
        log.warning("[ORTHANC] Connection refused — is Orthanc running on port 8042?")
        return None, False
    except requests.exceptions.Timeout:
        log.warning("[ORTHANC] Request timed out after 10s")
        return None, False
    except Exception as e:
        log.error(f"[ORTHANC] Unexpected error: {e}")
        return None, False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY — send_sr()
#  Called by flask_api.py /analyze after DB insert (Day 8 wiring)
# ══════════════════════════════════════════════════════════════════════════════

def send_sr(record_id: int,
            db: "DatabaseManager | None" = None,
            orthanc_url: str = ORTHANC_URL,
            output_dir: str = SR_OUTPUT_DIR) -> dict:
    """
    Full pipeline: fetch record → build SR → serialise → send to Orthanc → log.

    Called from flask_api.py after insert_record() on Day 8.

    Args:
        record_id   : inference_records.record_id from db_manager.insert_record()
        db          : DatabaseManager instance (shared from flask_api.py)
        orthanc_url : Orthanc base URL (default: http://127.0.0.1:8042)
        output_dir  : local folder to save .dcm SR files

    Returns:
        {
            "success":            bool,
            "sr_file_path":       str | None,
            "orthanc_instance_id":str | None,
            "receive_to_send_ms": int,          ← Day 16 timing target <5000ms
            "send_status":        "SENT" | "FAILED"
        }
    """
    t_start = time.time()

    if db is None and DatabaseManager is not None:
        db = DatabaseManager()

    # ── 1. Fetch inference record from DB ──────────────────────────────────────
    record = _fetch_record(record_id, db)
    if record is None:
        log.error(f"[send_sr] record_id={record_id} not found in DB")
        return {"success": False, "send_status": "FAILED",
                "sr_file_path": None, "orthanc_instance_id": None,
                "receive_to_send_ms": 0}

    # ── 2. Build DICOM SR dataset ──────────────────────────────────────────────
    try:
        ds = build_sr_dataset(record)
    except Exception as e:
        log.error(f"[send_sr] build_sr_dataset failed: {e}")
        _log_failure(db, record_id, None, str(e), t_start)
        return {"success": False, "send_status": "FAILED",
                "sr_file_path": None, "orthanc_instance_id": None,
                "receive_to_send_ms": int((time.time() - t_start) * 1000)}

    # ── 3. Serialise to disk + bytes ───────────────────────────────────────────
    try:
        sr_file_path, sr_bytes = serialise_sr(ds, output_dir)
    except Exception as e:
        log.error(f"[send_sr] serialise_sr failed: {e}")
        _log_failure(db, record_id, None, str(e), t_start)
        return {"success": False, "send_status": "FAILED",
                "sr_file_path": None, "orthanc_instance_id": None,
                "receive_to_send_ms": int((time.time() - t_start) * 1000)}

    # ── 4. Send to Orthanc ────────────────────────────────────────────────────
    orthanc_instance_id, success = send_to_orthanc(sr_bytes, orthanc_url)
    receive_to_send_ms = int((time.time() - t_start) * 1000)

    send_status = "SENT" if success else "FAILED"

    # ── 5. Log result to dicom_sr_log ─────────────────────────────────────────
    if db:
        db.log_sr_send(
            record_id          = record_id,
            sr_file_path       = sr_file_path,
            orthanc_instance_id= orthanc_instance_id,
            status             = send_status,
            receive_to_send_ms = receive_to_send_ms,
            error_message      = None if success else "Orthanc POST failed",
        )
        db._log(
            "INFO" if success else "WARNING",
            "send_dicom_results",
            f"SR send: record={record_id} status={send_status} "
            f"ms={receive_to_send_ms} instance={orthanc_instance_id}"
        )

    # ── 6. Day 16 timing check ────────────────────────────────────────────────
    if receive_to_send_ms > 5000:
        log.warning(f"[TIMING] receive_to_send_ms={receive_to_send_ms}ms EXCEEDS 5000ms target!")
    else:
        log.info(f"[TIMING] receive_to_send_ms={receive_to_send_ms}ms ✓ within 5s target")

    log.info(f"[send_sr] record_id={record_id} → {send_status} "
             f"({receive_to_send_ms}ms) | orthanc={orthanc_instance_id}")

    return {
        "success":             success,
        "sr_file_path":        sr_file_path,
        "orthanc_instance_id": orthanc_instance_id,
        "receive_to_send_ms":  receive_to_send_ms,
        "send_status":         send_status,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_record(record_id: int, db: "DatabaseManager | None") -> dict | None:
    """
    Fetch inference record + patient/study join from DB.
    Returns dict or None if not found.
    """
    if db is None:
        return None
    try:
        conn = db._get_connection()
        row = conn.execute("""
            SELECT
                ir.*,
                ds.patient_id,
                ds.study_date,
                ds.study_time,
                ds.description,
                p.patient_name,
                p.date_of_birth,
                p.sex
            FROM inference_records ir
            JOIN dicom_studies ds ON ir.study_uid = ds.study_uid
            JOIN patients p       ON ds.patient_id = p.patient_id
            WHERE ir.record_id = ?
        """, (record_id,)).fetchone()
        db._close(conn)
        return dict(row) if row else None
    except Exception as e:
        log.error(f"[_fetch_record] DB query failed: {e}")
        return None


def _log_failure(db, record_id, sr_file_path, error_msg, t_start):
    """Log a failed SR send to dicom_sr_log."""
    if db:
        ms = int((time.time() - t_start) * 1000)
        db.log_sr_send(
            record_id           = record_id,
            sr_file_path        = sr_file_path or "",
            orthanc_instance_id = None,
            status              = "FAILED",
            receive_to_send_ms  = ms,
            error_message       = error_msg,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE SELF-TEST
#  python send_dicom_results.py --test
#  Tests SR build + serialise without needing real DB or Orthanc
# ══════════════════════════════════════════════════════════════════════════════

def run_self_test():
    """
    Build and serialise a dummy SR for each modality.
    Does NOT require Orthanc to be running.
    Checks: SR builds without error, file written to disk, file is valid DICOM.
    """
    print("\n" + "═"*60)
    print("  send_dicom_results.py — M3 Day 6 Self-Test")
    print("═"*60)

    test_records = [
        {
            "record_id":         1,
            "study_uid":         "1.2.3.4.5.6.1",
            "patient_id":        "PT-1",
            "patient_name":      "Test Brain Patient",
            "date_of_birth":     "1985-03-15",
            "sex":               "M",
            "study_date":        "20260504",
            "study_time":        "120000",
            "modality":          "brain_mri",
            "predicted_class":   "Glioma",
            "mean_confidence":   92.5,
            "uncertainty_percent": 1.2,
            "routing_tier":      3,
            "ood_flag":          0,
            "gpio_alert_type":   "RED",
            "llm_report":        "Lesion consistent with high-grade glioma. Urgent neurosurgery referral recommended.",
            "inferred_at":       datetime.utcnow().isoformat(),
        },
        {
            "record_id":         2,
            "study_uid":         "1.2.3.4.5.6.2",
            "patient_id":        "PT-2",
            "patient_name":      "Test Chest Patient",
            "date_of_birth":     "1990-07-22",
            "sex":               "F",
            "study_date":        "20260504",
            "study_time":        "130000",
            "modality":          "chest_xray",
            "predicted_class":   "Pneumonia",
            "mean_confidence":   88.3,
            "uncertainty_percent": 3.1,
            "routing_tier":      3,
            "ood_flag":          0,
            "gpio_alert_type":   "RED",
            "llm_report":        "Bilateral infiltrates consistent with pneumonia. IV antibiotics recommended.",
            "inferred_at":       datetime.utcnow().isoformat(),
        },
        {
            "record_id":         3,
            "study_uid":         "1.2.3.4.5.6.3",
            "patient_id":        "PT-3",
            "patient_name":      "Test ECG Patient",
            "date_of_birth":     "1975-11-01",
            "sex":               "M",
            "study_date":        "20260504",
            "study_time":        "140000",
            "modality":          "ecg",
            "predicted_class":   "Normal",
            "mean_confidence":   96.1,
            "uncertainty_percent": 0.8,
            "routing_tier":      1,
            "ood_flag":          0,
            "gpio_alert_type":   "GREEN",
            "llm_report":        "ECG within normal limits. No significant arrhythmia detected.",
            "inferred_at":       datetime.utcnow().isoformat(),
        },
    ]

    passed = 0
    for rec in test_records:
        print(f"\n[TEST] patient={rec['patient_id']} modality={rec['modality']} "
              f"class={rec['predicted_class']}")
        try:
            # Build SR
            ds = build_sr_dataset(rec)
            assert ds.SOPClassUID    == SR_SOP_CLASS_UID, "Wrong SOP Class"
            assert ds.PatientID      == rec["patient_id"],  "Patient ID mismatch"
            assert ds.file_meta.TransferSyntaxUID == TRANSFER_SYNTAX, "Wrong transfer syntax"
            assert len(ds.ContentSequence) >= 9, "Too few content items"
            print(f"  ✓ SR dataset built — {len(ds.ContentSequence)} content items")

            # Serialise
            sr_path, sr_bytes = serialise_sr(ds, output_dir="./test_sr_outputs")
            assert os.path.exists(sr_path), "SR file not written to disk"
            assert len(sr_bytes) > 200, "SR bytes suspiciously small"
            print(f"  ✓ Serialised → {sr_path} ({len(sr_bytes)} bytes)")

            # Verify it reads back as valid DICOM
            ds_readback = pydicom.dcmread(sr_path)
            assert ds_readback.SOPClassUID == SR_SOP_CLASS_UID
            assert ds_readback.PatientID   == rec["patient_id"]
            print(f"  ✓ Verified DICOM readback OK")

            # Timing check
            print(f"  ✓ Transfer syntax: {ds.file_meta.TransferSyntaxUID} (ExplicitVRLittleEndian)")
            passed += 1

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'═'*60}")
    print(f"  Results: {passed}/{len(test_records)} tests passed")
    if passed == len(test_records):
        print("  All tests passed. send_dicom_results.py is ready.")
        print("  Note: Orthanc test requires --test-orthanc flag (Orthanc must be running)")
    print("═"*60 + "\n")


def run_orthanc_test(record_id: int, orthanc_url: str):
    """
    Full end-to-end test: fetch real DB record → build SR → send to Orthanc.
    Requires: fyp_medical.db to exist + Orthanc running on orthanc_url.
    """
    print(f"\n[ORTHANC TEST] record_id={record_id} → {orthanc_url}")
    db     = DatabaseManager() if DatabaseManager else None
    result = send_sr(record_id, db=db, orthanc_url=orthanc_url)
    print(f"  success            : {result['success']}")
    print(f"  send_status        : {result['send_status']}")
    print(f"  sr_file_path       : {result['sr_file_path']}")
    print(f"  orthanc_instance_id: {result['orthanc_instance_id']}")
    print(f"  receive_to_send_ms : {result['receive_to_send_ms']}ms "
          f"{'✓' if result['receive_to_send_ms'] < 5000 else '✗ EXCEEDS 5s TARGET'}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DICOM SR Sender — FYP Medical Imaging Pipeline (M3 Day 6)"
    )
    parser.add_argument("--test",         action="store_true",
                        help="Run SR build + serialise self-test (no Orthanc needed)")
    parser.add_argument("--test-orthanc", action="store_true",
                        help="Run full end-to-end test including Orthanc POST")
    parser.add_argument("--record-id",    type=int, default=1,
                        help="DB record_id to send as SR (used with --test-orthanc)")
    parser.add_argument("--orthanc",      default=ORTHANC_URL,
                        help=f"Orthanc base URL (default: {ORTHANC_URL})")
    parser.add_argument("--output-dir",   default=SR_OUTPUT_DIR,
                        help=f"Local SR output folder (default: {SR_OUTPUT_DIR})")
    args = parser.parse_args()

    if args.test:
        run_self_test()
    elif args.test_orthanc:
        run_orthanc_test(args.record_id, args.orthanc)
    else:
        print("Usage:")
        print("  python send_dicom_results.py --test")
        print("  python send_dicom_results.py --test-orthanc --record-id 1")
        print("  python send_dicom_results.py --test-orthanc --orthanc http://192.168.1.x:8042")