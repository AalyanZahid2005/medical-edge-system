"""
db_manager.py — FYP: Accelerated Medical Imaging using ML Architectures
Ahmed Rasheed (M3) — Day 2 Deliverable

Built to match ALL actual project files:
  - pipeline_skeleton.py  → payload keys: patient_id, modality, logits, features
  - mc_dropout_head.py    → mean_probs, std_probs, uncertainty > 5.0, Tier routing
  - flask_api.py          → /analyze, /analyze_patient, /correct contracts
  - generate_dummies.py   → fc_weights (3x1280), logits (1,3), features (1,1280,7,7)
  - gpio_alerts.py        → CRITICAL_FINDINGS list, uncertainty_percent > 5.0
  - batch_resize.py       → 224x224 images

Run unit tests:
    python db_manager.py
"""

import sqlite3
import os
import json
import unittest
from datetime import datetime

DB_PATH     = os.path.join(os.path.dirname(__file__), "fyp_medical.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "db_schema.sql")

# ── Matches gpio_alerts.py CRITICAL_FINDINGS list exactly ─────────────────────
CRITICAL_FINDINGS = [
    "Glioma", "Meningioma", "Pituitary",
    "1", "2", "3", "4",
    "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC",
    "Pneumonia",
    "Fracture",
    'L','R','B','A','a','J','S','V','r','F','e','j','n','E','/','f','Q','?'
]

# ══════════════════════════════════════════════════════════════════════════════
#  DatabaseManager
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    Central SQLite manager for the Medical Imaging pipeline.

    Aligned to:
      flask_api.py    → /analyze, /analyze_patient, /correct endpoints
      pipeline_skeleton.py → stage payloads (patient_id, modality, logits, features)
      mc_dropout_head.py   → mean_probs, std_probs arrays (20-pass MC Dropout)
      gpio_alerts.py       → CRITICAL_FINDINGS, uncertainty_percent > 5.0
      generate_dummies.py  → ONNX model specs, fc_weights shape
    """

    # ── Embedded schema (matches db_schema.sql exactly) ───────────────────────
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id    TEXT PRIMARY KEY,
            patient_name  TEXT,
            date_of_birth TEXT,
            sex           TEXT CHECK(sex IN ('M','F','O','U')) DEFAULT 'U',
            created_at    DATETIME DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS dicom_studies (
            study_uid       TEXT PRIMARY KEY,
            patient_id      TEXT NOT NULL,
            modality        TEXT NOT NULL CHECK(modality IN (
                                'brain_mri','retinal','skin','chest_xray','bone','ecg'
                            )),
            study_date      TEXT,
            study_time      TEXT,
            description     TEXT,
            dicom_file_path TEXT,
            received_at     DATETIME DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
        CREATE TABLE IF NOT EXISTS inference_records (
            record_id                INTEGER PRIMARY KEY AUTOINCREMENT,
            study_uid                TEXT NOT NULL,
            modality                 TEXT NOT NULL,
            predicted_class          TEXT,
            mean_confidence          REAL,
            uncertainty_percent      REAL,
            routing_tier             INTEGER CHECK(routing_tier IN (1,2,3)),
            mc_mean_probs            TEXT,
            mc_std_probs             TEXT,
            num_mc_passes            INTEGER DEFAULT 20,
            logits_raw               TEXT,
            features_shape           TEXT,
            ood_flag                 INTEGER DEFAULT 0,
            ood_reconstruction_error REAL,
            llm_report               TEXT,
            gradcam_base64           TEXT,
            gradcam_file_path        TEXT,
            gpio_triggered           INTEGER DEFAULT 0,
            gpio_alert_type          TEXT CHECK(gpio_alert_type IN ('GREEN','RED','NONE')) DEFAULT 'NONE',
            pipeline_ms              INTEGER,
            inferred_at              DATETIME DEFAULT (datetime('now')),
            FOREIGN KEY (study_uid) REFERENCES dicom_studies(study_uid)
        );
        CREATE TABLE IF NOT EXISTS fusion_results (
            fusion_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id        TEXT NOT NULL,
            modalities_used   TEXT,
            fusion_risk_score TEXT CHECK(fusion_risk_score IN ('Normal','Moderate','High','Critical')),
            fusion_details    TEXT,
            fusion_score_raw  REAL,
            fused_at          DATETIME DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
        CREATE TABLE IF NOT EXISTS active_learning_queue (
            queue_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id           INTEGER NOT NULL,
            uncertainty_trigger REAL,
            status              TEXT DEFAULT 'PENDING'
                                    CHECK(status IN ('PENDING','REVIEWED','CORRECTED','SKIPPED')),
            corrected_label     TEXT,
            correction_notes    TEXT,
            ewc_status          TEXT CHECK(ewc_status IN ('PENDING','RUNNING','COMPLETE','FAILED'))
                                    DEFAULT 'PENDING',
            ewc_message         TEXT,
            queued_at           DATETIME DEFAULT (datetime('now')),
            reviewed_at         DATETIME,
            FOREIGN KEY (record_id) REFERENCES inference_records(record_id)
        );
        CREATE TABLE IF NOT EXISTS dicom_sr_log (
            sr_id               INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id           INTEGER NOT NULL,
            sr_file_path        TEXT,
            orthanc_instance_id TEXT,
            transfer_syntax     TEXT DEFAULT 'ExplicitVRLittleEndian',
            send_status         TEXT DEFAULT 'PENDING'
                                    CHECK(send_status IN ('PENDING','SENT','FAILED')),
            receive_to_send_ms  INTEGER,
            sent_at             DATETIME,
            error_message       TEXT,
            FOREIGN KEY (record_id) REFERENCES inference_records(record_id)
        );
        CREATE TABLE IF NOT EXISTS pipeline_benchmark (
            bench_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            run_label       TEXT,
            test_duration_s REAL,
            total_frames    INTEGER,
            fps             REAL,
            stage1_ms       REAL,
            stage2_ms       REAL,
            stage3_ms       REAL,
            power_watts     REAL,
            temp_celsius    REAL,
            trt_precision   TEXT CHECK(trt_precision IN ('FP32','FP16','INT8')),
            recorded_at     DATETIME DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            modality              TEXT NOT NULL,
            model_type            TEXT CHECK(model_type IN ('ONNX','TRT_FP16','PYTORCH','OOD_AE')),
            file_path             TEXT NOT NULL,
            input_shape           TEXT,
            output_logits_shape   TEXT,
            output_features_shape TEXT,
            fc_weights_path       TEXT,
            num_classes           INTEGER,
            opset_version         INTEGER DEFAULT 17,
            is_active             INTEGER DEFAULT 1,
            registered_at         DATETIME DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS system_logs (
            log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            level      TEXT CHECK(level IN ('INFO','WARNING','ERROR','DEBUG','ALERT')) DEFAULT 'INFO',
            component  TEXT,
            message    TEXT,
            logged_at  DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_studies_patient  ON dicom_studies(patient_id);
        CREATE INDEX IF NOT EXISTS idx_studies_modality ON dicom_studies(modality);
        CREATE INDEX IF NOT EXISTS idx_infer_study      ON inference_records(study_uid);
        CREATE INDEX IF NOT EXISTS idx_infer_tier       ON inference_records(routing_tier);
        CREATE INDEX IF NOT EXISTS idx_infer_ood        ON inference_records(ood_flag);
        CREATE INDEX IF NOT EXISTS idx_infer_gpio       ON inference_records(gpio_triggered);
        CREATE INDEX IF NOT EXISTS idx_alq_status       ON active_learning_queue(status);
        CREATE INDEX IF NOT EXISTS idx_alq_ewc          ON active_learning_queue(ewc_status);
        CREATE INDEX IF NOT EXISTS idx_sr_status        ON dicom_sr_log(send_status);
        CREATE INDEX IF NOT EXISTS idx_model_modality   ON model_registry(modality);
        CREATE INDEX IF NOT EXISTS idx_logs_component   ON system_logs(component);
        CREATE INDEX IF NOT EXISTS idx_logs_level       ON system_logs(level);
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._shared_conn = None
        self._init_db()

    # ── Connection handling ────────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            conn = self._shared_conn
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # concurrent Flask + pipeline writes
        return conn

    def _close(self, conn):
        if self.db_path != ":memory:":
            conn.close()

    def _init_db(self):
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
        conn = self._get_connection()
        for stmt in self._SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
        self._close(conn)
        print(f"[DB] Ready → {self.db_path}")

    def _log(self, level: str, component: str, message: str):
        try:
            conn = self._get_connection()
            conn.execute(
                "INSERT INTO system_logs (level, component, message) VALUES (?,?,?)",
                (level, component, message)
            )
            conn.commit()
            self._close(conn)
        except Exception:
            pass  # logging must never crash the caller

    # ══════════════════════════════════════════════════════════════════════════
    #  PATIENTS
    # ══════════════════════════════════════════════════════════════════════════

    def upsert_patient(self, patient_id: str, patient_name: str = None,
                       dob: str = None, sex: str = "U") -> bool:
        """
        Insert or update a patient record.
        patient_id format: "PT-{n}" from pipeline_skeleton.py stage_1 payload.
        """
        sql = """
            INSERT INTO patients (patient_id, patient_name, date_of_birth, sex)
            VALUES (?,?,?,?)
            ON CONFLICT(patient_id) DO UPDATE SET
                patient_name  = excluded.patient_name,
                date_of_birth = excluded.date_of_birth,
                sex           = excluded.sex
        """
        try:
            conn = self._get_connection()
            conn.execute(sql, (patient_id, patient_name, dob, sex))
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager", f"Patient upserted: {patient_id}")
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"upsert_patient failed: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    #  DICOM STUDIES
    # ══════════════════════════════════════════════════════════════════════════

    def insert_study(self, study_uid: str, patient_id: str, modality: str,
                     dicom_file_path: str = None, description: str = None,
                     study_date: str = None, study_time: str = None) -> bool:
        """
        Insert a DICOM study record.
        Called by dicom_listener.py after routing an incoming .dcm file.
        modality must match pipeline_skeleton.py payload["modality"] values.
        """
        sql = """
            INSERT OR IGNORE INTO dicom_studies
                (study_uid, patient_id, modality, dicom_file_path,
                 description, study_date, study_time)
            VALUES (?,?,?,?,?,?,?)
        """
        try:
            conn = self._get_connection()
            conn.execute(sql, (study_uid, patient_id, modality,
                               dicom_file_path, description, study_date, study_time))
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager", f"Study inserted: {study_uid} [{modality}]")
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"insert_study failed: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    #  CORE — insert_record()
    #  Maps directly to flask_api.py /analyze response contract AND
    #  mc_dropout_head.py run_mc_inference() output
    # ══════════════════════════════════════════════════════════════════════════

    def insert_record(self, record: dict) -> int:
        """
        Insert one ML inference result from the 3-stage pipeline.

        Key mapping to project files:
        ┌─────────────────────────┬──────────────────────────────────────────┐
        │ Field                   │ Source                                   │
        ├─────────────────────────┼──────────────────────────────────────────┤
        │ study_uid               │ dicom_listener.py → DICOM StudyUID tag   │
        │ modality                │ pipeline_skeleton.py payload["modality"] │
        │ predicted_class         │ flask_api.py prediction.class            │
        │ mean_confidence         │ flask_api.py prediction.mean_confidence  │
        │ uncertainty_percent     │ flask_api.py prediction.uncertainty      │
        │                         │ gpio_alerts.py → trigger if > 5.0        │
        │ routing_tier            │ flask_api.py routing_tier (1/2/3)        │
        │ mc_mean_probs           │ mc_dropout_head.py mean_probs array      │
        │ mc_std_probs            │ mc_dropout_head.py std_probs array       │
        │ num_mc_passes           │ mc_dropout_head.py num_passes=20         │
        │ logits_raw              │ pipeline_skeleton.py dummy_logits (1,3)  │
        │ features_shape          │ pipeline_skeleton.py dummy_features      │
        │                         │ generate_dummies.py (1,1280,7,7)         │
        │ ood_flag                │ flask_api.py ood_flag                    │
        │ llm_report              │ flask_api.py llm_report (Phi-2 text)     │
        │ gradcam_base64          │ flask_api.py gradcam_base64              │
        │ gpio_triggered          │ gpio_alerts.py trigger_alert() result    │
        │ gpio_alert_type         │ gpio_alerts.py GREEN / RED               │
        │ pipeline_ms             │ Day 16 timing target <5000ms             │
        └─────────────────────────┴──────────────────────────────────────────┘

        Returns: new record_id (int > 0) or -1 on failure.
        """
        sql = """
            INSERT INTO inference_records (
                study_uid, modality, predicted_class,
                mean_confidence, uncertainty_percent, routing_tier,
                mc_mean_probs, mc_std_probs, num_mc_passes,
                logits_raw, features_shape,
                ood_flag, ood_reconstruction_error,
                llm_report, gradcam_base64, gradcam_file_path,
                gpio_triggered, gpio_alert_type, pipeline_ms
            ) VALUES (
                :study_uid, :modality, :predicted_class,
                :mean_confidence, :uncertainty_percent, :routing_tier,
                :mc_mean_probs, :mc_std_probs, :num_mc_passes,
                :logits_raw, :features_shape,
                :ood_flag, :ood_reconstruction_error,
                :llm_report, :gradcam_base64, :gradcam_file_path,
                :gpio_triggered, :gpio_alert_type, :pipeline_ms
            )
        """
        # ── Resolve routing_tier from gpio_alerts.py logic if not provided ──
        uncertainty = record.get("uncertainty_percent", 0.0)
        predicted   = record.get("predicted_class", "")

        if record.get("routing_tier") is None:
            if predicted in CRITICAL_FINDINGS or uncertainty > 5.0:
                auto_tier = 3   # RED alert
            elif uncertainty > 2.5:
                auto_tier = 2   # Review
            else:
                auto_tier = 1   # Auto-approve
        else:
            auto_tier = record["routing_tier"]

        # ── GPIO alert type from gpio_alerts.py trigger logic ───────────────
        if predicted in CRITICAL_FINDINGS or uncertainty > 5.0:
            gpio_type = "RED"
            gpio_trig = 1
        else:
            gpio_type = "GREEN"
            gpio_trig = 0

        full = {
            "study_uid":                record.get("study_uid"),
            "modality":                 record.get("modality"),
            "predicted_class":          predicted,
            "mean_confidence":          record.get("mean_confidence"),
            "uncertainty_percent":      uncertainty,
            "routing_tier":             auto_tier,
            # mc_dropout_head.py arrays → stored as JSON strings
            "mc_mean_probs":  json.dumps(record["mc_mean_probs"].tolist()
                              if hasattr(record.get("mc_mean_probs"), "tolist")
                              else record.get("mc_mean_probs")),
            "mc_std_probs":   json.dumps(record["mc_std_probs"].tolist()
                              if hasattr(record.get("mc_std_probs"), "tolist")
                              else record.get("mc_std_probs")),
            "num_mc_passes":            record.get("num_mc_passes", 20),
            # pipeline_skeleton.py Stage 2 tensor info
            "logits_raw":     json.dumps(record["logits_raw"].tolist()
                              if hasattr(record.get("logits_raw"), "tolist")
                              else record.get("logits_raw")),
            "features_shape":           record.get("features_shape", "1,1280,7,7"),
            # flask_api.py contract fields
            "ood_flag":                 int(record.get("ood_flag", False)),
            "ood_reconstruction_error": record.get("ood_reconstruction_error"),
            "llm_report":               record.get("llm_report"),
            "gradcam_base64":           record.get("gradcam_base64"),
            "gradcam_file_path":        record.get("gradcam_file_path"),
            # gpio_alerts.py result
            "gpio_triggered":           gpio_trig if record.get("gpio_triggered") is None else record["gpio_triggered"],
            "gpio_alert_type":          gpio_type if record.get("gpio_alert_type") is None else record["gpio_alert_type"],
            "pipeline_ms":              record.get("pipeline_ms"),
        }

        try:
            conn = self._get_connection()
            cursor = conn.execute(sql, full)
            new_id = cursor.lastrowid
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager",
                      f"Record inserted: id={new_id} class={predicted} "
                      f"conf={full['mean_confidence']} tier={auto_tier} gpio={gpio_type}")
            return new_id
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"insert_record failed: {e}")
            return -1

    # ══════════════════════════════════════════════════════════════════════════
    #  CORE — query_history()
    #  Used by flask_api.py /history endpoint and GUI patient tab
    # ══════════════════════════════════════════════════════════════════════════

    def query_history(self, patient_id: str, modality: str = None,
                      limit: int = 50) -> list:
        """
        Retrieve full inference history for a patient.
        Returns most recent first (ORDER BY record_id DESC).

        Args:
            patient_id : "PT-{n}" format from pipeline_skeleton.py
            modality   : optional filter — "brain_mri","retinal","skin",
                         "chest_xray","bone","ecg"
            limit      : max rows (default 50)

        Returns: list of dicts matching flask_api.py /analyze response contract
        """
        sql = """
            SELECT
                ir.record_id,
                ir.study_uid,
                ir.modality,
                ir.predicted_class,
                ir.mean_confidence,
                ir.uncertainty_percent,
                ir.routing_tier,
                ir.mc_mean_probs,
                ir.mc_std_probs,
                ir.ood_flag,
                ir.llm_report,
                ir.gradcam_file_path,
                ir.gpio_alert_type,
                ir.pipeline_ms,
                ir.inferred_at,
                ds.patient_id,
                ds.dicom_file_path
            FROM inference_records ir
            JOIN dicom_studies ds ON ir.study_uid = ds.study_uid
            WHERE ds.patient_id = ?
        """
        params = [patient_id]
        if modality:
            sql += " AND ir.modality = ?"
            params.append(modality)
        sql += " ORDER BY ir.record_id DESC LIMIT ?"
        params.append(limit)

        try:
            conn = self._get_connection()
            rows = conn.execute(sql, params).fetchall()
            self._close(conn)
            results = [dict(row) for row in rows]
            self._log("INFO", "db_manager",
                      f"query_history: {len(results)} rows for {patient_id}")
            return results
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"query_history failed: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════════════
    #  FUSION RESULTS
    #  Maps to flask_api.py /analyze_patient response contract
    # ══════════════════════════════════════════════════════════════════════════

    def insert_fusion_result(self, patient_id: str, modalities_used: list,
                             fusion_risk_score: str, fusion_details: str = None,
                             fusion_score_raw: float = None) -> int:
        """
        Store patient fusion MLP result.
        fusion_risk_score matches flask_api.py: "Normal"|"Moderate"|"High"|"Critical"
        modalities_used e.g. ["brain_mri", "chest_xray", "ecg"]
        """
        sql = """
            INSERT INTO fusion_results
                (patient_id, modalities_used, fusion_risk_score,
                 fusion_details, fusion_score_raw)
            VALUES (?,?,?,?,?)
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(sql, (
                patient_id,
                ",".join(modalities_used),
                fusion_risk_score,
                fusion_details,
                fusion_score_raw
            ))
            new_id = cursor.lastrowid
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager",
                      f"Fusion result inserted: id={new_id} patient={patient_id} risk={fusion_risk_score}")
            return new_id
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"insert_fusion_result failed: {e}")
            return -1

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTIVE LEARNING QUEUE
    #  Maps to flask_api.py /correct endpoint + mc_dropout_head.py uncertainty
    # ══════════════════════════════════════════════════════════════════════════

    def queue_for_review(self, record_id: int, uncertainty_trigger: float = None) -> bool:
        """
        Flag a high-uncertainty inference for clinician review.
        Called when mc_dropout_head.py std_probs > 5.0% (Tier 3 routing).
        uncertainty_trigger stores the exact std value that caused the flag.
        """
        sql = """
            INSERT OR IGNORE INTO active_learning_queue
                (record_id, uncertainty_trigger)
            VALUES (?,?)
        """
        try:
            conn = self._get_connection()
            conn.execute(sql, (record_id, uncertainty_trigger))
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager",
                      f"Queued record {record_id} for review (uncertainty={uncertainty_trigger})")
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"queue_for_review failed: {e}")
            return False

    def correct_label(self, record_id: int, corrected_label: str,
                      notes: str = None) -> bool:
        """
        Apply clinician correction from flask_api.py /correct endpoint.
        Triggers EWC fine-tuning loop (handled by M1's ewc_trainer.py).
        Sets ewc_status = 'PENDING' so M1 knows to pick it up.
        """
        sql = """
            UPDATE active_learning_queue
            SET status          = 'CORRECTED',
                corrected_label = ?,
                correction_notes= ?,
                ewc_status      = 'PENDING',
                ewc_message     = 'Awaiting EWC fine-tuning by ewc_trainer.py',
                reviewed_at     = datetime('now')
            WHERE record_id = ?
        """
        try:
            conn = self._get_connection()
            conn.execute(sql, (corrected_label, notes, record_id))
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager",
                      f"Label corrected: record={record_id} → '{corrected_label}'")
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"correct_label failed: {e}")
            return False

    def get_pending_queue(self) -> list:
        """
        Return all records awaiting clinician review.
        Used by GUI Active Learning Queue tab (M4 Day 8).
        """
        sql = """
            SELECT alq.*, ir.modality, ir.predicted_class,
                   ir.mean_confidence, ir.uncertainty_percent
            FROM active_learning_queue alq
            JOIN inference_records ir ON alq.record_id = ir.record_id
            WHERE alq.status = 'PENDING'
            ORDER BY alq.queued_at ASC
        """
        try:
            conn = self._get_connection()
            rows = conn.execute(sql).fetchall()
            self._close(conn)
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"get_pending_queue failed: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════════════
    #  DICOM SR LOG
    #  Maps to send_dicom_results.py (Day 6) ExplicitVRLittleEndian
    # ══════════════════════════════════════════════════════════════════════════

    def log_sr_send(self, record_id: int, sr_file_path: str,
                    orthanc_instance_id: str = None,
                    status: str = "SENT",
                    receive_to_send_ms: int = None,
                    error_message: str = None) -> bool:
        """
        Log every DICOM SR sent back to Orthanc.
        receive_to_send_ms = end-to-end timing for Day 16 <5s target.
        transfer_syntax defaults to ExplicitVRLittleEndian (send_dicom_results.py).
        """
        sql = """
            INSERT INTO dicom_sr_log
                (record_id, sr_file_path, orthanc_instance_id,
                 send_status, receive_to_send_ms, sent_at, error_message)
            VALUES (?,?,?,?,?,datetime('now'),?)
        """
        try:
            conn = self._get_connection()
            conn.execute(sql, (record_id, sr_file_path, orthanc_instance_id,
                               status, receive_to_send_ms, error_message))
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager",
                      f"SR logged: record={record_id} status={status} ms={receive_to_send_ms}")
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"log_sr_send failed: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════════
    #  PIPELINE BENCHMARK
    #  Maps directly to pipeline_skeleton.py run_throughput_test() output
    # ══════════════════════════════════════════════════════════════════════════

    def log_benchmark(self, run_label: str, test_duration_s: float,
                      total_frames: int, fps: float,
                      stage1_ms: float = 15.0,   # pipeline_skeleton simulated sleep
                      stage2_ms: float = 45.0,   # pipeline_skeleton simulated sleep
                      stage3_ms: float = 30.0,   # pipeline_skeleton simulated sleep
                      power_watts: float = None,
                      temp_celsius: float = None,
                      trt_precision: str = "FP16") -> int:
        """
        Store one benchmark run from pipeline_skeleton.py run_throughput_test().

        pipeline_skeleton.py output maps:
            processed_frames → total_frames
            total_time       → test_duration_s
            fps              → fps
            Stage sleeps     → stage1/2/3_ms (15ms, 45ms, 30ms simulated)

        Day 15: M1 fills power_watts + temp_celsius from real tegrastats CSV.
        """
        sql = """
            INSERT INTO pipeline_benchmark
                (run_label, test_duration_s, total_frames, fps,
                 stage1_ms, stage2_ms, stage3_ms,
                 power_watts, temp_celsius, trt_precision)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(sql, (
                run_label, test_duration_s, total_frames, fps,
                stage1_ms, stage2_ms, stage3_ms,
                power_watts, temp_celsius, trt_precision
            ))
            new_id = cursor.lastrowid
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager",
                      f"Benchmark logged: {run_label} → {fps:.2f} FPS")
            return new_id
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"log_benchmark failed: {e}")
            return -1

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL REGISTRY
    #  Maps to generate_dummies.py output files
    # ══════════════════════════════════════════════════════════════════════════

    def register_model(self, modality: str, model_type: str, file_path: str,
                       fc_weights_path: str = None, num_classes: int = 3,
                       input_shape: str = "1,3,224,224",
                       output_logits_shape: str = "1,3",
                       output_features_shape: str = "1,1280,7,7",
                       opset_version: int = 17) -> int:
        """
        Register an ONNX or TRT model in the registry.

        Defaults match generate_dummies.py:
          input_shape          = "1,3,224,224"  (batch_resize.py target size)
          output_logits_shape  = "1,3"          (3 brain tumor classes)
          output_features_shape= "1,1280,7,7"   (EfficientNet B0 feature map)
          fc_weights_path      = "dummy_fc_weights.npy"  (shape 3x1280)
          opset_version        = 17
        """
        sql = """
            INSERT INTO model_registry
                (modality, model_type, file_path, fc_weights_path, num_classes,
                 input_shape, output_logits_shape, output_features_shape, opset_version)
            VALUES (?,?,?,?,?,?,?,?,?)
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(sql, (
                modality, model_type, file_path, fc_weights_path, num_classes,
                input_shape, output_logits_shape, output_features_shape, opset_version
            ))
            new_id = cursor.lastrowid
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager",
                      f"Model registered: {modality} [{model_type}] → {file_path}")
            return new_id
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"register_model failed: {e}")
            return -1


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
#  Tests use dummy data that exactly matches your project files
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseManager(unittest.TestCase):
    """
    Day 2 unit tests for insert_record() and query_history().
    Dummy data sourced from actual project files:
      - pipeline_skeleton.py Stage payloads
      - mc_dropout_head.py output format
      - flask_api.py /analyze response contract
      - gpio_alerts.py CRITICAL_FINDINGS
      - generate_dummies.py tensor shapes
    """

    def setUp(self):
        self.db = DatabaseManager(db_path=":memory:")
        # Seed patient + study (matches pipeline_skeleton.py format)
        self.db.upsert_patient("PT-1", "Ahmed Test", "1990-01-01", "M")
        self.db.insert_study("STUDY-001", "PT-1", "brain_mri",
                             description="Brain MRI Scan")

    def _base_record(self, **overrides):
        """Base record matching pipeline_skeleton + mc_dropout_head output."""
        import numpy as np
        r = {
            "study_uid":          "STUDY-001",
            "modality":           "brain_mri",
            # flask_api.py /analyze contract
            "predicted_class":    "Glioma",
            "mean_confidence":    92.5,
            "uncertainty_percent": 1.2,
            "routing_tier":       1,
            "ood_flag":           False,
            "llm_report":         "Lesion consistent with glioma. AI confidence: 92.5%.",
            "gradcam_base64":     "data:image/jpeg;base64,AAAA",
            # mc_dropout_head.py run_mc_inference() output
            "mc_mean_probs":  np.array([0.925, 0.050, 0.025]),
            "mc_std_probs":   np.array([0.012, 0.008, 0.004]),
            "num_mc_passes":  20,
            # pipeline_skeleton.py Stage 2 output
            "logits_raw":     np.array([[1.2, -0.5, -0.7]]),
            "features_shape": "1,1280,7,7",
            # gpio_alerts.py result
            "gpio_triggered":  1,
            "gpio_alert_type": "RED",
            "pipeline_ms":    90,
        }
        r.update(overrides)
        return r

    # ── insert_record() ────────────────────────────────────────────────────

    def test_insert_record_returns_positive_id(self):
        """insert_record() must return positive integer on success."""
        rid = self.db.insert_record(self._base_record())
        self.assertGreater(rid, 0)

    def test_insert_record_stores_flask_contract_fields(self):
        """All flask_api.py /analyze contract fields must be stored correctly."""
        rid = self.db.insert_record(self._base_record())
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT * FROM inference_records WHERE record_id=?", (rid,)
        ).fetchone()
        self.assertEqual(row["predicted_class"],     "Glioma")
        self.assertAlmostEqual(row["mean_confidence"],    92.5,  places=1)
        self.assertAlmostEqual(row["uncertainty_percent"], 1.2,  places=1)
        self.assertEqual(row["routing_tier"],        1)
        self.assertEqual(row["ood_flag"],            0)

    def test_insert_record_stores_mc_dropout_arrays(self):
        """mc_dropout_head.py numpy arrays must be stored as JSON strings."""
        import numpy as np
        rid = self.db.insert_record(self._base_record())
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT mc_mean_probs, mc_std_probs FROM inference_records WHERE record_id=?",
            (rid,)
        ).fetchone()
        mean = json.loads(row["mc_mean_probs"])
        std  = json.loads(row["mc_std_probs"])
        self.assertAlmostEqual(mean[0], 0.925, places=3)
        self.assertAlmostEqual(std[0],  0.012, places=3)

    def test_insert_record_stores_pipeline_skeleton_fields(self):
        """pipeline_skeleton.py Stage 2 fields must be stored."""
        rid = self.db.insert_record(self._base_record())
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT logits_raw, features_shape, pipeline_ms FROM inference_records WHERE record_id=?",
            (rid,)
        ).fetchone()
        logits = json.loads(row["logits_raw"])
        self.assertEqual(len(logits[0]), 3)          # shape (1,3)
        self.assertEqual(row["features_shape"], "1,1280,7,7")
        self.assertEqual(row["pipeline_ms"],    90)

    def test_insert_record_gpio_critical_finding(self):
        """gpio_alerts.py: Glioma in CRITICAL_FINDINGS → RED alert auto-set."""
        rid = self.db.insert_record(self._base_record(
            predicted_class="Glioma",
            routing_tier=None,       # let db_manager auto-compute
            gpio_triggered=None,
            gpio_alert_type=None
        ))
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT routing_tier, gpio_alert_type, gpio_triggered FROM inference_records WHERE record_id=?",
            (rid,)
        ).fetchone()
        self.assertEqual(row["gpio_alert_type"], "RED")
        self.assertEqual(row["gpio_triggered"],  1)
        self.assertEqual(row["routing_tier"],    3)

    def test_insert_record_gpio_normal_finding(self):
        """gpio_alerts.py: Normal + low uncertainty → GREEN alert."""
        rid = self.db.insert_record(self._base_record(
            predicted_class="Normal Skin",
            uncertainty_percent=1.2,
            routing_tier=None,
            gpio_triggered=None,
            gpio_alert_type=None
        ))
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT gpio_alert_type, gpio_triggered, routing_tier FROM inference_records WHERE record_id=?",
            (rid,)
        ).fetchone()
        self.assertEqual(row["gpio_alert_type"], "GREEN")
        self.assertEqual(row["gpio_triggered"],  0)
        self.assertEqual(row["routing_tier"],    1)

    def test_insert_record_high_uncertainty_tier3(self):
        """mc_dropout_head.py: uncertainty > 5.0 → Tier 3 even if class is not critical."""
        rid = self.db.insert_record(self._base_record(
            predicted_class="Normal",
            uncertainty_percent=6.8,   # gpio_alerts.py threshold: > 5.0
            routing_tier=None,
            gpio_triggered=None,
            gpio_alert_type=None
        ))
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT routing_tier, gpio_alert_type FROM inference_records WHERE record_id=?",
            (rid,)
        ).fetchone()
        self.assertEqual(row["routing_tier"],    3)
        self.assertEqual(row["gpio_alert_type"], "RED")

    def test_insert_record_bad_study_uid_returns_minus1(self):
        """FK violation on bad study_uid must return -1."""
        result = self.db.insert_record(self._base_record(study_uid="BAD-UID"))
        self.assertEqual(result, -1)

    # ── query_history() ────────────────────────────────────────────────────

    def test_query_history_returns_list(self):
        self.assertIsInstance(self.db.query_history("PT-1"), list)

    def test_query_history_matches_inserted_record(self):
        """Record inserted must appear in patient history with correct fields."""
        self.db.insert_record(self._base_record())
        rows = self.db.query_history("PT-1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["predicted_class"],  "Glioma")
        self.assertEqual(rows[0]["patient_id"],       "PT-1")
        self.assertEqual(rows[0]["modality"],         "brain_mri")

    def test_query_history_modality_filter(self):
        """Modality filter must exclude non-matching modalities."""
        self.db.insert_record(self._base_record())
        self.db.insert_study("STUDY-002", "PT-1", "chest_xray")
        self.db.insert_record(self._base_record(
            study_uid="STUDY-002", modality="chest_xray",
            predicted_class="Pneumonia"
        ))
        brain_only = self.db.query_history("PT-1", modality="brain_mri")
        self.assertEqual(len(brain_only), 1)
        self.assertEqual(brain_only[0]["modality"], "brain_mri")

    def test_query_history_unknown_patient_empty(self):
        self.assertEqual(self.db.query_history("PT-UNKNOWN"), [])

    def test_query_history_most_recent_first(self):
        """Most recent record_id must appear first (DESC order)."""
        rid1 = self.db.insert_record(self._base_record(predicted_class="First"))
        rid2 = self.db.insert_record(self._base_record(predicted_class="Second"))
        self.assertGreater(rid2, rid1)
        rows = self.db.query_history("PT-1")
        self.assertEqual(rows[0]["record_id"], rid2)

    # ── Fusion result (flask_api.py /analyze_patient) ──────────────────────

    def test_insert_fusion_result(self):
        """Fusion result matches flask_api.py /analyze_patient contract."""
        fid = self.db.insert_fusion_result(
            patient_id        = "PT-1",
            modalities_used   = ["brain_mri", "chest_xray"],
            fusion_risk_score = "Critical",      # flask contract value
            fusion_details    = "Patient exhibits multi-modal Glioma and Pneumonia signatures.",
            fusion_score_raw  = 0.91
        )
        self.assertGreater(fid, 0)
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT * FROM fusion_results WHERE fusion_id=?", (fid,)
        ).fetchone()
        self.assertEqual(row["fusion_risk_score"], "Critical")
        self.assertIn("brain_mri", row["modalities_used"])

    # ── Active learning queue (/correct endpoint) ──────────────────────────

    def test_queue_and_correct_label_ewc_flow(self):
        """
        Full /correct flow:
        queue_for_review → get_pending_queue → correct_label → queue empty
        EWC status set to PENDING for M1's ewc_trainer.py.
        """
        rid = self.db.insert_record(self._base_record(
            uncertainty_percent=6.5   # above 5.0 → should be queued
        ))
        self.db.queue_for_review(rid, uncertainty_trigger=6.5)
        pending = self.db.get_pending_queue()
        self.assertEqual(len(pending), 1)
        self.assertAlmostEqual(pending[0]["uncertainty_trigger"], 6.5, places=1)

        # Clinician corrects via flask_api.py /correct
        self.db.correct_label(rid, "Meningioma", "Radiologist override")
        # EWC should be pending for M1
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT ewc_status, corrected_label FROM active_learning_queue WHERE record_id=?",
            (rid,)
        ).fetchone()
        self.assertEqual(row["ewc_status"],      "PENDING")
        self.assertEqual(row["corrected_label"], "Meningioma")
        # Queue now empty
        self.assertEqual(len(self.db.get_pending_queue()), 0)

    # ── Benchmark (pipeline_skeleton.py run_throughput_test) ──────────────

    def test_log_benchmark_pipeline_skeleton_output(self):
        """
        Benchmark fields match pipeline_skeleton.py run_throughput_test() output.
        processed_frames=312, total_time=10.02s → fps=31.14
        """
        bid = self.db.log_benchmark(
            run_label      = "mock_numpy_test",
            test_duration_s= 10.02,
            total_frames   = 312,
            fps            = 31.14,
            stage1_ms      = 15.0,   # pipeline_skeleton.py time.sleep(0.015)
            stage2_ms      = 45.0,   # pipeline_skeleton.py time.sleep(0.045)
            stage3_ms      = 30.0,   # pipeline_skeleton.py time.sleep(0.030)
        )
        self.assertGreater(bid, 0)
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT * FROM pipeline_benchmark WHERE bench_id=?", (bid,)
        ).fetchone()
        self.assertAlmostEqual(row["fps"],       31.14, places=2)
        self.assertEqual(row["total_frames"],    312)
        self.assertAlmostEqual(row["stage2_ms"], 45.0,  places=1)

    # ── Model registry (generate_dummies.py) ──────────────────────────────

    def test_register_dummy_onnx_model(self):
        """
        Model registration must match generate_dummies.py specs:
          file: dummy_dual_output.onnx
          logits output: (1,3), features: (1,1280,7,7), opset=17
        """
        mid = self.db.register_model(
            modality              = "brain_mri",
            model_type            = "ONNX",
            file_path             = "dummy_dual_output.onnx",
            fc_weights_path       = "dummy_fc_weights.npy",
            num_classes           = 3,
            input_shape           = "1,3,224,224",     # batch_resize.py target
            output_logits_shape   = "1,3",
            output_features_shape = "1,1280,7,7",
            opset_version         = 17                 # generate_dummies.py value
        )
        self.assertGreater(mid, 0)
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT * FROM model_registry WHERE model_id=?", (mid,)
        ).fetchone()
        self.assertEqual(row["file_path"],             "dummy_dual_output.onnx")
        self.assertEqual(row["output_features_shape"], "1,1280,7,7")
        self.assertEqual(row["opset_version"],         17)