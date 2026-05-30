"""
db_manager.py — WORKING VERSION for M3 Networking/Backend

This file is intentionally clean and import-safe.
It creates/uses fyp_medical.db and provides the methods used by:
- flask_api.py
- dicom_listener.py
- dicom_cfind.py
- confidence_router.py
- active_learning_auto.py
- ewc_trainer.py
- send_dicom_results.py
"""

import os
import json
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "fyp_medical.db")

CRITICAL_FINDINGS = [
    "Glioma", "Meningioma", "Pituitary",
    "1", "2", "3", "4",
    "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC",
    "Pneumonia",
    "Fracture",
    "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
]


class DatabaseManager:
    """
    SQLite manager for the M3/backend part.
    """

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
        record_id           INTEGER NOT NULL UNIQUE,
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

        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON")

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            conn = self._shared_conn
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)

        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")

        return conn

    def _close(self, conn):
        if self.db_path != ":memory:" and conn is not None:
            conn.close()

    def _init_db(self):
        conn = self._get_connection()
        try:
            conn.executescript(self._SCHEMA)
            conn.commit()
        finally:
            self._close(conn)

        print(f"[DB] Ready -> {self.db_path}")

    def _json(self, value):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        return json.dumps(value)

    def _log(self, level: str, component: str, message: str):
        try:
            conn = self._get_connection()
            conn.execute(
                "INSERT INTO system_logs (level, component, message) VALUES (?,?,?)",
                (str(level), str(component), str(message))
            )
            conn.commit()
            self._close(conn)
        except Exception:
            pass

    def log_event(self, level: str, component: str, message: str):
        self._log(level, component, message)

    def update_ewc_status(self, record_id: int, status, message: str = None) -> bool:
        status = status.value if hasattr(status, "value") else str(status)
        try:
            conn = self._get_connection()
            conn.execute(
                '''
                UPDATE active_learning_queue
                SET ewc_status = ?, ewc_message = ?
                WHERE record_id = ?
                ''',
                (status, message, record_id)
            )
            conn.commit()
            self._close(conn)
            return True
        except Exception as e:
            print(f"[DB ERROR] update_ewc_status failed: {e}")
            return False

    def upsert_patient(self, patient_id: str, patient_name: str = None,
                       dob: str = None, sex: str = "U", **kwargs) -> bool:
        if patient_name is None and "name" in kwargs:
            patient_name = kwargs["name"]

        if not patient_id:
            patient_id = "PT-UNKNOWN"

        if sex not in ("M", "F", "O", "U"):
            sex = "U"

        sql = '''
        INSERT INTO patients (patient_id, patient_name, date_of_birth, sex)
        VALUES (?,?,?,?)
        ON CONFLICT(patient_id) DO UPDATE SET
            patient_name  = COALESCE(excluded.patient_name, patients.patient_name),
            date_of_birth = COALESCE(excluded.date_of_birth, patients.date_of_birth),
            sex           = excluded.sex
        '''
        try:
            conn = self._get_connection()
            conn.execute(sql, (patient_id, patient_name, dob, sex))
            conn.commit()
            self._close(conn)
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"upsert_patient failed: {e}")
            return False

    def insert_study(self, study_uid: str, patient_id: str, modality: str,
                     dicom_file_path: str = None, description: str = None,
                     study_date: str = None, study_time: str = None) -> bool:
        if not study_uid:
            study_uid = f"STUDY-{int(datetime.utcnow().timestamp() * 1000)}"

        if not patient_id:
            patient_id = "PT-UNKNOWN"

        if modality not in ("brain_mri", "retinal", "skin", "chest_xray", "bone", "ecg"):
            modality = "brain_mri"

        self.upsert_patient(patient_id)

        sql = '''
        INSERT OR IGNORE INTO dicom_studies
            (study_uid, patient_id, modality, dicom_file_path, description, study_date, study_time)
        VALUES (?,?,?,?,?,?,?)
        '''
        try:
            conn = self._get_connection()
            conn.execute(sql, (
                study_uid, patient_id, modality, dicom_file_path,
                description, study_date, study_time
            ))
            conn.commit()
            self._close(conn)
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"insert_study failed: {e}")
            return False

    def insert_record(self, record: dict) -> int:
        if record is None:
            return -1

        study_uid = record.get("study_uid")
        modality = record.get("modality", "brain_mri")
        predicted = record.get("predicted_class", "Unknown")
        uncertainty = float(record.get("uncertainty_percent") or 0.0)
        confidence = record.get("mean_confidence")

        routing_tier = record.get("routing_tier")
        if routing_tier is None:
            if record.get("ood_flag", False) or predicted in CRITICAL_FINDINGS or uncertainty > 10.0:
                routing_tier = 3
            elif uncertainty >= 5.0:
                routing_tier = 2
            else:
                routing_tier = 1

        gpio_alert_type = record.get("gpio_alert_type")
        gpio_triggered = record.get("gpio_triggered")

        if gpio_alert_type is None:
            gpio_alert_type = "RED" if routing_tier == 3 else "GREEN"

        if gpio_triggered is None:
            gpio_triggered = 1 if gpio_alert_type == "RED" else 0

        full = {
            "study_uid": study_uid,
            "modality": modality,
            "predicted_class": predicted,
            "mean_confidence": confidence,
            "uncertainty_percent": uncertainty,
            "routing_tier": routing_tier,
            "mc_mean_probs": self._json(record.get("mc_mean_probs")),
            "mc_std_probs": self._json(record.get("mc_std_probs")),
            "num_mc_passes": record.get("num_mc_passes", 20),
            "logits_raw": self._json(record.get("logits_raw")),
            "features_shape": record.get("features_shape", "1,1280,7,7"),
            "ood_flag": int(bool(record.get("ood_flag", False))),
            "ood_reconstruction_error": record.get("ood_reconstruction_error"),
            "llm_report": record.get("llm_report"),
            "gradcam_base64": record.get("gradcam_base64"),
            "gradcam_file_path": record.get("gradcam_file_path"),
            "gpio_triggered": gpio_triggered,
            "gpio_alert_type": gpio_alert_type,
            "pipeline_ms": record.get("pipeline_ms"),
        }

        sql = '''
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
        '''

        try:
            conn = self._get_connection()
            cur = conn.execute(sql, full)
            record_id = cur.lastrowid
            conn.commit()
            self._close(conn)
            self._log("INFO", "db_manager", f"Record inserted: {record_id}")
            return int(record_id)
        except sqlite3.Error as e:
            print(f"[DB ERROR] insert_record failed: {e}")
            self._log("ERROR", "db_manager", f"insert_record failed: {e}")
            return -1

    def query_history(self, patient_id: str, modality: str = None, limit: int = 50) -> list:
        sql = '''
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
            ds.dicom_file_path,
            ds.study_date,
            ds.study_time,
            ds.description,
            p.patient_name,
            p.date_of_birth,
            p.sex
        FROM inference_records ir
        JOIN dicom_studies ds ON ir.study_uid = ds.study_uid
        JOIN patients p ON ds.patient_id = p.patient_id
        WHERE ds.patient_id = ?
        '''
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
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"query_history failed: {e}")
            return []

    def insert_fusion_result(self, patient_id: str, modalities_used: list,
                             fusion_risk_score: str, fusion_details: str = None,
                             fusion_score_raw: float = None) -> int:
        try:
            conn = self._get_connection()
            cur = conn.execute(
                '''
                INSERT INTO fusion_results
                    (patient_id, modalities_used, fusion_risk_score, fusion_details, fusion_score_raw)
                VALUES (?,?,?,?,?)
                ''',
                (
                    patient_id,
                    ",".join(modalities_used or []),
                    fusion_risk_score,
                    fusion_details,
                    fusion_score_raw
                )
            )
            fid = cur.lastrowid
            conn.commit()
            self._close(conn)
            return int(fid)
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"insert_fusion_result failed: {e}")
            return -1

    def queue_for_review(self, record_id: int, uncertainty_trigger: float = None) -> bool:
        try:
            conn = self._get_connection()
            conn.execute(
                '''
                INSERT OR IGNORE INTO active_learning_queue (record_id, uncertainty_trigger)
                VALUES (?,?)
                ''',
                (record_id, uncertainty_trigger)
            )
            conn.commit()
            self._close(conn)
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"queue_for_review failed: {e}")
            return False

    def correct_label(self, record_id: int, corrected_label: str, notes: str = None) -> bool:
        try:
            conn = self._get_connection()
            conn.execute(
                '''
                UPDATE active_learning_queue
                SET status = 'CORRECTED',
                    corrected_label = ?,
                    correction_notes = ?,
                    ewc_status = 'PENDING',
                    ewc_message = 'Awaiting EWC fine-tuning',
                    reviewed_at = datetime('now')
                WHERE record_id = ?
                ''',
                (corrected_label, notes, record_id)
            )
            conn.commit()
            self._close(conn)
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"correct_label failed: {e}")
            return False

    def get_pending_queue(self) -> list:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                '''
                SELECT alq.*, ir.modality, ir.predicted_class,
                       ir.mean_confidence, ir.uncertainty_percent
                FROM active_learning_queue alq
                JOIN inference_records ir ON alq.record_id = ir.record_id
                WHERE alq.status = 'PENDING'
                ORDER BY alq.uncertainty_trigger DESC, alq.queued_at ASC
                '''
            ).fetchall()
            self._close(conn)
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"get_pending_queue failed: {e}")
            return []

    def log_sr_send(self, record_id: int, sr_file_path: str,
                    orthanc_instance_id: str = None, status: str = "SENT",
                    receive_to_send_ms: int = None, error_message: str = None) -> bool:
        try:
            conn = self._get_connection()
            conn.execute(
                '''
                INSERT INTO dicom_sr_log
                    (record_id, sr_file_path, orthanc_instance_id,
                     send_status, receive_to_send_ms, sent_at, error_message)
                VALUES (?,?,?,?,?,datetime('now'),?)
                ''',
                (
                    record_id,
                    sr_file_path,
                    orthanc_instance_id,
                    status,
                    receive_to_send_ms,
                    error_message
                )
            )
            conn.commit()
            self._close(conn)
            return True
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"log_sr_send failed: {e}")
            return False

    def log_benchmark(self, run_label: str, test_duration_s: float,
                      total_frames: int, fps: float,
                      stage1_ms: float = 15.0,
                      stage2_ms: float = 45.0,
                      stage3_ms: float = 30.0,
                      power_watts: float = None,
                      temp_celsius: float = None,
                      trt_precision: str = "FP16") -> int:
        try:
            conn = self._get_connection()
            cur = conn.execute(
                '''
                INSERT INTO pipeline_benchmark
                    (run_label, test_duration_s, total_frames, fps,
                     stage1_ms, stage2_ms, stage3_ms,
                     power_watts, temp_celsius, trt_precision)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ''',
                (
                    run_label, test_duration_s, total_frames, fps,
                    stage1_ms, stage2_ms, stage3_ms,
                    power_watts, temp_celsius, trt_precision
                )
            )
            bid = cur.lastrowid
            conn.commit()
            self._close(conn)
            return int(bid)
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"log_benchmark failed: {e}")
            return -1

    def register_model(self, modality: str, model_type: str, file_path: str,
                       fc_weights_path: str = None, num_classes: int = 3,
                       input_shape: str = "1,3,224,224",
                       output_logits_shape: str = "1,3",
                       output_features_shape: str = "1,1280,7,7",
                       opset_version: int = 17) -> int:
        try:
            conn = self._get_connection()
            cur = conn.execute(
                '''
                INSERT INTO model_registry
                    (modality, model_type, file_path, fc_weights_path, num_classes,
                     input_shape, output_logits_shape, output_features_shape, opset_version)
                VALUES (?,?,?,?,?,?,?,?,?)
                ''',
                (
                    modality, model_type, file_path, fc_weights_path, num_classes,
                    input_shape, output_logits_shape, output_features_shape, opset_version
                )
            )
            mid = cur.lastrowid
            conn.commit()
            self._close(conn)
            return int(mid)
        except sqlite3.Error as e:
            self._log("ERROR", "db_manager", f"register_model failed: {e}")
            return -1


if __name__ == "__main__":
    db = DatabaseManager()
    print("[OK] db_manager.py imported and database initialized successfully.")

    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    conn.close()

    print("[OK] Tables:")
    for t in tables:
        print(" -", t)
# ============================================================
# Compatibility wrapper for master_pipeline_v2.py
# master_pipeline_v2.py expects:
# db_manager.insert_record(patient_id, pred_name, report, conf_pct, routing)
# ============================================================


def insert_record(patient_id, diagnosis=None, report=None, confidence=0.0, routing="N/A", **kwargs):
    """
    Final compatibility wrapper used by Flask/master_pipeline_v2.

    Fix:
    - Older wrapper tried inserting patient_id directly into inference_records.
    - Current schema stores patient_id in dicom_studies and links inference_records by study_uid.
    - This function creates/links patient + study, inserts inference record, and returns a real record_id.
    """
    import os
    import sqlite3
    import time
    import uuid
    import json

    # Support dict-style call too.
    if isinstance(patient_id, dict):
        rec = patient_id
        patient_id = rec.get("patient_id") or rec.get("patient") or "PT-UNKNOWN"
        diagnosis = rec.get("predicted_class") or rec.get("diagnosis") or rec.get("prediction") or diagnosis or "UNKNOWN"
        report = rec.get("llm_report") or rec.get("report") or rec.get("clinical_report") or report or ""
        confidence = rec.get("mean_confidence") or rec.get("confidence") or rec.get("confidence_percent") or confidence or 0.0
        routing = rec.get("routing_decision") or rec.get("routing") or rec.get("routing_tier") or routing or "N/A"
        modality = rec.get("modality") or kwargs.get("modality") or "unknown"
        uncertainty = rec.get("uncertainty") or rec.get("uncertainty_percent") or rec.get("std_dev") or 0.0
        ood_flag = rec.get("ood_flag", False)
        gpio_alert_type = rec.get("gpio_alert_type", "NONE")
        gpio_triggered = rec.get("gpio_triggered", 0)
        pipeline_ms = rec.get("pipeline_ms") or rec.get("latency_ms")
    else:
        patient_id = patient_id or "PT-UNKNOWN"
        diagnosis = diagnosis or "UNKNOWN"
        report = report or ""
        modality = kwargs.get("modality", "unknown")
        uncertainty = kwargs.get("uncertainty", kwargs.get("uncertainty_percent", 0.0))
        ood_flag = kwargs.get("ood_flag", False)
        gpio_alert_type = kwargs.get("gpio_alert_type", "NONE")
        gpio_triggered = kwargs.get("gpio_triggered", 0)
        pipeline_ms = kwargs.get("pipeline_ms")

    try:
        confidence = float(confidence or 0.0)
    except Exception:
        confidence = 0.0

    try:
        uncertainty = float(uncertainty or 0.0)
    except Exception:
        uncertainty = 0.0

    # Normalize routing tier for DB schema:
    # inference_records.routing_tier is INTEGER CHECK(routing_tier IN (1,2,3))
    routing_text = str(routing or "")
    if "Tier 1" in routing_text or "Auto-SR" in routing_text:
        routing_tier_value = 1
    elif "Tier 2" in routing_text or "AI-Assisted" in routing_text:
        routing_tier_value = 2
    elif "Tier 3" in routing_text or "Urgent" in routing_text or "Rejected" in routing_text:
        routing_tier_value = 3
    else:
        if confidence >= 90.0:
            routing_tier_value = 1
        elif confidence >= 70.0:
            routing_tier_value = 2
        else:
            routing_tier_value = 3

    # Normalize GPIO alert.
    # We will migrate DB to allow YELLOW too.
    gpio_alert_type = str(gpio_alert_type or "NONE").upper()
    if gpio_alert_type not in ("GREEN", "YELLOW", "RED", "NONE"):
        gpio_alert_type = "NONE"

    db_path = (
        os.environ.get("MEDAI_DB_PATH")
        or os.environ.get("DB_PATH")
        or ("/app/fyp_medical.db" if os.path.exists("/app") else "fyp_medical.db")
    )

    study_uid = kwargs.get("study_uid") or "STUDY-{}-{}".format(
        int(time.time() * 1000),
        uuid.uuid4().hex[:8]
    )

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("PRAGMA foreign_keys = ON")

        # Ensure minimum schema exists. CREATE IF NOT EXISTS will not damage existing tables.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT,
                date_of_birth TEXT,
                sex TEXT DEFAULT 'U'
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dicom_studies (
                study_uid TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                modality TEXT,
                dicom_file_path TEXT,
                study_date TEXT,
                study_time TEXT,
                description TEXT,
                created_at DATETIME DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS inference_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_uid TEXT,
                modality TEXT,
                predicted_class TEXT,
                mean_confidence REAL,
                uncertainty_percent REAL,
                routing_tier TEXT,
                mc_mean_probs TEXT,
                mc_std_probs TEXT,
                num_mc_passes INTEGER,
                logits_raw TEXT,
                features_shape TEXT,
                ood_flag INTEGER DEFAULT 0,
                ood_reconstruction_error REAL,
                llm_report TEXT,
                gradcam_base64 TEXT,
                gradcam_file_path TEXT,
                gpio_triggered INTEGER DEFAULT 0,
                gpio_alert_type TEXT DEFAULT 'NONE',
                pipeline_ms INTEGER,
                inferred_at DATETIME DEFAULT (datetime('now')),
                FOREIGN KEY (study_uid) REFERENCES dicom_studies(study_uid)
            )
        """)

        cur.execute(
            "INSERT OR IGNORE INTO patients (patient_id, patient_name, sex) VALUES (?, ?, ?)",
            (str(patient_id), str(patient_id), "U")
        )

        cur.execute(
            """
            INSERT OR IGNORE INTO dicom_studies
                (study_uid, patient_id, modality, description)
            VALUES (?, ?, ?, ?)
            """,
            (study_uid, str(patient_id), str(modality), "API inference upload")
        )

        # Dynamic insert: only use columns that exist in the actual DB table.
        table_cols = [
            row["name"]
            for row in cur.execute("PRAGMA table_info(inference_records)").fetchall()
        ]
        table_cols_set = set(table_cols)

        candidate = {
            "study_uid": study_uid,
            "patient_id": str(patient_id),  # used only if legacy table has this column
            "modality": str(modality),
            "predicted_class": str(diagnosis),
            "diagnosis": str(diagnosis),    # used only if legacy table has this column
            "mean_confidence": confidence,
            "confidence": confidence,       # used only if legacy table has this column
            "uncertainty_percent": uncertainty,
            "routing_tier": int(routing_tier_value),
            "routing": str(routing),        # used only if legacy table has this column
            "mc_mean_probs": json.dumps(kwargs.get("mc_mean_probs", [])),
            "mc_std_probs": json.dumps(kwargs.get("mc_std_probs", [])),
            "num_mc_passes": int(kwargs.get("num_mc_passes", 0) or 0),
            "logits_raw": json.dumps(kwargs.get("logits_raw", [])),
            "features_shape": json.dumps(kwargs.get("features_shape", [])),
            "ood_flag": 1 if bool(ood_flag) else 0,
            "ood_reconstruction_error": kwargs.get("ood_reconstruction_error", kwargs.get("mse", None)),
            "llm_report": str(report),
            "full_report": str(report),     # used only if legacy table has this column
            "gradcam_base64": kwargs.get("gradcam_base64"),
            "gradcam_file_path": kwargs.get("gradcam_file_path"),
            "gpio_triggered": int(gpio_triggered or 0),
            "gpio_alert_type": str(gpio_alert_type or "NONE"),
            "pipeline_ms": pipeline_ms,
        }

        insert_cols = [
            c for c in candidate.keys()
            if c in table_cols_set and c not in ("record_id", "inferred_at")
        ]

        if not insert_cols:
            raise sqlite3.Error("No compatible columns found in inference_records.")

        placeholders = ",".join(["?"] * len(insert_cols))
        sql = "INSERT INTO inference_records ({}) VALUES ({})".format(
            ",".join(insert_cols),
            placeholders
        )

        values = [candidate[c] for c in insert_cols]
        cur.execute(sql, values)

        record_id = int(cur.lastrowid)
        conn.commit()
        conn.close()

        print("[DB] Inserted inference record:", record_id)
        return record_id

    except Exception as e:
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        print("[DB WARNING] insert_record failed:", e)
        return -1

