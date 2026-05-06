-- ============================================================
-- Accelerated Medical Imaging using ML Architectures
-- Built from actual project files:
--   flask_api.py, pipeline_skeleton.py, mc_dropout_head.py,
--   gpio_alerts.py, generate_dummies.py, batch_resize.py
-- ============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;   -- Supports concurrent Flask + pipeline writes

-- ============================================================
-- TABLE 1: PATIENTS
-- Source: flask_api.py /analyze + /analyze_patient endpoints
--         pipeline_skeleton.py → payload["patient_id"] = "PT-{n}"
-- ============================================================
CREATE TABLE IF NOT EXISTS patients (
    patient_id      TEXT PRIMARY KEY,   -- Matches pipeline "PT-{n}" format
    patient_name    TEXT,
    date_of_birth   TEXT,               -- YYYY-MM-DD
    sex             TEXT CHECK(sex IN ('M','F','O','U')) DEFAULT 'U',
    created_at      DATETIME DEFAULT (datetime('now'))
);

-- ============================================================
-- TABLE 2: DICOM_STUDIES
-- Source: pipeline_skeleton.py → payload["modality"] = "brain_mri"
--         dicom_listener.py (Day 3) will populate dicom_file_path
--         Modalities match gpio_alerts.py CRITICAL_FINDINGS classes:
--         Brain, Retinal, Skin, Chest, Bone, ECG
-- ============================================================
CREATE TABLE IF NOT EXISTS dicom_studies (
    study_uid       TEXT PRIMARY KEY,   -- DICOM StudyInstanceUID
    patient_id      TEXT NOT NULL,
    modality        TEXT NOT NULL CHECK(modality IN (
                        'brain_mri','retinal','skin','chest_xray','bone','ecg'
                    )),                 -- Matches pipeline_skeleton payload["modality"]
    study_date      TEXT,               -- YYYYMMDD from DICOM tag
    study_time      TEXT,               -- HHMMSS from DICOM tag
    description     TEXT,
    dicom_file_path TEXT,               -- Local .dcm path (set by dicom_listener.py)
    received_at     DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- ============================================================
-- TABLE 3: INFERENCE_RECORDS
-- Source: flask_api.py /analyze response contract:
--   "modality", "prediction.class", "prediction.mean_confidence",
--   "prediction.uncertainty", "ood_flag", "routing_tier",
--   "llm_report", "gradcam_base64"
-- Source: mc_dropout_head.py → mean_probs, std_probs (20-pass)
-- Source: gpio_alerts.py → CRITICAL_FINDINGS, uncertainty > 5.0
-- Source: pipeline_skeleton.py → Stage 2 dummy_logits, dummy_features
-- ============================================================
CREATE TABLE IF NOT EXISTS inference_records (
    record_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    study_uid               TEXT NOT NULL,

    -- From pipeline_skeleton.py payload
    modality                TEXT NOT NULL,  -- "brain_mri", "retinal", etc.

    -- From flask_api.py /analyze response contract
    predicted_class         TEXT,           -- e.g. "Glioma" (matches gpio CRITICAL_FINDINGS)
    mean_confidence         REAL,           -- flask: prediction.mean_confidence (0-100 scale)
    uncertainty_percent     REAL,           -- flask: prediction.uncertainty — used by gpio_alerts.py
                                            --        trigger: uncertainty_percent > 5.0
    routing_tier            INTEGER CHECK(routing_tier IN (1,2,3)),
                                            -- flask: routing_tier (1=auto, 2=review, 3=urgent)

    -- From mc_dropout_head.py run_mc_inference() 20-pass output
    mc_mean_probs           TEXT,           -- JSON array e.g. "[0.92, 0.05, 0.03]"
    mc_std_probs            TEXT,           -- JSON array e.g. "[0.012, 0.008, 0.004]"
    num_mc_passes           INTEGER DEFAULT 20,

    -- From generate_dummies.py / pipeline Stage 2
    logits_raw              TEXT,           -- JSON array from dummy_logits shape (1,3)
    features_shape          TEXT,           -- e.g. "1,1280,7,7" from dummy_features

    -- From flask_api.py /analyze response
    ood_flag                INTEGER DEFAULT 0,  -- flask: ood_flag (bool → 0/1)
    ood_reconstruction_error REAL,              -- from OOD autoencoder (Day 8)

    -- From flask_api.py /analyze response
    llm_report              TEXT,           -- flask: llm_report (Phi-2 clinical text)
    gradcam_base64          TEXT,           -- flask: gradcam_base64 (stored if small)
    gradcam_file_path       TEXT,           -- OR file path for large CAM images (Day 13)

    -- gpio_alerts.py result
    gpio_triggered          INTEGER DEFAULT 0,  -- 1 = RED LED + Buzzer fired
    gpio_alert_type         TEXT CHECK(gpio_alert_type IN ('GREEN','RED','NONE'))
                                DEFAULT 'NONE',

    -- Timing (for Day 16 end-to-end <5s target)
    pipeline_ms             INTEGER,        -- Stage1+Stage2+Stage3 total ms
    inferred_at             DATETIME DEFAULT (datetime('now')),

    FOREIGN KEY (study_uid) REFERENCES dicom_studies(study_uid)
);

-- ============================================================
-- TABLE 4: FUSION_RESULTS
-- Source: flask_api.py /analyze_patient response contract:
--   "fusion_risk_score": "Critical"
--   "details": "Patient exhibits multi-modal Glioma and Pneumonia signatures."
-- Source: generate_dummies.py → 10,000 synthetic patient profiles (Day 6)
-- ============================================================
CREATE TABLE IF NOT EXISTS fusion_results (
    fusion_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          TEXT NOT NULL,
    modalities_used     TEXT,           -- e.g. "brain_mri,chest_xray,ecg"
    fusion_risk_score   TEXT CHECK(fusion_risk_score IN ('Normal','Moderate','High','Critical')),
                                        -- Matches flask /analyze_patient: "Critical"
    fusion_details      TEXT,           -- flask: details field
    fusion_score_raw    REAL,           -- Raw MLP output (0.0-1.0) before label mapping
    fused_at            DATETIME DEFAULT (datetime('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- ============================================================
-- TABLE 5: ACTIVE_LEARNING_QUEUE
-- Source: flask_api.py /correct endpoint
--   EWC fine-tuning loop — doctor sends corrected label
--   confidence_router.py (Day 7) flags high-uncertainty records
--   mc_dropout_head.py: uncertainty > 5.0 → Tier 3 Worklist
-- ============================================================
CREATE TABLE IF NOT EXISTS active_learning_queue (
    queue_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id           INTEGER NOT NULL,

    -- Populated when mc_dropout_head uncertainty > 5.0%
    uncertainty_trigger REAL,           -- The std_probs value that triggered queuing

    status              TEXT DEFAULT 'PENDING'
                            CHECK(status IN ('PENDING','REVIEWED','CORRECTED','SKIPPED')),

    -- From flask_api.py /correct endpoint body
    corrected_label     TEXT,           -- Doctor's override class name
    correction_notes    TEXT,

    -- EWC status (flask /correct response)
    ewc_status          TEXT CHECK(ewc_status IN ('PENDING','RUNNING','COMPLETE','FAILED'))
                            DEFAULT 'PENDING',
    ewc_message         TEXT,           -- flask: "EWC Fine-tuning complete. Model updated..."

    queued_at           DATETIME DEFAULT (datetime('now')),
    reviewed_at         DATETIME,
    FOREIGN KEY (record_id) REFERENCES inference_records(record_id)
);

-- ============================================================
-- TABLE 6: DICOM_SR_LOG
-- Source: send_dicom_results.py (Day 6) — ExplicitVRLittleEndian SR
--         dicom_listener.py (Day 3) auto-routes incoming files
--         Day 16: end-to-end timing target < 5000ms
-- ============================================================
CREATE TABLE IF NOT EXISTS dicom_sr_log (
    sr_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id           INTEGER NOT NULL,
    sr_file_path        TEXT,           -- Local path to generated .dcm SR file
    orthanc_instance_id TEXT,           -- Instance ID from Orthanc REST response
    transfer_syntax     TEXT DEFAULT 'ExplicitVRLittleEndian',
                                        -- Matches send_dicom_results.py Day 6

    send_status         TEXT DEFAULT 'PENDING'
                            CHECK(send_status IN ('PENDING','SENT','FAILED')),
    -- Day 16: End-to-end DICOM workflow timing (target <5000ms)
    receive_to_send_ms  INTEGER,        -- ms from DICOM received → SR sent back
    sent_at             DATETIME,
    error_message       TEXT,
    FOREIGN KEY (record_id) REFERENCES inference_records(record_id)
);

-- ============================================================
-- TABLE 7: PIPELINE_BENCHMARK
-- Source: pipeline_skeleton.py run_throughput_test()
--   processed_frames, total_time, fps
--   Stage timings: Stage1=15ms, Stage2=45ms, Stage3=30ms (simulated)
--   Day 15: M1 measures final FP16 TRT latency + thermal test
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_benchmark (
    bench_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_label           TEXT,           -- e.g. "FP16_TRT_final", "mock_numpy"
    test_duration_s     REAL,           -- pipeline_skeleton: test_duration=10
    total_frames        INTEGER,        -- pipeline_skeleton: processed_frames
    fps                 REAL,           -- pipeline_skeleton: fps = frames/time
    -- Per-stage latency (ms) from pipeline_skeleton simulated sleeps
    stage1_ms           REAL,           -- fetch+preprocess (simulate: 15ms)
    stage2_ms           REAL,           -- TRT inference (simulate: 45ms)
    stage3_ms           REAL,           -- MC Dropout+CAM+DB (simulate: 30ms)
    -- Day 15: thermal test fields
    power_watts         REAL,           -- Jetson Nano power draw
    temp_celsius        REAL,           -- Board temperature
    trt_precision       TEXT CHECK(trt_precision IN ('FP32','FP16','INT8')),
    recorded_at         DATETIME DEFAULT (datetime('now'))
);

-- ============================================================
-- TABLE 8: MODEL_REGISTRY
-- Source: generate_dummies.py → dummy_dual_output.onnx, dummy_fc_weights.npy
--         pipeline_skeleton.py Stage 2 → TRT engine execution
--         6 modalities: Brain, Retinal, Skin, Bone, Chest, ECG
-- ============================================================
CREATE TABLE IF NOT EXISTS model_registry (
    model_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    modality            TEXT NOT NULL,  -- "brain_mri", "retinal", etc.
    model_type          TEXT CHECK(model_type IN ('ONNX','TRT_FP16','PYTORCH','OOD_AE')),
    file_path           TEXT NOT NULL,  -- e.g. "dummy_dual_output.onnx"
    -- generate_dummies.py model specs
    input_shape         TEXT,           -- e.g. "1,3,224,224" (batch_resize target)
    output_logits_shape TEXT,           -- e.g. "1,3" (num_classes)
    output_features_shape TEXT,         -- e.g. "1,1280,7,7" (for CAM)
    fc_weights_path     TEXT,           -- e.g. "dummy_fc_weights.npy" shape (3,1280)
    num_classes         INTEGER,        -- Brain=3, Skin=7, ECG=19, etc.
    opset_version       INTEGER DEFAULT 17,  -- generate_dummies.py opset_version=17
    is_active           INTEGER DEFAULT 1,
    registered_at       DATETIME DEFAULT (datetime('now'))
);

-- ============================================================
-- TABLE 9: SYSTEM_LOGS
-- Source: All components use [SYSTEM], [ALERT], [STATUS] prefixes
--   gpio_alerts.py: "[ALERT] Critical Finding..."
--   pipeline_skeleton.py: "[SYSTEM] 3-Stage Pipeline Initialized"
--   mc_dropout_head.py: "[CLINICAL ROUTING CHECK]"
-- ============================================================
CREATE TABLE IF NOT EXISTS system_logs (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    level       TEXT CHECK(level IN ('INFO','WARNING','ERROR','DEBUG','ALERT'))
                    DEFAULT 'INFO',
    component   TEXT,   -- "pipeline_stage1", "mc_dropout", "gpio_alerts",
                        -- "dicom_listener", "flask_api", "confidence_router"
    message     TEXT,
    logged_at   DATETIME DEFAULT (datetime('now'))
);

-- ============================================================
-- INDEXES — Optimised for Flask API query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_studies_patient    ON dicom_studies(patient_id);
CREATE INDEX IF NOT EXISTS idx_studies_modality   ON dicom_studies(modality);
CREATE INDEX IF NOT EXISTS idx_infer_study        ON inference_records(study_uid);
CREATE INDEX IF NOT EXISTS idx_infer_tier         ON inference_records(routing_tier);
CREATE INDEX IF NOT EXISTS idx_infer_ood          ON inference_records(ood_flag);
CREATE INDEX IF NOT EXISTS idx_infer_gpio         ON inference_records(gpio_triggered);
CREATE INDEX IF NOT EXISTS idx_alq_status         ON active_learning_queue(status);
CREATE INDEX IF NOT EXISTS idx_alq_ewc            ON active_learning_queue(ewc_status);
CREATE INDEX IF NOT EXISTS idx_sr_status          ON dicom_sr_log(send_status);
CREATE INDEX IF NOT EXISTS idx_model_modality     ON model_registry(modality);
CREATE INDEX IF NOT EXISTS idx_logs_component     ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_logs_level         ON system_logs(level);