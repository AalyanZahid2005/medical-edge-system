"""
flask_api_full.py  —  FYP: Accelerated Medical Imaging using ML Architectures
Ahmed Rasheed (M3) — FULL Production API (Days 5–20 integrated)

Replaces flask_api.py stub.  Wires:
  - db_manager.py         (M3 Day 2)
  - confidence_router.py  (M3 Day 7)
  - send_dicom_results.py (M3 Day 6)
  - ewc_trainer.py        (M3 Day 6)
  - active_learning_auto.py (M3 Day 7)
  - dicom_cfind.py        (M3 Day 18) — started as a side thread

Endpoints:
    POST /analyze               Single-modality inference
    POST /analyze_patient       Multi-modal Transformer Fusion (M2 Day 16)
    POST /correct               Clinician label correction → EWC
    GET  /history/<patient_id>  Patient inference history
    GET  /queue                 Active learning queue
    GET  /health                Health check
    POST /federated_sync        Trigger top-3 AL auto-corrections (Day 7)
    GET  /fusion_compare        Compare Transformer vs MLP fusion (Day 16)
    POST /stream                WebSocket endpoint stub (Day 11, see note)
    GET  /stats                 Pipeline / system stats

Run:
    python flask_api_full.py
    python flask_api_full.py --no-stub          # real TRT (Day 7+)
    python flask_api_full.py --test             # self-test
    python flask_api_full.py --enable-cfind     # start DICOM C-FIND daemon
    python flask_api_full.py --precision fp16   # set precision mode (Day 12)

Install:
    pip install flask flask-sock pydicom requests
"""

import os
import sys
import json
import re
import time
import random
import argparse
import traceback
import threading
from datetime import datetime

from flask import Flask, request, jsonify

# ── WebSocket (Day 11) ─────────────────────────────────────────────────────────
try:
    from flask_sock import Sock
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from db_manager import DatabaseManager, CRITICAL_FINDINGS
except ImportError:
    DatabaseManager   = None
    CRITICAL_FINDINGS = []
    print("[WARNING] db_manager.py not found.")

try:
    from confidence_router import ConfidenceRouter
    _router = ConfidenceRouter()
except ImportError:
    _router = None
    print("[WARNING] confidence_router.py not found.")

try:
    from ewc_trainer import run_ewc_async, EWCStatus
    EWC_AVAILABLE = True
except ImportError:
    EWC_AVAILABLE = False
    print("[WARNING] ewc_trainer.py not found.")

try:
    from active_learning_auto import ActiveLearningAutomation
    AL_AVAILABLE = True
except ImportError:
    AL_AVAILABLE = False

try:
    from dicom_cfind import DicomCFindPoller
    CFIND_AVAILABLE = True
except ImportError:
    CFIND_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
STUB_MODE    = False   # Flip False on Day 7 (--no-stub)
FLASK_HOST   = "0.0.0.0"
FLASK_PORT   = 5000
FLASK_DEBUG  = False

# Precision mode (Day 12): "fp32" | "fp16" | "int8"
PRECISION_MODE = os.environ.get("PRECISION_MODE", "fp16")

UNCERTAINTY_TIER3 = 5.0
UNCERTAINTY_TIER2 = 2.5
VALID_MODALITIES  = {"brain_mri", "retinal", "skin", "chest_xray", "bone", "ecg"}

STUB_CLASSES = {
    "brain_mri":  ["Glioma", "Meningioma", "Pituitary", "Normal"],
    "retinal":    ["Mild DR", "Moderate DR", "Severe DR",
                   "Proliferative DR", "No DR"],
    "skin":       ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    "chest_xray": ["Pneumonia", "Normal"],
    "bone":       ["Fracture", "Normal"],
    "ecg":        ["Normal", "A", "V", "L", "R", "1", "2", "3", "4"],
}

FUSION_RISK_MAP = {
    (0.0, 0.25): "Normal",
    (0.25, 0.50): "Moderate",
    (0.50, 0.75): "High",
    (0.75, 1.01): "Critical",
}

# ══════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# ============================================================
# FINAL JSON GUARD: OOD rejection response cleaner
# Ensures OOD cases return status=rejected, not status=ok.
# ============================================================
@app.after_request
def final_ood_json_guard(response):
    try:
        if request.path != "/analyze":
            return response

        if not response.is_json:
            return response

        data = response.get_json(silent=True)
        if not isinstance(data, dict):
            return response

        pred_obj = data.get("prediction", {})
        pred_class = ""

        if isinstance(pred_obj, dict):
            pred_class = str(pred_obj.get("class", ""))
        else:
            pred_class = str(data.get("predicted_class", ""))

        is_ood = bool(data.get("ood_flag", False)) or pred_class in [
            "UNKNOWN_PATHOLOGY",
            "OOD_REJECTED",
            "WORKER_ERROR"
        ]

        # Do not override modality mismatch wording; that guard handles mismatch.
        if not is_ood or bool(data.get("modality_mismatch", False)):
            return response

        data["status"] = "rejected"
        data["ood_flag"] = True
        data["routing_tier"] = 3
        data["routing_decision"] = "Tier 3: Rejected / OOD Review"
        data["risk_tier"] = "Rejected / OOD Review"
        data["gpio_alert_type"] = "RED"
        data["gpio_triggered"] = 1
        data["manual_review_required"] = True
        data["auto_report_allowed"] = False
        data["routing_reason"] = "Out-of-distribution or rejected scan requires manual review."

        response.set_data(json.dumps(data))
        response.headers["Content-Type"] = "application/json"
        response.headers["Content-Length"] = len(response.get_data())

    except Exception as e:
        print("[FINAL OOD GUARD WARNING]", str(e))

    return response


# ============================================================
# FINAL JSON GUARD: modality mismatch response cleaner
# Registered early so it runs late in Flask after_request order.
# ============================================================
@app.after_request
def final_modality_mismatch_json_guard(response):
    try:
        if request.path != "/analyze":
            return response

        if not response.is_json:
            return response

        data = response.get_json(silent=True)
        if not isinstance(data, dict):
            return response

        pred_obj = data.get("prediction", {})
        pred_class = ""

        if isinstance(pred_obj, dict):
            pred_class = str(pred_obj.get("class", ""))
        else:
            pred_class = str(data.get("predicted_class", ""))

        if pred_class != "MODALITY_MISMATCH" and not bool(data.get("modality_mismatch", False)):
            return response

        selected = str(data.get("modality", data.get("selected_modality", "")))

        # If pipeline did not expose detected_modality, infer from the report text.
        report_text = (
            str(data.get("llm_report", "")) + " " +
            str(data.get("clinical_report", "")) + " " +
            str(data.get("report", ""))
        )

        detected = data.get("detected_modality")
        if not detected:
            import re
            m = re.search(r"closest to ['\\\"]([^'\\\"]+)['\\\"]", report_text)
            detected = m.group(1) if m else "brain_mri"

        data["status"] = "rejected"
        data["ood_flag"] = True
        data["modality_mismatch"] = True
        data["selected_modality"] = selected
        data["detected_modality"] = detected

        data["routing_tier"] = 3
        data["routing_decision"] = "Tier 3: Modality Mismatch / Manual Review"
        data["risk_tier"] = "Rejected / Wrong Modality"
        data["gpio_alert_type"] = "RED"
        data["gpio_triggered"] = 1
        data["manual_review_required"] = True
        data["auto_report_allowed"] = False
        data["routing_reason"] = (
            "Selected modality does not match uploaded image. "
            "Selected: {}. Detected: {}."
        ).format(selected, detected)

        response.set_data(json.dumps(data))
        response.headers["Content-Type"] = "application/json"
        response.headers["Content-Length"] = len(response.get_data())

    except Exception as e:
        print("[FINAL MISMATCH GUARD WARNING]", str(e))

    return response



# ============================================================
# Final response safety hook for /analyze routing metadata
# Ensures API output always exposes full clinical routing proof.
# ============================================================
@app.after_request
def inject_routing_metadata(response):
    try:
        if request.path != "/analyze":
            return response

        if not response.is_json:
            return response

        data = response.get_json(silent=True)
        if not isinstance(data, dict):
            return response

        if "routing_tier" not in data:
            return response

        tier = int(data.get("routing_tier", 3))

        if tier == 1:
            routing_decision = "Tier 1: Auto-SR"
            risk_tier = "High Confidence / Auto-SR"
            gpio_alert_type = "GREEN"
            manual_review_required = False
            auto_report_allowed = True
            routing_reason = "Confidence is >= 90%."
        elif tier == 2:
            routing_decision = "Tier 2: AI-Assisted Review"
            risk_tier = "Medium Confidence / Doctor Review"
            gpio_alert_type = "YELLOW"
            manual_review_required = True
            auto_report_allowed = True
            routing_reason = "Confidence is between 70% and 90%."
        else:
            routing_decision = "Tier 3: Urgent Manual Review"
            risk_tier = "Low Confidence / Urgent Review"
            gpio_alert_type = "RED"
            manual_review_required = True
            auto_report_allowed = False
            routing_reason = "Confidence is below 70% or scan requires urgent review."

        data["routing_decision"] = data.get("routing_decision", routing_decision)
        data["risk_tier"] = data.get("risk_tier", risk_tier)
        data["manual_review_required"] = bool(data.get("manual_review_required", manual_review_required))
        data["auto_report_allowed"] = bool(data.get("auto_report_allowed", auto_report_allowed))
        data["routing_reason"] = data.get("routing_reason", routing_reason)
        data["gpio_alert_type"] = data.get("gpio_alert_type", gpio_alert_type)
        data["gpio_triggered"] = int(data.get("gpio_triggered", 1))

        response.set_data(json.dumps(data))
        response.headers["Content-Type"] = "application/json"
        response.headers["Content-Length"] = len(response.get_data())

    except Exception as e:
        print("[ROUTING HOOK WARNING]", str(e))

    return response

app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

if WEBSOCKET_AVAILABLE:
    sock = Sock(app)

# Global singletons
_db:        "DatabaseManager"          = None
_al_auto:   "ActiveLearningAutomation" = None
_cfind:     "DicomCFindPoller"         = None

def get_db():
    global _db
    if _db is None and DatabaseManager is not None:
        _db = DatabaseManager()
    return _db


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stub_inference(modality: str) -> dict:
    classes     = STUB_CLASSES.get(modality, ["Normal", "Abnormal"])
    raw         = [random.random() for _ in range(len(classes))]
    total       = sum(raw)
    mean_probs  = [round(r / total, 4) for r in raw]
    std_probs   = [round(random.uniform(0.003, 0.025), 4) for _ in range(len(classes))]
    top_idx     = mean_probs.index(max(mean_probs))
    predicted   = classes[top_idx]
    confidence  = round(mean_probs[top_idx] * 100, 2)
    uncertainty = round(max(std_probs) * 100, 2)
    logits_raw  = [[round(random.gauss(0, 1), 4) for _ in range(len(classes))]]
    llm_report  = (
        f"[STUB] Modality: {modality.upper()}. "
        f"AI prediction: {predicted} ({confidence:.1f}%, ±{uncertainty:.2f}%). "
        f"{'Urgent review.' if predicted in CRITICAL_FINDINGS else 'Routine follow-up.'}"
    )
    return {
        "predicted_class":          predicted,
        "mean_confidence":          confidence,
        "uncertainty":              uncertainty,
        "mc_mean_probs":            mean_probs,
        "mc_std_probs":             std_probs,
        "logits_raw":               logits_raw,
        "features_shape":           "1,1280,7,7",
        "num_mc_passes":            20,
        "llm_report":               llm_report,
        "gradcam_base64":           None,
        "ood_flag":                 False,
        "ood_reconstruction_error": None,
    }


def _map_fusion_risk(score: float) -> str:
    for (lo, hi), label in FUSION_RISK_MAP.items():
        if lo <= score < hi:
            return label
    return "Critical"


def _get_routing(predicted_class, mean_confidence, uncertainty, modality,
                 ood_flag=False):
    """Use ConfidenceRouter if available, else inline fallback."""
    if _router:
        dec = _router.route(
            predicted_class     = predicted_class,
            mean_confidence     = mean_confidence,
            uncertainty_percent = uncertainty,
            modality            = modality,
            ood_flag            = ood_flag,
        )
        return dec.tier, dec.gpio_alert, dec.gpio_triggered
    # Inline fallback
    if predicted_class in CRITICAL_FINDINGS or uncertainty > UNCERTAINTY_TIER3:
        return 3, "RED", 1
    if uncertainty > UNCERTAINTY_TIER2:
        return 2, "GREEN", 0
    return 1, "GREEN", 0


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1: POST /analyze
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Run single-modality ML inference.
    Called by: dicom_listener.py, dicom_cfind.py, M4 GUI Scan Upload tab.

    Request: multipart/form-data  OR  application/json
        patient_id  : str
        modality    : str (auto-detected from DICOM if not provided)
        study_uid   : str (optional — created if absent)
        file        : .dcm or image file (optional in stub mode)
        precision   : "fp32" | "fp16" | "int8" (Day 12, optional)
    """
    t_start = time.time()
    db      = get_db()

    # ── Parse inputs ──────────────────────────────────────────────────────────
    if request.is_json:
        body       = request.get_json(silent=True) or {}
        patient_id = body.get("patient_id", "PT-UNKNOWN")
        modality   = body.get("modality", "brain_mri")
        study_uid  = body.get("study_uid", f"AUTO-{int(time.time()*1000)}")
        precision  = body.get("precision", PRECISION_MODE)
    else:
        patient_id = request.form.get("patient_id", "PT-UNKNOWN")
        modality   = request.form.get("modality", "brain_mri")
        study_uid  = request.form.get("study_uid",
                                       f"AUTO-{int(time.time()*1000)}")
        precision  = request.form.get("precision", PRECISION_MODE)

    modality = modality.strip().lower()
    if modality not in VALID_MODALITIES:
        return jsonify({
            "status": "error",
            "message": f"Unknown modality '{modality}'. "
                       f"Valid: {sorted(VALID_MODALITIES)}"
        }), 400

    # ── Auto-seed DB ──────────────────────────────────────────────────────────
    if db:
        try:
            db.upsert_patient(patient_id)
            db.insert_study(study_uid, patient_id, modality)
        except Exception:
            pass

    # ── Inference (stub or real) ───────────────────────────────────────────────
    if STUB_MODE:
        inf = _stub_inference(modality)
    else:
        # 1. Grab the image file Ahmed forgot to parse
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "error", "message": "No file uploaded!"}), 400
            
        file_path = f"/tmp/{study_uid}.jpg"
        file.save(file_path)

        # 2. Load Hamid's Phase 5 Pipeline (Cached to protect memory)
# 2. Load Hamid's Phase 5 Pipeline (Cached to protect memory)
        if "real_pipeline" not in app.config:
            import sys
            sys.path.append("./src/pipeline")
            import src.pipeline.master_pipeline_v2 as mp2
            # Dynamically grab Hamid's class
            PipelineClass = next(attr for name, attr in vars(mp2).items() if isinstance(attr, type) and "Pipeline" in name)
            app.config["real_pipeline"] = PipelineClass()            
        # 3. Convert image to PyTorch Tensor
        from PIL import Image
        import torchvision.transforms as T
        img_tensor = T.ToTensor()(Image.open(file_path).convert("RGB")).unsqueeze(0)
        
        # 4. Fire the true TensorRT GPU inference!
        inf = app.config["real_pipeline"].process_scan(
            patient_id=patient_id, 
            image_tensor=img_tensor, 
            class_names=STUB_CLASSES.get(modality, ["Abnormal", "Normal"]), 
            modality_name=modality
        )

    predicted   = inf["predicted_class"]
    confidence  = inf["mean_confidence"]
    uncertainty = inf["uncertainty"]
    ood_flag    = inf["ood_flag"]

    # ── Routing ───────────────────────────────────────────────────────────────
    tier, gpio_alert, gpio_triggered = _get_routing(
        predicted, confidence, uncertainty, modality, ood_flag
    )

    # ── DB insert ─────────────────────────────────────────────────────────────
    record_id = -1
    if db:
        try:
            record_id = db.insert_record({
                "study_uid":               study_uid,
                "modality":                modality,
                "predicted_class":         predicted,
                "mean_confidence":         confidence,
                "uncertainty_percent":     uncertainty,
                "routing_tier":            tier,
                "routing_decision": inf.get("routing_decision", inf.get("routing", "")),
                "risk_tier": inf.get("risk_tier", ""),
                "manual_review_required": bool(inf.get("manual_review_required", routing_tier == 3)),
                "auto_report_allowed": bool(inf.get("auto_report_allowed", routing_tier in [1, 2])),
                "routing_reason": inf.get("routing_reason", ""),
                "mc_mean_probs":           json.dumps(inf.get("mc_mean_probs", [])),
                "mc_std_probs":            json.dumps(inf.get("mc_std_probs", [])),
                "num_mc_passes":           inf.get("num_mc_passes", 20),
                "logits_raw":              json.dumps(inf.get("logits_raw", [])),
                "features_shape":          inf.get("features_shape", ""),
                "ood_flag":                1 if ood_flag else 0,
                "ood_reconstruction_error": inf.get("ood_reconstruction_error"),
                "llm_report":              inf.get("llm_report", ""),
                "gradcam_base64":          inf.get("gradcam_base64"),
                "gpio_triggered":          gpio_triggered,
                "gpio_alert_type":         gpio_alert,
                "pipeline_ms":             int((time.time() - t_start) * 1000),
            })
        except Exception as e:
            print(f"[WARN] DB insert failed: {e}")

    # ── Auto-queue Tier 3 to active_learning_queue ────────────────────────────
    if db and tier == 3 and record_id > 0:
        try:
            db.queue_for_review(record_id, uncertainty_trigger=uncertainty)
        except Exception:
            pass

    pipeline_ms = int((time.time() - t_start) * 1000)

    return jsonify({
        "status":         "ok",
        "modality":        modality,
        "precision":       precision,
        "prediction": {
            "class":            predicted,
            "mean_confidence":  confidence,
            "uncertainty":      uncertainty,
        },
        "ood_flag":        ood_flag,
        "routing_tier":    tier,
        "gpio_alert_type": gpio_alert,
        "gpio_triggered":  gpio_triggered,
        "llm_report":      inf.get("llm_report", ""),
        "gradcam_base64":  inf.get("gradcam_base64"),
        "mc_mean_probs":   inf.get("mc_mean_probs", []),
        "mc_std_probs":    inf.get("mc_std_probs", []),
        "record_id":       record_id,
        "pipeline_ms":     pipeline_ms,
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2: POST /analyze_patient
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/analyze_patient", methods=["POST"])
def analyze_patient():
    """
    Multi-modal patient fusion.
    Day 16: uses M2's CrossModalAttentionFusion (transformer_fusion.pth).
    Until then: stub MLP simulation.
    """
    db   = get_db()
    body = request.get_json(silent=True) or {}
    patient_id = body.get("patient_id", "PT-UNKNOWN")

    # Fetch all inference records for this patient
    history = db.query_history(patient_id) if db else []
    modalities_used = list({r["modality"] for r in history}) if history else []

    # Stub fusion score (Day 16: M2 wires real transformer fusion here)
    fusion_score_raw = random.uniform(0.0, 1.0)
    fusion_risk      = _map_fusion_risk(fusion_score_raw)

    # Build details message
    critical_found = [r["predicted_class"] for r in history
                      if r.get("predicted_class") in CRITICAL_FINDINGS]
    details = (
        f"Patient {patient_id} — {len(history)} scan(s) across "
        f"{len(modalities_used)} modality(ies)."
    )
    if critical_found:
        details += f" Critical findings: {', '.join(set(critical_found))}."

    # Attention weights placeholder (Day 16: real transformer weights)
    attention_weights = {m: round(random.random(), 3) for m in modalities_used}

    # DB insert fusion result
    fusion_id = -1
    if db:
        try:
            fusion_id = db.insert_fusion_result(
                patient_id        = patient_id,
                modalities_used   = modalities_used,
                fusion_risk_score = fusion_risk,
                fusion_details    = details,
                fusion_score_raw  = fusion_score_raw,
            )
        except Exception as e:
            print(f"[WARN] fusion DB insert: {e}")

    return jsonify({
        "status":           "ok",
        "patient_id":       patient_id,
        "fusion_risk_score": fusion_risk,
        "fusion_score_raw": round(fusion_score_raw, 4),
        "modalities_used":  modalities_used,
        "details":          details,
        "attention_weights": attention_weights,   # Day 16 radar chart (M4)
        "fusion_id":        fusion_id,
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3: POST /correct
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/correct", methods=["POST"])
def correct_label():
    """
    Clinician label correction → EWC fine-tune loop.

    Request JSON:
        record_id       : int
        corrected_label : str
        notes           : str (optional)
    """
    db   = get_db()
    body = request.get_json(silent=True) or {}

    record_id       = body.get("record_id")
    corrected_label = body.get("corrected_label", "")
    notes           = body.get("notes", "")

    if not record_id or not corrected_label:
        return jsonify({"status": "error",
                        "message": "record_id and corrected_label required"}), 400

    # Fetch the record to get modality
    modality = "brain_mri"
    if db:
        try:
            conn = db._get_connection()
            row  = conn.execute(
                "SELECT modality FROM inference_records WHERE record_id=?",
                (record_id,)
            ).fetchone()
            if row:
                modality = row["modality"]
            else:
                return jsonify({"status": "error",
                                "message": f"record_id {record_id} not found"}), 404
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # Write correction to DB
    if db:
        try:
            db.correct_label(record_id, corrected_label, notes)
        except Exception as e:
            return jsonify({"status": "error",
                            "message": f"DB correct_label failed: {e}"}), 500

    # Kick off async EWC fine-tune
    if EWC_AVAILABLE:
        run_ewc_async(record_id, corrected_label, modality,
                      features_numpy=None, db=db)
        ewc_status = EWCStatus.RUNNING
        msg = "EWC Fine-tuning started asynchronously."
    else:
        ewc_status = EWCStatus.PENDING
        msg = "Correction logged. EWC trainer not available (ewc_trainer.py missing)."

    return jsonify({
        "status":          "ok",
        "record_id":       record_id,
        "corrected_label": corrected_label,
        "modality":        modality,
        "ewc_status":      ewc_status,
        "message":         msg,
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4: GET /history/<patient_id>
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/history/<patient_id>", methods=["GET"])
def get_history(patient_id):
    db = get_db()
    if not db:
        return jsonify({"status": "error", "message": "DB unavailable"}), 503

    modality = request.args.get("modality")
    limit    = int(request.args.get("limit", 50))

    try:
        records = db.query_history(patient_id, modality=modality, limit=limit)
        return jsonify({
            "status":     "ok",
            "patient_id": patient_id,
            "count":      len(records),
            "records":    records,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 5: GET /queue
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/queue", methods=["GET"])
def get_queue():
    db = get_db()
    if not db:
        return jsonify({"status": "error", "message": "DB unavailable"}), 503
    try:
        items = db.get_pending_queue()
        return jsonify({
            "status": "ok",
            "count":  len(items),
            "items":  items,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 6: GET /health
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    db     = get_db()
    cfind  = _cfind is not None and _cfind.running
    al     = _al_auto is not None and _al_auto.running

    return jsonify({
        "status":        "ok",
        "stub_mode":     STUB_MODE,
        "precision":     PRECISION_MODE,
        "db":            "connected" if db else "unavailable",
        "cfind_running": cfind,
        "al_auto":       al,
        "websocket":     WEBSOCKET_AVAILABLE,
        "timestamp":     datetime.utcnow().isoformat(),
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 7: POST /federated_sync  (Day 7)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/federated_sync", methods=["POST"])
def federated_sync():
    """
    Trigger top-3 active learning auto-corrections immediately.
    Called by M1's FedAvg server after weight aggregation.
    """
    db = get_db()
    if AL_AVAILABLE:
        al = ActiveLearningAutomation(db=db,
                                      flask_url=f"http://127.0.0.1:{FLASK_PORT}")
        result = al.run_once()
    else:
        result = {"triggered": 0, "error": "active_learning_auto not available"}

    return jsonify({
        "status":   "ok",
        "al_result": result,
        "message":  f"Federated sync triggered {result.get('triggered', 0)} "
                    f"AL auto-corrections.",
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 8: GET /fusion_compare  (Day 16)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/fusion_compare", methods=["GET"])
def fusion_compare():
    """
    Compare Transformer Fusion vs MLP fusion for a patient.
    Day 16: M2 wires real transformer_fusion.pth here.
    Until then: returns stub comparison.
    """
    patient_id = request.args.get("patient_id", "PT-UNKNOWN")
    db         = get_db()

    history  = db.query_history(patient_id) if db else []
    mlp_score = random.uniform(0.0, 1.0)
    tf_score  = random.uniform(0.0, 1.0)

    return jsonify({
        "status":              "ok",
        "patient_id":          patient_id,
        "num_records":         len(history),
        "mlp_fusion_risk":     _map_fusion_risk(mlp_score),
        "mlp_score_raw":       round(mlp_score, 4),
        "transformer_fusion_risk": _map_fusion_risk(tf_score),
        "transformer_score_raw":   round(tf_score, 4),
        "preferred_model":     "transformer" if abs(tf_score - 0.5) < abs(mlp_score - 0.5)
                               else "mlp",
        "note": "Day 16: M2 wires real CrossModalAttentionFusion here.",
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 9: GET /stats
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/stats", methods=["GET"])
def get_stats():
    db = get_db()
    stats = {
        "stub_mode":    STUB_MODE,
        "precision":    PRECISION_MODE,
        "cfind_stats":  _cfind.get_stats() if _cfind else None,
        "al_stats":     _al_auto.get_stats() if _al_auto else None,
    }
    if db:
        try:
            conn = db._get_connection()
            stats["total_records"]  = conn.execute(
                "SELECT COUNT(*) FROM inference_records").fetchone()[0]
            stats["total_patients"] = conn.execute(
                "SELECT COUNT(*) FROM patients").fetchone()[0]
            stats["pending_queue"]  = conn.execute(
                "SELECT COUNT(*) FROM active_learning_queue "
                "WHERE status='PENDING'").fetchone()[0]
            stats["tier_breakdown"] = {
                str(r["routing_tier"]): r["cnt"]
                for r in conn.execute(
                    "SELECT routing_tier, COUNT(*) as cnt FROM inference_records "
                    "GROUP BY routing_tier"
                ).fetchall()
            }
        except Exception as e:
            stats["db_error"] = str(e)

    return jsonify({"status": "ok", "stats": stats}), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 10: WebSocket /stream  (Day 11)
# ══════════════════════════════════════════════════════════════════════════════

if WEBSOCKET_AVAILABLE:
    @sock.route("/stream")
    def stream(ws):
        """
        WebSocket endpoint — pushes live inference results to M4 GUI camera tab.
        M4 wires this to the Camera Mode tab (Day 11).
        """
        db = get_db()
        ws.send(json.dumps({"status": "connected", "message": "Stream ready"}))
        while True:
            try:
                msg = ws.receive(timeout=1)
                if msg is None:
                    break
                body = json.loads(msg)
                modality = body.get("modality", "brain_mri")
                # In stub mode: return a stub result
                inf   = _stub_inference(modality)
                tier, gpio_alert, gpio_triggered = _get_routing(
                    inf["predicted_class"], inf["mean_confidence"],
                    inf["uncertainty"], modality
                )
                ws.send(json.dumps({
                    "status":         "ok",
                    "modality":       modality,
                    "prediction": {
                        "class":           inf["predicted_class"],
                        "mean_confidence": inf["mean_confidence"],
                        "uncertainty":     inf["uncertainty"],
                    },
                    "routing_tier":    tier,
                    "gpio_alert_type": gpio_alert,
                    "llm_report":      inf.get("llm_report", ""),
                    "timestamp":       datetime.utcnow().isoformat(),
                }))
            except Exception:
                break


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "error", "message": "Bad request", "detail": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"status": "error", "message": "File too large (max 64 MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    tb = traceback.format_exc()
    print(f"[ERROR] 500:\n{tb}")
    return jsonify({"status": "error", "message": "Internal server error"}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_self_test():
    import io
    client = app.test_client()

    print("\n" + "="*60)
    print("  flask_api_full.py — M3 Full API Self-Test")
    print("="*60)

    db = get_db()
    if db:
        db.upsert_patient("PT-TEST", "Test Patient", "1990-01-01", "M")
        db.insert_study("STUDY-FULLTEST-001", "PT-TEST", "brain_mri",
                        description="Full API self-test study")

    tests = [
        ("GET  /health",        lambda: client.get("/health")),
        ("POST /analyze (json)",
         lambda: client.post("/analyze",
                             json={"patient_id": "PT-TEST",
                                   "modality": "brain_mri",
                                   "study_uid": "STUDY-FULLTEST-001"})),
        ("POST /analyze (chest)",
         lambda: client.post("/analyze",
                             json={"patient_id": "PT-TEST",
                                   "modality": "chest_xray",
                                   "study_uid": "STUDY-FULLTEST-002"})),
        ("GET  /history/PT-TEST",  lambda: client.get("/history/PT-TEST")),
        ("GET  /queue",            lambda: client.get("/queue")),
        ("POST /analyze_patient",
         lambda: client.post("/analyze_patient",
                             json={"patient_id": "PT-TEST"},
                             content_type="application/json")),
        ("GET  /fusion_compare",
         lambda: client.get("/fusion_compare?patient_id=PT-TEST")),
        ("POST /federated_sync",
         lambda: client.post("/federated_sync",
                             json={}, content_type="application/json")),
        ("GET  /stats",            lambda: client.get("/stats")),
        ("POST /analyze (invalid modality)",
         lambda: client.post("/analyze",
                             json={"patient_id": "PT-TEST",
                                   "modality": "INVALID"})),
    ]

    passed = 0
    for name, fn in tests:
        try:
            r    = fn()
            data = r.get_json()
            ok   = data.get("status") in ("ok", "error")
            sym  = "✓" if r.status_code < 500 else "✗"
            print(f"  {sym} {name} → HTTP {r.status_code}")
            if r.status_code < 500:
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} → EXCEPTION: {e}")

    # /correct test
    if db:
        pending = db.get_pending_queue()
        if pending:
            r    = client.post("/correct",
                               json={"record_id": pending[0]["record_id"],
                                     "corrected_label": "Meningioma",
                                     "notes": "self-test"},
                               content_type="application/json")
            sym  = "✓" if r.status_code == 200 else "✗"
            print(f"  {sym} POST /correct → HTTP {r.status_code}")
            if r.status_code == 200:
                passed += 1
        else:
            print("  ⚠ POST /correct — no pending queue items yet")

    print(f"\n  {passed}/{len(tests)} tests passed")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════



# ============================================================
# Final routing correction hook
# Always recomputes routing from final prediction confidence.
# This prevents mismatch like 99% confidence but Tier 3.
# ============================================================
@app.after_request
def force_routing_from_prediction_confidence(response):
    try:
        if request.path != "/analyze":
            return response

        if not response.is_json:
            return response

        data = response.get_json(silent=True)
        if not isinstance(data, dict):
            return response

        # Preserve explicit modality mismatch rejection from pipeline
        pred_check = data.get("prediction", {})
        pred_class_check = ""
        if isinstance(pred_check, dict):
            pred_class_check = str(pred_check.get("class", ""))
        else:
            pred_class_check = str(data.get("predicted_class", ""))

        if pred_class_check == "MODALITY_MISMATCH" or bool(data.get("modality_mismatch", False)):
            data["routing_tier"] = 3
            data["routing_decision"] = "Tier 3: Modality Mismatch / Manual Review"
            data["risk_tier"] = "Rejected / Wrong Modality"
            data["gpio_alert_type"] = "RED"
            data["gpio_triggered"] = 1
            data["manual_review_required"] = True
            data["auto_report_allowed"] = False
            data["routing_reason"] = data.get(
                "routing_reason",
                "Selected modality does not match uploaded image."
            )

            response.set_data(json.dumps(data))
            response.headers["Content-Type"] = "application/json"
            response.headers["Content-Length"] = len(response.get_data())
            return response

        pred = data.get("prediction", {})
        if isinstance(pred, dict):
            conf = float(pred.get("mean_confidence", data.get("mean_confidence", 0.0)))
        else:
            conf = float(data.get("mean_confidence", 0.0))

        ood_flag = bool(data.get("ood_flag", False))

        # Preserve modality mismatch as a rejected safety case
        pred_obj = data.get("prediction", {})
        pred_class = ""
        if isinstance(pred_obj, dict):
            pred_class = str(pred_obj.get("class", ""))
        else:
            pred_class = str(data.get("predicted_class", ""))

        if pred_class == "MODALITY_MISMATCH" or bool(data.get("modality_mismatch", False)):
            report_text = str(data.get("llm_report", "")) + " " + str(data.get("report", ""))
            detected = data.get("detected_modality")

            if not detected:
                m = re.search(r"closest to ['\"]([^'\"]+)['\"]", report_text)
                detected = m.group(1) if m else "unknown"

            data["status"] = "rejected"
            data["ood_flag"] = True
            data["modality_mismatch"] = True
            data["selected_modality"] = data.get("modality", "")
            data["detected_modality"] = detected

            data["routing_tier"] = 3
            data["routing_decision"] = "Tier 3: Modality Mismatch / Manual Review"
            data["risk_tier"] = "Rejected / Wrong Modality"
            data["gpio_alert_type"] = "RED"
            data["gpio_triggered"] = 1
            data["manual_review_required"] = True
            data["auto_report_allowed"] = False
            data["routing_reason"] = (
                "Selected modality does not match uploaded image."
                if detected == "unknown"
                else "Selected modality does not match uploaded image. Detected image type: {}.".format(detected)
            )

            response.set_data(json.dumps(data))
            response.headers["Content-Type"] = "application/json"
            response.headers["Content-Length"] = len(response.get_data())
            return response

        if ood_flag or str(data.get("status", "")).lower() == "rejected":
            tier = 3
            routing_decision = "Tier 3: Rejected / OOD Review"
            risk_tier = "Rejected / OOD Review"
            gpio_alert_type = "RED"
            manual_review_required = True
            auto_report_allowed = False
            routing_reason = "Out-of-distribution or rejected scan requires manual review."
        elif conf >= 90.0:
            tier = 1
            routing_decision = "Tier 1: Auto-SR"
            risk_tier = "High Confidence / Auto-SR"
            gpio_alert_type = "GREEN"
            manual_review_required = False
            auto_report_allowed = True
            routing_reason = "Confidence is >= 90%."
        elif conf >= 70.0:
            tier = 2
            routing_decision = "Tier 2: AI-Assisted Review"
            risk_tier = "Medium Confidence / Doctor Review"
            gpio_alert_type = "YELLOW"
            manual_review_required = True
            auto_report_allowed = True
            routing_reason = "Confidence is between 70% and 90%."
        else:
            tier = 3
            routing_decision = "Tier 3: Urgent Manual Review"
            risk_tier = "Low Confidence / Urgent Review"
            gpio_alert_type = "RED"
            manual_review_required = True
            auto_report_allowed = False
            routing_reason = "Confidence is below 70% or scan requires urgent review."

        data["routing_tier"] = tier
        data["routing_decision"] = routing_decision
        data["risk_tier"] = risk_tier
        data["gpio_alert_type"] = gpio_alert_type
        data["gpio_triggered"] = 1 if gpio_alert_type == "RED" else 0
        data["manual_review_required"] = manual_review_required
        data["auto_report_allowed"] = auto_report_allowed
        data["routing_reason"] = routing_reason

        response.set_data(json.dumps(data))
        response.headers["Content-Type"] = "application/json"
        response.headers["Content-Length"] = len(response.get_data())

    except Exception as e:
        print("[FORCE ROUTING HOOK WARNING]", str(e))

    return response



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask API Full — FYP Medical Imaging (M3 Days 5-20)"
    )
    parser.add_argument("--host",           default=FLASK_HOST)
    parser.add_argument("--port",           type=int, default=FLASK_PORT)
    parser.add_argument("--no-stub",        action="store_true",
                        help="Disable stub mode (real TRT, Day 7+)")
    parser.add_argument("--precision",      default=PRECISION_MODE,
                        choices=["fp32", "fp16", "int8"],
                        help="TRT precision mode (Day 12)")
    parser.add_argument("--enable-cfind",   action="store_true",
                        help="Start DICOM C-FIND autonomous polling (Day 18)")
    parser.add_argument("--enable-al-auto", action="store_true",
                        help="Start Active Learning auto-correction daemon (Day 7)")
    parser.add_argument("--test",           action="store_true",
                        help="Run self-test and exit")
    args = parser.parse_args()

    if args.no_stub:
        STUB_MODE = False
        print("[CONFIG] Stub mode DISABLED")

    PRECISION_MODE = args.precision
    print(f"[CONFIG] Precision mode: {PRECISION_MODE}")

    if args.test:
        run_self_test()
        sys.exit(0)

    # Init DB
    database = get_db()
    print(f"[DB] {'Connected → ' + database.db_path if database else 'WARNING: no database'}")

    # Start DICOM C-FIND daemon (Day 18)
    if args.enable_cfind and CFIND_AVAILABLE:
        _cfind = DicomCFindPoller(db=database)
        _cfind.start()

    # Start Active Learning automation (Day 7)
    if args.enable_al_auto and AL_AVAILABLE:
        _al_auto = ActiveLearningAutomation(db=database)
        _al_auto.start()

    print(f"[SYSTEM] Flask API starting on http://{args.host}:{args.port}")
    print(f"[SYSTEM] Stub mode   : {'ON' if STUB_MODE else 'OFF'}")
    print(f"[SYSTEM] Precision   : {PRECISION_MODE}")
    print(f"[SYSTEM] C-FIND      : {'ON' if args.enable_cfind else 'OFF'}")
    print(f"[SYSTEM] AL auto     : {'ON' if args.enable_al_auto else 'OFF'}")
    print(f"[SYSTEM] WebSocket   : {'ON (/stream)' if WEBSOCKET_AVAILABLE else 'OFF (pip install flask-sock)'}")
    print(f"[SYSTEM] Endpoints   : /analyze /analyze_patient /correct /history "
          f"/queue /health /federated_sync /fusion_compare /stats /stream")

    app.run(host=args.host, port=args.port,
            debug=FLASK_DEBUG, threaded=True)
