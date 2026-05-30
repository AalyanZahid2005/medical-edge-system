import os
import sys
import json
import time
import uuid
import subprocess
import threading
import numpy as np

from stage3_postprocessing import Stage3_PostProcessor
from model_registry import get_model_config
from modality_bouncer import ModalityBouncer

import gpio_alerts
import db_manager
from voice_alert import VoiceAnnouncer
from ewc_trainer_real import EWCTrainer
from fedavg_server import FederatedAggregationServer


class MedicalEdgePipelineV2:
    def __init__(self):
        print("==================================================")
        print("===   INITIALIZING NEUROEDGE HUB V2.0          ===")
        print("==================================================")

        self.voice = VoiceAnnouncer()
        self.post_processor = Stage3_PostProcessor(mc_passes=20)

        # Cross-modality safety bouncer:
        # prevents MRI image being analyzed as X-ray/ECG/skin/etc.
        self.modality_bouncer = ModalityBouncer()
        self.ewc_trainer = EWCTrainer(ewc_lambda=0.4)
        self.fed_server = FederatedAggregationServer()

        print("\n=== SYSTEM ONLINE AND READY FOR ISOLATED MULTI-MODAL INFERENCE ===")

    def _safe_voice(self, message, priority="NORMAL"):
        try:
            self.voice.announce(message, priority=priority)
        except Exception as e:
            print("[VOICE WARNING] Skipped voice announcement: {}".format(e))

    def _safe_gpio_success(self):
        try:
            if hasattr(gpio_alerts, "trigger_success_led"):
                gpio_alerts.trigger_success_led()
            elif hasattr(gpio_alerts, "trigger_green_led"):
                gpio_alerts.trigger_green_led()
            else:
                print("[GPIO] SUCCESS simulated")
        except Exception as e:
            print("[GPIO WARNING] Success LED skipped: {}".format(e))

    def _safe_gpio_warning(self):
        try:
            if hasattr(gpio_alerts, "trigger_warning_led"):
                gpio_alerts.trigger_warning_led()
            elif hasattr(gpio_alerts, "trigger_yellow_led"):
                gpio_alerts.trigger_yellow_led()
            else:
                print("[GPIO] WARNING simulated")
        except Exception as e:
            print("[GPIO WARNING] Warning LED skipped: {}".format(e))

    def _safe_gpio_emergency(self):
        try:
            if hasattr(gpio_alerts, "trigger_emergency_alarm"):
                gpio_alerts.trigger_emergency_alarm()
            elif hasattr(gpio_alerts, "trigger_emergency_alert"):
                gpio_alerts.trigger_emergency_alert()
            elif hasattr(gpio_alerts, "trigger_error_led"):
                gpio_alerts.trigger_error_led()
            else:
                print("[GPIO] EMERGENCY simulated")
        except Exception as e:
            print("[GPIO WARNING] Emergency alert skipped: {}".format(e))

    def _safe_db_insert(self, patient_id, pred_name, report, conf_pct, routing):
        try:
            if hasattr(db_manager, "insert_record"):
                return db_manager.insert_record(patient_id, pred_name, report, conf_pct, routing)
            return False
        except Exception as e:
            print("[DB WARNING] insert_record failed: {}".format(e))
            return False

    def _decide_routing(self, conf_pct, ood_flag=False):
        try:
            conf_pct = float(conf_pct)
        except Exception:
            conf_pct = 0.0

        if ood_flag:
            return {
                "routing_tier": 3,
                "routing_decision": "Tier 3: Rejected / OOD Review",
                "risk_tier": "Rejected / OOD Review",
                "gpio_alert_type": "RED",
                "manual_review_required": True,
                "auto_report_allowed": False,
                "reason": "Out-of-distribution scan rejected by OOD guard."
            }

        if conf_pct >= 90.0:
            return {
                "routing_tier": 1,
                "routing_decision": "Tier 1: Auto-SR",
                "risk_tier": "High Confidence / Auto-SR",
                "gpio_alert_type": "GREEN",
                "manual_review_required": False,
                "auto_report_allowed": True,
                "reason": "Confidence is >= 90%."
            }

        if conf_pct >= 70.0:
            return {
                "routing_tier": 2,
                "routing_decision": "Tier 2: AI-Assisted Review",
                "risk_tier": "Medium Confidence / Doctor Review",
                "gpio_alert_type": "YELLOW",
                "manual_review_required": True,
                "auto_report_allowed": True,
                "reason": "Confidence is between 70% and 90%."
            }

        return {
            "routing_tier": 3,
            "routing_decision": "Tier 3: Urgent Manual Review",
            "risk_tier": "Low Confidence / Urgent Review",
            "gpio_alert_type": "RED",
            "manual_review_required": True,
            "auto_report_allowed": False,
            "reason": "Confidence is below 70%."
        }

    def _run_isolated_worker(self, image_tensor, modality_name):
        token = str(uuid.uuid4())
        input_path = "/tmp/medai_input_{}.npy".format(token)
        output_path = "/tmp/medai_output_{}.json".format(token)

        np.save(input_path, np.asarray(image_tensor, dtype=np.float32))

        worker_path = os.path.join(os.path.dirname(__file__), "trt_isolated_worker.py")

        cmd = [
            sys.executable,
            worker_path,
            "--modality", str(modality_name),
            "--input", input_path,
            "--output", output_path
        ]

        print("[ISOLATED WORKER] Running:", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=220
        )

        if proc.stdout:
            print("[WORKER STDOUT]", proc.stdout[-2000:])

        if proc.stderr:
            print("[WORKER STDERR]", proc.stderr[-2000:])

        if not os.path.exists(output_path):
            return {
                "status": "ERROR",
                "error_type": "NO_WORKER_OUTPUT",
                "predicted_class": "WORKER_CRASH",
                "mean_confidence": 0.0,
                "uncertainty": 100.0,
                "ood_flag": True,
                "mse": 0.0,
                "message": "Worker did not produce JSON output. Return code: {}".format(proc.returncode),
                "stderr": proc.stderr[-2000:]
            }

        with open(output_path, "r") as f:
            result = json.load(f)

        try:
            os.remove(input_path)
        except Exception:
            pass

        try:
            os.remove(output_path)
        except Exception:
            pass

        return result

    def process_scan(self, patient_id, image_tensor, class_names=None, modality_name="brain_mri"):
        print("\n[PIPELINE] Processing Scan for Patient {} | Modality: {}".format(
            patient_id,
            modality_name
        ))

        start_time = time.time()

        cfg = get_model_config(modality_name)
        if cfg is None:
            cfg = {
                "display_name": modality_name,
                "class_names": []
            }

        # ------------------------------------------------------------
        # Stage 0: Cross-modality validation
        # ------------------------------------------------------------
        bouncer = self.modality_bouncer.validate(modality_name, image_tensor)

        if not bouncer.get("valid", True):
            total_time = (time.time() - start_time) * 1000.0

            selected = str(modality_name)
            detected = str(bouncer.get("detected_modality", "unknown"))
            reason = str(bouncer.get("reason", "Selected modality does not match uploaded image."))

            pred_name = "MODALITY_MISMATCH"
            conf_pct = 0.0
            uncertainty = 100.0

            routing_decision = "Tier 3: Modality Mismatch / Manual Review"
            risk_tier = "Rejected / Wrong Modality"
            gpio_alert_type = "RED"

            report = (
                "SCAN REJECTED: Modality mismatch detected. "
                "Selected modality was '{}', but the image appears closest to '{}'. "
                "No diagnostic classifier was executed. Manual review is required."
            ).format(selected, detected)

            self._safe_gpio_emergency()
            self._safe_db_insert(patient_id, pred_name, report, conf_pct, routing_decision)

            return {
                "status": "REJECTED",
                "api_status": "error",
                "patient_id": str(patient_id),
                "modality": selected,
                "modality_display": selected,

                "predicted_class": pred_name,
                "prediction": pred_name,
                "diagnosis": pred_name,

                "confidence": conf_pct,
                "confidence_percent": conf_pct,
                "mean_confidence": conf_pct,

                "uncertainty": uncertainty,
                "uncertainty_score": uncertainty,
                "std_confidence": uncertainty,
                "std_dev": uncertainty,
                "standard_deviation": uncertainty,

                "report": report,
                "report_text": report,
                "llm_report": report,
                "clinical_report": report,

                "routing": routing_decision,
                "routing_decision": routing_decision,
                "risk_tier": risk_tier,
                "routing_tier": 3,
                "gpio_alert_type": gpio_alert_type,
                "gpio_triggered": 1,
                "manual_review_required": True,
                "auto_report_allowed": False,
                "routing_reason": reason,

                "latency_ms": float(total_time),
                "inference_time_ms": float(total_time),

                "ood_flag": True,
                "modality_mismatch": True,
                "detected_modality": detected,
                "selected_modality": selected,
                "modality_scores": bouncer.get("scores", {}),
                "mse": float(bouncer.get("selected_score") or 0.0),

                "worker_status": "SKIPPED",
                "worker_error_type": "MODALITY_MISMATCH"
            }

        worker = self._run_isolated_worker(image_tensor, modality_name)

        pred_name = str(worker.get("predicted_class", "UNKNOWN"))
        conf_pct = float(worker.get("mean_confidence", 0.0))
        uncertainty = float(worker.get("uncertainty", 100.0))
        ood_flag = bool(worker.get("ood_flag", False))

        route = self._decide_routing(conf_pct, ood_flag=ood_flag)

        if route["gpio_alert_type"] == "GREEN":
            self._safe_gpio_success()
        elif route["gpio_alert_type"] == "YELLOW":
            self._safe_gpio_warning()
        else:
            self._safe_gpio_emergency()

        total_time = (time.time() - start_time) * 1000.0

        if worker.get("status") != "SUCCESS":
            report = "SCAN REJECTED: {}.".format(worker.get("message", route["reason"]))
        else:
            probs = np.asarray(worker.get("probs", []), dtype=np.float32)
            if probs.size == 0:
                probs = np.asarray([conf_pct / 100.0], dtype=np.float32)

            display_std = np.ones_like(probs, dtype=np.float32) * (uncertainty / 100.0)

            try:
                report = self.post_processor.generate_phi2_report_stub(
                    pred_name,
                    probs,
                    display_std
                )
            except Exception:
                report = (
                    "CLINICAL SUMMARY: The AI edge node analyzed a {} scan and detected '{}' "
                    "with confidence {:.1f}%. Routing decision: {}."
                ).format(
                    cfg.get("display_name", modality_name),
                    pred_name,
                    conf_pct,
                    route["routing_decision"]
                )

        self._safe_db_insert(patient_id, pred_name, report, conf_pct, route["routing_decision"])

        self._safe_voice(
            "{} result for patient {}. Prediction {} with confidence {:.1f} percent.".format(
                cfg.get("display_name", modality_name),
                patient_id,
                pred_name,
                conf_pct
            ),
            priority="NORMAL"
        )

        print("[PIPELINE SUCCESS] {} | {} | {:.2f}% | {:.2f} ms".format(
            modality_name,
            pred_name,
            conf_pct,
            total_time
        ))

        return {
            "status": "SUCCESS" if worker.get("status") == "SUCCESS" else "REJECTED",
            "api_status": "success" if worker.get("status") == "SUCCESS" else "error",
            "patient_id": str(patient_id),
            "modality": str(modality_name),
            "modality_display": str(cfg.get("display_name", modality_name)),

            "predicted_class": pred_name,
            "prediction": pred_name,
            "diagnosis": pred_name,

            "confidence": conf_pct,
            "confidence_percent": conf_pct,
            "mean_confidence": conf_pct,

            "uncertainty": uncertainty,
            "uncertainty_score": uncertainty,
            "std_confidence": uncertainty,
            "std_dev": uncertainty,
            "standard_deviation": uncertainty,

            "report": str(report),
            "report_text": str(report),
            "llm_report": str(report),
            "clinical_report": str(report),

            "routing": str(route["routing_decision"]),
            "routing_decision": str(route["routing_decision"]),
            "risk_tier": str(route["risk_tier"]),
            "routing_tier": int(route["routing_tier"]),
            "gpio_alert_type": str(route["gpio_alert_type"]),
            "gpio_triggered": 1 if route["gpio_alert_type"] == "RED" else 0,
            "manual_review_required": bool(route["manual_review_required"]),
            "auto_report_allowed": bool(route["auto_report_allowed"]),
            "routing_reason": str(route["reason"]),

            "latency_ms": float(total_time),
            "inference_time_ms": float(total_time),

            "ood_flag": ood_flag,
            "mse": float(worker.get("mse", 0.0)),
            "worker_status": worker.get("status"),
            "worker_error_type": worker.get("error_type", "")
        }

    def trigger_ewc_correction(self, patient_id, old_pred, true_label, modality):
        self._safe_voice(
            "Doctor correction received for {}. Initiating continual learning.".format(patient_id),
            priority="NORMAL"
        )

        def _train_and_swap():
            try:
                self.ewc_trainer.process_doctor_correction(
                    patient_id,
                    old_pred,
                    true_label,
                    modality
                )
            except Exception as e:
                print("[EWC WARNING] Correction training failed: {}".format(e))

        t = threading.Thread(target=_train_and_swap)
        t.daemon = True
        t.start()

    def run_federated_sync(self):
        self._safe_voice("Initiating global federated synchronization.", priority="NORMAL")

        def _sync_nodes():
            try:
                dummy_weights = np.random.randn(512, 128).astype(np.float32)
                self.fed_server.receive_hospital_update("Remote Node A", dummy_weights, 1000)
                self.fed_server.receive_hospital_update("Remote Node B", dummy_weights, 1500)
                self.fed_server.execute_fedavg()
            except Exception as e:
                print("[FEDERATED WARNING] Sync failed: {}".format(e))

        t = threading.Thread(target=_sync_nodes)
        t.daemon = True
        t.start()


if __name__ == "__main__":
    pipeline = MedicalEdgePipelineV2()
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for m in ["brain_mri", "chest_xray", "bone", "ecg", "retinal", "skin"]:
        print(pipeline.process_scan("PAT-TEST", x, None, m))
