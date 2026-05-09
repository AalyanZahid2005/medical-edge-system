import numpy as np
import time
import threading

# Core AI Stages
from stage1_ood_guard import OODSecurityGuard
from stage2_tensorrt import TensorRT_Stage2
from stage3_postprocessing import Stage3_PostProcessor

# System Integrations
import gpio_alerts
import db_manager
from voice_alert import VoiceAnnouncer

# MLOps (Continual & Federated)
from ewc_trainer import EWCTrainer
from fedavg_server import FederatedAggregationServer

class MedicalEdgePipelineV2:
    def __init__(self):
        print("==================================================")
        print("===   INITIALIZING NEUROEDGE HUB V2.0          ===")
        print("==================================================")
        
        # 1. Voice Subsystem
        self.voice = VoiceAnnouncer()
        self.voice.announce("System booting. Initializing hardware and neural engines.", priority="NORMAL")

        # 2. AI Pipeline Stages
        self.ood_guard = OODSecurityGuard("../../jetson_deployment_package/mri_ood_guard.onnx", threshold=0.08)
        self.trt_engine = TensorRT_Stage2("../../jetson_deployment_package/brain_mri_fp16.engine", "brain_mri")
        self.post_processor = Stage3_PostProcessor(mc_passes=20)
        
        # 3. Hardware & DB
        gpio_alerts.setup_pins()
        
        # 4. MLOps Engines
        self.ewc_trainer = EWCTrainer(ewc_lambda=0.4)
        self.fed_server = FederatedAggregationServer()
        
        print("\n=== SYSTEM ONLINE AND READY FOR INFERENCE ===")
        self.voice.announce("Boot sequence complete. Ready for medical scans.", priority="NORMAL")

    def process_scan(self, patient_id, image_tensor, class_names, modality_name="brain_mri"):
        print(f"\n[PIPELINE] Processing Scan for Patient {patient_id}...")
        start_time = time.time()
        
        # --- STAGE 1: OOD SECURITY CHECK ---
        if self.ood_guard.check_scan_validity(image_tensor)["ood_flag"]:
            print(">> PIPELINE ABORTED: Anomaly Detected.")
            gpio_alerts.trigger_emergency_alarm()
            db_manager.insert_record(patient_id, "UNKNOWN PATHOLOGY", "Rejected: OOD", ood_flag=True)
            self.voice.announce(f"Scan rejected for patient {patient_id}. Corrupted or invalid data.", priority="CRITICAL")
            return {"status": "REJECTED"}
            
        # --- STAGE 2: TENSORRT FP16 INFERENCE ---
        trt_output = self.trt_engine.execute_inference(image_tensor)
        
        # --- STAGE 3: MC DROPOUT & CONFIDENCE ---
        mean_conf, std_dev = self.post_processor.apply_mc_dropout(trt_output)
        pred_idx = np.argmax(mean_conf)
        pred_name = class_names[pred_idx]
        conf_pct = float(np.max(mean_conf)) * 100
        
        # --- ROUTING LOGIC & HARDWARE TRIGGER ---
        if conf_pct > 90.0:
            gpio_alerts.trigger_success_led()
            routing = "Tier 1: Auto-SR"
            self.voice.announce(f"High confidence diagnosis for {patient_id}. {pred_name} detected.", priority="NORMAL")
        elif conf_pct > 70.0:
            gpio_alerts.trigger_warning_led()
            routing = "Tier 2: AI-Assisted"
            self.voice.announce(f"Medium confidence for {patient_id}. Flagged for doctor review.", priority="NORMAL")
        else:
            gpio_alerts.trigger_emergency_alarm()
            routing = "Tier 3: Urgent Review"
            self.voice.announce(f"Critical. Low confidence on {patient_id}. Immediate review required.", priority="CRITICAL")

        # --- SQLITE LOGGING ---
        report = self.post_processor.generate_phi2_report_stub(pred_name, mean_conf, std_dev)
        db_manager.insert_record(patient_id, pred_name, report, conf_pct, routing)
        
        # --- LATENCY MEASUREMENT ---
        total_time = (time.time() - start_time) * 1000
        print(f"\n[PIPELINE SUCCESS] End-to-End Latency: {total_time:.2f} ms")
        return {"status": "SUCCESS", "patient_id": patient_id, "prediction": pred_name, "latency_ms": total_time}

    def trigger_ewc_correction(self, patient_id, old_pred, true_label, modality):
        self.voice.announce(f"Doctor correction received for {patient_id}. Initiating Continual Learning.", priority="NORMAL")
        
        # Run EWC and TRT Recompilation in a background thread so the hospital doesn't pause
        def _train_and_swap():
            self.ewc_trainer.process_doctor_correction(patient_id, old_pred, true_label, modality)
            self.voice.announce(f"Learning complete. TensorRT engine hot-swapped for {modality}.", priority="NORMAL")
            
        t = threading.Thread(target=_train_and_swap)
        t.daemon = True
        t.start()

    def run_federated_sync(self):
        self.voice.announce("Initiating global federated synchronization.", priority="NORMAL")
        
        def _sync_nodes():
            dummy_weights = np.random.randn(512, 128).astype(np.float32)
            self.fed_server.receive_hospital_update("Remote Node A", dummy_weights, 1000)
            self.fed_server.receive_hospital_update("Remote Node B", dummy_weights, 1500)
            self.fed_server.execute_fedavg()
            self.voice.announce("Federated sync complete. Global model updated securely.", priority="NORMAL")
            
        t = threading.Thread(target=_sync_nodes)
        t.daemon = True
        t.start()

# ==========================================
# EXECUTE MERGE DAY 2 DEMO
# ==========================================
if __name__ == "__main__":
    pipeline = MedicalEdgePipelineV2()
    time.sleep(3) # Let boot voice finish
    
    print("\n\n" + "="*50)
    print("SCENE 1: High-Speed End-to-End Inference")
    print("="*50)
    mri_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    valid_scan = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = pipeline.process_scan("PAT-992", valid_scan, mri_classes, "brain_mri")
    
    print("\n\n" + "="*50)
    print("SCENE 2: Doctor Triggers Active Learning (EWC)")
    print("="*50)
    time.sleep(4) # Let audio catch up
    if result["status"] == "SUCCESS":
        pipeline.trigger_ewc_correction(result["patient_id"], result["prediction"], "Corrected-Diagnosis", "brain_mri")
    
    print("\n\n" + "="*50)
    print("SCENE 3: Global Federated Network Sync")
    print("="*50)
    time.sleep(4) 
    pipeline.run_federated_sync()
    
    # Keep the main process alive so background threads can finish their work
    time.sleep(15)
