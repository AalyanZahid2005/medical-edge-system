import numpy as np
import time

# Core AI Stages
from stage1_ood_guard import OODSecurityGuard
from stage2_tensorrt import TensorRT_Stage2
from stage3_postprocessing import Stage3_PostProcessor

# System Integrations (From Day 0 and Day 1)
import gpio_alerts
import db_manager

class MedicalEdgePipeline:
    def __init__(self):
        print("=== INITIALIZING NEUROEDGE DIAGNOSTIC HUB ===")
        self.ood_guard = OODSecurityGuard("../../jetson_deployment_package/mri_ood_guard.onnx", threshold=0.08)
        self.trt_engine = TensorRT_Stage2("../../jetson_deployment_package/brain_mri_fp16.engine", "brain_mri")
        self.post_processor = Stage3_PostProcessor(mc_passes=20)
        
        # Initialize Hardware Pins
        gpio_alerts.setup_pins()
        print("=== SYSTEM ONLINE AND READY FOR INFERENCE ===")

    def process_scan(self, patient_id, image_tensor, class_names):
        print(f"\n[PIPELINE] Processing Scan for Patient {patient_id}...")
        start_time = time.time()
        
        # 1. OOD SECURITY CHECK
        security_report = self.ood_guard.check_scan_validity(image_tensor)
        if security_report["ood_flag"]:
            print(">> PIPELINE ABORTED: Anomaly Detected.")
            
            # HARDWARE ALERT: Trigger Red LED + Buzzer for rejected scan
            gpio_alerts.trigger_emergency_alarm()
            
            # DATABASE: Log the rejection
            db_manager.insert_record(patient_id, "UNKNOWN PATHOLOGY", "Rejected: OOD", ood_flag=True)
            return {"status": "REJECTED"}
            
        # 2. TENSORRT FP16 INFERENCE
        trt_output = self.trt_engine.execute_inference(image_tensor)
        
        # 3. MC DROPOUT & LLM REPORT
        mean_conf, std_dev = self.post_processor.apply_mc_dropout(trt_output)
        predicted_class_idx = np.argmax(mean_conf)
        predicted_class_name = class_names[predicted_class_idx]
        confidence_pct = float(np.max(mean_conf)) * 100
        
        final_report = self.post_processor.generate_phi2_report_stub(
            class_name=predicted_class_name, mean_conf=mean_conf, std_dev=std_dev
        )
        
        # 4. ROUTING LOGIC & HARDWARE TRIGGER
        if confidence_pct > 90.0:
            print(">> HIGH CONFIDENCE: Auto-Routing to SR.")
            gpio_alerts.trigger_success_led() # Green LED
            routing_tier = "Tier 1: Auto-SR"
        elif confidence_pct > 70.0:
            print(">> MEDIUM CONFIDENCE: Flagging for AI-Assisted Review.")
            gpio_alerts.trigger_warning_led() # Yellow/Blinking LED
            routing_tier = "Tier 2: AI-Assisted"
        else:
            print(">> LOW CONFIDENCE: Emergency Worklist Routing.")
            gpio_alerts.trigger_emergency_alarm() # Red LED + Buzzer
            routing_tier = "Tier 3: Urgent Review"

        # 5. SQLITE LOGGING
        db_manager.insert_record(patient_id, predicted_class_name, final_report, confidence_pct, routing_tier)
        
        total_time = (time.time() - start_time) * 1000
        print(f"\n[PIPELINE SUCCESS] End-to-End Latency: {total_time:.2f} ms")
        return {"status": "SUCCESS", "report": final_report}
