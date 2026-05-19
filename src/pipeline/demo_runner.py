import time
import cv2
import os
import numpy as np
import Jetson.GPIO as GPIO
from master_pipeline_v2 import MedicalEdgePipelineV2

def run_defense_demo():
    print("\n" + "★"*60)
    print("🚀 INITIATING FINAL FYP DEFENSE DEMO: 2022-FYDP-28")
    print("★"*60)
    
    # 0. Initialize System
    pipeline = MedicalEdgePipelineV2()
    time.sleep(3) # Let voice finish "System Booting"

    # ---------------------------------------------------------
    # SCENE 1: The Perfect Scan (Tier 1 Routing) -> GREEN LED
    # ---------------------------------------------------------
    input("\n[PAUSED] Press ENTER to run SCENE 1: High-Speed Inference (Tier 1)...")
    print("\n" + "="*50)
    print("🎬 SCENE 1: Routine Chest X-Ray (High Confidence)")
    print("="*50)
    
    chest_classes = ["Normal", "Pneumonia"]
    img_path = "sample_chest_xray.jpg"
    
    # Check if real image exists
    if not os.path.exists(img_path):
        print(f"\n[ERROR] '{img_path}' not found! Run the wget command first.")
        return 

    print(f"[SYSTEM] Loading real medical image from {img_path}...")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    # Preprocess: Normalize and add Batch dimension
    valid_scan = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    valid_scan = np.expand_dims(valid_scan, axis=0)
    
    # Trigger AI Pipeline
    pipeline.process_scan("PAT-001", valid_scan, chest_classes, "chest_xray")
    
    # ---------------------------------------------------------
    # SCENE 2: The Security Bouncer -> RED LED + BUZZER
    # ---------------------------------------------------------
    input("\n[PAUSED] Press ENTER to run SCENE 2: Security Gate (OOD Bouncer)...")
    print("\n" + "="*50)
    print("🎬 SCENE 2: Non-Medical Image Upload Attempt")
    print("="*50)
    
    # Create random noise to trigger the bouncer
    noise_scan = np.random.uniform(0, 10, (1, 3, 224, 224)).astype(np.float32) 
    print("[SYSTEM] Simulating massive MSE spike in Autoencoder...")
    
    pipeline.process_scan("PAT-002-CORRUPT", noise_scan, chest_classes, "chest_xray")

    # ---------------------------------------------------------
    # SCENE 3: Continual Learning & Federated Sync
    # ---------------------------------------------------------
    input("\n[PAUSED] Press ENTER to run SCENE 3: Continual Learning (EWC & FedAvg)...")
    print("\n" + "="*50)
    print("🎬 SCENE 3: Doctor Correction & Network Sync")
    print("="*50)
    
    # Turn off the alarm before starting MLOps scene
    GPIO.output([13, 15], GPIO.LOW)
    
    print("\n[GUI SIMULATION] Doctor selects true label: 'Pneumonia' for PAT-003")
    time.sleep(1)
    
    pipeline.trigger_ewc_correction(
        patient_id="PAT-003", 
        old_pred="Normal", 
        true_label="Pneumonia", 
        modality="chest_xray"
    )
    
    print("\n[WAITING 8 SECONDS] Allowing background EWC Hot-Swap to finish...")
    time.sleep(8) 
    
    pipeline.run_federated_sync()
    time.sleep(5)
    
    print("\n" + "★"*60)
    print("✅ DEMO SEQUENCE COMPLETE. SYSTEM AWAITING LIVE CAMERA INPUT.")
    print("★"*60)

if __name__ == "__main__":
    try:
        run_defense_demo()
    except KeyboardInterrupt:
        print("\n\n[USER STOP] Demo halted. Initiating Emergency GPIO Reset...")
    except Exception as e:
        print(f"\n[SYSTEM CRASH] Error: {e}")
    finally:
        # THE AUTO-RESET GUARD:
        # This turns off everything even if the script crashes or you Ctrl+C.
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup([11, 13, 15], GPIO.OUT)
        GPIO.output([11, 13, 15], GPIO.LOW)
        GPIO.cleanup()
        print("[SYSTEM] GPIO Cleanup Complete. All hardware indicators are OFF.")
