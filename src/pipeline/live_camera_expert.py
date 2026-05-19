import cv2
import numpy as np
import time
from master_pipeline_v2 import MedicalEdgePipelineV2
import pyttsx3 # Ensure voice is wired

class LiveEdgeExpert:
    def __init__(self):
        self.pipeline = MedicalEdgePipelineV2()
        self.modalities = ["brain_mri", "chest_xray", "skin_lesion", "retinal", "bone_fracture", "ecg"]
        self.current_idx = 0
        
        self.modality_names = {
            "brain_mri": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
            "chest_xray": ["Normal", "Pneumonia"],
            "skin_lesion": ["Melanoma", "Nevus", "Basal Cell"],
            "retinal": ["Mild", "Moderate", "Severe", "Proliferative DR"],
            "bone_fracture": ["Fractured", "Normal"],
            "ecg": ["Normal", "Arrhythmia", "Myocardial Infarction"]
        }
        
        # Initialize Voice Engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def run(self):
        # 0 is the default USB camera port on Jetson Nano
        cap = cv2.VideoCapture(0) 
        
        print("\n==================================================")
        print("📷 [CAMERA] LIVE Expert System Online")
        print("==================================================")
        
        if not cap.isOpened():
            print("[ERROR] Cannot open physical camera. Is it plugged in?")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            modality = self.modalities[self.current_idx]
            
            # Draw Live HUD
            cv2.putText(frame, f"EXPERT MODE: {modality.upper()}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'N' to switch | 'SPACE' to scan | 'Q' to quit", (20, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('NeuroEdge Live Hub', frame)

            key = cv2.waitKey(1) & 0xFF
            
            # Switch Modality
            if key == ord('n'):
                self.current_idx = (self.current_idx + 1) % len(self.modalities)
                print(f"\n[SYSTEM] Switched to {self.modalities[self.current_idx]} expert.")

            # Trigger Analysis
            elif key == ord(' '):
                print(f"\n[ANALYSIS] Capturing physical scan for {modality}...")
                
                # Format frame for TensorRT
                resized = cv2.resize(frame, (224, 224))
                tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor = np.expand_dims(tensor, axis=0)
                
                classes = self.modality_names.get(modality, ["Class A", "Class B"])
                self.pipeline.process_scan("LIVE-CAM-USER", tensor, classes, modality)

            elif key == ord('q'):
                print("Shutting down physical camera...")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    expert = LiveEdgeExpert()
    expert.run()
