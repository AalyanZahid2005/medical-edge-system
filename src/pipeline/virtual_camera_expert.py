import cv2
import numpy as np
import time
from master_pipeline_v2 import MedicalEdgePipelineV2

class VirtualEdgeExpert:
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

    def generate_virtual_frame(self):
        # Generates a synthetic camera feed in memory (no display needed)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 30, 20) 
        noise = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
        return cv2.add(frame, noise)

    def run(self):
        print("\n==================================================")
        print("📷 [CAMERA] Virtual Expert System Online (Headless)")
        print("==================================================")
        
        while True:
            modality = self.modalities[self.current_idx]
            print(f"\n--- CURRENT EXPERT: {modality.upper()} ---")
            cmd = input("COMMANDS: Type 's' to Scan | 'n' to Next Modality | 'q' to Quit: ").strip().lower()
            
            if cmd == 'n':
                self.current_idx = (self.current_idx + 1) % len(self.modalities)
                print(f"[SYSTEM] Switched to {self.modalities[self.current_idx]} expert.")
                
            elif cmd == 's':
                print(f"\n[ANALYSIS] Capturing virtual scan for {modality}...")
                
                # Generate frame in memory and process it
                frame = self.generate_virtual_frame()
                resized = cv2.resize(frame, (224, 224))
                tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                tensor = np.expand_dims(tensor, axis=0)
                
                classes = self.modality_names.get(modality, ["Class A", "Class B"])
                self.pipeline.process_scan("V-CAM-USER", tensor, classes, modality)
                
            elif cmd == 'q':
                print("Shutting down NeuroEdge Hub...")
                break

if __name__ == "__main__":
    expert = VirtualEdgeExpert()
    expert.run()
