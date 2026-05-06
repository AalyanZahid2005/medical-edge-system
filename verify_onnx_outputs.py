import torch
import onnxruntime as ort
import numpy as np
import os
from ood_autoencoder import MedicalOODAutoencoder

# 1. Read PyTorch models from here
INPUT_WEIGHTS_DIR = r"D:\medical-edge-system\production_weights"

# 2. Read the new ONNX models from here
OUTPUT_ONNX_DIR = r"D:\medical-edge-system\jetson_deployment_package"

MODALITIES = ["eyes", "bone", "skin", "pneumonia", "mri", "ecg"]

def verify_models():
    print("============================================================")
    print("[SYSTEM] Commencing PyTorch vs ONNX Verification...")
    
    for mod in MODALITIES:
        pth_path = os.path.join(INPUT_WEIGHTS_DIR, f"{mod}_ood_guard.pth")
        onnx_path = os.path.join(OUTPUT_ONNX_DIR, f"{mod}_ood_guard.onnx")
        
        if not os.path.exists(pth_path) or not os.path.exists(onnx_path):
            print(f"[WARNING] Missing files for {mod.upper()}. Skipping.")
            continue
            
        print(f"\n-> Verifying {mod.upper()}...")
        
        # 1. Generate a single random test tensor
        test_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        
        # 2. Get PyTorch Output
        model_pt = MedicalOODAutoencoder()
        model_pt.load_state_dict(torch.load(pth_path, map_location='cpu'))
        model_pt.eval()
        with torch.no_grad():
            pytorch_output = model_pt(test_tensor).numpy()
            
        # 3. Get ONNX Output
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: test_tensor.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 4. Compare the Math Strictness
        diff = np.abs(pytorch_output - onnx_output).max()
        
        if np.allclose(pytorch_output, onnx_output, atol=1e-4):
            print(f"   [PASS] Max difference is {diff:.8f}. ONNX is perfectly aligned.")
        else:
            print(f"   [FAIL] Max difference is {diff:.8f}. TRT CRASH RISK HIGH.")

if __name__ == '__main__':
    verify_models()