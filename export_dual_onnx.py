import torch
import os
from ood_autoencoder import MedicalOODAutoencoder

# Where to read the .pth files from
INPUT_WEIGHTS_DIR = r"D:\medical-edge-system\production_weights"

# Where to save the new ONNX files
OUTPUT_ONNX_DIR = r"D:\medical-edge-system\jetson_deployment_package"

MODALITIES = ["eyes", "bone", "skin", "pneumonia", "mri", "ecg"]

def export_models():
    print("============================================================")
    print("[SYSTEM] Commencing ONNX Export Pipeline...")
    
    # Auto-create the deployment folder if it doesn't exist yet
    if not os.path.exists(OUTPUT_ONNX_DIR):
        os.makedirs(OUTPUT_ONNX_DIR)
        print(f"[SYSTEM] Created new directory: {OUTPUT_ONNX_DIR}")
    
    # We create a "dummy" image. The exporter feeds this through the model 
    # to trace exactly how the math flows from input to output.
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

    for mod in MODALITIES:
        pth_path = os.path.join(INPUT_WEIGHTS_DIR, f"{mod}_ood_guard.pth")
        onnx_path = os.path.join(OUTPUT_ONNX_DIR, f"{mod}_ood_guard.onnx")
        
        if not os.path.exists(pth_path):
            print(f"[WARNING] Could not find {pth_path}. Skipping.")
            continue
        
        # 1. Load the PyTorch Model
        model = MedicalOODAutoencoder()
        model.load_state_dict(torch.load(pth_path, map_location='cpu'))
        model.eval()
        
        # 2. Export to ONNX
        print(f"-> Exporting {mod.upper()} to ONNX...")
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=11, # Crucial for Jetson Nano TensorRT compatibility
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"   [SUCCESS] Saved: {onnx_path}")

if __name__ == '__main__':
    export_models()