import os
import torch
import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
WEIGHTS_DIR = r"D:\medical-edge-system\production_weights"
EXPORT_DIR = r"D:\medical-edge-system\jetson_deployment_package"

# Ensure the deployment folder exists
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# We need the class numbers to load the correct architecture size before converting
MODALITIES = {
    "pneumonia": 2,
    "mri": 4,
    "retina": 5,
    "skin": 7,
    "bone": 2,
    "ecg": 5
}

def export_to_onnx():
    print("============================================================")
    print("[SYSTEM] Starting ONNX Export for TensorRT Compilation...")
    print("============================================================\n")
    
    # TensorRT needs to know the exact shape of the incoming images.
    # We create a "dummy" tensor representing one 224x224 RGB image to trace the graph.
    dummy_input = torch.randn(1, 3, 224, 224)

    for modality, num_classes in MODALITIES.items():
        pth_path = os.path.join(WEIGHTS_DIR, f"{modality}_classifier.pth")
        onnx_path = os.path.join(EXPORT_DIR, f"{modality}_classifier.onnx")

        if not os.path.exists(pth_path):
            print(f"[WARNING] {pth_path} not found. Skipping.")
            continue

        # 1. Load the exact EfficientNet blueprint we used for training
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        # 2. Inject your trained weights into the blueprint
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        
        # CRITICAL: Put the model in evaluation mode. 
        # This disables Dropout so the ONNX graph doesn't randomly delete nodes!
        model.eval() 

        # 3. Export the graph
        print(f"-> Exporting {modality.upper()}...")
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=13, # Opset 13 is highly stable for TensorRT 8.x
            do_constant_folding=True, # Optimizes the math for faster inference
            input_names=['input'],
            output_names=['output'],
            # Allows the Jetson to process batches of images later if needed
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
        )
        print(f"   [SUCCESS] Saved to {onnx_path}")
        
    print("\n[SYSTEM] All models successfully converted to ONNX!")
    print("[SYSTEM] Ready for TensorRT engine compilation.")

if __name__ == '__main__':
    export_to_onnx()