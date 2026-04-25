import torch
import torch.nn as nn
import numpy as np

def generate_dummy_assets():
    print("[SYSTEM] Generating dummy assets to unblock M3 and M4...")
    
    # 1. Generate Dummy FC Weights for True CAM
    # Shape: [num_classes, 1280] (Assuming 3 classes for Brain Tumor prototype)
    dummy_fc_weights = np.random.rand(3, 1280).astype(np.float32)
    np.save('dummy_fc_weights.npy', dummy_fc_weights)
    print("[SAVED] dummy_fc_weights.npy (Shape: 3x1280)")

    # 2. Generate Dummy Dual-Output ONNX Model
    class DummyDualOutputModel(nn.Module):
        """Simulates the EfficientNet B0 dual output for the pipeline."""
        def forward(self, x):
            batch_size = x.size(0)
            # Simulated logits (Classification)
            logits = torch.randn(batch_size, 3)
            # Simulated feature maps (for GradCAM/True CAM)
            features = torch.randn(batch_size, 1280, 7, 7)
            return logits, features

    model = DummyDualOutputModel()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export with dynamic axes for variable batch sizing
    torch.onnx.export(
        model, 
        dummy_input, 
        "dummy_dual_output.onnx",
        export_params=True,
        opset_version=17,
        input_names=['input_image'],
        output_names=['logits', 'features'], 
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    print("[SAVED] dummy_dual_output.onnx (Outputs: logits, features)")

if __name__ == "__main__":
    generate_dummy_assets()