import onnxruntime as ort
import numpy as np

class OODSecurityGuard:
    def __init__(self, ood_model_path, threshold=0.05):
        print(f"[STAGE 1] Initializing Security Guard: {ood_model_path}")
        # Using CPUExecutionProvider for OOD to save GPU RAM for Stage 2 TRT
        self.session = ort.InferenceSession(ood_model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def check_scan_validity(self, image_tensor):
        """
        Passes the image through the autoencoder. 
        If it cannot reconstruct the image well (high MSE), it's Out-Of-Distribution (fake/corrupted).
        """
        # Forward pass through the autoencoder
        reconstructed_image = self.session.run(None, {self.input_name: image_tensor})[0]
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean(np.square(image_tensor - reconstructed_image))
        
        if mse > self.threshold:
            print(f"[SECURITY ALERT] Scan Rejected! MSE: {mse:.4f} > {self.threshold}")
            return {"ood_flag": True, "status": "UNKNOWN PATHOLOGY", "mse": mse}
        else:
            print(f"[SECURITY PASSED] Valid Medical Scan. MSE: {mse:.4f}")
            return {"ood_flag": False, "status": "APPROVED", "mse": mse}

# ==========================================
# QUICK VERIFICATION TEST
# ==========================================
if __name__ == "__main__":
    # Point to Hamid's MRI OOD Guard
    ood_path = "../jetson_deployment_package/mri_ood_guard.onnx"
    
    guard = OODSecurityGuard(ood_path, threshold=0.08)
    
    # 1. Test with a blank/random image (Should be REJECTED)
    print("\n--- Testing Random Noise (Fake Scan) ---")
    fake_scan = np.random.randn(1, 3, 224, 224).astype(np.float32)
    guard.check_scan_validity(fake_scan)
