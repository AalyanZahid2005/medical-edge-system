import onnxruntime as ort
import torch
import torch.nn.functional as F
import numpy as np


class OODSecurityGuard:
    def __init__(self, ood_model_path, threshold=0.05):
        print(f"[STAGE 1] Initializing Security Guard: {ood_model_path}")

        # CPUExecutionProvider saves GPU RAM for TensorRT inference
        self.session = ort.InferenceSession(
            ood_model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def _prepare_for_ood(self, image_tensor):
        """
        Converts input to NCHW float32 and resizes it to 224x224,
        because the OOD autoencoder expects/reconstructs 224x224.
        """
        if isinstance(image_tensor, torch.Tensor):
            arr = image_tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(image_tensor)

        arr = arr.astype(np.float32)

        # If image is CHW, make it NCHW
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)

        if arr.ndim != 4:
            raise ValueError(f"OOD expected 4D tensor, got shape {arr.shape}")

        # If NHWC, convert to NCHW
        if arr.shape[1] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))

        # Resize only for OOD guard
        if arr.shape[2] != 224 or arr.shape[3] != 224:
            t = torch.from_numpy(arr)
            t = F.interpolate(
                t,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )
            arr = t.cpu().numpy()

        return np.ascontiguousarray(arr, dtype=np.float32)

    def check_scan_validity(self, image_tensor):
        """
        Passes the image through the autoencoder.
        High reconstruction MSE means Out-Of-Distribution.
        """
        ood_input = self._prepare_for_ood(image_tensor)

        reconstructed_image = self.session.run(
            None,
            {self.input_name: ood_input}
        )[0].astype(np.float32)

        if reconstructed_image.shape != ood_input.shape:
            raise ValueError(
                f"OOD shape mismatch after resize: input {ood_input.shape}, "
                f"reconstructed {reconstructed_image.shape}"
            )

        mse = float(np.mean(np.square(ood_input - reconstructed_image)))

        if mse > self.threshold:
            print(f"[SECURITY ALERT] Scan Rejected! MSE: {mse:.4f} > {self.threshold}")
            return {"ood_flag": True, "status": "UNKNOWN PATHOLOGY", "mse": mse}
        else:
            print(f"[SECURITY PASSED] Valid Medical Scan. MSE: {mse:.4f}")
            return {"ood_flag": False, "status": "APPROVED", "mse": mse}


if __name__ == "__main__":
    ood_path = "../jetson_deployment_package/mri_ood_guard.onnx"

    guard = OODSecurityGuard(ood_path, threshold=0.08)

    print("\n--- Testing Random Noise Fake Scan ---")
    fake_scan = np.random.randn(1, 3, 512, 512).astype(np.float32)
    guard.check_scan_validity(fake_scan)
