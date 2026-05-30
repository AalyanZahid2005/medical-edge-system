import os
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from model_registry import MODEL_REGISTRY


class ModalityBouncer:
    def __init__(self):
        self.sessions = {}

        for modality, cfg in MODEL_REGISTRY.items():
            ood_path = cfg.get("ood_path")

            if ood_path and os.path.exists(ood_path):
                print("[BOUNCER] Loading OOD guard for {}: {}".format(modality, ood_path))
                session = ort.InferenceSession(
                    ood_path,
                    providers=["CPUExecutionProvider"]
                )
                self.sessions[modality] = {
                    "session": session,
                    "input_name": session.get_inputs()[0].name
                }
            else:
                print("[BOUNCER WARNING] Missing OOD guard for {}".format(modality))

    def _prepare(self, image_tensor):
        arr = np.asarray(image_tensor, dtype=np.float32)

        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)

        if arr.ndim != 4:
            raise ValueError("Expected 4D tensor, got {}".format(arr.shape))

        # NHWC to NCHW
        if arr.shape[1] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))

        if arr.shape[2] != 224 or arr.shape[3] != 224:
            t = torch.from_numpy(arr.astype(np.float32))
            t = F.interpolate(
                t,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )
            arr = t.cpu().numpy()

        return np.ascontiguousarray(arr, dtype=np.float32)

    def score_all(self, image_tensor):
        x = self._prepare(image_tensor)
        scores = {}

        for modality, item in self.sessions.items():
            session = item["session"]
            input_name = item["input_name"]

            recon = session.run(None, {input_name: x})[0].astype(np.float32)

            if recon.shape != x.shape:
                recon = self._prepare(recon)

            mse = float(np.mean(np.square(x - recon)))
            scores[modality] = mse

        return scores

    def validate(self, selected_modality, image_tensor, margin_ratio=1.35):
        """
        Returns whether the selected modality is plausible.

        Lower MSE means the OOD autoencoder reconstructs that modality better.
        If another modality reconstructs much better than the selected one,
        this is a modality mismatch.
        """
        scores = self.score_all(image_tensor)

        if not scores:
            return {
                "valid": True,
                "reason": "No OOD modality guards available.",
                "scores": {},
                "detected_modality": selected_modality,
                "selected_score": None,
                "best_score": None
            }

        best_modality = min(scores, key=scores.get)
        best_score = scores[best_modality]
        selected_score = scores.get(selected_modality)

        if selected_score is None:
            return {
                "valid": False,
                "reason": "No OOD guard exists for selected modality.",
                "scores": scores,
                "detected_modality": best_modality,
                "selected_score": None,
                "best_score": best_score
            }

        # If selected modality is clearly worse than another modality, reject.
        mismatch = (
            best_modality != selected_modality
            and selected_score > best_score * margin_ratio
        )

        if mismatch:
            return {
                "valid": False,
                "reason": "Selected modality '{}' does not match detected image type '{}'."
                          .format(selected_modality, best_modality),
                "scores": scores,
                "detected_modality": best_modality,
                "selected_score": selected_score,
                "best_score": best_score
            }

        return {
            "valid": True,
            "reason": "Selected modality matches modality-bouncer check.",
            "scores": scores,
            "detected_modality": best_modality,
            "selected_score": selected_score,
            "best_score": best_score
        }
