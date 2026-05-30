import argparse
import json
import os
import sys
import traceback
import numpy as np

sys.path.append(os.path.dirname(__file__))

from model_registry import get_model_config
from stage1_ood_guard import OODSecurityGuard
from stage2_tensorrt import TensorRT_Stage2


def softmax_confidence(raw_output, class_names):
    raw = np.asarray(raw_output, dtype=np.float32).flatten()
    n = min(len(raw), len(class_names))

    if n <= 0:
        return {
            "pred_idx": 0,
            "predicted_class": "UNKNOWN",
            "mean_confidence": 0.0,
            "probs": []
        }

    raw = raw[:n]
    class_names = class_names[:n]

    shifted = raw - np.max(raw)
    exp_conf = np.exp(shifted)
    probs = exp_conf / (np.sum(exp_conf) + 1e-8)

    pred_idx = int(np.argmax(probs))
    pred_name = str(class_names[pred_idx])
    conf_pct = float(np.max(probs)) * 100.0

    return {
        "pred_idx": pred_idx,
        "predicted_class": pred_name,
        "mean_confidence": conf_pct,
        "probs": probs.tolist()
    }


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        cfg = get_model_config(args.modality)

        if cfg is None:
            write_json(args.output, {
                "status": "REJECTED",
                "error_type": "UNKNOWN_MODALITY",
                "modality": args.modality,
                "predicted_class": "UNKNOWN_MODALITY",
                "mean_confidence": 0.0,
                "uncertainty": 100.0,
                "ood_flag": True,
                "mse": 0.0,
                "message": "Selected modality is not present in model registry."
            })
            return

        if not cfg["runtime_available"]:
            write_json(args.output, {
                "status": "REJECTED",
                "error_type": "MODEL_NOT_AVAILABLE",
                "modality": args.modality,
                "predicted_class": "MODEL_NOT_AVAILABLE",
                "mean_confidence": 0.0,
                "uncertainty": 100.0,
                "ood_flag": True,
                "mse": 0.0,
                "message": "TensorRT engine missing or disabled for this modality."
            })
            return

        image_tensor = np.load(args.input).astype(np.float32)

        ood_result = {"ood_flag": False, "mse": 0.0}
        if cfg.get("ood_exists"):
            guard = OODSecurityGuard(cfg["ood_path"], threshold=0.08)
            ood_result = guard.check_scan_validity(image_tensor)

            if ood_result.get("ood_flag", False):
                write_json(args.output, {
                    "status": "REJECTED",
                    "error_type": "OOD_REJECTED",
                    "modality": args.modality,
                    "predicted_class": "UNKNOWN_PATHOLOGY",
                    "mean_confidence": 0.0,
                    "uncertainty": 100.0,
                    "ood_flag": True,
                    "mse": float(ood_result.get("mse", 0.0)),
                    "message": "OOD guard rejected this scan for selected modality."
                })
                return

        trt_model = TensorRT_Stage2(cfg["engine_path"], args.modality)
        raw_output = trt_model.execute_inference(image_tensor)

        pred = softmax_confidence(raw_output, cfg["class_names"])

        uncertainty = 1.0
        if pred["probs"]:
            uncertainty = 1.0 + float(np.std(np.asarray(pred["probs"], dtype=np.float32))) * 2.0
            uncertainty = min(max(uncertainty, 0.0), 100.0)

        write_json(args.output, {
            "status": "SUCCESS",
            "modality": args.modality,
            "modality_display": cfg["display_name"],
            "predicted_class": pred["predicted_class"],
            "mean_confidence": pred["mean_confidence"],
            "uncertainty": uncertainty,
            "probs": pred["probs"],
            "ood_flag": False,
            "mse": float(ood_result.get("mse", 0.0)),
            "engine_path": cfg["engine_path"]
        })

    except Exception as e:
        write_json(args.output, {
            "status": "ERROR",
            "error_type": "WORKER_EXCEPTION",
            "modality": args.modality,
            "predicted_class": "WORKER_ERROR",
            "mean_confidence": 0.0,
            "uncertainty": 100.0,
            "ood_flag": True,
            "mse": 0.0,
            "message": str(e),
            "traceback": traceback.format_exc()
        })


if __name__ == "__main__":
    main()
