import os

MODEL_REGISTRY = {
    "brain_mri": {
        "enabled": True,
        "engine_path": "./jetson_deployment_package/brain_mri_fp16.engine",
        "ood_path": "./jetson_deployment_package/mri_ood_guard.onnx",
        "class_names": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "display_name": "Brain MRI"
    },
    "chest_xray": {
        "enabled": True,
        "engine_path": "./jetson_deployment_package/chest_xray_fp16.engine",
        "ood_path": "./jetson_deployment_package/pneumonia_ood_guard.onnx",
        "class_names": ["Normal", "Pneumonia"],
        "display_name": "Chest X-ray"
    },
    "bone": {
        "enabled": True,
        "engine_path": "./jetson_deployment_package/bone_fracture_fp16.engine",
        "ood_path": "./jetson_deployment_package/bone_ood_guard.onnx",
        "class_names": ["Fracture", "Normal"],
        "display_name": "Bone X-ray"
    },
    "ecg": {
        "enabled": True,
        "engine_path": "./jetson_deployment_package/ecg_fp16.engine",
        "ood_path": "./jetson_deployment_package/ecg_ood_guard.onnx",
        "class_names": ["Normal", "Abnormal"],
        "display_name": "ECG"
    },
    "retinal": {
        "enabled": True,
        "engine_path": "./jetson_deployment_package/retinal_fp16.engine",
        "ood_path": "./jetson_deployment_package/eyes_ood_guard.onnx",
        "class_names": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
        "display_name": "Retinal Fundus"
    },
    "skin": {
        "enabled": True,
        "engine_path": "./jetson_deployment_package/skin_lesion_fp16.engine",
        "ood_path": "./jetson_deployment_package/skin_ood_guard.onnx",
        "class_names": ["NV", "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"],
        "display_name": "Skin Lesion"
    }
}

def get_model_config(modality):
    if modality not in MODEL_REGISTRY:
        return None

    cfg = dict(MODEL_REGISTRY[modality])
    cfg["engine_exists"] = os.path.exists(cfg["engine_path"])
    cfg["ood_exists"] = bool(cfg.get("ood_path")) and os.path.exists(cfg["ood_path"])
    cfg["runtime_available"] = bool(cfg["enabled"]) and cfg["engine_exists"]
    return cfg
