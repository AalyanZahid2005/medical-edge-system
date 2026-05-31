import base64
import io
import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
from torchvision import models, transforms

from model_registry import MODEL_REGISTRY


_MODEL_CACHE = {}

WEIGHT_FILES = {
    "brain_mri": "mri_classifier.pth",
    "chest_xray": "pneumonia_classifier.pth",
    "bone": "bone_classifier.pth",
    "ecg": "ecg_classifier.pth",
    "retinal": "retina_classifier.pth",
    "skin": "skin_classifier.pth",
}


def _build_efficientnet(num_classes):
    try:
        model = models.efficientnet_b0(weights=None)
    except TypeError:
        model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _load_model(modality, device="cpu"):
    key = (modality, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if modality not in MODEL_REGISTRY:
        raise ValueError("Unknown modality: {}".format(modality))

    class_names = MODEL_REGISTRY[modality]["class_names"]
    weight_name = WEIGHT_FILES.get(modality)

    if not weight_name:
        raise ValueError("No PyTorch weight mapping for modality: {}".format(modality))

    weight_path = os.path.join("/app/production_weights", weight_name)
    if not os.path.exists(weight_path):
        weight_path = os.path.join("production_weights", weight_name)

    if not os.path.exists(weight_path):
        raise FileNotFoundError("Missing Grad-CAM weight file: {}".format(weight_path))

    model = _build_efficientnet(len(class_names))
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    _MODEL_CACHE[key] = (model, class_names)
    return model, class_names


def _make_overlay(original_rgb, cam, output_size=(384, 384)):
    original = original_rgb.resize(output_size).convert("RGB")

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(output_size)
    cam_img = cam_img.filter(ImageFilter.GaussianBlur(radius=2))

    cam_np = np.asarray(cam_img).astype(np.float32) / 255.0

    # Keep only strongest attention area so the affected region is highlighted.
    threshold = np.percentile(cam_np, 72)
    mask = (cam_np - threshold) / (cam_np.max() - threshold + 1e-8)
    mask = np.clip(mask, 0, 1)

    # Red/yellow heatmap
    heat_np = np.zeros((output_size[1], output_size[0], 4), dtype=np.uint8)
    heat_np[..., 0] = 255
    heat_np[..., 1] = np.uint8(120 * mask)
    heat_np[..., 2] = 20
    heat_np[..., 3] = np.uint8(190 * mask)

    heat = Image.fromarray(heat_np, mode="RGBA")
    base = original.convert("RGBA")
    overlay = Image.alpha_composite(base, heat)

    out = io.BytesIO()
    overlay.convert("RGB").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("utf-8")


def generate_real_gradcam_base64(image_path, modality, target_class=None, device="cpu"):
    """
    True Grad-CAM using PyTorch gradients and EfficientNet feature maps.
    No cv2 and no pytorch_grad_cam dependency required.
    """
    try:
        torch.set_num_threads(1)

        model, class_names = _load_model(modality, device=device)

        img = Image.open(image_path).convert("RGB")
        img = ImageOps.exif_transpose(img)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        x = preprocess(img).unsqueeze(0).to(device)

        activations = {}
        gradients = {}

        target_layer = model.features[-1]

        def forward_hook(module, inp, out):
            activations["value"] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0].detach()

        fh = target_layer.register_forward_hook(forward_hook)

        try:
            bh = target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            bh = target_layer.register_backward_hook(backward_hook)

        logits = model(x)

        if target_class and target_class in class_names:
            target_idx = class_names.index(target_class)
        else:
            target_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_idx]

        model.zero_grad()
        score.backward()

        fh.remove()
        bh.remove()

        acts = activations["value"]
        grads = gradients["value"]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)

        cam = cam[0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return _make_overlay(img, cam)

    except Exception as e:
        print("[REAL GRADCAM WARNING]", e)
        return None
