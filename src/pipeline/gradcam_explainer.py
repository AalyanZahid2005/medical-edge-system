import os
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class FastMedicalExplainer:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        
        # EfficientNet's final "thinking" layer
        self.target_layers = [self.model.features[-1]]
        
        # Initialize the lightning-fast GradCAM
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, image_path, save_path="fast_heatmap.png"):
        print(f"[Grad-CAM] Analyzing {image_path}...")
        
        # 1. Load and format image
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # BGR to RGB
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img_float = np.float32(rgb_img) / 255

        # 2. PyTorch Preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(rgb_img).unsqueeze(0).to(self.device)

        # 3. Generate Heatmap (Takes milliseconds!)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # 4. Overlay red/blue heatmap on the original X-ray
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        # 5. Save
        cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"[SUCCESS] Instant Heatmap saved to {save_path}!")

# --- Test Execution ---
if __name__ == "__main__":
    hardware_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[SYSTEM] Initializing Grad-CAM on: {hardware_device.upper()}")

    # Load Model 
    dummy_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) 
    explainer = FastMedicalExplainer(dummy_model, device=hardware_device)
    
    # Bulletproof File Finder 
    base_path = r"D:\medical-edge-system\FYP_Demo_Assets\01_Pneumonia"
    
    if os.path.exists(base_path + ".jpg"): demo_image = base_path + ".jpg"
    elif os.path.exists(base_path + ".jpeg"): demo_image = base_path + ".jpeg"
    elif os.path.exists(base_path + ".jpg.jpg"): demo_image = base_path + ".jpg.jpg"
    else:
        print("[CRITICAL ERROR] Image missing!")
        exit()
        
    explainer.generate_heatmap(demo_image, "pneumonia_gradcam_result.png")