import os
import torch
import numpy as np
import cv2
import shap
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models

class MedicalShapExplainer:
    def __init__(self, model, device="cpu"):
        """
        Initializes the SHAP Explainer.
        We use GradientExplainer because it is optimized for CNNs like EfficientNet.
        """
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        
        # Preprocessing matching your main pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def generate_heatmap(self, image_path, save_path="explanation_heatmap.png"):
        print(f"[SHAP] Generating explainability heatmap for {image_path}...")
        
        # 1. Load and prepare the image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # 2. We need a "Background" tensor for SHAP to compare against
        background = torch.zeros((1, 3, 224, 224)).to(self.device)
        
        # 3. Initialize the Gradient Explainer
        explainer = shap.GradientExplainer(self.model, background)
        
        # 4. Calculate SHAP values (The heavy math)
        shap_values, indexes = explainer.shap_values(input_tensor)
        
        # 5. Format for visualization (C, H, W) -> (H, W, C)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        image_numpy = np.swapaxes(np.swapaxes(input_tensor.cpu().numpy(), 1, -1), 1, 2)
        
        # 6. Plot and save the overlay
        plt.figure()
        shap.image_plot(shap_numpy, -image_numpy, show=False)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"[SUCCESS] Heatmap saved to {save_path}. Ready for doctor review.")

# --- Test Execution ---
if __name__ == "__main__":
    # 1. Dynamic Hardware Check (Looks for your RTX 2050)
    hardware_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[SYSTEM] Initializing SHAP Explainer on: {hardware_device.upper()}")
    
    if hardware_device == "cpu":
        print("[WARNING] PyTorch cannot find your GPU. Processing will take 1-3 minutes.")
    else:
        print("[INFO] RTX 2050 detected! Processing will be fast.")

    # 2. Load Model 
    dummy_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) 
    
    # 3. Initialize Explainer with the detected hardware
    explainer = MedicalShapExplainer(dummy_model, device=hardware_device)
    
    # 4. The Bulletproof File Finder 
    base_path = r"D:\medical-edge-system\FYP_Demo_Assets\01_Pneumonia"
    
    if os.path.exists(base_path + ".jpg"):
        demo_image = base_path + ".jpg"
    elif os.path.exists(base_path + ".jpeg"):
        demo_image = base_path + ".jpeg"
    elif os.path.exists(base_path + ".jpg.jpg"): 
        demo_image = base_path + ".jpg.jpg"
    else:
        print(f"\n[CRITICAL ERROR] The image is completely missing from FYP_Demo_Assets!")
        exit()
        
    print(f"[INFO] Successfully found image at: {demo_image}")
    
    # 5. Generate Heatmap
    explainer.generate_heatmap(demo_image, "pneumonia_shap_result.png")