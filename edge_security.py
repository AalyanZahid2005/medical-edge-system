import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Import your blueprint
from ood_autoencoder import MedicalOODAutoencoder

class EdgeSecuritySystem:
    def __init__(self, weights_directory):
        self.weights_dir = weights_directory
        # Automatically use Jetson Nano GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SYSTEM] Edge Security Initialized on: {self.device.type.upper()}")
        
        # Must match exact preprocessing from training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def scan_image(self, image_path, modality):
        """Scans an incoming image to see if it belongs to the claimed modality."""
        modality = modality.lower()
        
        # 1. Locate the specific security guard files
        model_path = os.path.join(self.weights_dir, f"{modality}_ood_guard.pth")
        threshold_path = os.path.join(self.weights_dir, f"{modality}_threshold.npy")
        
        if not os.path.exists(model_path) or not os.path.exists(threshold_path):
            return f"[ERROR] Missing security files for {modality.upper()}."

        # 2. Load the mathematical threshold
        # .item() extracts the single number from the numpy array
        max_allowed_error = np.load(threshold_path).item() 

        # 3. Load the model into the GPU
        model = MedicalOODAutoencoder().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval() # Set to evaluation mode (turns off training mechanics)

        # 4. Load and preprocess the new patient image
        try:
            # Convert to RGB to ensure 3 color channels even if grayscale
            image = Image.open(image_path).convert('RGB') 
            img_tensor = self.transform(image).unsqueeze(0).to(self.device) # Add batch dimension
        except Exception as e:
            return f"[ERROR] Could not read image: {e}"

        # 5. Run the Security Scan
        extract_loss_fn = nn.MSELoss(reduction='none')
        
        with torch.no_grad():
            reconstructed = model(img_tensor)
            # Calculate error across color, height, and width
            image_error = extract_loss_fn(reconstructed, img_tensor).mean(dim=[1, 2, 3]).item()

        # 6. Make the Decision
        print(f"\n--- SCAN RESULTS FOR {modality.upper()} ---")
        print(f"Patient Image Error:  {image_error:.6f}")
        print(f"Security Threshold:   {max_allowed_error:.6f}")
        
        if image_error <= max_allowed_error:
            return "[RESULT] ACCEPTED: Image structure matches training data."
        else:
            return "[RESULT] REJECTED: Out-of-Distribution (OOD) data detected. Invalid scan."

# ==========================================
# TEST BED: Run this to test the system!
# ==========================================
if __name__ == "__main__":
    # Point this to your folder with the 18 files
    WEIGHTS_FOLDER = r"D:\medical-edge-system\production_weights"
    
    # Initialize the scanner
    scanner = EdgeSecuritySystem(WEIGHTS_FOLDER)
    
    # --- HOW TO TEST ---
    # 1. Find an image of a chest x-ray inside your Pneumonia training folder.
    # 2. Paste its path below. It should say "ACCEPTED".
    # 3. Then, pass that SAME chest x-ray path, but claim it is an "EYES" scan. 
    #    The system should immediately say "REJECTED".
    
    test_image = r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\train\NORMAL\IM-0115-0001.jpeg" # <--- REPLACE WITH A REAL PATH
    claimed_modality = "EYES" # Try changing this to "EYES" or "BONE"
    
    decision = scanner.scan_image(test_image, claimed_modality)
    print(decision)