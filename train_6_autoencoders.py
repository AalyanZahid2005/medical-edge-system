import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Import your verified architecture
from ood_autoencoder import MedicalOODAutoencoder

# ---------------------------------------------------------
# 1. Paths for all Datasets
# ---------------------------------------------------------
DATA_PATHS = {
    # "EYES": r"D:\medical-edge-system\EYES\dataset\aptos_split\train",  # COMPLETED!
    # "BONE": r"D:\medical-edge-system\BONE\dataset\bone_split\train",  # COMPLETED!
    # "SKIN": r"D:\medical-edge-system\SKIN\dataset\skin_split\train",  # COMPLETED!
    
    "PNEUMONIA": r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\train",
    "MRI": r"D:\medical-edge-system\MRI\Data Set\Stage1_Detection\Training", 
    "ECG": r"D:\medical-edge-system\ECG\ecg_split\train" 
}

# ---------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 10 
LEARNING_RATE = 1e-3
PERCENTILE_THRESHOLD = 95 # Reject the worst 5% of reconstructions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Compute device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor() 
])

# ---------------------------------------------------------
# 3. Training & Calibration Loop
# ---------------------------------------------------------
def train_and_calibrate():
    for name, path in DATA_PATHS.items():
        print(f"\n{'='*60}")
        print(f"[SYSTEM] Commencing Pipeline for: {name}")
        
        if not os.path.exists(path):
            print(f"[WARNING] Path missing: {path}. Skipping {name}...")
            continue
            
        # Load specific dataset
        ds = datasets.ImageFolder(path, transform=transform)
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        # Initialize fresh model for this specific modality
        model = MedicalOODAutoencoder().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        scaler = GradScaler()
        
        # Two loss functions: one for training, one for extracting per-image math
        train_loss_fn = nn.MSELoss() 
        extract_loss_fn = nn.MSELoss(reduction='none') 
        
        # --- PHASE A: TRAIN ---
        print(f"-> Training {name} Autoencoder ({EPOCHS} Epochs)...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                
                with autocast():
                    reconstructed = model(inputs)
                    loss = train_loss_fn(reconstructed, inputs)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                
            print(f"   Epoch {epoch+1}/{EPOCHS} | MSE: {running_loss/len(dataloader):.6f}")
            
        # Save the model
        model_name = f"{name.lower()}_ood_guard.pth"
        torch.save(model.state_dict(), model_name)
        print(f"-> Model saved as: {model_name}")

        # --- PHASE B: CALIBRATE & EXPORT .npy ---
        print(f"-> Calibrating Statistical Threshold for {name}...")
        model.eval()
        all_errors = []
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                reconstructed = model(inputs)
                
                # Calculate MSE for *each individual image* in the batch
                loss_per_image = extract_loss_fn(reconstructed, inputs).mean(dim=[1, 2, 3])
                all_errors.extend(loss_per_image.cpu().numpy())
                
        # Calculate the threshold
        errors_array = np.array(all_errors)
        threshold = np.percentile(errors_array, PERCENTILE_THRESHOLD)
        
        # Save arrays for the Jetson Nano
        errors_file = f"{name.lower()}_errors.npy"
        thresh_file = f"{name.lower()}_threshold.npy"
        
        np.save(errors_file, errors_array)
        np.save(thresh_file, np.array([threshold]))
        
        print(f"[SUCCESS] {name} 95th Percentile Threshold: {threshold:.6f}")
        print(f"[SUCCESS] Exported {errors_file} and {thresh_file}")

if __name__ == '__main__':
    train_and_calibrate()