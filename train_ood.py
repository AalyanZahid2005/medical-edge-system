import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast

# Import the architecture we just verified!
from ood_autoencoder import MedicalOODAutoencoder

# ---------------------------------------------------------
# 1. Configuration & Master Paths
# ---------------------------------------------------------
BATCH_SIZE = 64 # Autoencoders are lightweight, we can push the batch size up!
EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "ood_guard_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Utilizing compute device: {device}")

# We only want the training folders to teach it what "normal" 2D images look like.
# NOTE: Ensure the internal folder names (like 'chest_xray\train') match your actual setup!
TRAIN_DIRS = [
    r"D:\medical-edge-system\EYES\dataset\aptos_split\train",
    r"D:\medical-edge-system\BONE\dataset\bone_split\train",
    r"D:\medical-edge-system\SKIN\dataset\skin_split\train",
    r"D:\medical-edge-system\Pneumonia\chest_xray\train", # Updated path!
    r"D:\medical-edge-system\MRI\train"                   # Updated path!
]

# ---------------------------------------------------------
# 2. Universal Data Loading
# ---------------------------------------------------------
# Note: Autoencoders output a Sigmoid (0 to 1), so we ONLY use ToTensor() 
# which scales pixels to 0-1. We DO NOT use Normalize() here!
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor() 
])

print("[SYSTEM] Stitching Master Medical Dataset...")
valid_datasets = []

for path in TRAIN_DIRS:
    if os.path.exists(path):
        print(f" -> Found dataset at: {path}")
        ds = datasets.ImageFolder(path, transform=transform)
        valid_datasets.append(ds)
    else:
        print(f" -> [WARNING] Could not find: {path}. Check your folder names!")

if not valid_datasets:
    print("[FATAL ERROR] No datasets found. Check your paths.")
    exit()

# Combine them all into one giant dataset
master_dataset = ConcatDataset(valid_datasets)
dataloader = DataLoader(master_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"[SYSTEM] Master Dataset Ready! Total Images: {len(master_dataset)}")

# ---------------------------------------------------------
# 3. Model, Loss, and Optimizer
# ---------------------------------------------------------
model = MedicalOODAutoencoder().to(device)

# Mean Squared Error (MSE) measures the difference between the original and rebuilt image
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler() # Using your mixed-precision trick for speed!

# ---------------------------------------------------------
# 4. The Training Loop
# ---------------------------------------------------------
if __name__ == '__main__':
    print("\n[SYSTEM] Commencing OOD Autoencoder Training...")
    
    # We only care about the reconstruction loss dropping
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, _) in enumerate(dataloader):
            # We don't care about labels (_) for Autoencoders!
            # The input IS the target.
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            with autocast():
                reconstructed = model(inputs)
                loss = criterion(reconstructed, inputs)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Reconstruction Error (MSE): {avg_loss:.6f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] OOD Security Guard Trained & Saved as '{MODEL_SAVE_PATH}'!")