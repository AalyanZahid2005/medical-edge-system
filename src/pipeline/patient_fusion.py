import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# ---------------------------------------------------------
# 1. Configuration & Hyperparameters
# ---------------------------------------------------------
N_MODALITIES = 6        # Brain, Chest, Retina, Skin, Bone, ECG
FEATURE_DIM = 1280      # Size of the vector from each CNN (e.g., EfficientNet)
INPUT_SIZE = N_MODALITIES * FEATURE_DIM  # 6 * 1280 = 7680
HIDDEN_SIZE = 512
NUM_CLASSES = 4         # 4 Risk Tiers (Normal, Mild, Severe, Critical)

NUM_SAMPLES = 10000     # Synthetic patient profiles
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3

WEIGHTS_DIR = r"D:\medical-edge-system\production_weights"

# Ensure weights directory exists
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Initializing Fusion Network on: {device.type.upper()}")

# ---------------------------------------------------------
# 2. The Baseline Fusion MLP Architecture
# ---------------------------------------------------------
class PatientFusionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PatientFusionMLP, self).__init__()
        
        # Linear(1280*N -> 512) -> ReLU -> Dropout(0.3) -> Linear -> 4
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------
# 3. Synthetic Data Generator
# ---------------------------------------------------------
def generate_synthetic_patients(num_samples, input_size, num_classes):
    print(f"\n[DATA] Generating {num_samples} synthetic patient profiles...")
    
    # Generate random features (mean=0, std=1) to simulate CNN outputs
    X = torch.randn(num_samples, input_size)
    
    # To ensure the model actually learns (instead of guessing pure random noise),
    # we create a fixed mathematical projection to assign classes based on the noise.
    secret_pattern = torch.randn(input_size, num_classes)
    logits = torch.matmul(X, secret_pattern)
    y = torch.argmax(logits, dim=1)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("[DATA] Synthetic generation complete.")
    return dataloader

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
def train_fusion_model():
    # 1. Prepare Data
    dataloader = generate_synthetic_patients(NUM_SAMPLES, INPUT_SIZE, NUM_CLASSES)
    
    # 2. Initialize Model, Loss, and Optimizer
    model = PatientFusionMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n{'='*50}")
    print(f"[TRAIN] Commencing Fusion MLP Training ({EPOCHS} Epochs)")
    print(f"{'='*50}")
    
    # 3. Execute Training
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"   Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # 4. Save the Final Weights
    save_path = os.path.join(WEIGHTS_DIR, "fusion_mlp_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Fusion MLP trained and saved to: {save_path}")

if __name__ == '__main__':
    train_fusion_model()