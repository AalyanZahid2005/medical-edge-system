import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Import the architecture we just defined
sys.path.append(r'D:\medical-edge-system\src\models')
from transformer_fusion import CrossModalAttentionFusion

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalAttentionFusion().to(device)
    
    # Standard training setup [cite: 213]
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Load your synthetic 10k profiles [cite: 118]
    # Here we simulate the loading of (batch, 6 modalities, 1280 features)
    print(f"[PHASE 5] Training on {device}...")
    
    # Dummy data loop for demonstration
    # In practice, replace with: torch.load('synthetic_profiles.pth')
    for epoch in range(25):
        inputs = torch.randn(32, 6, 1280).to(device)
        targets = torch.rand(32, 1).to(device)
        
        optimizer.zero_grad()
        risk, _ = model(inputs)
        loss = criterion(risk, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/25 - Loss: {loss.item():.4f}")

    # Save the final weights to your production folder [cite: 213]
    save_path = r"D:\medical-edge-system\production_weights\transformer_fusion.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Model saved to: {save_path}")

if __name__ == "__main__":
    run_training()