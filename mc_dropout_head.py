import os
import torch
import torch.nn as nn
import numpy as np

class MCDropoutClassifier(nn.Module):
    def __init__(self, weights_path, num_classes=3, dropout_rate=0.5):
        super(MCDropoutClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(1280, num_classes)
        
        # Load the weights generated on Day 1 into this PyTorch layer
        if os.path.exists(weights_path):
            fc_weights = np.load(weights_path)
            self.fc.weight.data = torch.tensor(fc_weights).float()
            # Initialize bias to zero for the sake of this local test
            self.fc.bias.data = torch.zeros(num_classes).float()
            print(f"[SYSTEM] Loaded FC weights from {weights_path}")
        else:
            print(f"[ERROR] Missing {weights_path}. Run generate_dummies.py first.")

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

def run_mc_inference(features_numpy, mc_head_model, num_passes=20):
    """Runs the 1280-d TRT feature vector through the dropout head 20 times."""
    # Force the model into training mode so dropout remains ACTIVE during inference
    mc_head_model.train()
    
    # Convert incoming numpy array (simulating TRT output) to tensor
    x = torch.tensor(features_numpy).float()
    
    predictions = []

    # Disable gradient tracking to save memory
    with torch.no_grad():
        for _ in range(num_passes):
            logits = mc_head_model(x)
            # Apply softmax to get probability percentages
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.numpy())

    # Stack the 20 passes into a single array. Shape: (20, 1, num_classes)
    predictions = np.stack(predictions)

    # Compute Mean and Standard Deviation across the 20 passes (axis=0)
    mean_probs = np.mean(predictions, axis=0)[0]
    std_probs = np.std(predictions, axis=0)[0]

    return mean_probs, std_probs

if __name__ == "__main__":
    print("--- Testing MC Dropout Variance Logic ---")
    
    # 1. Initialize the MC Head using Day 1's dummy weights
    weights_file = "dummy_fc_weights.npy"
    mc_head = MCDropoutClassifier(weights_path=weights_file, num_classes=3)
    
    # 2. Simulate the 1280-dimensional feature vector output from TensorRT
    # We use a static random seed here so the input is identical, proving that
    # any variance in the output comes purely from the MC Dropout.
    np.random.seed(42)
    fake_trt_features = np.random.randn(1, 1280).astype(np.float32)
    
    # 3. Execute 20-pass Monte Carlo inference
    print("\n[SYSTEM] Running 20 forward passes...")
    mean_p, std_p = run_mc_inference(fake_trt_features, mc_head, num_passes=20)
    
    # 4. Display the Clinical Output
    classes = ["Glioma", "Meningioma", "Pituitary"]
    target_idx = np.argmax(mean_p)
    
    print("\n--- BAYESIAN UNCERTAINTY RESULTS ---")
    for i in range(len(classes)):
        print(f"{classes[i]}: {mean_p[i]*100:.2f}% (± {std_p[i]*100:.2f}%)")
        
    print("\n[CLINICAL ROUTING CHECK]")
    uncertainty = std_p[target_idx] * 100
    if uncertainty > 5.0:
        print(f"[ALERT] High Uncertainty ({uncertainty:.2f}%). Routing to Tier 3 Worklist.")
    else:
        print(f"[STATUS] Confidence stable. Auto-approving Tier 1.")