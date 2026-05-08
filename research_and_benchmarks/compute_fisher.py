import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Import your baseline Autoencoder or CNN (Using a standard CNN block for example)
# In production, you will import your specific trained model here.
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = r"D:\medical-edge-system\production_weights"

def compute_fisher_matrix():
    print("============================================================")
    print("[SYSTEM] FRONT-LOADING: Computing Fisher Information Matrix...")
    
    # 1. Load your Dataset (Example using Pneumonia, update path if needed)
    dataset_path = r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\train"
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Cannot find dataset at {dataset_path}")
        return
        
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    # We only need a small sample (e.g., 200 images) to calculate the Fisher matrix
    subset_indices = torch.randperm(len(dataset))[:200]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=16, shuffle=True)

    # 2. Load your trained model (Using EfficientNet-B0 as the likely winner)
    model = models.efficientnet_b0(pretrained=False) 
    # NOTE: Normally you would load your .pth weights here!
    model = model.to(device)
    model.eval() # Must be in eval mode so dropout doesn't mess up the math

    # 3. Initialize an empty dictionary to hold the Fisher values for every parameter
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)

    # 4. The Math: Calculate the squared gradients
    print("-> Calculating gradients across dataset sample...")
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # We use Log Softmax and Negative Log Likelihood for Empirical Fisher
        log_probs = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(log_probs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Accumulate the squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Add (gradient^2 / number of batches) to get the average
                fisher_dict[name].data += (param.grad.data ** 2) / len(dataloader)

    # 5. Save the Fisher Matrix
    fisher_path = os.path.join(WEIGHTS_DIR, "pneumonia_fisher.pth")
    torch.save(fisher_dict, fisher_path)
    
    print(f"[SUCCESS] Fisher Information Matrix computed and saved to: {fisher_path}")
    print("[SYSTEM] EWC Protection is now ready for Day 6!")

if __name__ == '__main__':
    compute_fisher_matrix()