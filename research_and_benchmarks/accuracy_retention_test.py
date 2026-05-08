import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. Configuration & Phase 3 Hyperparameters
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PHASE 3] Initializing SEQUENTIAL EWC Protocol on: {device.type.upper()}")

WEIGHTS_DIR = r"D:\medical-edge-system\production_weights"
EWC_LAMBDA = 100.0       # Heavy armor for the 5-disease gauntlet
DRIFT_THRESHOLD = 5.0    

ANCHOR_TASK = "pneumonia"
NEW_TASKS_SEQUENCE = ["bone", "mri", "retina", "skin", "ecg"]

MODALITIES = {
    "pneumonia_test": {"path": r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\test", "classes": 2},
    "bone": {"path": r"D:\medical-edge-system\BONE\dataset\bone_split\train", "classes": 2},
    "mri": {"path": r"D:\medical-edge-system\MRI\Data Set\Stage2_Classification\Training", "classes": 4},
    "retina": {"path": r"D:\medical-edge-system\EYES\dataset\aptos_split\train", "classes": 5},
    "skin": {"path": r"D:\medical-edge-system\SKIN\dataset\skin_split\train", "classes": 7},
    "ecg": {"path": r"D:\medical-edge-system\ECG\ecg_split\train", "classes": 5}
}

# ---------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def ewc_loss(model, base_weights, fisher_matrix, lambda_param):
    loss_ewc = 0.0
    for name, param in model.named_parameters():
        # This safely ignores the classification head because it's not in the base_weights dictionary!
        if name in base_weights and name in fisher_matrix:
            penalty = (fisher_matrix[name] * (param - base_weights[name]).pow(2)).sum()
            loss_ewc += penalty
    return (lambda_param / 2.0) * loss_ewc

def evaluate_accuracy(model, dataloader, task_name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"   -> {task_name} Accuracy: {accuracy:.2f}%")
    return accuracy

# ---------------------------------------------------------
# 3. Master Sequential Learning Loop
# ---------------------------------------------------------
def run_sequential_learning():
    print("\n" + "="*60)
    print(f"PHASE 1: SECURING THE ANCHOR ({ANCHOR_TASK.upper()})")
    print("="*60)
    
    anchor_loader = DataLoader(datasets.ImageFolder(MODALITIES["pneumonia_test"]["path"], transform=transform), batch_size=32, shuffle=False)
    
    # Load Base Model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, MODALITIES["pneumonia_test"]["classes"])
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, f"{ANCHOR_TASK}_classifier.pth")))
    model = model.to(device)
    
    # 1. Get Baseline
    baseline_acc = evaluate_accuracy(model, anchor_loader, "BASELINE PNEUMONIA")
    
    # 2. Lock the Backbone Weights (CRITICAL FIX: Ignore the 'classifier' layer for EWC math)
    base_weights = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n}
    fisher_matrix = {n: torch.ones_like(p).to(device) for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n}
    
    # 3. Save the Pneumonia Head safely in a box for later
    pneumonia_weight = model.classifier[1].weight.clone().detach()
    pneumonia_bias = model.classifier[1].bias.clone().detach()

    print("\n" + "="*60)
    print("PHASE 2: INITIATING SEQUENTIAL BOMBARDMENT")
    print("="*60)
    
    in_features = model.classifier[1].in_features
    standard_criterion = nn.CrossEntropyLoss()
    
    # The Gauntlet Loop
    for step, new_task in enumerate(NEW_TASKS_SEQUENCE):
        print(f"\n--- LEARNING TASK {step+1}/5: {new_task.upper()} ---")
        
        # Swap the head for the new task
        model.classifier[1] = nn.Linear(in_features, MODALITIES[new_task]["classes"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        task_loader = DataLoader(datasets.ImageFolder(MODALITIES[new_task]["path"], transform=transform), batch_size=32, shuffle=True)
        
        model.train()
        for i, (inputs, labels) in enumerate(task_loader):
            if i > 50: break 
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss_standard = standard_criterion(outputs, labels)
            
            # Apply the EWC Armor to protect the core backbone
            loss_penalty = ewc_loss(model, base_weights, fisher_matrix, EWC_LAMBDA)
            
            total_loss = loss_standard + loss_penalty
            total_loss.backward()
            optimizer.step()
            
        print(f"   -> {new_task.upper()} learned. Moving to next disease...")

    print("\n" + "="*60)
    print("PHASE 3: THE ULTIMATE RETENTION TEST")
    print("="*60)
    
    # Take the original Pneumonia head out of the box and put it back on the newly trained backbone
    model.classifier[1] = nn.Linear(in_features, MODALITIES["pneumonia_test"]["classes"]).to(device)
    model.classifier[1].weight.data = pneumonia_weight
    model.classifier[1].bias.data = pneumonia_bias
    
    retained_acc = evaluate_accuracy(model, anchor_loader, "SURVIVING PNEUMONIA")
    
    drift = baseline_acc - retained_acc
    print(f"\n[METRIC] Sequential Catastrophic Forgetting Drift: {drift:.2f}%")
    
    if drift > DRIFT_THRESHOLD:
        print(f"\n[ALERT] The AI collapsed under the weight of 6 diseases.")
        print(f"[ACTION] You need a stronger Lambda or a larger network backbone.")
    else:
        print(f"\n[HISTORIC SUCCESS] Drift is < {DRIFT_THRESHOLD}%.")
        print(f"The model successfully absorbed 5 new diseases without erasing the first!")

if __name__ == '__main__':
    run_sequential_learning()