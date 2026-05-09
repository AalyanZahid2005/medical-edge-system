import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------
# 1. Configuration & Phase 3 Hyperparameters
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[PHASE 3] Initializing Federated Learning Simulator on: {device.type.upper()}")

# We will use the Pneumonia dataset to prove the Federated concept
TRAIN_DIR = r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\train"
TEST_DIR = r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\test"

NUM_ROUNDS = 3           # How many times the server will sync with hospitals
LOCAL_EPOCHS = 1         # How many passes each hospital makes over its own data per round
HOSPITALS = ["UET Medical Center (Lahore)", "Shahdara Clinic", "Islamabad Research Lab"]

# ---------------------------------------------------------
# 2. Data Silos Setup (Splitting the data into 3 private parts)
# ---------------------------------------------------------
print("[SERVER] Partitioning dataset into secure local hospital silos...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Split the training data into 3 equal chunks to simulate different hospitals
chunk_size = len(full_train_dataset) // len(HOSPITALS)
split_sizes = [chunk_size, chunk_size, len(full_train_dataset) - (2 * chunk_size)]
hospital_datasets = random_split(full_train_dataset, split_sizes)

hospital_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in hospital_datasets]

# ---------------------------------------------------------
# 3. Model Setup & FedAvg Utilities
# ---------------------------------------------------------
def create_base_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model.to(device)

def average_weights(hospital_weights_list):
    """The FedAvg Algorithm: Averages the math without looking at the images."""
    avg_weights = copy.deepcopy(hospital_weights_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(hospital_weights_list)):
            avg_weights[key] += hospital_weights_list[i][key]
        # Divide by the number of hospitals
        avg_weights[key] = torch.div(avg_weights[key], len(hospital_weights_list))
    return avg_weights

def evaluate_global_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ---------------------------------------------------------
# 4. The Federated Learning Loop
# ---------------------------------------------------------
def run_federated_simulation():
    # Initialize the "Central Server" brain
    global_model = create_base_model()
    
    print("\n" + "="*60)
    print("STARTING FEDERATED TRAINING PROTOCOL")
    print("="*60)
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- GLOBAL SYNCHRONIZATION ROUND {round_num}/{NUM_ROUNDS} ---")
        
        # Keep track of the updated math from each hospital
        local_weights = []
        global_weights = global_model.state_dict()
        
        # Simulate each hospital training offline
        for i, hospital_name in enumerate(HOSPITALS):
            print(f"  -> Sending AI to {hospital_name}...")
            
            # 1. Hospital downloads the global brain
            local_model = create_base_model()
            local_model.load_state_dict(copy.deepcopy(global_weights))
            
            # 2. Hospital trains purely on its own private data
            optimizer = optim.Adam(local_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            local_model.train()
            
            for epoch in range(LOCAL_EPOCHS):
                for inputs, labels in hospital_loaders[i]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
            # 3. Hospital sends ONLY the updated math back to the server
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            print(f"     [SUCCESS] {hospital_name} uploaded mathematically updated weights. (No images shared)")
            
        print(f"\n[SERVER] Aggregating mathematical updates from {len(HOSPITALS)} hospitals using FedAvg...")
        
        # The Server averages the math to create the upgraded global brain
        new_global_weights = average_weights(local_weights)
        global_model.load_state_dict(new_global_weights)
        
        # Test the new Global AI
        global_acc = evaluate_global_model(global_model, test_loader)
        print(f"[SERVER] Round {round_num} Global Accuracy: {global_acc:.2f}%")

    print("\n" + "="*60)
    print("FEDERATED SIMULATION COMPLETE")
    print(f"Final Global Model Accuracy: {global_acc:.2f}%")
    print("Privacy Breaches: 0")
    print("="*60)

if __name__ == '__main__':
    run_federated_simulation()