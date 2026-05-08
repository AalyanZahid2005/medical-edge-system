import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------------
WEIGHTS_DIR = r"D:\medical-edge-system\production_weights"

if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Initializing Training on: {device.type.upper()}")

BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 0.001

# ---------------------------------------------------------
# 2. Modality Dictionary (ALL 6 EXACT PATHS)
# ---------------------------------------------------------
MODALITIES = {
    "pneumonia": {
        "dataset_path": r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\train",
        "num_classes": 2  
    },
    "mri": {
        "dataset_path": r"D:\medical-edge-system\MRI\Data Set\Stage2_Classification\Training", 
        "num_classes": 4 
    },
    "retina": {
        "dataset_path": r"D:\medical-edge-system\EYES\dataset\aptos_split\train", 
        "num_classes": 5
    },
    "skin": {
        "dataset_path": r"D:\medical-edge-system\SKIN\dataset\skin_split\train", 
        "num_classes": 7
    },
    "bone": {
        "dataset_path": r"D:\medical-edge-system\BONE\dataset\bone_split\train", 
        "num_classes": 2
    },
    "ecg": {
        "dataset_path": r"D:\medical-edge-system\ECG\ecg_split\train", 
        "num_classes": 5
    }
}

# ---------------------------------------------------------
# 3. Data Preprocessing 
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 4. Master Training Engine
# ---------------------------------------------------------
def build_and_train_model(modality_name, config):
    print(f"\n{'='*50}")
    print(f"[TRAIN] Commencing Primary Classifier: {modality_name.upper()}")
    print(f"{'='*50}")

    dataset_path = config["dataset_path"]
    num_classes = config["num_classes"]

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Cannot find dataset at {dataset_path}. Skipping {modality_name.upper()}...")
        return

    train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"[DATA] Loaded {len(train_dataset)} images belonging to {num_classes} classes.")

    # Load Transfer Learning Blueprint
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Adjust final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"   Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save to production folder
    save_path = os.path.join(WEIGHTS_DIR, f"{modality_name}_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] {modality_name.upper()} model saved to: {save_path}")

# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------
if __name__ == '__main__':
    for modality, config in MODALITIES.items():
        # We already trained pneumonia, so we can skip it to save you time. 
        # If you want to retrain it anyway, just remove this if-statement.
        if modality == "pneumonia":
            print(f"\n[SYSTEM] Skipping PNEUMONIA (Already Trained).")
            continue
            
        build_and_train_model(modality, config)
        
    print("\n[SYSTEM] All primary classifiers have been successfully generated!")
    print("[SYSTEM] Please hand the new .pth files over to Aalyan.")