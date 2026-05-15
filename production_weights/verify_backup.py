import os
import torch
import numpy as np

# Updated to match your exact folder from the screenshot
BACKUP_DIR = r"D:\medical-edge-system\production_weights"

def verify_production_weights():
    print(f"--- Starting Phase 4 Model Verification ---")
    print(f"Scanning directory: {BACKUP_DIR}\n")
    
    if not os.path.exists(BACKUP_DIR):
        print(f"[CRITICAL ERROR] The directory {BACKUP_DIR} does not exist!")
        return

    all_passed = True
    files_found = os.listdir(BACKUP_DIR)
    
    if len(files_found) == 0:
        print("[CRITICAL ERROR] The folder is empty!")
        return

    for file_name in files_found:
        file_path = os.path.join(BACKUP_DIR, file_name)
        
        # 1. Check if it is a 0 KB empty file
        if os.path.getsize(file_path) == 0:
            print(f"  [X] CORRUPTED (0 KB): {file_name}")
            all_passed = False
            continue

        # 2. Deep check: Try to read the headers to ensure it isn't corrupted
        try:
            if file_name.endswith(".pth"):
                # Load dictionary structure safely to CPU
                _ = torch.load(file_path, map_location="cpu") 
                print(f"  [OK] {file_name} - PyTorch Weights Healthy")
            elif file_name.endswith(".npy"):
                # Load Numpy arrays
                _ = np.load(file_path, mmap_mode='r')
                print(f"  [OK] {file_name} - NumPy Array Healthy")
            else:
                print(f"  [-] Skipped unknown file type: {file_name}")
        except Exception as e:
            print(f"  [X] READ ERROR: {file_name} is corrupted. ({e})")
            all_passed = False

    print("\n--- Verification Summary ---")
    if all_passed:
        print(f"[SUCCESS] All {len([f for f in files_found if f.endswith(('.pth', '.npy'))])} model files are uncorrupted!")
        print("ACTION: You are cleared to copy the 'production_weights' folder to your USB drive.")
    else:
        print("[WARNING] Some files are corrupted. Do not back up yet.")

if __name__ == "__main__":
    verify_production_weights()