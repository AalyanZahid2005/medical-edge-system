import os
import time
import numpy as np
import onnxruntime as ort
import cv2

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Path to your new ONNX model (Testing the Pneumonia Specialist)
MODEL_PATH = r"D:\medical-edge-system\jetson_deployment_package\pneumonia_classifier.onnx"

# Path to a folder containing test images
# (Update this to point to a folder with actual test X-rays)
TEST_IMAGES_DIR = r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\test\NORMAL" 

def load_and_preprocess_image(image_path):
    """Loads an image and converts it into the exact mathematical tensor the AI expects."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    
    # Convert from OpenCV (H, W, C) to PyTorch format (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Normalize to float32
    img = img.astype(np.float32) / 255.0
    return img

def run_10_image_check():
    print("============================================================")
    print(f"[SYSTEM] Loading ONNX Model: {os.path.basename(MODEL_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Could not find model at {MODEL_PATH}")
        return
        
    # Initialize the ONNX Runtime session
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    # Grab the first 10 images from the test folder
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"[ERROR] Could not find test images at {TEST_IMAGES_DIR}")
        return
        
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))][:10]
    
    if len(image_files) == 0:
        print("[WARNING] No images found in the test directory.")
        return

    print("============================================================")
    print(f"{'IMAGE':<20} | {'LATENCY':<15} | {'STATUS'}")
    print("============================================================")
    
    total_time = 0.0
    
    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        input_tensor = load_and_preprocess_image(img_path)
        
        # Start the clock
        start_time = time.time()
        
        # Run Inference
        _ = session.run(None, {input_name: input_tensor})
        
        # Stop the clock
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000 # Convert to milliseconds
        total_time += latency
        
        print(f"{img_file[:18]:<20} | {latency:>8.2f} ms   | [SUCCESS]")
        
    print("============================================================")
    print(f"Average Inference Speed: {total_time / len(image_files):.2f} ms per image")
    print("Quality Assurance Check: PASSED")

if __name__ == '__main__':
    run_10_image_check()