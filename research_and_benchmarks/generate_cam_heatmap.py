import cv2
import numpy as np
import os

def generate_cam_heatmap(image_path, feature_maps, fc_weights, class_index, output_path, alpha=0.5):
    """
    Overlays a Class Activation Map (CAM) heatmap onto a medical image.
    """
    # 1. Load the original medical image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return False
        
    original_img = cv2.imread(image_path)
    height, width, _ = original_img.shape

    # 2. Get the specific weights for the predicted class
    # For example, if it predicts "Pneumonia" (Class 1), we pull the weights for Class 1.
    class_weights = fc_weights[class_index]

    # 3. Multiply the weights by the AI's feature maps to find the "Hotspots"
    # We do a dot product across the 1280 channels
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(class_weights):
        cam += w * feature_maps[i]

    # 4. Clean up the math (ReLU: ignore negative values)
    cam = np.maximum(cam, 0)
    
    # Normalize between 0 and 1
    cam = cam / (np.max(cam) + 1e-8)

    # 5. Resize the tiny mathematical map to match the full image resolution
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

    # 6. Apply OpenCV's "JET" colormap (Blue = Low Importance, Red = High Importance)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # 7. Blend the original image and the heatmap together
    blended_image = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

    # 8. Save the final visual proof
    cv2.imwrite(output_path, blended_image)
    print(f"[SUCCESS] Clinical Heatmap saved to: {output_path}")
    return True

# ==========================================
# TEST BED: Run this to test your OpenCV Logic!
# ==========================================
if __name__ == "__main__":
    # --- 1. Define Paths ---
    # Pick ANY real image from your dataset to test on
    TEST_IMAGE_PATH = r"D:\medical-edge-system\Pneumonia\Chest X ray\chest_xray\train\NORMAL\IM-0115-0001.jpeg" 
    OUTPUT_PATH = r"D:\medical-edge-system\sample_heatmap_proof.jpg"
    WEIGHTS_PATH = r"D:\medical-edge-system\dummy_fc_weights.npy" # The file you uploaded to GitHub

    if os.path.exists(TEST_IMAGE_PATH):
        print("[SYSTEM] Simulating AI Feature Extraction...")
        
        # --- 2. Load the Fully Connected Weights ---
        if os.path.exists(WEIGHTS_PATH):
            fc_weights = np.load(WEIGHTS_PATH)
        else:
            print(f"[WARNING] {WEIGHTS_PATH} not found. Generating mock weights for testing.")
            # Shape: (Num Classes, Feature Dimensions) -> e.g., (4 classes, 1280 features)
            fc_weights = np.random.randn(4, 1280) 

        # --- 3. Simulate the AI's internal spatial map (usually a 7x7 grid before the final layer) ---
        mock_feature_maps = np.random.rand(1280, 7, 7)
        
        # We manually inject a "hotspot" so you can clearly see the red zone on the test image
        mock_feature_maps[:, 2:5, 4:6] += 2.0 

        # --- 4. Generate the visual! ---
        print("[SYSTEM] Generating Visual Proof (CAM)...")
        generate_cam_heatmap(TEST_IMAGE_PATH, mock_feature_maps, fc_weights, class_index=1, output_path=OUTPUT_PATH)
    else:
        print("[WARNING] Please update TEST_IMAGE_PATH to a real image on your PC to test this script.")