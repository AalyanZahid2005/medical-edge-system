import os
import cv2
import argparse

def batch_resize(input_dir, output_dir, size=(224, 224)):
    """Resizes all images in a directory structure to 224x224 for EfficientNet B0."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.png', '.jpg', '.jpeg')
    processed_count = 0

    print(f"[SYSTEM] Starting batch resize from {input_dir}...")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                input_path = os.path.join(root, file)
                
                # Replicate subfolder structure in the output directory
                relative_path = os.path.relpath(root, input_dir)
                target_folder = os.path.join(output_dir, relative_path)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                output_path = os.path.join(target_folder, file)
                
                # Read, resize, and save
                img = cv2.imread(input_path)
                if img is not None:
                    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(output_path, resized_img)
                    processed_count += 1
                    
                    if processed_count % 500 == 0:
                        print(f"[STATUS] Processed {processed_count} images...")

    print(f"[SUCCESS] Resized {processed_count} images to {size}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Image Resizer")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save resized images")
    args = parser.parse_args()
    
    batch_resize(args.input, args.output)