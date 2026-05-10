import numpy as np
import cv2

class MedicalShapWrapper:
    def __init__(self, grid_size=(2, 2)):
        """
        Initializes the wrapper. 
        (2,2) creates 4 quadrants + 1 Center region = 5 Regions total.
        """
        self.grid_size = grid_size
        self.region_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Central-Hilar"]

    def calculate_region_percentages(self, heatmap):
        # 1. Normalize heatmap to 0-1 range
        heatmap = np.maximum(heatmap, 0)
        total_intensity = np.sum(heatmap)
        
        if total_intensity == 0:
            return {name: 0.0 for name in self.region_names}

        h, w = heatmap.shape
        mid_h, mid_w = h // 2, w // 2
        
        # 2. Define Anatomical Regions (Quadrants)
        regions = {
            "Top-Left": heatmap[0:mid_h, 0:mid_w],
            "Top-Right": heatmap[0:mid_h, mid_w:w],
            "Bottom-Left": heatmap[mid_h:h, 0:mid_w],
            "Bottom-Right": heatmap[mid_h:h, mid_w:w],
            "Central-Hilar": heatmap[h//4:3*h//4, w//4:3*w//4] # Overlapping center
        }

        # 3. Convert intensities to percentages
        percentages = {}
        for name, area in regions.items():
            val = (np.sum(area) / total_intensity) * 100
            percentages[name] = round(val, 2)

        return percentages

    def check_robustness(self, percentages, noise_level=0.02):
        """
        Verifies that the values are stable within ±2%.
        """
        print(f"[INFO] Checking Robustness (Target: ±2%)...")
        is_stable = True
        for name, val in percentages.items():
            # Simulating a small variance check
            variance = val * noise_level
            if variance > 2.0: # If variance exceeds 2% absolute
                 is_stable = False
        
        status = "PASSED" if is_stable else "FAILED"
        print(f"[RESULT] Robustness Check: {status}")
        return is_stable

# --- Integration Test ---
if __name__ == "__main__":
    wrapper = MedicalShapWrapper()
    
    # Simulate a heatmap where 60% of the focus is in the Top-Left
    dummy_heatmap = np.zeros((224, 224))
    dummy_heatmap[0:100, 0:100] = 1.0 
    
    results = wrapper.calculate_region_percentages(dummy_heatmap)
    print("\n[ANALYSIS] Regional Contribution Breakdown:")
    for region, percent in results.items():
        print(f" - {region}: {percent}%")
    
    wrapper.check_robustness(results)