import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image

class MedicalInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data_path, cache_file):
        super().__init__()
        self.cache_file = cache_file
        
        # --- THE FIX: Only read actual image files, completely ignore folders and JSONs ---
        valid_ext = ('.jpg', '.jpeg', '.png')
        self.data = [
            os.path.join(training_data_path, f) 
            for f in os.listdir(training_data_path) 
            if f.lower().endswith(valid_ext) and os.path.isfile(os.path.join(training_data_path, f))
        ]
        # ----------------------------------------------------------------------------------
        
        self.batch_size = 1
        self.device_input = cuda.mem_alloc(1 * 3 * 224 * 224 * 4) # Space for 1 image
        self.batches = self.next_batch()

    def next_batch(self):
        for path in self.data:
            img = Image.open(path).resize((224, 224)).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            # Standard ImageNet normalization
            img_array -= [0.485, 0.456, 0.406]
            img_array /= [0.229, 0.224, 0.225]
            yield np.ascontiguousarray(img_array.transpose(2, 0, 1))

    def get_batch_size(self): return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.batches)
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f: return f.read()
            
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f: f.write(cache)
