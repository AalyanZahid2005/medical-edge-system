import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# ==========================================
# CRITICAL ARCHITECTURE REQUIREMENT
# Store runtime here to prevent Python Garbage Collection (GC)
# ==========================================
RUNTIMES = {}

class TensorRT_Stage2:
    def __init__(self, engine_path, model_name):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 1. Initialize Runtime and store in global dictionary
        RUNTIMES[model_name] = trt.Runtime(self.logger)
        
        print(f"[STAGE 2] Loading highly optimized FP16 Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = RUNTIMES[model_name].deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()

        # 2. Allocate GPU Memory Buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate Host (CPU) and Device (GPU) memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def execute_inference(self, input_array):
        # Copy input data from CPU to GPU
        np.copyto(self.inputs[0]["host"], input_array.ravel())
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])
        
        # 3. CRITICAL REQUIREMENT: Use execute_v2 only!
        self.context.execute_v2(bindings=self.bindings)
        
        # Copy output data back from GPU to CPU
        cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])
        
        return self.outputs[0]["host"]

# ==========================================
# QUICK VERIFICATION TEST
# ==========================================
if __name__ == "__main__":
    # Point to the engine you are building right now
    engine_path = "../jetson_deployment_package/brain_mri_fp16.engine"
    
    try:
        trt_model = TensorRT_Stage2(engine_path, "brain_mri")
        
        # Generate dummy tensor (Batch 1, 3 Channels, 224x224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        print("[STAGE 2] Firing execute_v2 forward pass...")
        output = trt_model.execute_inference(dummy_input)
        
        print("\n=== TENSORRT STAGE 2 SUCCESS ===")
        print(f"Raw Output Array: {output[:4]}...")
    except Exception as e:
        print(f"Waiting for trtexec to finish building the engine... Error: {e}")
