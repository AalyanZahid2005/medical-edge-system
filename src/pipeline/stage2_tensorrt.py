import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import torch.nn.functional as F

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

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

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
                self.inputs.append({
                    "name": binding,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": tuple(self.engine.get_binding_shape(binding))
                })
            else:
                self.outputs.append({
                    "name": binding,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": tuple(self.engine.get_binding_shape(binding))
                })

        if not self.inputs:
            raise RuntimeError("TensorRT engine has no input binding.")

        if not self.outputs:
            raise RuntimeError("TensorRT engine has no output binding.")

    def _prepare_input(self, input_array):
        """
        Prepare API image tensor for TensorRT.

        The uploaded/API image may arrive as:
        - 512x512
        - NHWC
        - CHW
        - NCHW

        The TensorRT engine expects fixed NCHW size, normally:
        1 x 3 x 224 x 224
        """

        input_array = np.asarray(input_array, dtype=np.float32)

        # If CHW, convert to NCHW
        if input_array.ndim == 3:
            input_array = np.expand_dims(input_array, axis=0)

        if input_array.ndim != 4:
            raise ValueError(f"TensorRT expected 4D input, got shape {input_array.shape}")

        # If NHWC, convert to NCHW
        if input_array.shape[1] not in (1, 3) and input_array.shape[-1] in (1, 3):
            input_array = np.transpose(input_array, (0, 3, 1, 2))

        expected_size = self.inputs[0]["host"].size
        input_shape = self.inputs[0]["shape"]

        # Detect expected H,W from engine shape if possible
        target_h, target_w = 224, 224

        if len(input_shape) == 4:
            # Common explicit shape: (1, 3, 224, 224)
            if input_shape[2] > 0 and input_shape[3] > 0:
                target_h, target_w = int(input_shape[2]), int(input_shape[3])
        elif expected_size == 1 * 3 * 224 * 224:
            target_h, target_w = 224, 224

        # Resize if API sends 512x512 or any wrong spatial size
        if input_array.size != expected_size:
            t = torch.from_numpy(input_array.astype(np.float32))
            t = F.interpolate(
                t,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            )
            input_array = t.cpu().numpy()

        input_array = np.ascontiguousarray(input_array, dtype=np.float32)

        # Final safety check before np.copyto
        if input_array.size != expected_size:
            raise ValueError(
                "TRT input size mismatch after resize: got shape {}, size {}, "
                "expected buffer size {}, engine input shape {}".format(
                    input_array.shape,
                    input_array.size,
                    expected_size,
                    input_shape
                )
            )

        return input_array

    def execute_inference(self, input_array):
        """
        Runs TensorRT inference using execute_v2.
        """

        input_array = self._prepare_input(input_array)

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
    engine_path = "../jetson_deployment_package/brain_mri_fp16.engine"

    try:
        trt_model = TensorRT_Stage2(engine_path, "brain_mri")

        # Test with 512x512 because API upload was sending this size
        dummy_input = np.random.randn(1, 3, 512, 512).astype(np.float32)

        print("[STAGE 2] Firing execute_v2 forward pass...")
        output = trt_model.execute_inference(dummy_input)

        print("\n=== TENSORRT STAGE 2 SUCCESS ===")
        print(f"Raw Output Array: {output[:4]}...")

    except Exception as e:
        print(f"Waiting for trtexec to finish building the engine... Error: {e}")
