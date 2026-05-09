import subprocess
import os
import time

class EngineRecompiler:
    def __init__(self, onnx_dir="/home/jetson-nano/medical-edge-system/jetson_deployment_package/", engine_dir="/home/jetson-nano/medical-edge-system/jetson_deployment_package/"):
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        print("[MLOps] TRT Hot-Swap Recompiler Initialized.")

    def recompile_engine(self, modality_name):
        """
        Takes an updated ONNX model (post-EWC training) and recompiles it into an FP16 TRT Engine.
        """
        onnx_path = os.path.join(self.onnx_dir, f"{modality_name}_classifier.onnx")
        engine_path = os.path.join(self.engine_dir, f"{modality_name}_fp16.engine")
        
        print(f"\n[EWC RECOMPILE] Starting automated TRT build for {modality_name}...")
        start_time = time.time()
        
        # Build the exact trtexec command
        trt_cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16"
        ]
        
        try:
            # Run the compilation process in the background
            process = subprocess.Popen(trt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            # Read output continuously to monitor progress
            for line in process.stdout:
                if "PASSED" in line:
                    print(f"[SUCCESS] TRT Engine Recompiled in {(time.time() - start_time):.1f} seconds.")
                    return True
                elif "FAILED" in line:
                    print(f"[CRITICAL ERROR] TRT Compilation Failed: {line}")
                    return False
                    
            process.wait()
            
        except Exception as e:
            print(f"[FATAL] Recompiler crashed: {str(e)}")
            return False

# ==========================================
# TEST THE RECOMPILER
# ==========================================
if __name__ == "__main__":
    recompiler = EngineRecompiler()
    # Test triggering a background build of the Skin Lesion model
    recompiler.recompile_engine("skin")
