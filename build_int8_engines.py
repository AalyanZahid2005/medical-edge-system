import tensorrt as trt
import os
import sys

# Import Hamid's calibrator class
sys.path.append(os.path.abspath("src/optimization"))
from int8_calibrator import MedicalInt8Calibrator

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_int8_engine(onnx_path, engine_path, calib_dir):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 1. Parse ONNX
    print(f"\n[*] Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(f"[ERROR] Failed to parse {onnx_path}")
            return

    # --- THE FIX: LOCK IN THE DYNAMIC SHAPES ---
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    # Force the shape to exactly 1 batch, 3 color channels, 224x224 pixels
    profile.set_shape(input_tensor.name, (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
    config.add_optimization_profile(profile)
    # -------------------------------------------

    # 2. Configure INT8 Builder
    config.max_workspace_size = 1 << 30 # 1GB Memory Limit
    config.set_flag(trt.BuilderFlag.INT8)
    
    # Attach Hamid's Calibrator
    calib_cache = engine_path + ".cache"
    calibrator = MedicalInt8Calibrator(calib_dir, calib_cache)
    config.int8_calibrator = calibrator

    # 3. Build Engine
    print(f"[*] Compiling INT8 Engine... The Jetson Nano fan will spin up. Please wait.")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes:
        with open(engine_path, 'wb') as f:
            f.write(engine_bytes)
        print(f"[SUCCESS] Saved highly compressed engine to: {engine_path}")
    else:
        print(f"[ERROR] Failed to build INT8 engine for {onnx_path}")

if __name__ == "__main__":
    print("==================================================")
    print("🚀 INITIATING TENSORRT INT8 COMPILER SEQUENCE")
    print("==================================================")
    
    onnx_dir = "jetson_deployment_package"
    calib_dir = "Final_Defense_Assets" 
    
    onnx_models = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
    
    if not onnx_models:
        print("[ERROR] No ONNX files found!")
        sys.exit(1)
        
    for onnx_file in onnx_models:
        onnx_path = os.path.join(onnx_dir, onnx_file)
        engine_file = onnx_file.replace(".onnx", "_int8.engine")
        engine_path = os.path.join(onnx_dir, engine_file)
        
        if os.path.exists(engine_path):
            print(f"\n[SKIP] {engine_file} already exists.")
            continue
            
        build_int8_engine(onnx_path, engine_path, calib_dir)
        
    print("\n==================================================")
    print("✅ COMPILER SEQUENCE FINISHED")
    print("==================================================")
