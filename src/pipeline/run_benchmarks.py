import time
import numpy as np
from master_pipeline_v2 import TensorRT_Stage2

def run_hardware_benchmark(engine_path, precision_name, num_runs=100):
    print(f"\n==================================================")
    print(f"🚀 INITIALIZING BENCHMARK: {precision_name}")
    print(f"==================================================")
    
    try:
        # Load the hardware engine
        trt_engine = TensorRT_Stage2(engine_path, "benchmark_test")
        
        # Create a dummy tensor (Batch Size 1, 3 Channels, 224x224)
        # This simulates a preprocessed medical scan
        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        print("[GPU] Engine loaded. Warming up Maxwell CUDA Cores...")
        # Warmup passes (GPU clocks take a second to spin up to max speed)
        for _ in range(20):
            trt_engine.execute_inference(dummy_input)
            
        print(f"[TEST] Running {num_runs} consecutive inferences...")
        
        start_time = time.time()
        
        for _ in range(num_runs):
            trt_engine.execute_inference(dummy_input)
            
        end_time = time.time()
        
        # Calculate Metrics
        total_time = end_time - start_time
        avg_latency_ms = (total_time / num_runs) * 1000
        fps = num_runs / total_time
        
        print(f"\n✅ BENCHMARK RESULTS ({precision_name}):")
        print(f"   -> Average Latency: {avg_latency_ms:.2f} ms")
        print(f"   -> Throughput (FPS): {fps:.2f} images/sec")
        print(f"==================================================\n")
        
    except Exception as e:
        print(f"[ERROR] Could not run benchmark. Did you forge the {precision_name} engine?")
        print(f"Details: {e}")

if __name__ == "__main__":
    # Test the FP16 Engine we forged earlier
    run_hardware_benchmark("../../jetson_deployment_package/brain_mri_fp16.engine", "Brain MRI (FP16 TRT)")
    
    # If Hamid gave you the INT8 engines, you would uncomment this line:
    # run_hardware_benchmark("../../jetson_deployment_package/brain_mri_int8.engine", "Brain MRI (INT8 TRT)")
