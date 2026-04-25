import threading
import queue
import time
import numpy as np

# Thread-safe queues to decouple the stages and prevent CPU/GPU blocking
# Maxsize prevents RAM overflow if Stage 1 is faster than Stage 2
STAGE_1_QUEUE = queue.Queue(maxsize=20)
STAGE_2_QUEUE = queue.Queue(maxsize=20)

class PipelineSkeleton:
    def __init__(self):
        self.running = True
        self.processed_frames = 0
        self.start_time = time.time()

        # Start the background daemon threads
        self.t1 = threading.Thread(target=self.stage_1_fetch_preprocess, daemon=True)
        self.t2 = threading.Thread(target=self.stage_2_tensorrt_compute, daemon=True)
        self.t3 = threading.Thread(target=self.stage_3_postprocess_io, daemon=True)

        self.t1.start()
        self.t2.start()
        self.t3.start()
        print("[SYSTEM] 3-Stage Asynchronous Pipeline Initialized.")

    def stage_1_fetch_preprocess(self):
        """STAGE 1 (CPU Bound): Fetch image, decode, resize to 224x224, normalize."""
        patient_id = 1
        while self.running:
            # Simulate I/O fetch and OpenCV preprocessing latency (~15ms)
            time.sleep(0.015) 
            
            # Simulate the CHW normalized float32 tensor
            dummy_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)

            payload = {
                "patient_id": f"PT-{patient_id}",
                "modality": "brain_mri",
                "tensor": dummy_tensor
            }
            
            STAGE_1_QUEUE.put(payload) # Blocks if queue is full
            patient_id += 1

    def stage_2_tensorrt_compute(self):
        """STAGE 2 (GPU Bound): Memory copy, TensorRT execution, copy back."""
        # NOTE: In production, you MUST initialize PyCUDA context here for this thread
        # cuda.init()
        # cuda_ctx = cuda.Device(0).retain_primary_context()

        while self.running:
            payload = STAGE_1_QUEUE.get()

            # CRITICAL: PyCUDA context push required before TRT execution in threaded environments
            # cuda_ctx.push()

            # Simulate Host-to-Device memory transfer & TRT inference (~45ms on Nano)
            time.sleep(0.045) 
            
            # Simulate TRT execute_v2(bindings) outputting dual tensors
            # bindings = [int(rt["d_input"]), int(rt["d_logits"]), int(rt["d_features"])]
            # rt["context"].execute_v2(bindings=bindings)
            
            # Simulate the extracted logits and 1280-d features
            dummy_logits = np.random.randn(1, 3).astype(np.float32)
            dummy_features = np.random.randn(1, 1280, 7, 7).astype(np.float32)

            # CRITICAL: PyCUDA context pop after execution
            # cuda_ctx.pop()

            payload["logits"] = dummy_logits
            payload["features"] = dummy_features
            
            STAGE_2_QUEUE.put(payload)
            STAGE_1_QUEUE.task_done()

    def stage_3_postprocess_io(self):
        """STAGE 3 (CPU Bound): MC Dropout, True CAM generation, DB insertion."""
        while self.running:
            payload = STAGE_2_QUEUE.get()

            # Simulate MC Dropout math, True CAM generation using fc_weights.npy, and SQLite DB insert (~30ms)
            time.sleep(0.030)

            self.processed_frames += 1
            STAGE_2_QUEUE.task_done()

    def run_throughput_test(self, test_duration=10):
        """Measures the pipeline's Images Per Second (FPS)."""
        print(f"[SYSTEM] Running throughput benchmark for {test_duration} seconds...")
        self.start_time = time.time()
        self.processed_frames = 0
        
        time.sleep(test_duration)
        
        self.running = False
        total_time = time.time() - self.start_time
        fps = self.processed_frames / total_time
        
        print("\n--- BENCHMARK RESULTS ---")
        print(f"Total Images Processed: {self.processed_frames}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Throughput: {fps:.2f} FPS")
        print("-------------------------\n")

if __name__ == "__main__":
    pipeline = PipelineSkeleton()
    try:
        pipeline.run_throughput_test(test_duration=10)
    except KeyboardInterrupt:
        pipeline.running = False
        print("\n[SYSTEM] Pipeline terminated by user.")