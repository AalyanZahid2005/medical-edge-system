import torch
import torchvision.models as models
import time
import os

# ---------------------------------------------------------
# 1. Setup the Competitors
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Running Ablation Study on: {device.type.upper()}")

# We load the un-trained blueprints just to test their architectural physics
competitors = {
    "EfficientNet-B0": models.efficientnet_b0(),
    "EfficientNet-B2": models.efficientnet_b2(),
    "MobileNetV3-Large": models.mobilenet_v3_large()
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_tensor, iterations=100):
    """Measures how fast the model processes an image in milliseconds."""
    model.eval()
    
    # GPU Warmup (First few passes are always slow)
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_tensor)
            
    # Actual timed race
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
            
    # Calculate average time in milliseconds
    end_time = time.time()
    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    return avg_time_ms

# ---------------------------------------------------------
# 2. Execute the Drag Race
# ---------------------------------------------------------
def run_ablation():
    print("\n" + "="*60)
    print(f"{'MODEL':<20} | {'PARAMETERS':<12} | {'LATENCY (ms)':<15}")
    print("="*60)
    
    dummy_image = torch.randn(1, 3, 224, 224).to(device)

    for name, model in competitors.items():
        model = model.to(device)
        
        # 1. Count how heavy the brain is (Parameters)
        params = count_parameters(model)
        
        # 2. Measure the speed
        latency = measure_latency(model, dummy_image)
        
        # 3. Print the benchmark row
        print(f"{name:<20} | {params/1e6:>5.2f} Million | {latency:>8.2f} ms")

if __name__ == '__main__':
    run_ablation()