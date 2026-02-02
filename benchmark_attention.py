
import os
import sys
import time
import torch
import torch.nn as nn
import math

# Add path to find quantum_transformer
project_root = '/home/zzn/qfl_tq/LD-Diffusion-quantum-v3'
training_path = os.path.join(project_root, 'Training Codes of LD-Diffusion')
sys.path.append(training_path)

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    from training.quantum_transformer import QuantumAttention64, ClassicAttention64
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def benchmark():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Settings
    B, S, D = 16, 64, 64 # Batch 16, Seq 64 (8x8 patch), Dim 64
    
    # Models
    models = {
        "Classic": ClassicAttention64().to(device),
        "Quantum": QuantumAttention64(N_QUBITS=6, Q_DEPTH=2, device_name=str(device)).to(device)
    }
    
    # Input
    x = torch.randn(B, S, D).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for name, model in models.items():
            try:
                _ = model(x)
            except Exception as e:
                print(f"Warmup failed for {name}: {e}")
                import traceback
                traceback.print_exc()

    # Benchmark
    iterations = 20
    print(f"\nStarting Benchmark ({iterations} iterations)...")
    
    results = {}
    
    for name, model in models.items():
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
                
        torch.cuda.synchronize()
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        results[name] = avg_time
        print(f"{name}: {avg_time*1000:.2f} ms per batch")
        
    print("\nOverview:")
    base_time = results["Classic"]
    for name, avg_time in results.items():
        ratio = avg_time / base_time
        print(f"{name}: {avg_time*1000:.2f} ms ({ratio:.2f}x)")

if __name__ == "__main__":
    benchmark()
