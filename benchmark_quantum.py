
import os
import sys
import time
import torch
import torch.nn as nn
import math
import numpy as np

# Add path to find quantum_transformer
project_root = '/home/zzn/qfl_tq/LD-Diffusion-quantum-v3'
training_path = os.path.join(project_root, 'Training Codes of LD-Diffusion')
sys.path.append(training_path)

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    from training.quantum_transformer import QuantumFrontEndQCNN, QuantumMLP
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def benchmark():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Settings
    B, C, H, W = 16, 64, 16, 16
    style_dim = 128
    n_qubits_data = 6
    n_layers = 2
    
    # Models to test
    models = {}
    
    # 1. Original (Baseline)
    qmlp_out_dim_orig = n_qubits_data * 3
    qmlp_orig = QuantumMLP(in_features=style_dim, out_features=qmlp_out_dim_orig, n_qubits=6, q_depth=2, encoding='angle').to(device)
    models["Original (6q)"] = QuantumFrontEndQCNN(channels=C, style_dim=qmlp_out_dim_orig, n_qubits_data=n_qubits_data, n_qubits_ancilla=0, time_emb_module=qmlp_orig, device_name=str(device)).to(device)

    # 2. Final Improved Fusion
    # QMLP has 6 qubits (High Capacity)
    # Interaction uses 4 qubits (Fast Entanglement)
    # Excess 2 qubits are idled (Partial Trace)
    n_anc_total = 6
    n_anc_interact = 4
    
    qmlp_final = QuantumMLP(in_features=style_dim, out_features=style_dim, n_qubits=n_anc_total, q_depth=2, encoding='angle').to(device)
    models[f"Final Improved ({n_anc_total}q->{n_anc_interact}q)"] = QuantumFrontEndQCNN(channels=C, style_dim=style_dim, n_qubits_data=n_qubits_data, n_qubits_ancilla=n_anc_interact, time_emb_module=qmlp_final, device_name=str(device)).to(device)
    
    # 4. JIT Optimized
    # We will simulate JIT by decorating the function.
    # Note: torch.jit.script on 'torchquantum' functional calls might fail if they are not scriptable.
    # We will wrap it in a try-except to report if JIT fails.
    try:
        # We need to decorate the METHOD of the CLASS INSTANCE or the CLASS itself?
        # torch.jit.script works on functions or nn.Modules.
        # Let's try to script the method we just extracted.
        # Access the class method
        # Problem: 'qdev' is a complex object. JIT might not handle it well.
        # Real JIT with torchquantum is tricky. 
        # Instead of failing, let's just run the code we extracted (which is cleaner) and call it "Refactored".
        # The extraction itself might save some lookup overhead.
        pass
    except Exception as e:
        print(f"JIT Setup failed: {e}")

    # Inputs
    x = torch.randn(B, C, H, W).to(device)
    style = torch.randn(B, style_dim).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for name, model in models.items():
            try:
                _ = model(x, style)
            except Exception as e:
                print(f"Warmup failed for {name}: {e}")
                # Print details to debug
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
                _ = model(x, style)
                
        torch.cuda.synchronize()
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        results[name] = avg_time
        print(f"{name}: {avg_time*1000:.2f} ms per batch")
        
    print("\nOverview:")
    base_time = results["Original (6q)"]
    for name, avg_time in results.items():
        ratio = avg_time / base_time
        print(f"{name}: {avg_time*1000:.2f} ms ({ratio:.2f}x)")

if __name__ == "__main__":
    benchmark()
