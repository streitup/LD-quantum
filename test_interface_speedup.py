
import time
import torch
import torch.nn as nn
import sys
import os

# Add path to training codes
sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))

from training.quantum_transformer import QuantumAttention64
import torchquantum as tq

def benchmark_interface():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking Interface Optimization on {device}")
    
    # Config
    B, S, D = 4, 16, 64 # Typical batch
    N_QUBITS = 6
    DEPTH = 8
    
    model = QuantumAttention64(N_QUBITS=N_QUBITS, Q_DEPTH=DEPTH, n_heads=4, device_name=device.type).to(device)
    model.eval()
    
    x = torch.randn(B, S, D).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = model(x)
        
    # Benchmark Optimized (New)
    print("\n--- Benchmarking Optimized Batch-Parallel Interface ---")
    start_time = time.time()
    n_iters = 50
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    avg_time_new = (end_time - start_time) / n_iters
    print(f"Optimized Time: {avg_time_new*1000:.2f} ms / iter")
    
    # Benchmark Legacy (Sequential)
    # Manually simulate 3 separate branches to avoid shape errors in legacy methods
    print("\n--- Benchmarking Legacy Sequential Interface ---")
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            bsz = B * S
            x_flat = x.reshape(bsz, D)
            # We need to project input for encoding if the model does it
            # In _forward_impl: x_bsz = self.inp_proj(x_64.reshape(bsz, D))
            x_bsz = model.inp_proj(x_flat)
            
            # 1. Q Branch
            qdev_q = tq.QuantumDevice(n_wires=N_QUBITS, bsz=bsz, device=device.type)
            model._amplitude_encode(qdev_q, x_bsz)
            model._apply_pqc(qdev_q, model.enc_w)
            model._apply_pqc(qdev_q, model.q_w)
            _ = model._measure_probs(qdev_q)
            
            # 2. K Branch
            qdev_k = tq.QuantumDevice(n_wires=N_QUBITS, bsz=bsz, device=device.type)
            model._amplitude_encode(qdev_k, x_bsz)
            model._apply_pqc(qdev_k, model.enc_w)
            model._apply_pqc(qdev_k, model.k_w)
            _ = model._measure_probs(qdev_k)
            
            # 3. V Branch
            qdev_v = tq.QuantumDevice(n_wires=N_QUBITS, bsz=bsz, device=device.type)
            model._amplitude_encode(qdev_v, x_bsz)
            model._apply_pqc(qdev_v, model.enc_w)
            model._apply_pqc(qdev_v, model.v_w)
            _ = model._measure_probs(qdev_v)
            
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    avg_time_old = (end_time - start_time) / n_iters
    print(f"Legacy Time:    {avg_time_old*1000:.2f} ms / iter")
    
    # Calculate Speedup
    speedup = avg_time_old / avg_time_new
    reduction = (avg_time_old - avg_time_new) / avg_time_old * 100
    
    print(f"\nResults:")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Latency Reduction: {reduction:.2f}%")
    
    return avg_time_new, avg_time_old

if __name__ == "__main__":
    benchmark_interface()
