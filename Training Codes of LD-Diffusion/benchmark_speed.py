
import torch
import torch.nn as nn
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.quantum_transformer import QuantumFrontEndQCNN

def benchmark_speed():
    print("\n=== Benchmark: Training Speed vs Qubit Count ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    B = 4
    C = 128
    H, W = 16, 16
    emb_dim = 512
    
    x = torch.randn(B, C, H, W).to(device)
    emb = torch.randn(B, emb_dim).to(device)
    
    configs = [
        (4, 8), # Baseline
        (6, 8),
        (8, 8),
        (6, 4), # Optimization Candidate 1
        (8, 4), # Optimization Candidate 2
        (8, 2), # Optimization Candidate 3
    ]
    
    print(f"Input: {x.shape}")
    
    results = {}
    
    for n, g in configs:
        print(f"\n--- Testing {n} Qubits, {g} Groups ---")
        try:
            model = QuantumFrontEndQCNN(
                channels=C,
                style_dim=emb_dim,
                n_qubits_data=n,
                n_qubits_ancilla=2,
                n_layers=4,
                n_groups=g,
                stride=1,
                reupload_data=True,
                encoding_type='tanh',
                projection_type='mlp'
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Warmup
            print("  Warmup...")
            for _ in range(5):
                out = model(x, emb)
                loss = out.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            torch.cuda.synchronize() if device == 'cuda' else None
            
            # Benchmark
            print("  Running Benchmark (20 iters)...")
            start_time = time.time()
            for _ in range(20):
                out = model(x, emb)
                loss = out.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 20
            key = f"Q{n}_G{g}"
            results[key] = avg_time
            print(f"  Avg Time: {avg_time:.4f} s/iter")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[key] = None

    print("\n=== Summary ===")
    base_time = results.get("Q4_G8")
    for n, g in configs:
        key = f"Q{n}_G{g}"
        t = results.get(key)
        if t:
            ratio = t / base_time if base_time else 0
            print(f"{key}: {t:.4f} s/iter (x{ratio:.2f})")

if __name__ == "__main__":
    benchmark_speed()
