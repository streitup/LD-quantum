import torch
import torch.nn as nn
import time
import sys
import os

# Add path to training codes
sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))

from training.quantum_transformer import (
    QuantumAttention64, 
    QuantumAttentionAngle, 
    QuantumAttentionHybrid, 
    ClassicAttention64,
    QuantumAttentionLight
)
from training.networks import UNetBlock

def benchmark_attention(name, model, x, emb, device='cuda'):
    model.to(device)
    x = x.to(device)
    emb = emb.to(device)
    target = torch.randn_like(x).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Warmup
    print(f"Warming up {name}...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(x, emb)
            
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Run loop (Training)
    n_iters = 30
    initial_loss = 0.0
    final_loss = 0.0
    
    print(f"Training {name} for {n_iters} iterations...")
    for i in range(n_iters):
        optimizer.zero_grad()
        out = model(x, emb)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        if i == 0: initial_loss = current_loss
        final_loss = current_loss
        
        if (i + 1) % 5 == 0:
            print(f"  Iter {i+1}/{n_iters} | Loss: {current_loss:.6f}")
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_iters * 1000 # ms
    
    # Memory
    torch.cuda.reset_peak_memory_stats()
    # Backward pass memory is harder to measure statically, but max_memory_allocated during the loop captures it.
    # We'll just report the peak memory during the training loop.
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    
    # Parameter count
    param_count = sum(p.numel() for p in model.parameters()) / 1000 / 1000 # M
    
    print(f"{name:<40} | Time: {avg_time:.2f} ms | Mem: {max_mem:.2f} MB | Params: {param_count:.2f} M | Init Loss: {initial_loss:.4f} | Final Loss: {final_loss:.4f}")
    return avg_time, max_mem, param_count, initial_loss, final_loss

def run_benchmark():
    print("Benchmarking Attention + Convolution Architectures on [B=32, C=64, H=16, W=16] (Training Mode)")
    print("-" * 120)
    
    # Setup
    B = 32
    C = 64
    H = 16
    W = 16
    emb_channels = 256
    
    x = torch.randn(B, C, H, W)
    emb = torch.randn(B, emb_channels)
    
    # Common kwargs for UNetBlock
    block_kwargs = dict(
        in_channels=C,
        out_channels=C,
        emb_channels=emb_channels,
        num_heads=4,
        dropout=0.1,
        attention=True,
    )
    
    # Define Architectures
    configs = [
        (
            "Classic Attn + Classic Conv",
            dict(
                use_quantum_transformer=False,
                use_qcnn_frontend=False
            )
        ),
        (
            "Quantum Attn (Lite) + Classic Conv",
            dict(
                use_quantum_transformer=True,
                quantum_adapter="training.quantum_transformer:QuantumAdapterHybridLite",
                quantum_adapter_kwargs=dict(device_name='cuda'),
                use_qcnn_frontend=False
            )
        ),
        (
            "Classic Attn + Quantum Conv",
            dict(
                use_quantum_transformer=False,
                use_qcnn_frontend=True,
                qcnn_chunk_size=2048 # Increased chunk size for speed
            )
        ),
        (
            "Quantum Attn (Lite) + Quantum Conv",
            dict(
                use_quantum_transformer=True,
                quantum_adapter="training.quantum_transformer:QuantumAdapterHybridLite",
                quantum_adapter_kwargs=dict(device_name='cuda'),
                use_qcnn_frontend=True,
                qcnn_chunk_size=2048
            )
        ),
    ]
    
    results = []
    
    for name, specific_kwargs in configs:
        try:
            # Merge kwargs
            kwargs = block_kwargs.copy()
            kwargs.update(specific_kwargs)
            
            # Instantiate UNetBlock directly
            model = UNetBlock(**kwargs)
            
            res = benchmark_attention(name, model, x, emb)
            results.append((name, *res))
            
            # Clean up to free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{name:<40} | Failed: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 120)
    print(f"{'Architecture':<40} | {'Time (ms)':<10} | {'Params (M)':<10} | {'Init Loss':<10} | {'Final Loss':<10}")
    print("-" * 120)
    for r in results:
        name, t, m, p, il, fl = r
        print(f"{name:<40} | {t:<10.2f} | {p:<10.2f} | {il:<10.4f} | {fl:<10.4f}")

if __name__ == "__main__":
    run_benchmark()
