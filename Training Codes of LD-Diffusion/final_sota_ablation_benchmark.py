
import torch
import time
import sys
import os
import numpy as np
from torch.cuda.amp import autocast

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from training.quantum_transformer import (
        QuantumAttention64, 
        QuantumAttentionAngle, 
        QuantumAttentionAngleDense,
        ClassicAttention64
    )
except ImportError:
    # Fallback if running directly from folder
    from training.quantum_transformer import (
        QuantumAttention64, 
        QuantumAttentionAngle, 
        QuantumAttentionAngleDense,
        ClassicAttention64
    )

def benchmark_model(name, model, input_tensor, device, num_runs=50):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # Warmup
    print(f"[{name}] Warming up...")
    for _ in range(5):
        with autocast():
            out = model(input_tensor)
            loss = out.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Forward Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        with autocast():
            out = model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    forward_time = start_event.elapsed_time(end_event) / num_runs
    
    # Backward Benchmark
    start_event.record()
    for _ in range(num_runs):
        with autocast():
            out = model(input_tensor)
            loss = out.mean()
        scaler.scale(loss).backward()
        optimizer.zero_grad() # Don't step, just backward
    end_event.record()
    torch.cuda.synchronize()
    backward_time = start_event.elapsed_time(end_event) / num_runs
    
    # Memory
    torch.cuda.reset_peak_memory_stats()
    with autocast():
        out = model(input_tensor)
        loss = out.mean()
    scaler.scale(loss).backward()
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    
    # Parameter Count
    params = sum(p.numel() for p in model.parameters())
    
    return {
        "Forward (ms)": forward_time,
        "Backward (ms)": backward_time,
        "Peak Mem (MB)": peak_mem,
        "Params": params
    }

def run_ablation_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Benchmark on: {device}")
    
    # Settings
    B, S, C = 32, 16, 64 # Typical Batch, Sequence, Channels
    input_tensor = torch.randn(B, S, C)
    
    print(f"Input Shape: {input_tensor.shape}")
    
    results = {}
    
    # 1. Baseline: Classical Attention
    print("\n--- 1. Baseline: Classical Attention ---")
    model_classic = ClassicAttention64(num_heads=4)
    results['Classic (Baseline)'] = benchmark_model('Classic', model_classic, input_tensor, device)
    
    # 2. QSANN (Original - Amplitude Encoding)
    print("\n--- 2. QSANN (Original: Amplitude) ---")
    # Note: Using QuantumAttention64 which is the base class implementing Amplitude Encoding
    model_qsann = QuantumAttention64(n_heads=4, device_name='cuda')
    results['QSANN (Amplitude)'] = benchmark_model('QSANN', model_qsann, input_tensor, device)
    
    # 3. QSANN (Angle Encoding)
    print("\n--- 3. QSANN (Ablation: Angle Encoding) ---")
    # Tests the switch from Amplitude to Angle (Rx/Ry)
    model_angle = QuantumAttentionAngle(n_heads=4, device_name='cuda')
    results['QSANN (Angle)'] = benchmark_model('Angle', model_angle, input_tensor, device)
    
    # 4. SOTA: Dense Angle Encoding (with Batch Parallel)
    print("\n--- 4. SOTA: Dense Angle Encoding (Layer-wise) ---")
    # Tests Layer-wise Injection + Batch Parallel Optimizations
    model_sota = QuantumAttentionAngleDense(n_heads=4, device_name='cuda')
    results['SOTA (Dense)'] = benchmark_model('SOTA', model_sota, input_tensor, device)
    
    # Print Results Table
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'Params':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Mem (MB)':<10}")
    print("-" * 80)
    
    for name, res in results.items():
        print(f"{name:<25} | {res['Params']:<10} | {res['Forward (ms)']:<10.2f} | {res['Backward (ms)']:<10.2f} | {res['Peak Mem (MB)']:<10.2f}")
    print("="*80)
    
    # Analysis
    print("\n[Analysis]")
    base_fwd = results['Classic (Baseline)']['Forward (ms)']
    qsann_fwd = results['QSANN (Amplitude)']['Forward (ms)']
    sota_fwd = results['SOTA (Dense)']['Forward (ms)']
    
    print(f"1. Quantum Overhead vs Classical: {qsann_fwd/base_fwd:.1f}x slower")
    print(f"2. SOTA Speedup vs Original QSANN: {qsann_fwd/sota_fwd:.1f}x faster")
    
    sota_params = results['SOTA (Dense)']['Params']
    qsann_params = results['QSANN (Amplitude)']['Params']
    print(f"3. Parameter Efficiency: {sota_params/qsann_params:.2f}x params (Dense adds projections)")

if __name__ == "__main__":
    run_ablation_benchmark()
