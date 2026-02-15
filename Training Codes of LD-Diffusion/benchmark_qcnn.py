
import time
import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from training.quantum_transformer import QuantumFrontEndQCNN, QuantumAttentionAngleDense

def benchmark_module(model, inputs, name="Module", n_iter=10):
    model.cuda()
    
    # Unpack inputs
    if isinstance(inputs, tuple):
        args = [x.cuda() for x in inputs]
    else:
        args = [inputs.cuda()]
        
    optimizer = torch.optim.Adam(model.parameters())
    
    # Warmup
    print(f"Warming up {name}...")
    try:
        for _ in range(3):
            y = model(*args)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    except Exception as e:
        print(f"Warmup failed for {name}: {e}")
        return
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"Benchmarking {name}...")
    torch.cuda.reset_peak_memory_stats()
    start_event.record()
    
    try:
        for _ in range(n_iter):
            y = model(*args)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    except Exception as e:
        print(f"Benchmark failed for {name}: {e}")
        return
        
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0 # seconds
    avg_time = elapsed_time / n_iter
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    
    print(f"Results for {name}:")
    print(f"  Avg Time per Iter: {avg_time:.4f} s")
    print(f"  Max Memory: {max_mem:.2f} MB")
    print("-" * 30)

if __name__ == "__main__":
    # Config
    C = 128
    H, W = 32, 32
    SEQ_LEN = H * W # 1024
    STYLE_DIM = C

    # Loop over Batch Sizes
    for BS in [8, 16, 32]:
        print(f"\nTesting Batch Size: {BS}")
        print(f"Config: BS={BS}, C={C}, H={H}, W={W}, SEQ_LEN={SEQ_LEN}")
        print("-" * 30)
    
        # 1. Test QuantumFrontEndQCNN
        print("Initializing QuantumFrontEndQCNN...")
        qcnn = QuantumFrontEndQCNN(
            channels=C, 
            style_dim=STYLE_DIM, 
            n_qubits_data=4, 
            n_layers=8,
            n_groups=8,
            device_name='cuda',
            max_qdev_bsz=4096, 
            use_checkpoint=True
        )
        
        x_qcnn = torch.randn(BS, C, H, W)
        style_qcnn = torch.randn(BS, STYLE_DIM)
        
        benchmark_module(qcnn, (x_qcnn, style_qcnn), f"QuantumFrontEndQCNN (BS={BS})", n_iter=5)
    
        # 2. Test QuantumAttentionAngleDense
        print("Initializing QuantumAttentionAngleDense...")
        attn = QuantumAttentionAngleDense(
            in_channels=64, 
            N_QUBITS=6,
            Q_DEPTH=4,
            num_heads=4,
            device_name='cuda',
            chunk_size=2048, 
            use_checkpoint=True
        )
        
        x_attn = torch.randn(BS, SEQ_LEN, 64) 
        
        benchmark_module(attn, x_attn, f"QuantumAttentionAngleDense (BS={BS})", n_iter=5)
        
        # Clean up
        del qcnn, attn, x_qcnn, style_qcnn, x_attn
        torch.cuda.empty_cache()
