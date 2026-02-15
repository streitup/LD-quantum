
import time
import torch
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append('/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion')

from training.quantum_transformer import QuantumFrontEndQCNN, ClassicAttention64

def benchmark_model(model, x, style=None, name="Model", n_warmup=5, n_runs=20):
    device = x.device
    model.to(device)
    model.train()
    
    # Warmup
    print(f"Warming up {name}...")
    with torch.no_grad():
        for _ in range(n_warmup):
            if style is not None:
                _ = model(x, style)
            else:
                _ = model(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Benchmark Forward
    print(f"Benchmarking {name} Forward...")
    for _ in range(n_runs):
        if style is not None:
            _ = model(x, style)
        else:
            _ = model(x)
            
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) / n_runs
    print(f"{name} Average Forward Time: {avg_time*1000:.2f} ms")
    
    return avg_time

def run_benchmark():
    # Settings
    B = 8
    C = 64 # Using 64 to be lighter, or 128 if we want to stress test
    H = 32
    W = 32
    style_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Benchmark Settings: Batch={B}, Channels={C}, Size={H}x{W}, Device={device}")
    
    # Inputs
    x = torch.randn(B, C, H, W, device=device)
    style = torch.randn(B, style_dim, device=device)
    
    # 1. Classical Convolution (Baseline)
    # Matching QCNN: 3x3 kernel, padding 1, stride 1 (or 2)
    # QCNN defaults: kernel=3, padding=1
    cnn = nn.Conv2d(C, C, kernel_size=3, padding=1, stride=1).to(device)
    
    # 2. QuantumFrontEndQCNN
    # Using default params mostly, but ensuring valid groups
    # channels=64, n_groups=8 (default n_groups=1 in init? let's check. Default is 1.)
    # User mentioned "parallel quantum circuit", implying groups or batching.
    # Let's try with n_groups=8 to test grouped parallelism.
    qcnn = QuantumFrontEndQCNN(
        channels=C,
        style_dim=style_dim,
        n_qubits_data=6,
        n_qubits_ancilla=2,
        n_groups=8, # Grouped QCNN
        stride=1,
        device_name='cuda'
    ).to(device)
    
    # 3. Classic Attention (Alternative Baseline if QCNN replaces Attention)
    # ClassicAttention64 expects [B, S, 64].
    # We need to adapt input/output for it.
    class ClassicAttnWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ClassicAttention64(num_heads=8)
        def forward(self, x):
            # x: [B, C, H, W] -> [B, H*W, C] (Assuming C=64)
            b, c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
            out = self.attn(x_flat)
            return out.reshape(b, h, w, c).permute(0, 3, 1, 2)
            
    classic_attn = ClassicAttnWrapper().to(device)

    # Run Benchmarks
    print("\n--- Running Benchmarks ---")
    
    # CNN
    t_cnn = benchmark_model(cnn, x, name="Classical Conv2d (Baseline)", n_warmup=10, n_runs=50)
    
    # Classic Attention
    if C == 64:
        t_attn = benchmark_model(classic_attn, x, name="Classic Attention (Baseline)", n_warmup=10, n_runs=50)
    
    # QCNN
    # QCNN is expected to be slower, so fewer runs
    try:
        t_qcnn = benchmark_model(qcnn, x, style=style, name="QuantumFrontEndQCNN", n_warmup=2, n_runs=5)
        
        print(f"\n--- Results Summary ---")
        print(f"Classical Conv2d: {t_cnn*1000:.2f} ms")
        if C == 64:
            print(f"Classic Attention: {t_attn*1000:.2f} ms")
        print(f"Quantum QCNN:     {t_qcnn*1000:.2f} ms")
        print(f"Slowdown Factor (vs CNN): {t_qcnn/t_cnn:.1f}x")
        
    except Exception as e:
        print(f"QCNN Benchmark Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()
