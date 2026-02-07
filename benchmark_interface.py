import time
import torch
import torch.nn as nn
from training.quantum_transformer import QuantumAttentionHybridLite

def benchmark_interface():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    # Config
    B, S, D = 32, 64, 128
    # patch_size = 2 # Not used in HybridLite (assumes tokens are already processed)
    n_qubits = 8
    
    # QuantumAttentionHybridLite expects input_dim to match D
    # It inherits from QuantumAttention64 which has default qk_dim=16, n_heads=4 -> inner_dim=64
    # If D=128, we need to ensure dimensions match or projections handle it.
    # HybridLite has separate q_proj_lite (Conv1d) that maps input_dim -> inner_dim.
    
    model = QuantumAttentionHybridLite(
        input_dim=D,
        n_heads=4,
        qk_dim=32, # 4 * 32 = 128 inner dim
        N_QUBITS=n_qubits,
        Q_DEPTH=2,
        device_name=device.type
    ).to(device)
    
    x = torch.randn(B, S, D, device=device)
    
    # Warmup
    for _ in range(5):
        _ = model(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    iters = 50
    for _ in range(iters):
        _ = model(x)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iters
    print(f"Average Forward Time: {avg_time*1000:.2f} ms")

if __name__ == "__main__":
    try:
        benchmark_interface()
    except Exception as e:
        print(f"Error: {e}")
