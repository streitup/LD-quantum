
import time
import torch
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional.gate_wrapper import apply_unitary_einsum, apply_unitary_bmm

def get_ry_matrix(params):
    # params: [B]
    # returns: [B, 2, 2]
    theta = params
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    row1 = torch.stack([c, -s], dim=1)
    row2 = torch.stack([s, c], dim=1)
    return torch.stack([row1, row2], dim=1)

def benchmark_fast_tq():
    bsz = 10000
    n_wires = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    qdev = tq.QuantumDevice(n_wires=n_wires, bsz=bsz, device=device.type)
    # State: [B, 2, 2, ..., 2] (n_wires+1 dims)
    # qdev init creates states.
    
    params = torch.randn(bsz, device=device)
    mat = get_ry_matrix(params).to(dtype=torch.complex64)
    
    wires = [0] # Single wire application
    
    # Method 0: tqf.ry
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        tqf.ry(qdev, wires=0, params=params)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"tqf.ry time (100 runs): {(time.time() - start)*1000:.2f} ms")
    
    # Method 1: Einsum
    state = qdev.states # Use qdev states
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        # We need to clone state or modify in place? tq modifies state usually
        # But for benchmark, let's just run.
        res = apply_unitary_einsum(state, mat, wires)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"Einsum time (100 runs): {(time.time() - start)*1000:.2f} ms")
    
    # Method 2: BMM
    try:
        from torchquantum.functional.gate_wrapper import apply_unitary_bmm
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            res = apply_unitary_bmm(state, mat, wires)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"BMM time (100 runs): {(time.time() - start)*1000:.2f} ms")
    except ImportError:
        print("BMM not found")

    # Method 3: Compiled BMM (Einsum was slow)
    try:
        fast_bmm = torch.compile(apply_unitary_bmm)
        # Warmup
        res = fast_bmm(state, mat, wires)
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            res = fast_bmm(state, mat, wires)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"Compiled BMM time (100 runs): {(time.time() - start)*1000:.2f} ms")
    except Exception as e:
        print(f"Compilation failed: {e}")

if __name__ == "__main__":
    benchmark_fast_tq()
