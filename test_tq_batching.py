
import time
import torch
import torchquantum as tq
import torchquantum.functional as tqf
import traceback

def test_batching():
    bsz = 10000
    n_wires = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    qdev = tq.QuantumDevice(n_wires=n_wires, bsz=bsz, device=device.type)
    
    params_single = torch.randn(bsz, device=device)
    params_multi = torch.randn(bsz, n_wires, device=device)
    
    # Method 1: Loop
    qdev.reset_states(bsz)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    for i in range(n_wires):
        # params_multi[:, i] is [B]
        tqf.ry(qdev, wires=i, params=params_multi[:, i]) 
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"Loop time: {(time.time() - start)*1000:.2f} ms")
    
    # Method 2: List of wires (if supported)
    qdev.reset_states(bsz)
    try:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.time()
        # tqf.ry might expect params to be [B, N] if wires is list? 
        # Or it might loop internally.
        tqf.ry(qdev, wires=list(range(n_wires)), params=params_multi)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"Batch wires time: {(time.time() - start)*1000:.2f} ms")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_batching()
