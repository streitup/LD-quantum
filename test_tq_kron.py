
import time
import torch
import torchquantum as tq
import torchquantum.functional as tqf

def batch_kron(A, B):
    """
    Batched Kronecker product.
    A: [B, m, n]
    B: [B, p, q]
    Returns: [B, m*p, n*q]
    """
    bsz = A.shape[0]
    m, n = A.shape[1], A.shape[2]
    p, q = B.shape[1], B.shape[2]
    
    return torch.einsum('bik,bjl->bijkl', A, B).reshape(bsz, m*p, n*q)

def get_ry_matrix(params):
    # params: [B]
    # returns: [B, 2, 2]
    theta = params
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    row1 = torch.stack([c, -s], dim=1)
    row2 = torch.stack([s, c], dim=1)
    return torch.stack([row1, row2], dim=1)

def test_kron_speed():
    bsz = 2048 # Realistic chunk size
    n_wires = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    qdev = tq.QuantumDevice(n_wires=n_wires, bsz=bsz, device=device.type)
    
    params_multi = torch.randn(bsz, n_wires, device=device)
    
    # Method 1: Loop
    qdev.reset_states(bsz)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    for i in range(n_wires):
        tqf.ry(qdev, wires=i, params=params_multi[:, i])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"Loop time: {(time.time() - start)*1000:.2f} ms")
    
    # Method 2: Fused Unitary
    qdev.reset_states(bsz)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    
    # Construct unitary
    # Assume wires 0 to n_wires-1
    # U = RY(0) kron RY(1) ... kron RY(n-1) ? 
    # Actually state order is usually 0, 1, 2...
    # TorchQuantum state is [B, 2, 2, 2...]
    # If we apply U to all wires, U should be U0 kron U1 ... or Un-1 kron ... U0?
    # Usually it's U0 kron U1... but let's check convention.
    # We just want to benchmark speed of construction + application.
    
    mats = []
    for i in range(n_wires):
        mats.append(get_ry_matrix(params_multi[:, i]))
        
    U = mats[0]
    for i in range(1, n_wires):
        U = batch_kron(U, mats[i])
        
    # Apply
    # We need to manually call apply_unitary_bmm or similar
    # But qdev.apply_unitary works if we wrap it?
    # tqf.qubitunitary(qdev, wires=list(range(n_wires)), params=U)
    tqf.qubitunitary(qdev, wires=list(range(n_wires)), params=U)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"Fused time: {(time.time() - start)*1000:.2f} ms")

if __name__ == "__main__":
    test_kron_speed()
