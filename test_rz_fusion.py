
import torch
import torch.nn as nn
import time
import sys
import os

# Add path to include torchquantum and training modules
sys.path.append('/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion')

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:
    print("TorchQuantum not found")
    sys.exit(1)

def benchmark_rz_fusion():
    n_qubits = 8
    bsz = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random params
    params = torch.randn(bsz, n_qubits, device=device)
    
    # 1. Sequential Rz
    qdev_seq = tq.QuantumDevice(n_wires=n_qubits, bsz=bsz, device=device)
    
    start = time.time()
    for _ in range(100):
        for i in range(n_qubits):
            tqf.rz(qdev_seq, wires=i, params=params[:, i])
    torch.cuda.synchronize()
    end = time.time()
    print(f"Sequential Rz Time: {end - start:.4f}s")
    
    # 2. Fused Rz (Diagonal)
    # Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2})
    # Total diagonal is Kronecker product of diags.
    # Phase for basis state |k> where k = b_{n-1}...b_0
    # Phase = sum_{j} (if b_j=0 then -theta_j/2 else theta_j/2)
    #       = sum_{j} (-1)^{b_j + 1} * theta_j / 2
    # We can compute this efficiently.
    
    # Precompute basis indices bits
    # dim = 2^n
    dim = 2**n_qubits
    arange = torch.arange(dim, device=device)
    # bits: [dim, n_qubits]
    # We can extract bits using bitwise ops
    bits = ((arange.unsqueeze(1) >> torch.arange(n_qubits - 1, -1, -1, device=device)) & 1)
    # Note: tq usually uses big-endian or little-endian? 
    # tq is usually big-endian (wire 0 is MSB? or LSB?).
    # Let's verify with small example or assume standard.
    # If wire 0 is top, it is usually MSB in state vector representation |q0 q1 ...>.
    
    # Convert bits 0 -> -1, 1 -> 1
    signs = (bits * 2 - 1).float() # [dim, n]
    # Actually Rz definition: 
    # |0> -> e^{-i theta/2}
    # |1> -> e^{i theta/2}
    # So 0 corresponds to coeff -0.5, 1 corresponds to +0.5
    
    qdev_fused = tq.QuantumDevice(n_wires=n_qubits, bsz=bsz, device=device)
    
    start = time.time()
    for _ in range(100):
        # Calculate total phase for each state index for each batch item
        # params: [B, n]
        # signs: [dim, n]
        # We want [B, dim] phase sum
        # sum_j (signs[k, j] * params[b, j] * 0.5)
        # = 0.5 * (params @ signs.T)
        
        # params: [B, n], signs.T: [n, dim] -> [B, dim]
        phases = 0.5 * torch.matmul(params, signs.T)
        
        # rotation = exp(1j * phases)
        rot_diag = torch.complex(torch.cos(phases), torch.sin(phases))
        
        # Apply diagonal
        # qdev.states: [B, 2^n] (or [B, 2, 2, ...])
        # If flat:
        if qdev_fused.states.ndim > 2:
            states_flat = qdev_fused.states.reshape(bsz, -1)
            states_flat = states_flat * rot_diag
            qdev_fused.states = states_flat.reshape(qdev_fused.states.shape)
        else:
            qdev_fused.states = qdev_fused.states * rot_diag
            
    torch.cuda.synchronize()
    end = time.time()
    print(f"Fused Rz Time: {end - start:.4f}s")

if __name__ == '__main__':
    benchmark_rz_fusion()
