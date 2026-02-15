
import torch
import time
import sys

# Add path
sys.path.append('/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion')

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    from torchquantum.functional.gate_wrapper import apply_unitary_bmm
except ImportError:
    print("TorchQuantum not found")
    sys.exit(1)

def benchmark_ry_fusion():
    n_qubits = 8
    bsz = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    params = torch.randn(bsz, n_qubits, device=device)
    
    qdev_seq = tq.QuantumDevice(n_wires=n_qubits, bsz=bsz, device=device)
    
    # 1. Sequential Ry
    start = time.time()
    for _ in range(100):
        for i in range(n_qubits):
            tqf.ry(qdev_seq, wires=i, params=params[:, i])
    torch.cuda.synchronize()
    end = time.time()
    print(f"Sequential Ry Time: {end - start:.4f}s")
    
    # 2. Fused Ry (Matrix Mult)
    # Construct full unitary for Ry layer
    # Ry(theta) = [[c, -s], [s, c]]
    # This part is tricky to batch efficiently without loop if we don't use bmm on small matrices.
    # But we want to test apply_unitary_bmm with full matrix vs loop.
    
    # Let's assume we have the matrix constructed (cost of construction excluded for now, or included?)
    # Construction cost is non-negligible.
    # But let's test if applying a 256x256 matrix is faster than 8 calls to tqf.ry.
    
    dim = 2**n_qubits
    # Random unitary [B, dim, dim]
    # In reality it is structured.
    mat = torch.eye(dim, dtype=torch.complex64, device=device).unsqueeze(0).expand(bsz, -1, -1)
    
    qdev_fused = tq.QuantumDevice(n_wires=n_qubits, bsz=bsz, device=device)
    
    start = time.time()
    for _ in range(100):
        # Apply full matrix
        # qdev.states: [B, 2^n]
        # states = mat @ states.unsqueeze(-1) -> [B, 2^n, 1]
        states = qdev_fused.states.reshape(bsz, dim, 1)
        new_states = torch.bmm(mat, states).reshape(bsz, -1)
        qdev_fused.states = new_states
        
    torch.cuda.synchronize()
    end = time.time()
    print(f"Fused Ry (Full Matrix Apply) Time: {end - start:.4f}s")
    
    # 3. Hybrid: Vectorized Construction + Apply?
    # No, let's just see if matrix mult is faster.

if __name__ == '__main__':
    benchmark_ry_fusion()
