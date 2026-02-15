
import torch
import time
import sys
sys.path.append('/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion')
from training.quantum_transformer import QuantumFrontEndQCNN, apply_unitary_bmm

class MockQDev:
    def __init__(self, bsz, n_wires):
        self.bsz = bsz
        self.n_wires = n_wires
        # dim = 2**n_wires
        # Use tensor shape [bsz, 2, 2, ..., 2] for compatibility with apply_unitary_bmm
        shape = [bsz] + [2] * n_wires
        self.states = torch.randn(shape, dtype=torch.cfloat, device='cuda')

def benchmark_ry():
    print("Benchmarking RY Layer Strategies...")
    
    bsz = 8192 # Sub-batch size
    n_qubits = 6
    device = 'cuda'
    
    qcnn = QuantumFrontEndQCNN(channels=4, style_dim=4, n_qubits_data=n_qubits, device_name=device).to(device)
    qdev = MockQDev(bsz, n_qubits)
    params = torch.randn(bsz, n_qubits, device=device)
    
    # Method 1: Fused (Current)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        qcnn._fast_ry_layer(qdev, params)
    torch.cuda.synchronize()
    t_fused = (time.time() - start) / 10
    print(f"Fused RY Layer: {t_fused*1000:.3f} ms")
    
    # Method 2: Sequential Loop
    def sequential_ry(qdev, params):
        for i in range(n_qubits):
            # Inline logic of _fast_ry to avoid method lookup overhead measurement
            theta = params[:, i]
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            matrix = torch.stack([
                torch.stack([c, -s], dim=1),
                torch.stack([s, c], dim=1)
            ], dim=1).to(qdev.states.dtype)
            qdev.states = apply_unitary_bmm(qdev.states, matrix, [i])

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        sequential_ry(qdev, params)
    torch.cuda.synchronize()
    t_seq = (time.time() - start) / 10
    print(f"Sequential RY Loop: {t_seq*1000:.3f} ms")
    
    print(f"Speedup Sequential vs Fused: {t_fused/t_seq:.2f}x")

if __name__ == "__main__":
    benchmark_ry()
