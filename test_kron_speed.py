
import torch
import time
import sys

def benchmark_kron_speed():
    n_qubits = 8
    bsz = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random 2x2 matrices
    # [B, 2, 2]
    mats = [torch.randn(bsz, 2, 2, device=device) for _ in range(n_qubits)]
    
    start = time.time()
    for _ in range(100):
        res = mats[0]
        for m in mats[1:]:
            # einsum: bik, bjl -> bijkl -> reshape
            # [B, dim1, dim1] x [B, 2, 2] -> [B, dim1*2, dim1*2]
            res = torch.einsum('bik,bjl->bijkl', res, m).reshape(bsz, res.shape[1]*2, res.shape[2]*2)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Kron Construction Time: {end - start:.4f}s")

if __name__ == '__main__':
    benchmark_kron_speed()
