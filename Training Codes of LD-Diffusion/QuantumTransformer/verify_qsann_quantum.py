import os
import sys
import torch

# Enable debug prints inside QSANNAdapter
os.environ['QTRANSFORMER_DEBUG'] = '1'

print("[verify] Starting QSANNAdapter quantum attention verification...")
try:
    import QSANNAdapter as qsann_mod
    from QSANNAdapter import QSANNAdapter
except Exception as e:
    print(f"[verify][ERROR] Failed to import QSANNAdapter: {e}")
    sys.exit(1)

print(f"[verify] TorchQuantum available flag in module: {getattr(qsann_mod, '_TQ_AVAILABLE', None)}")

try:
    adapter = QSANNAdapter(N_QUBITS=8, Q_DEPTH=2, qk_dim=4, prefer_x_interface=True, device_name='cpu')
except Exception as e:
    print(f"[verify][ERROR] Failed to instantiate QSANNAdapter: {e}")
    sys.exit(1)

# Create a small input tensor and run forward in x-only mode
x = torch.randn(1, 8, 4, 4)
print(f"[verify] Input shape: {tuple(x.shape)}; dtype: {x.dtype}; device: {x.device}")

try:
    out = adapter(x, num_heads=2)
    print(f"[verify] Output shape: {tuple(out.shape)}; dtype: {out.dtype}; device: {out.device}")
    print("[verify] Completed forward pass.")
except Exception as e:
    print(f"[verify][ERROR] Forward pass failed: {e}")
    sys.exit(1)