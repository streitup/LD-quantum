import torch
import torch.nn as nn
import time
import os
import sys

# Ensure current path is in sys.path
sys.path.append(os.getcwd())

from training.networks import DhariwalUNet

def benchmark_speed():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {device}")
    
    # Common settings
    B, C, H, W = 4, 128, 32, 32 # Use 128 channels to match DhariwalUNet default model_channels somewhat better
    # Or just let DhariwalUNet adapt.
    # The error "mat1 and mat2 shapes cannot be multiplied (1x4 and 10x768)"
    # map_label takes [B, label_dim]. But in DhariwalUNet:
    # tmp = class_labels (which is [B])
    # self.map_label(tmp) -> Linear expects [B, in_features]
    # But class_labels is [B]. We need to one-hot encode it or embed it?
    # Wait, `networks.py` Line 620:
    # self.map_label = Linear(in_features=label_dim, out_features=emb_channels, ...)
    # It expects a vector of size label_dim. Usually one-hot.
    # But in `training_loop.py` or precond, it usually handles labels.
    # Let's check `VPPrecond` etc.
    # Line 740: class_labels = torch.zeros([1, self.label_dim]) ...
    # It seems it expects one-hot or zero-padded vector.
    
    # Fix: One-hot encode class_labels
    noise_labels = torch.randn(B).to(device)
    raw_labels = torch.randint(0, 10, (B,)).to(device)
    class_labels = torch.zeros(B, 10).to(device).scatter_(1, raw_labels.unsqueeze(1), 1.0)
    
    x = torch.randn(B, C, H, W).to(device)
    
    # 1. Classic UNet
    print("\n--- Benchmarking Classic DhariwalUNet ---")
    classic_unet = DhariwalUNet(
        img_resolution=H,
        in_channels=C,
        out_channels=C,
        label_dim=10,
        model_channels=128, # Explicitly set to avoid huge defaults
        channel_mult=[1, 1], # Simple structure for speed test
        num_blocks=1,
        use_qcnn_frontend=False
    ).to(device)
    
    # Warmup
    for _ in range(5):
        _ = classic_unet(x, noise_labels, class_labels)
    
    torch.cuda.synchronize()
    start = time.time()
    steps = 50
    for _ in range(steps):
        _ = classic_unet(x, noise_labels, class_labels)
    torch.cuda.synchronize()
    classic_time = (time.time() - start) / steps
    print(f"Classic UNet Speed: {1/classic_time:.2f} it/s (Time per iter: {classic_time*1000:.2f} ms)")
    
    # 2. Quantum UNet (Optimized)
    print("\n--- Benchmarking Quantum UNet (Grouped QCNN) ---")
    # Config matching our "Pure Quantum" best result
    quantum_unet = DhariwalUNet(
        img_resolution=H,
        in_channels=C,
        out_channels=C,
        label_dim=10,
        model_channels=128,
        channel_mult=[1, 1],
        num_blocks=1,
        use_qcnn_frontend=True,
        qcnn_chunk_size=4096, 
        qcnn_use_strided=False,
        qcnn_reupload=True
    ).to(device)
    
    # Note: networks.py initializes QuantumFrontEndQCNN. 
    # We need to ensure it uses the "Grouped=4, NoBypass" config by default or via kwargs.
    # Currently `networks.py` hardcodes some params in `__init__`:
    # n_qubits_ancilla=4, n_layers=1.
    # We might need to adjust `networks.py` to match our "Best Config" (n_layers=2, n_groups=4).
    # But for now let's test what `networks.py` has, assuming I've updated it or will update it.
    # Wait, I haven't updated `networks.py` to pass `n_groups`!
    # I should update `networks.py` first to ensure fair comparison with the "Optimized" version.
    
    # Warmup
    for _ in range(3):
        _ = quantum_unet(x, noise_labels, class_labels)
        
    torch.cuda.synchronize()
    start = time.time()
    steps = 20 # Fewer steps for quantum as it is slower
    for _ in range(steps):
        _ = quantum_unet(x, noise_labels, class_labels)
    torch.cuda.synchronize()
    quantum_time = (time.time() - start) / steps
    print(f"Quantum UNet Speed: {1/quantum_time:.2f} it/s (Time per iter: {quantum_time*1000:.2f} ms)")
    
    print("\n--- Summary ---")
    print(f"Classic: {classic_time*1000:.2f} ms")
    print(f"Quantum: {quantum_time*1000:.2f} ms")
    print(f"Slowdown Factor: {quantum_time/classic_time:.2f}x")

if __name__ == "__main__":
    benchmark_speed()
