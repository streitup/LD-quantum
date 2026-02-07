
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import math

# Import from training code
import sys
sys.path.append("/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion/training")

from quantum_transformer import (
    QuantumFrontEndQCNN, 
    QuantumFrontEndQCNNState, 
    QuantumAttentionAngle, 
    QuantumAttentionState
)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MeasuredBlock(nn.Module):
    def __init__(self, C, emb_dim, size):
        super().__init__()
        # Front-end: 128 channels input
        self.fe = QuantumFrontEndQCNN(
            channels=C, 
            style_dim=emb_dim, 
            n_layers=1,
            n_groups=1,
            encoding_type='amplitude'
        )
        # Projection: 128 (QCNN output) -> 64 (Attention input)
        self.proj = nn.Linear(C, 64)
        
        self.attn = QuantumAttentionAngle(
            N_QUBITS=6,
            qk_dim=16,
            n_heads=4,
            force_fp32_attention=False
        )
        
    def forward(self, x, style):
        # 1. Front End (Measurement included)
        # Output: [B, 128, H, W]
        feat = self.fe(x, style)
        
        # 2. Reshape & Project for Attention 
        # [B, 128, H, W] -> [B, 128, L] -> [B, L, 128] -> [B, L, 64]
        feat_seq = feat.flatten(2).transpose(1, 2)
        feat_seq = self.proj(feat_seq)
        
        # 3. Attention (Re-encoding included)
        out = self.attn(feat_seq)
        return out

class FusedBlock(nn.Module):
    def __init__(self, C, emb_dim, size):
        super().__init__()
        # Front-end: Returns State
        self.fe = QuantumFrontEndQCNNState(
            channels=C, 
            style_dim=emb_dim, 
            n_layers=1,
            n_groups=1,
            encoding_type='amplitude',
            n_qubits_ancilla=0
        )
        # Attention: Consumes State
        self.attn = QuantumAttentionState(
            N_QUBITS=6,
            qk_dim=16,
            n_heads=4,
            force_fp32_attention=False
        )
        
    def forward(self, x, style):
        # 1. Front End (Returns State [B, L, 2^6])
        state = self.fe(x, style)
        
        # 2. Attention (Uses State directly)
        out = self.attn(state)
        return out

def benchmark_fused_architecture():
    print("\n=== Benchmark: Fused QCNN-Attention (No Measurement) ===")
    print("Goal: Compare Standard (Measure -> Re-encode) vs Fused (State Transfer)")
    print("Resolution: 128x16x16 Input, Patch Size 8x8 (6 Qubits)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Params
    C = 128
    emb_dim = 256
    bsz = 4
    size = 16 # 16x16 image
    
    # Inputs
    x = torch.randn(bsz, C, size, size).to(device)
    style = torch.randn(bsz, emb_dim).to(device)
    
    # Models
    model_measured = MeasuredBlock(C, emb_dim, size).to(device)
    model_fused = FusedBlock(C, emb_dim, size).to(device)
    
    # Optimizer
    opt_measured = optim.Adam(model_measured.parameters(), lr=1e-3)
    opt_fused = optim.Adam(model_fused.parameters(), lr=1e-3)
    
    print("\n--- Parameter Count ---")
    print(f"Measured Block: {sum(p.numel() for p in model_measured.parameters())}")
    print(f"Fused Block:    {sum(p.numel() for p in model_fused.parameters())}")
    
    print("\n--- Training Speed (Forward + Backward) ---")
    steps = 50
    
    # Warmup
    for _ in range(5):
        _ = model_measured(x, style)
        _ = model_fused(x, style)
        
    # Measure Measured
    start = time.time()
    for _ in range(steps):
        opt_measured.zero_grad()
        out = model_measured(x, style)
        loss = out.mean()
        loss.backward()
        opt_measured.step()
    torch.cuda.synchronize()
    time_measured = time.time() - start
    print(f"Measured: {time_measured:.4f}s ({steps/time_measured:.2f} it/s)")
    
    # Measure Fused
    start = time.time()
    for _ in range(steps):
        opt_fused.zero_grad()
        out = model_fused(x, style)
        loss = out.mean()
        loss.backward()
        opt_fused.step()
    torch.cuda.synchronize()
    time_fused = time.time() - start
    print(f"Fused:    {time_fused:.4f}s ({steps/time_fused:.2f} it/s)")
    
    print("\n--- Conclusion ---")
    print("Fused architecture skips intermediate measurement and re-encoding.")
    if time_fused < time_measured:
        print(f"Result: Fused is {time_measured/time_fused:.2f}x Faster.")
    else:
        print("Result: Fused is slower (likely due to complex state handling overhead in Python).")

if __name__ == "__main__":
    benchmark_fused_architecture()
