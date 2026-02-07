
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

# Add path to find quantum_transformer
sys.path.append(os.path.join(os.getcwd(), "LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion/training"))

try:
    from quantum_transformer import (
        QuantumFrontEndQCNN, 
        QuantumAttention64, 
        QuantumFrontEndQCNNState, 
        QuantumAttentionState
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class MeasuredBlock(nn.Module):
    def __init__(self, C, emb_dim, size):
        super().__init__()
        # Standard QCNN: Returns [B, C, H, W]
        self.fe = QuantumFrontEndQCNN(
            channels=C, 
            style_dim=emb_dim, 
            n_layers=1, 
            n_groups=1, 
            encoding_type='amplitude'
        )
        
        # Projection to match Attention Input Dim (128 -> 64)
        # Assuming Attention expects 64 input features if N_QUBITS=6 for Angle Encoding?
        # Actually QuantumAttention64 defaults might vary, but let's project to 64 for safety/consistency
        self.proj = nn.Linear(C, 64)
        
        # Standard Attention: Consumes Classical Tensor
        self.attn = QuantumAttention64(
            N_QUBITS=6,
            qk_dim=16,
            n_heads=4,
            force_fp32_attention=False
        )
        
    def forward(self, x, style):
        # 1. Front End: [B, C, H, W]
        feat = self.fe(x, style)
        
        # 2. Reshape to Sequence: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = feat.shape
        feat_seq = feat.view(B, C, H * W).transpose(1, 2)
        
        # 3. Project: [B, L, 128] -> [B, L, 64]
        feat_proj = self.proj(feat_seq)
        
        # 4. Attention: [B, L, 64] -> [B, L, 64]
        out = self.attn(feat_proj)
        return out

class FusedBlock(nn.Module):
    def __init__(self, C, emb_dim, size):
        super().__init__()
        # Front-end: Returns State [B, L, 2^6]
        # n_qubits_ancilla=0 to match Attention N_QUBITS=6 (2^6=64)
        self.fe = QuantumFrontEndQCNNState(
            channels=C, 
            style_dim=emb_dim, 
            n_layers=1,
            n_groups=1,
            encoding_type='amplitude',
            n_qubits_ancilla=0
        )
        # Attention: Consumes State directly
        self.attn = QuantumAttentionState(
            N_QUBITS=6,
            qk_dim=16,
            n_heads=4,
            force_fp32_attention=False
        )
    
    def forward(self, x, style):
        # 1. Front End (Returns State [B, L, 64])
        state = self.fe(x, style)
        
        # 2. Attention (Uses State directly)
        out = self.attn(state)
        return out

def run_benchmark():
    print("=== Benchmark: Loss Comparison (Measured vs Fused) ===")
    
    # Settings
    B = 2 # Batch size (keep small for speed)
    C = 128
    H = 16
    W = 16
    emb_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Models
    model_measured = MeasuredBlock(C, emb_dim, (H, W)).to(device)
    model_fused = FusedBlock(C, emb_dim, (H, W)).to(device)
    
    # Data
    x = torch.randn(B, C, H, W, device=device)
    style = torch.randn(B, emb_dim, device=device)
    
    # Check output shape to set target
    with torch.no_grad():
        dummy_out = model_measured(x, style)
    print(f"Model Output Shape: {dummy_out.shape}")
    
    target = torch.randn_like(dummy_out) # Target matches output shape
    
    # Training Setup
    criterion = nn.MSELoss()
    lr = 1e-3
    opt_m = optim.Adam(model_measured.parameters(), lr=lr)
    opt_f = optim.Adam(model_fused.parameters(), lr=lr)
    
    steps = 100
    
    print(f"\nTraining for {steps} steps...")
    
    losses_m = []
    losses_f = []
    
    start_time = time.time()
    
    for i in range(steps):
        # Measured
        opt_m.zero_grad()
        out_m = model_measured(x, style)
        loss_m = criterion(out_m, target)
        loss_m.backward()
        opt_m.step()
        losses_m.append(loss_m.item())
        
        # Fused
        opt_f.zero_grad()
        out_f = model_fused(x, style)
        loss_f = criterion(out_f, target)
        loss_f.backward()
        opt_f.step()
        losses_f.append(loss_f.item())
        
        if (i+1) % 20 == 0:
            print(f"Step {i+1:03d} | Loss Measured: {loss_m.item():.6f} | Loss Fused: {loss_f.item():.6f}")
            
    total_time = time.time() - start_time
    print(f"\nTotal Time: {total_time:.2f}s")
    print(f"Final Loss Measured: {losses_m[-1]:.6f}")
    print(f"Final Loss Fused:    {losses_f[-1]:.6f}")
    
    # Analysis
    improvement = (losses_m[-1] - losses_f[-1]) / losses_m[-1] * 100
    print(f"\nLoss Improvement (Fused vs Measured): {improvement:.2f}%")
    if losses_f[-1] < losses_m[-1]:
        print("Result: Fused Architecture achieves LOWER loss.")
    else:
        print("Result: Fused Architecture achieves HIGHER loss.")

if __name__ == "__main__":
    run_benchmark()
