import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.quantum_transformer import QuantumFrontEndQCNNState, QuantumFrontEndQCNN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Models ---

class ClassicalBaseline(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x, emb):
        return x + self.conv(self.norm(x))

class QCNN_Fused_State(nn.Module):
    """
    Original Fused-NoAttn: Amplitude Encoding -> Circuit -> State -> Abs^2 -> Proj
    (Linear Encoding + Linear Output + Manual Non-linearity)
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        self.state_dim = 64
        # Note: We use the fixed QCNNState class that handles Re-uploading correctly
        self.fe = QuantumFrontEndQCNNState(
            channels=channels,
            style_dim=emb_dim,
            n_layers=1, 
            n_groups=self.n_groups,
            encoding_type='amplitude',
            n_qubits_ancilla=0,
            reupload_data=True,
            stride=1
        )
        self.proj_out = nn.Linear(self.n_groups * self.state_dim, channels)

    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        state = self.fe(x, emb) # [B, L, G*D]
        features = (state.abs() ** 2).float() # Manual Measurement
        x_out = self.proj_out(features)
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out + resid

class QCNN_Amplitude_Meas(nn.Module):
    """
    Fused-NoAttn with Measurement Added Back properly?
    Actually, QCNN_Fused_State ALREADY does measurement (Abs^2).
    What if we use the Standard QCNN Projection logic (Per-Group Projection)?
    And ensure we are using the 'Measurement' flow.
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.channels = channels
        self.n_groups = 4
        # We subclass QCNNState but modify forward to behave like Standard QCNN
        self.fe = QuantumFrontEndQCNNState(
            channels=channels,
            style_dim=emb_dim,
            n_layers=1, 
            n_groups=self.n_groups,
            encoding_type='amplitude',
            n_qubits_ancilla=0,
            reupload_data=True,
            stride=1
        )
        # Standard QCNN uses a Linear layer per group output (probs)
        # 1<<N -> Channels_per_group
        self.out_proj = nn.Linear(64, channels // 4) 

    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        # We need to hack FE to return probs inside or do it here
        state = self.fe(x, emb) # [B, L, G*D]
        
        # Reshape to groups
        state = state.reshape(b, -1, self.n_groups, 64) # [B, L, G, 64]
        probs = (state.abs() ** 2).float()
        
        # Apply Per-Group Projection (Standard QCNN style)
        # [B, L, G, 64] -> [B, L, G, C/G]
        out_g = self.out_proj(probs)
        
        # Flatten
        out_flat = out_g.reshape(b, -1, self.n_groups * (self.channels // 4))
        
        x_out = out_flat.transpose(1, 2).reshape(b, c, h, w)
        return x_out + resid

class QCNN_Angle_Meas(nn.Module):
    """
    Standard QCNN: Angle Encoding -> Circuit -> Measurement -> Proj
    (Non-linear Encoding + Non-linear Output)
    Using Base QuantumFrontEndQCNN (Angle mode is safe from the bug)
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        self.fe = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim,
            n_layers=1,
            n_groups=self.n_groups,
            encoding_type='tanh', # Angle Encoding
            n_qubits_ancilla=0,
            reupload_data=True,
            stride=1
        )
        # Base class handles projection internally

    def forward(self, x, emb):
        resid = x
        x_out = self.fe(x, emb)
        return x_out + resid

# --- Benchmark ---

class QCNN_Angle_Pure(nn.Module):
    """
    Pure QCNN (Angle Encoding) WITHOUT Classical Residual.
    Tests if the Quantum Circuit alone can learn the features.
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        # Standard QCNN uses Angle Encoding (tanh) and measures Pauli Z
        self.fe = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim,
            n_layers=1, 
            n_groups=self.n_groups,
            encoding_type='tanh', # Angle encoding
            n_qubits_ancilla=2,
            reupload_data=True,
            stride=1
        )
        # Note: QuantumFrontEndQCNN already includes output projection to 'channels'
        # But it usually expects a residual connection to be added outside.
        # Here we just use the QCNN output.

    def forward(self, x, emb):
        # No residual connection
        # x_out = self.fe(x, emb)
        # However, QuantumFrontEndQCNN returns [B, C, H, W] directly?
        # Let's check QuantumFrontEndQCNN in quantum_transformer.py
        # Yes, it returns [B, C, H_out, W_out]
        
        return self.fe(x, emb)

def benchmark_qcnn_variants(steps=150, device='cuda'):
    print("\n=== Benchmark: QCNN Variants Analysis ===")
    print("Comparing Classical, Hybrid, Fused, and Pure QCNN architectures.")
    set_seed(42)
    
    C = 128
    EMB = 128
    H, W = 16, 16
    BSZ = 8
    
    # 1. Classical: Standard Conv Block
    # 2. Hybrid: Standard Angle QCNN + Classical Residual (The "Standard-Angle" from before)
    # 3. Fused: Amplitude Encoding + Global Projection (The "Fused-State" from before)
    # 4. Pure QCNN: Standard Angle QCNN without Residual
    
    models = {
        'Classical': ClassicalBaseline(C).to(device),
        'Hybrid': QCNN_Angle_Meas(C, EMB).to(device),      # Previously Standard-Angle
        'Fused': QCNN_Fused_State(C, EMB).to(device),      # Previously Fused-State
        'Pure QCNN': QCNN_Angle_Pure(C, EMB).to(device),   # New
    }
    
    # Optimizers
    opts = {name: optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
    losses = {name: [] for name in models}
    
    print(f"Classical Params: {count_parameters(models['Classical'])}")
    print(f"Hybrid Params:    {count_parameters(models['Hybrid'])}")
    print(f"Fused Params:     {count_parameters(models['Fused'])}")
    print(f"Pure QCNN Params: {count_parameters(models['Pure QCNN'])}")
    
    print("\nTraining for {} steps...".format(steps))
    print("{:<6} | {:<10} | {:<10} | {:<10} | {:<10}".format(
        "Step", "Classical", "Hybrid", "Fused", "Pure QCNN"
    ))
    print("-" * 65)
    
    for i in range(steps + 1):
        # Synthetic Data
        x = torch.randn(BSZ, C, H, W).to(device)
        emb = torch.randn(BSZ, EMB).to(device)
        target = torch.tanh(x) * 0.5 # Dummy target
        
        log_line = f"{i:<6}"
        
        for name, model in models.items():
            opts[name].zero_grad()
            out = model(x, emb)
            loss = nn.MSELoss()(out, target)
            loss.backward()
            opts[name].step()
            
            losses[name].append(loss.item())
            
            if i % 25 == 0:
                log_line += f" | {loss.item():.4f}    "
        
        if i % 25 == 0:
            print(log_line)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu' # Force CPU if needed
    benchmark_qcnn_variants(steps=150, device=device)
