
import torch
import torch.nn as nn
import time
import numpy as np
import random
from training.quantum_transformer import QuantumFrontEndQCNN

def run_structured_comparison(device='cpu', n_steps=100):
    print(f"\n=== Structured Task Comparison (Noise vs Function) on {device} ===")
    
    # Configuration
    B = 4
    C = 128
    H, W = 16, 16
    style_dim = 512
    n_groups = 8 
    
    # 1. Classic Block
    class ClassicBlock(nn.Module):
        def __init__(self, channels, style_dim):
            super().__init__()
            self.norm0 = nn.GroupNorm(32, channels)
            self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
            self.affine = nn.Linear(style_dim, channels * 2)
            self.norm1 = nn.GroupNorm(32, channels)
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.act = nn.SiLU()
            
        def forward(self, x, emb):
            x = self.conv0(self.act(self.norm0(x)))
            params = self.affine(self.act(emb))
            scale, shift = params.chunk(2, dim=1)
            x = self.norm1(x)
            x = x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
            x = self.act(x)
            return self.conv1(x)

    # 2. Quantum Block (2 Qubits Amplitude)
    class QuantumBlock(nn.Module):
        def __init__(self, channels, style_dim, groups):
            super().__init__()
            self.norm0 = nn.GroupNorm(32, channels)
            self.norm1 = nn.GroupNorm(32, channels)
            self.act = nn.SiLU()
            
            self.qcnn = QuantumFrontEndQCNN(
                channels=channels,
                style_dim=style_dim,
                n_qubits_data=2, # Using the efficient 2Q setup
                n_qubits_ancilla=2,
                n_layers=4,
                n_groups=groups,
                reupload_data=True,
                stride=1,
                encoding_type='amplitude'
            )
            
        def forward(self, x, emb):
            x = self.norm0(x)
            x = self.norm1(x)
            return self.qcnn(x, self.act(emb))

    # Define Models
    models = {
        "Classic": ClassicBlock(C, style_dim).to(device),
        "Quantum-2Q": QuantumBlock(C, style_dim, n_groups).to(device)
    }
    
    # Data Setup
    x = torch.randn(B, C, H, W).to(device)
    emb = torch.randn(B, style_dim).to(device)
    
    # --- Task 1: Random Noise (Hard to learn, favors capacity) ---
    print("\n--- Task 1: Fitting Random Noise (Memorization) ---")
    target_noise = torch.randn(B, C, H, W).to(device)
    
    for name, model in models.items():
        # Reset parameters to ensure fairness (optional, but good practice)
        # Here we just re-instantiate or deepcopy, but for simplicity we rely on fresh init in real usage.
        # Actually, let's re-init optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        losses = []
        start = time.time()
        for i in range(50): # Short run
            optimizer.zero_grad()
            out = model(x, emb)
            loss = criterion(out, target_noise)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"{name:<10} | Initial: {losses[0]:.4f} | Final: {losses[-1]:.4f} | Time: {(time.time()-start)*1000/50:.1f}ms/step")

    # --- Task 2: Structured Function (Favors Inductive Bias) ---
    print("\n--- Task 2: Fitting Structured Function (Sin/Cos Modulation) ---")
    # Target = sin(x) + cos(style_modulation)
    # Use the same x and emb
    style_proj = nn.Linear(style_dim, C).to(device) # Random projection
    with torch.no_grad():
        s = style_proj(emb).unsqueeze(-1).unsqueeze(-1)
        # Structured Target:
        target_struct = torch.sin(x) * 0.5 + torch.cos(s) * 0.5
    
    # Re-initialize models to avoid transfer learning
    models = {
        "Classic": ClassicBlock(C, style_dim).to(device),
        "Quantum-2Q": QuantumBlock(C, style_dim, n_groups).to(device)
    }

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        losses = []
        start = time.time()
        for i in range(50):
            optimizer.zero_grad()
            out = model(x, emb)
            loss = criterion(out, target_struct)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        print(f"{name:<10} | Initial: {losses[0]:.4f} | Final: {losses[-1]:.4f} | Time: {(time.time()-start)*1000/50:.1f}ms/step")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_structured_comparison(device)
