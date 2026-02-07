import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import random
import numpy as np

# Add path to find training modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

try:
    from training.quantum_transformer import (
        QuantumFrontEndQCNN, 
        QuantumFrontEndQCNNState,
        QuantumAttentionAngle,
        QuantumAttentionLight,
        QuantumAttentionDeep,
        QuantumAttentionHybrid,
        QuantumAttentionHybridLite,
        QuantumAttentionState
    )
except ImportError as e:
    print(f"Error importing quantum modules: {e}")
    # Fallback or exit
    sys.exit(1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Block Definitions ---

class ClassicalAttnBlock(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        
    def forward(self, x, emb):
        h = self.conv1(F.relu(self.norm1(x)))
        x = x + h
        b, c, h, w = x.shape
        x_in = x.flatten(2).transpose(1, 2)
        x_in = self.norm2(x).flatten(2).transpose(1, 2)
        x_out, _ = self.attn(x_in, x_in, x_in)
        x = x + x_out.transpose(1, 2).reshape(b, c, h, w)
        return x

import torch.nn.functional as F

class HybridBlockFull(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.qcnn = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim,
            n_qubits_data=6,
            n_qubits_ancilla=2,
            n_layers=4,
            n_groups=4, 
            stride=1,
            reupload_data=True,
            encoding_type='tanh',
            projection_type='mlp'
        )
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        
    def forward(self, x, emb):
        resid = x
        x = self.norm1(x)
        x = self.qcnn(x, emb)
        x = x + resid
        
        b, c, h, w = x.shape
        resid = x
        x_in = self.norm2(x).flatten(2).transpose(1, 2)
        x_out, _ = self.attn(x_in, x_in, x_in)
        x = x + x_out.transpose(1, 2).reshape(b, c, h, w)
        return x

class QuantumBlockFused(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        # Using n_groups=1 to match State dimension (2^6 = 64) with Attention
        self.qcnn = QuantumFrontEndQCNNState(
            channels=channels,
            style_dim=emb_dim,
            n_qubits_data=6,
            n_qubits_ancilla=0,
            n_layers=4,
            n_groups=1, # CHANGED: 1 group to output [B, L, 64]
            stride=1,
            reupload_data=False, # Must be False for 'amplitude' encoding/state transfer without data re-upload
            encoding_type='tanh', # QCNNState handles this
            projection_type='mlp'
        )
        self.qcnn.encoding_type = 'amplitude'
        
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.qattn = QuantumAttentionState(
            N_QUBITS=6,
            qk_dim=16,
            n_heads=4,
            attn_dropout=0.0
        )
        
        self.proj_out = nn.Linear(64, channels) if channels != 64 else nn.Identity()
        
    def forward(self, x, emb):
        resid = x
        x = self.norm1(x)
        # QCNNState returns [B, L, 2^N]
        state = self.qcnn(x, emb)
        
        # Attention Consumes State
        x_out = self.qattn(state)
        x_out = self.proj_out(x_out)
        
        b, c, h, w = resid.shape
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out + resid

class QuantumBlockGeneric(nn.Module):
    def __init__(self, channels, emb_dim, attn_cls, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.qcnn = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim,
            n_qubits_data=6,
            n_qubits_ancilla=2,
            n_layers=4,
            n_groups=4, 
            stride=1,
            reupload_data=True,
            encoding_type='tanh',
            projection_type='mlp'
        )
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.proj_in = nn.Linear(channels, 64)
        self.qattn = attn_cls(**attn_kwargs) # attn works on 64 dim
        self.proj_out = nn.Linear(64, channels)
        
    def forward(self, x, emb):
        resid = x
        x = self.norm1(x)
        x = self.qcnn(x, emb)
        x = x + resid
        
        resid = x
        x = self.norm2(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_in = self.proj_in(x_flat)
        x_out = self.qattn(x_in)
        x_out = self.proj_out(x_out)
        x = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x + resid

def benchmark_full_block_comparison(steps=50, device='cpu'):
    print("\n=== Benchmark: Full Block Comparison (Classical vs Hybrid vs Quantum) ===")
    print(f"Resolution: 128x16x16")
    set_seed(42)
    
    C = 128
    EMB = 128
    H, W = 16, 16
    BSZ = 4 # Reduced batch size for speed
    
    models = {
        'Classical': ClassicalAttnBlock(C, EMB).to(device),
        'Hybrid': HybridBlockFull(C, EMB).to(device),
        'Q-Angle': QuantumBlockGeneric(C, EMB, QuantumAttentionAngle).to(device),
        'Q-Light': QuantumBlockGeneric(C, EMB, QuantumAttentionLight).to(device),
        'Q-Hybrid-Lite': QuantumBlockGeneric(C, EMB, QuantumAttentionHybridLite).to(device),
        'Fused': QuantumBlockFused(C, EMB).to(device),
    }

    for name, model in models.items():
        print(f"{name} Params: {count_parameters(model)}")
    
    x = torch.randn(BSZ, C, H, W).to(device)
    emb = torch.randn(BSZ, EMB).to(device)
    target = torch.randn(BSZ, C, H, W).to(device)
    
    criterion = nn.MSELoss()
    opts = {name: optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
    losses = {name: [] for name in models}
    times = {name: 0.0 for name in models}
    
    print("\nTraining...")
    for i in range(steps):
        for name, model in models.items():
            opts[name].zero_grad()
            t0 = time.time()
            out = model(x, emb)
            loss = criterion(out, target)
            loss.backward()
            opts[name].step()
            t1 = time.time()
            losses[name].append(loss.item())
            times[name] += (t1 - t0)
            
        if i % 10 == 0:
            log_str = f"Step {i}: " + ", ".join([f"{n}={losses[n][-1]:.4f}" for n in models])
            print(log_str)
            
    print("\nFinal Results (Loss):")
    for name in models:
        print(f"{name}: {losses[name][-1]:.5f}")
        
    print("\nAverage Time per Step (s):")
    for name in models:
        print(f"{name}: {times[name]/steps:.4f}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Check if 'cuda' is actually usable (sometimes installed but no GPU)
    if device == 'cuda':
        try:
            torch.tensor([1.0]).cuda()
        except:
            device = 'cpu'
    print(f"Running on {device}")
    benchmark_full_block_comparison(steps=50, device=device)
