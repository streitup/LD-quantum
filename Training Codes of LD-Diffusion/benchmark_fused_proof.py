import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os
import time

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.quantum_transformer import QuantumAttentionHybrid, QuantumFrontEndQCNNState, QuantumAttentionState

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Classical & Hybrid Baselines ---

class ClassicalConvPart(nn.Module):
    """Standard ResBlock with Conv + SiLU + Emb Injection"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, channels)
        
    def forward(self, x, emb):
        resid = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        emb_out = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        x = x + emb_out
        x = self.act(x)
        x = self.conv2(x)
        return x + resid

class ClassicalAttnBlock(nn.Module):
    """Classical Conv + Classical MultiheadAttention"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv_part = ClassicalConvPart(channels, emb_dim)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        
    def forward(self, x, emb):
        x = self.conv_part(x, emb)
        resid = x
        x = self.norm2(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_out, _ = self.attn(x_flat, x_flat, x_flat)
        x = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x + resid

class QuantumAttnHybridBlock(nn.Module):
    """Classical Conv + Quantum Attention (Hybrid)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv_part = ClassicalConvPart(channels, emb_dim)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.proj_in = nn.Linear(channels, 64) if channels != 64 else nn.Identity()
        self.qattn = QuantumAttentionHybrid(n_heads=4)
        self.proj_out = nn.Linear(64, channels) if channels != 64 else nn.Identity()
        
    def forward(self, x, emb):
        x = self.conv_part(x, emb)
        resid = x
        x = self.norm2(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_in = self.proj_in(x_flat)
        x_out = self.qattn(x_in)
        x_out = self.proj_out(x_out)
        x = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x + resid

# --- Fused Variants ---

class QuantumFusedBlock_Full(nn.Module):
    """Fused Architecture: QCNN (State) -> Quantum Attention (State)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        self.state_dim = 64
        
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
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.qattn = QuantumAttentionState(
            N_QUBITS=6, qk_dim=16, n_heads=4, force_fp32_attention=False
        )
        # Output dim = n_groups * inner_dim_of_attn (64)
        self.proj_out = nn.Linear(64 * self.n_groups, channels)
        
    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        # State: [B, L, G*D]
        state = self.fe(x, emb) 
        
        # Reshape for parallel attention: [B, L, G, D] -> [B*G, L, D]
        L = state.shape[1]
        state = state.reshape(b, L, self.n_groups, self.state_dim)
        state = state.permute(0, 2, 1, 3).reshape(b * self.n_groups, L, self.state_dim)
        
        # Attention on State
        x_out_q = self.qattn(state) # [B*G, L, 64]
        
        # Reshape back: [B*G, L, 64] -> [B, L, G*64]
        x_out_q = x_out_q.reshape(b, self.n_groups, L, -1).permute(0, 2, 1, 3).reshape(b, L, -1)
        
        x_out = self.proj_out(x_out_q)
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out + resid

class QuantumFusedBlock_NoAttn(nn.Module):
    """Fused Ablation: QCNN (State) -> Abs^2 -> Proj (No Attention)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        self.state_dim = 64
        
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
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.proj_out = nn.Linear(self.n_groups * self.state_dim, channels)

    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        
        state = self.fe(x, emb) # [B, L, G*D]
        
        # No Attention. Just measurement simulation (Abs^2)
        features = (state.abs() ** 2).float()
        
        x_out = self.proj_out(features)
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out + resid

class QuantumFusedBlock_Residual(nn.Module):
    """Fused + Classical Residual Branch"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        # 1. Full Fused Core
        self.n_groups = 4
        self.state_dim = 64
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
        self.qattn = QuantumAttentionState(
            N_QUBITS=6, qk_dim=16, n_heads=4, force_fp32_attention=False
        )
        self.proj_out = nn.Linear(64 * self.n_groups, channels)
        
        # 2. Classical Branch
        self.classical_branch = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        self.quantum_scale = nn.Parameter(torch.tensor(0.1)) 
        
    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        
        # Classical Path
        x_c = self.classical_branch(x)
        
        # Quantum Path
        state = self.fe(x, emb) 
        L = state.shape[1]
        state = state.reshape(b, L, self.n_groups, self.state_dim)
        state = state.permute(0, 2, 1, 3).reshape(b * self.n_groups, L, self.state_dim)
        x_q = self.qattn(state)
        x_q = x_q.reshape(b, self.n_groups, L, -1).permute(0, 2, 1, 3).reshape(b, L, -1)
        x_q = self.proj_out(x_q)
        x_q = x_q.transpose(1, 2).reshape(b, c, h, w)
        
        # Combine
        x_out = x_q * self.quantum_scale + x_c
        
        return x_out + resid

# --- Benchmark Runner ---

def benchmark_fused_proof(steps=150, device='cuda'):
    print("\n=== Benchmark: Proof of Fused Architecture Issues ===")
    print("Comparing 5 Variants to prove Fused Architecture flaws.")
    set_seed(42)
    
    C = 128
    EMB = 128
    H, W = 16, 16
    BSZ = 8
    
    models = {
        'Classical': ClassicalAttnBlock(C, EMB).to(device),
        'Hybrid': QuantumAttnHybridBlock(C, EMB).to(device),
        'Fused-Full': QuantumFusedBlock_Full(C, EMB).to(device),
        'Fused-NoAttn': QuantumFusedBlock_NoAttn(C, EMB).to(device),
        'Fused-Residual': QuantumFusedBlock_Residual(C, EMB).to(device),
    }
    
    for name, model in models.items():
        print(f"{name} Params: {count_parameters(model)}")
        
    # Dummy Data (Regression Task)
    x = torch.randn(BSZ, C, H, W).to(device)
    emb = torch.randn(BSZ, EMB).to(device)
    target = torch.randn(BSZ, C, H, W).to(device)
    
    opts = {name: optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
    criterion = nn.MSELoss()
    
    print(f"\nTraining for {steps} steps...")
    print(f"{'Step':<5} | {'Classical':<10} | {'Hybrid':<10} | {'Fused-Full':<10} | {'Fused-NoAttn':<12} | {'Fused-Res':<10}")
    print("-" * 75)
    
    for i in range(steps + 1):
        losses = {}
        for name, model in models.items():
            opts[name].zero_grad()
            out = model(x, emb)
            loss = criterion(out, target)
            loss.backward()
            opts[name].step()
            losses[name] = loss.item()
            
        if i % 25 == 0:
            print(f"{i:<5} | {losses['Classical']:<10.4f} | {losses['Hybrid']:<10.4f} | {losses['Fused-Full']:<10.4f} | {losses['Fused-NoAttn']:<12.4f} | {losses['Fused-Residual']:<10.4f}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    benchmark_fused_proof(steps=150, device=device)
