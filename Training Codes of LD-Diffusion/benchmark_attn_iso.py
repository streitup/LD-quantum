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

from training.quantum_transformer import QuantumAttention64, QuantumAttentionAngle, QuantumAttentionLight, QuantumAttentionDeep, QuantumAttentionHybrid, QuantumAttentionHybridLite, QuantumFrontEndQCNNState, QuantumAttentionState

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Models for Attention Isolation Benchmark ---

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
        # Add emb
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
        # Conv Part
        x = self.conv_part(x, emb)
        
        # Attn Part
        resid = x
        x = self.norm2(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, L, C]
        
        x_out, _ = self.attn(x_flat, x_flat, x_flat)
        
        x = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x + resid

class QuantumAttnBlock(nn.Module):
    """Classical Conv + Quantum Attention (Amplitude Encoding)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv_part = ClassicalConvPart(channels, emb_dim)
        
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        
        # Quantum Attention expects 64 channels
        self.proj_in = nn.Linear(channels, 64) if channels != 64 else nn.Identity()
        self.qattn = QuantumAttention64(n_heads=4) # Using optimized 4-head version
        self.proj_out = nn.Linear(64, channels) if channels != 64 else nn.Identity()
        
    def forward(self, x, emb):
        # Conv Part
        x = self.conv_part(x, emb)
        
        # Attn Part
        resid = x
        x = self.norm2(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, L, C]
        
        # Project if needed (QuantumAttn works on 64 dim)
        x_in = self.proj_in(x_flat)
        
        x_out = self.qattn(x_in)
        
        x_out = self.proj_out(x_out)
        
        x = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x + resid

class QuantumAttnAngleBlock(nn.Module):
    """Classical Conv + Quantum Attention (Angle Encoding - Tanh)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv_part = ClassicalConvPart(channels, emb_dim)
        
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        
        # Quantum Attention expects 64 channels
        self.proj_in = nn.Linear(channels, 64) if channels != 64 else nn.Identity()
        self.qattn = QuantumAttentionAngle(n_heads=4) # Angle Encoding Version
        self.proj_out = nn.Linear(64, channels) if channels != 64 else nn.Identity()
        
    def forward(self, x, emb):
        # Conv Part
        x = self.conv_part(x, emb)
        
        # Attn Part
        resid = x
        x = self.norm2(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, L, C]
        
        # Project if needed (QuantumAttn works on 64 dim)
        x_in = self.proj_in(x_flat)
        
        x_out = self.qattn(x_in)
        
        x_out = self.proj_out(x_out)
        
        x = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x + resid

class QuantumAttnLightBlock(nn.Module):
    """Classical Conv + Quantum Attention (Lightweight)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv_part = ClassicalConvPart(channels, emb_dim)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.proj_in = nn.Linear(channels, 64) if channels != 64 else nn.Identity()
        self.qattn = QuantumAttentionLight(n_heads=4)
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

class QuantumAttnDeepBlock(nn.Module):
    """Classical Conv + Quantum Attention (Deep)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv_part = ClassicalConvPart(channels, emb_dim)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.proj_in = nn.Linear(channels, 64) if channels != 64 else nn.Identity()
        self.qattn = QuantumAttentionDeep(n_heads=4)
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

class QuantumAttnHybridLiteBlock(nn.Module):
    """Classical Conv + Quantum Attention (Hybrid Lite)"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv_part = ClassicalConvPart(channels, emb_dim)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.proj_in = nn.Linear(channels, 64) if channels != 64 else nn.Identity()
        self.qattn = QuantumAttentionHybridLite(n_heads=4)
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

class QuantumFusedBlock(nn.Module):
    """
    Quantum QCNN (State) + Quantum Attention (State)
    Replaces Classical Conv with Quantum Frontend to enable Fused State Transfer.
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4 # Option 1: Multi-Group State Alignment
        self.state_dim = 64 # 2^6
        
        # Use QCNNState instead of ClassicalConvPart
        self.fe = QuantumFrontEndQCNNState(
            channels=channels,
            style_dim=emb_dim,
            n_layers=1, 
            n_groups=self.n_groups, # Increased capacity
            encoding_type='amplitude',
            n_qubits_ancilla=0,
            reupload_data=True,
            stride=1
        )
        
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        
        # Attention Consumes State
        # We will run Attention in parallel for each group (Option 1 & 3 logic)
        self.qattn = QuantumAttentionState(
            N_QUBITS=6,
            qk_dim=16,
            n_heads=4,
            force_fp32_attention=False
        )
        
        # Output dim = n_groups * inner_dim_of_attn (which is usually 64)
        # But since we removed Attention, we project directly from State Dimension
        # State Dim = n_groups * 2^N (where N=6, so 64)
        self.proj_out = nn.Linear(self.n_groups * self.state_dim, channels)

    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        
        # FE returns State [B, L, groups * 2^N]
        # This is the "Fused State" representing the features
        state = self.fe(x, emb) 
        
        # === Ablation: No Quantum Attention ===
        # We directly use the Quantum State as features
        # Flatten state: [B, L, G*D] -> [B, L, G*D] (Already flattened by FE usually, let's check)
        # FE returns [B, L, -1] which is [B, L, n_groups * 2^N]
        
        # Convert Complex State to Real Features
        # Strategy: Use Probabilities (abs^2) to simulate measurement in computational basis
        # This keeps the "State Generation" part but reads it out classically.
        features = (state.abs() ** 2).float()
        
        x_out = self.proj_out(features) # [B, L, C]
        
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        
        return x_out + resid

def benchmark_attn_isolation(steps=200, device='cpu'):
    print("\n=== Benchmark: Attention Encoding Comparison (Classical Conv + X) ===")
    print(f"Comparing: [Classical], [Quantum Amplitude], [Quantum Angle (Tanh)], [Light], [Deep], [Hybrid], [Hybrid-Lite], [Fused (Q-Front)]")
    print(f"Resolution: 128x16x16")
    set_seed(42)
    
    C = 128
    EMB = 128
    H, W = 16, 16 # Benchmark resolution 128*16*16
    BSZ = 8
    
    # Models
    models = {
        'Classical': ClassicalAttnBlock(C, EMB).to(device),
        'Q-Amp': QuantumAttnBlock(C, EMB).to(device),
        'Q-Ang': QuantumAttnAngleBlock(C, EMB).to(device),
        'Q-Light': QuantumAttnLightBlock(C, EMB).to(device),
        'Q-Deep': QuantumAttnDeepBlock(C, EMB).to(device),
        'Q-Hybrid': QuantumAttnHybridBlock(C, EMB).to(device),
        'Q-Hybrid-Lite': QuantumAttnHybridLiteBlock(C, EMB).to(device),
        'Fused': QuantumFusedBlock(C, EMB).to(device),
    }

    for name, model in models.items():
        print(f"{name} Params: {count_parameters(model)}")
    
    # Data
    x = torch.randn(BSZ, C, H, W).to(device)
    emb = torch.randn(BSZ, EMB).to(device)
    target = torch.randn(BSZ, C, H, W).to(device) # Regression target
    
    # Optimizers
    opts = {name: optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
    losses = {name: [] for name in models}
    
    criterion = nn.MSELoss()
    
    print("\nTraining...")
    for i in range(steps):
        log_items = []
        for name, model in models.items():
            opts[name].zero_grad()
            out = model(x, emb)
            loss = criterion(out, target)
            loss.backward()
            opts[name].step()
            losses[name].append(loss.item())
            
        if i % 20 == 0 or i == steps - 1:
            log_str = f"Step {i}: " + ", ".join([f"{n}={losses[n][-1]:.5f}" for n in models])
            print(log_str)
            
    print("\nFinal Results:")
    for name in models:
        print(f"{name}: {losses[name][-1]:.5f}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    benchmark_attn_isolation(steps=30, device=device)
