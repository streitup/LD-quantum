
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys

# Import from training code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/training")

from quantum_transformer import (
    QuantumFrontEndQCNN, 
    QuantumFrontEndQCNNState, 
    QuantumAttentionAngle, 
    QuantumAttentionHybrid,
    QuantumAttentionState
)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Algorithm 1: Classical Baseline ---
class ClassicalConvPart(nn.Module):
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

class ClassicalBaseline(nn.Module):
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

# --- Algorithm 2: Q-Hybrid-baseline (MLP + Silu) ---
class QHybridBaseline(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        # Classical Convolution (Matching proof.py Hybrid)
        self.fe = ClassicalConvPart(channels, emb_dim)
        
        # MLP + Silu
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 64) # Project to 64 for Q-Attn
        )
        # Quantum Attention
        self.qattn = QuantumAttentionHybrid(input_dim=64, n_heads=4)
        self.proj_out = nn.Linear(64, channels)

    def forward(self, x, emb):
        # Conv
        x = self.fe(x, emb) # [B, C, H, W]
        b, c, h, w = x.shape
        
        # Reshape & MLP
        x_flat = x.flatten(2).transpose(1, 2) # [B, L, C]
        x_in = self.mlp(x_flat) # [B, L, 64]
        
        # Q-Attn
        x_out = self.qattn(x_in) # [B, L, 64]
        
        # Out Proj
        x_out = self.proj_out(x_out)
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out

# --- Algorithm 3: Q-Hybrid-Lite (Linear) ---
class QHybridLite(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        # Classical Convolution (Matching proof.py Hybrid)
        self.fe = ClassicalConvPart(channels, emb_dim)
        
        # Linear Only
        self.proj = nn.Linear(channels, 64)
        
        # Quantum Attention
        self.qattn = QuantumAttentionHybrid(input_dim=64, n_heads=4)
        self.proj_out = nn.Linear(64, channels)

    def forward(self, x, emb):
        # Conv
        x = self.fe(x, emb)
        b, c, h, w = x.shape
        
        # Reshape & Linear
        x_flat = x.flatten(2).transpose(1, 2)
        x_in = self.proj(x_flat)
        
        # Q-Attn
        x_out = self.qattn(x_in)
        
        # Out Proj
        x_out = self.proj_out(x_out)
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out

# --- Algorithm 4: Q-Pure-Fused (No Intermediate Measurement) ---
class QPureFused(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        # Shared params logic is inside QuantumFrontEndQCNNState?
        # The doc says "Share rotation parameters".
        # We'll use QuantumFrontEndQCNNState which returns state.
        self.n_groups = 4 # Q-Attn splits to groups usually? 
        # Actually QPureFused in fused_test.md implies connecting QCNN state to QAttn.
        # benchmark_fused_proof.py uses n_groups=4.
        # But QHybrid uses n_groups=1 (implied by default).
        # Let's stick to n_groups=1 for fair comparison if possible, or match architecture.
        # QuantumAttentionState expects [B, L, 2^N].
        # QuantumFrontEndQCNNState returns state.
        
        self.fe = QuantumFrontEndQCNNState(
            channels=channels,
            style_dim=emb_dim,
            n_layers=8, 
            n_groups=1,
            encoding_type='amplitude',
            n_qubits_ancilla=0,
            reupload_data=True,
            stride=1
        )
        
        # Quantum Attention on State
        self.qattn = QuantumAttentionState(
            N_QUBITS=6, # 2^6 = 64 state dim
            qk_dim=16,
            n_heads=4,
            force_fp32_attention=False
        )
        
        # Output projection from Q-Attn state/output?
        # QuantumAttentionState returns classical values after measurement?
        # Let's check QuantumAttentionState.
        # Assuming it returns [B, L, 64] (measured).
        self.proj_out = nn.Linear(64, channels)

    def forward(self, x, emb):
        b, c, h, w = x.shape
        # QCNN State
        state = self.fe(x, emb) # [B, L, 2^N] (assuming N=6 matches C=64?)
        # Wait, C=128. QCNN state dim depends on qubits.
        # QuantumFrontEndQCNNState with C=128, n_groups=1.
        # We need to check output dim.
        
        # Q-Attn
        x_out = self.qattn(state)
        
        x_out = self.proj_out(x_out)
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out

def run_experiment():
    print("=== Fused Metrics Experiment (100-shot-obama Simulation) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    C = 128
    EMB = 128
    H, W = 16, 16
    BSZ = 8
    
    # Models
    models = {
        'Classical baseline': ClassicalBaseline(C, EMB).to(device),
        'Q-Hybrid-baseline': QHybridBaseline(C, EMB).to(device),
        'Q-Hybrid-Lite': QHybridLite(C, EMB).to(device),
        'Q-Pure-Fused': QPureFused(C, EMB).to(device)
    }
    
    # Params
    print("\n[Parameter Count]")
    for name, model in models.items():
        print(f"{name:<20}: {count_parameters(model)}")
        
    # Data (Simulated 100-shot)
    # Simulate Overfitting/Reconstruction Task (Input = Noise, Target = Image)
    # This matches benchmark_fused_proof.py setup which tests memorization capacity.
    set_seed(42)
    x = torch.randn(BSZ, C, H, W).to(device) # Random Noise Input
    target = torch.randn(BSZ, C, H, W).to(device) # "Original Image" (Random Features)
    emb = torch.randn(BSZ, EMB).to(device)
    
    # Training
    steps = 150
    criterion = nn.MSELoss()
    optimizers = {name: optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
    
    results = {name: [] for name in models}
    
    print(f"\n[Training {steps} epochs...]")
    print(f"{'Epoch':<5} | {'Classical':<10} | {'Hybrid-Base':<12} | {'Hybrid-Lite':<12} | {'Pure-Fused':<10}")
    
    for i in range(1, steps+1):
        loss_vals = {}
        for name, model in models.items():
            opt = optimizers[name]
            opt.zero_grad()
            out = model(x, emb)
            loss = criterion(out, target)
            loss.backward()
            opt.step()
            results[name].append(loss.item())
            loss_vals[name] = loss.item()
            
        if i % 10 == 0:
            print(f"{i:<5} | {loss_vals['Classical baseline']:<10.4f} | {loss_vals['Q-Hybrid-baseline']:<12.4f} | {loss_vals['Q-Hybrid-Lite']:<12.4f} | {loss_vals['Q-Pure-Fused']:<10.4f}")
            
    print("\n[Final Results]")
    print(f"{'Algorithm':<20} | {'Params':<10} | {'Final Loss':<10}")
    print("-" * 45)
    for name in models:
        final_loss = results[name][-1]
        params = count_parameters(models[name])
        print(f"{name:<20} | {params:<10} | {final_loss:.6f}")

if __name__ == "__main__":
    run_experiment()
