import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import time
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training.quantum_transformer import QuantumAttentionPatch
    print("Successfully imported QuantumAttentionPatch")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ClassicalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: [B, S, D]
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        return self.proj_out(attn_out)

# --- Data Generators ---

def generate_image_data(batch_size, height, width, dim, device):
    """
    Generates synthetic 2D image data with spatial correlations (stripes/grids).
    Flattened to [B, H*W, D].
    Task: Denoising (Input = Signal + Noise, Target = Signal)
    """
    # Create spatial grid
    x = torch.linspace(0, 4*math.pi, width, device=device)
    y = torch.linspace(0, 4*math.pi, height, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Signal: sin(x) * cos(y) pattern, repeated across dim
    signal_base = torch.sin(grid_x) * torch.cos(grid_y) # [H, W]
    signal = signal_base.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, dim) # [B, H, W, D]
    
    # Add random phase shifts per batch and channel to make it harder
    phase = torch.randn(batch_size, 1, 1, dim, device=device)
    signal = torch.sin(grid_x.unsqueeze(0).unsqueeze(-1) + phase) * torch.cos(grid_y.unsqueeze(0).unsqueeze(-1) + phase)
    
    # Flatten
    signal_flat = signal.reshape(batch_size, height * width, dim)
    
    # Noise
    noise = torch.randn_like(signal_flat) * 0.5
    
    inputs = signal_flat + noise
    targets = signal_flat
    
    return inputs, targets

def generate_sequence_data(batch_size, seq_len, dim, device):
    """
    Generates synthetic 1D sequence data with temporal correlations.
    Shape: [B, L, D].
    Task: Denoising / Reconstruction of smooth curves.
    """
    t = torch.linspace(0, 8*math.pi, seq_len, device=device) # [L]
    
    # Signal: Sum of sines with different frequencies
    signal = torch.zeros(batch_size, seq_len, dim, device=device)
    
    for i in range(3): # Superimpose 3 frequencies
        freq = torch.randn(batch_size, 1, dim, device=device) * 2.0 + 1.0
        phase = torch.randn(batch_size, 1, dim, device=device) * 2 * math.pi
        signal += torch.sin(t.view(1, seq_len, 1) * freq + phase)
        
    # Normalize
    signal = signal / 3.0
    
    # Noise
    noise = torch.randn_like(signal) * 0.3
    
    inputs = signal + noise
    targets = signal
    
    return inputs, targets

# --- Training Loop ---

def train_task(model, inputs, targets, steps=200, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    criterion = nn.MSELoss()
    losses = []
    
    start_time = time.time()
    
    best_loss = float('inf')
    
    for i in range(steps):
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        curr_loss = loss.item()
        losses.append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            
        if (i+1) % 100 == 0:
            print(f"    Step {i+1}/{steps}: Loss = {curr_loss:.6f} (Best: {best_loss:.6f})")
            
    end_time = time.time()
    return best_loss, end_time - start_time

# --- Main Benchmark ---

def run_benchmarks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Benchmarks on {device}")
    
    # Configuration
    B = 8
    D = 64
    H, W = 16, 16 # Image Size
    L = 256       # Sequence Length (Same as H*W for comparison)
    STEPS = 600
    
    # Models
    print("\n--- Initializing Models ---")
    classical_img = ClassicalAttention(dim=D).to(device)
    # Use patch_size=1 to match resolution (fine-grained feature extraction)
    # n_qubits=6 (2^6=64) matches dim=64
    # Increased q_depth to 4 and lora_rank to 16 for better expressivity
    quantum_img = QuantumAttentionPatch(dim=D, patch_size=1, n_qubits=6, q_depth=4, lora_rank=16).to(device)
    
    classical_seq = ClassicalAttention(dim=D).to(device)
    quantum_seq = QuantumAttentionPatch(dim=D, patch_size=1, n_qubits=6, q_depth=4, lora_rank=16).to(device)
    
    c_params = count_parameters(classical_img)
    q_params = count_parameters(quantum_img)
    
    print(f"Classical Params: {c_params}")
    print(f"Quantum Params:   {q_params}")
    print(f"Reduction Ratio:  {1.0 - q_params/c_params:.1%}")
    
    # --- Task 1: Image Feature Extraction (Denoising) ---
    print("\n=== Task 1: Image Pattern Denoising (2D Spatial) ===")
    inputs, targets = generate_image_data(B, H, W, D, device)
    
    print("[1] Classical Attention (Image)...")
    c_loss, c_time = train_task(classical_img, inputs, targets, steps=STEPS)
    print(f"-> Final Loss: {c_loss:.6f}, Time: {c_time:.2f}s")
    
    print("[2] Quantum Attention (Image)...")
    q_loss, q_time = train_task(quantum_img, inputs, targets, steps=STEPS)
    print(f"-> Final Loss: {q_loss:.6f}, Time: {q_time:.2f}s")
    
    # --- Task 2: Sequence Feature Extraction (Denoising) ---
    print("\n=== Task 2: Sequence Reconstruction (1D Temporal) ===")
    inputs, targets = generate_sequence_data(B, L, D, device)
    
    print("[1] Classical Attention (Seq)...")
    c_loss_seq, c_time_seq = train_task(classical_seq, inputs, targets, steps=STEPS)
    print(f"-> Final Loss: {c_loss_seq:.6f}, Time: {c_time_seq:.2f}s")
    
    print("[2] Quantum Attention (Seq)...")
    q_loss_seq, q_time_seq = train_task(quantum_seq, inputs, targets, steps=STEPS)
    print(f"-> Final Loss: {q_loss_seq:.6f}, Time: {q_time_seq:.2f}s")
    
    # --- Summary ---
    print("\n=== Benchmark Summary ===")
    print(f"Image Task Improvement:    {(c_loss - q_loss)/c_loss:.1%} reduction in MSE")
    print(f"Sequence Task Improvement: {(c_loss_seq - q_loss_seq)/c_loss_seq:.1%} reduction in MSE")
    print(f"Parameter Efficiency:      Quantum model uses {q_params} params vs {c_params} params")

if __name__ == "__main__":
    run_benchmarks()
