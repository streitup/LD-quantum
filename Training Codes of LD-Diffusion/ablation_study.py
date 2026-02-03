
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from training.quantum_transformer import QuantumFrontEndQCNN

def run_ablation(device='cpu', n_steps=100):
    print(f"\n=== QCNN Qubit Ablation Study (2->8 Qubits) on {device} ===")
    print(f"Training for {n_steps} steps per model...")
    
    # Configuration
    B = 4
    C = 128      # Channels
    H, W = 16, 16 # Spatial Dim
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

    # 2. Quantum Block Wrapper
    class QuantumBlock(nn.Module):
        def __init__(self, channels, style_dim, groups, n_qubits_data):
            super().__init__()
            self.norm0 = nn.GroupNorm(32, channels)
            self.norm1 = nn.GroupNorm(32, channels)
            self.act = nn.SiLU()
            
            self.qcnn = QuantumFrontEndQCNN(
                channels=channels,
                style_dim=style_dim,
                n_qubits_data=n_qubits_data,
                n_qubits_ancilla=2,
                n_layers=4,
                n_groups=groups,
                reupload_data=True,
                stride=1,
                encoding_type='amplitude' # Fixed to Amplitude as per previous success
            )
            
        def forward(self, x, emb):
            x = self.norm0(x)
            x = self.norm1(x)
            return self.qcnn(x, self.act(emb))

    # Define Models
    models = {
        "Classic Baseline": ClassicBlock(C, style_dim).to(device),
    }
    
    # Add Quantum Models with varying qubits
    qubit_counts = [2, 4, 6, 7, 8]
    for q in qubit_counts:
        models[f"Quantum-{q}Q"] = QuantumBlock(C, style_dim, n_groups, n_qubits_data=q).to(device)
    
    # Data
    x = torch.randn(B, C, H, W).to(device)
    emb = torch.randn(B, style_dim).to(device)
    target = torch.randn(B, C, H, W).to(device)
    criterion = nn.MSELoss()
    
    # Storage for results
    history = {name: [] for name in models.keys()}
    times = {name: [] for name in models.keys()}
    
    # Training Loop
    for name, model in models.items():
        print(f"\nTraining {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Warmup
        try:
            with torch.no_grad():
                _ = model(x, emb)
        except Exception as e:
            print(f"Skipping {name} due to init error: {e}")
            continue

        start_total = time.time()
        
        for step in range(n_steps):
            step_start = time.time()
            optimizer.zero_grad()
            
            try:
                out = model(x, emb)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                
                # Record
                loss_val = loss.item()
                history[name].append(loss_val)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                step_time = (time.time() - step_start) * 1000 # ms
                times[name].append(step_time)
                
                if step % 10 == 0 or step == n_steps - 1:
                    print(f"  Step {step:3d}: Loss={loss_val:.4f} (Time: {step_time:.1f}ms)")
                    
            except Exception as e:
                print(f"  Step {step}: Failed ({e})")
                break
        
        total_time = time.time() - start_total
        avg_time = sum(times[name]) / len(times[name]) if times[name] else 0
        print(f"  > Avg Step Time: {avg_time:.2f} ms")
        print(f"  > Final Loss: {history[name][-1]:.4f}")

    # Summary
    print("\n=== Ablation Summary ===")
    print(f"{'Model':<20} | {'Final Loss':<10} | {'Avg Time (ms)':<15} | {'Min Loss':<10}")
    print("-" * 65)
    for name in models.keys():
        if history[name]:
            final_loss = history[name][-1]
            min_loss = min(history[name])
            avg_t = sum(times[name]) / len(times[name])
            print(f"{name:<20} | {final_loss:.4f}     | {avg_t:.2f}          | {min_loss:.4f}")
        else:
            print(f"{name:<20} | Failed")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_ablation(device, n_steps=50)
