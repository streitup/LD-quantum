
import torch
import torch.nn as nn
import time
import numpy as np
from training.quantum_transformer import QuantumFrontEndQCNN

def benchmark_speed(device='cpu'):
    print(f"\n=== QCNN Speed Benchmark (Grouped vs Classic vs Amplitude) on {device} ===")
    
    # Configuration to match real usage
    B = 4
    C = 128      # Channels
    H, W = 16, 16 # Spatial Dim
    style_dim = 512
    n_groups = 8 # channels_per_group = 16
    
    # 1. Classic Block (Conv-Affine-Conv)
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

    # 2. Configurable Quantum Block
    class QuantumBlock(nn.Module):
        def __init__(self, channels, style_dim, groups, encoding_type='tanh', n_qubits_data=4, freeze_circuit=False):
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
                encoding_type=encoding_type,
                freeze_qcnn=freeze_circuit
            )
            
        def forward(self, x, emb):
            x = self.norm0(x)
            x = self.norm1(x)
            return self.qcnn(x, self.act(emb))

    # 3. Ablation: Pure Linear Bottleneck (No Quantum)
    # Mimics the dimension reduction of 2-qubit Amplitude Encoding
    # 144 (Patch) -> 4 (Latent) -> 16 (Channel)
    class LinearBottleneckBlock(nn.Module):
        def __init__(self, channels, style_dim, groups, bottleneck_dim=4):
            super().__init__()
            self.norm0 = nn.GroupNorm(32, channels)
            self.norm1 = nn.GroupNorm(32, channels)
            self.act = nn.SiLU()
            
            self.channels = channels
            self.groups = groups
            self.kernel_size = 3
            self.stride = 1
            self.padding = 1
            self.patch_dim = (channels // groups) * 3 * 3
            
            self.unfold = nn.Unfold(kernel_size=3, padding=1, stride=1)
            
            # Mimic QuantumFrontEndQCNN structure
            self.down_proj = nn.Linear(self.patch_dim, bottleneck_dim)
            self.up_proj = nn.Linear(bottleneck_dim, channels // groups)
            
            # Style modulation (mimic fusion)
            self.style_proj = nn.Linear(style_dim, bottleneck_dim)

        def forward(self, x, emb):
            B, C, H, W = x.shape
            x_in = self.norm1(self.norm0(x))
            style = self.act(emb)
            
            # Unfold
            patches = self.unfold(x_in) # [B, C*9, L]
            L = patches.shape[-1]
            patches = patches.transpose(1, 2).reshape(B*L, self.groups, -1) # [B*L, G, P]
            
            # Linear Down (Encoder)
            latents = self.down_proj(patches) # [B*L, G, 4]
            
            # Simple Style Modulation (Add)
            s = self.style_proj(style).unsqueeze(1).unsqueeze(1) # [B, 1, 1, 4]
            # Expand style to match latents [B*L, G, 4] is tricky without reshape
            # Let's simplify: style [B, 4] -> [B, 1, 1, 4]
            # We need to broadcast to [B*L, G, 4]. 
            # latents is shaped [B*L, G, 4]. We can reshape latents to [B, L, G, 4]
            latents = latents.view(B, L, self.groups, -1)
            latents = latents + s # Fusion
            
            # Non-linearity (mimic Quantum Measurement |x|^2 or Tanh)
            latents = torch.sigmoid(latents) # Sigmoid is closer to [0,1] probs
            
            # Linear Up (Decoder/Measurement)
            out = self.up_proj(latents) # [B, L, G, 16]
            
            # Fold/Reshape back
            out = out.view(B, L, -1).transpose(1, 2) # [B, C, L]
            H_out = int((H + 2 * self.padding - 3) / self.stride + 1)
            out = out.reshape(B, C, H_out, H_out)
            
            return out

    models = {
        "Classic": ClassicBlock(C, style_dim).to(device),
        "Quantum (Amp, 2Q)": QuantumBlock(C, style_dim, n_groups, 'amplitude', 2).to(device),
        "Quantum (Amp, 2Q, Frozen)": QuantumBlock(C, style_dim, n_groups, 'amplitude', 2, freeze_circuit=True).to(device),
        "Linear Only (Dim=4)": LinearBottleneckBlock(C, style_dim, n_groups, bottleneck_dim=4).to(device)
    }
    
    x = torch.randn(B, C, H, W).to(device)
    emb = torch.randn(B, style_dim).to(device)
    
    results = {}
    
    # Warmup
    print("Warming up...")
    for name, model in models.items():
        try:
            for _ in range(5):
                _ = model(x, emb)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error warming up {name}: {e}")
    
    print("\nRunning Benchmarks...")
    for name, model in models.items():
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        try:
            for _ in range(50):
                _ = model(x, emb)
            torch.cuda.synchronize() if device == 'cuda' else None
            end = time.time()
            avg = (end - start) / 50 * 1000
            results[name] = avg
            print(f"{name}: {avg:.2f} ms")
        except Exception as e:
            print(f"{name}: Failed ({e})")
            results[name] = float('inf')

    # Loss/Output Stats (Quick Check)
    print("\nOutput Statistics (Std Dev of output):")
    with torch.no_grad():
        for name, model in models.items():
            if results[name] != float('inf'):
                out = model(x, emb)
                print(f"{name}: Mean={out.mean().item():.4f}, Std={out.std().item():.4f}")

    # Backward Pass / Training Check
    print("\nTraining Step Benchmark (Forward + Backward):")
    criterion = nn.MSELoss()
    target = torch.randn(B, C, H, W).to(device)
    
    for name, model in models.items():
        if results[name] == float('inf'): continue
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        try:
            for i in range(10):
                optimizer.zero_grad()
                out = model(x, emb)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                if i == 0 or i == 9:
                    print(f"  {name} Iter {i}: Loss={loss.item():.4f}")
            
            torch.cuda.synchronize() if device == 'cuda' else None
            end = time.time()
            avg = (end - start) / 10 * 1000
            print(f"  {name} Avg Train Step: {avg:.2f} ms")
            
        except Exception as e:
            print(f"  {name}: Failed Backward ({e})")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmark_speed(device)
