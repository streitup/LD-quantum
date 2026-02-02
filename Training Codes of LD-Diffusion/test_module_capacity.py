import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import random
from training.quantum_transformer import QuantumMLP, QuantumFrontEndQCNN

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Benchmark 1: Time Embedding Capacity ---
def benchmark_time_embedding(steps=200, device='cpu'):
    print("\n=== Benchmark 1: Time Embedding Capacity ===")
    set_seed(42)
    B = 32
    # 纯量子测试：维度固定为64，与量子态空间对齐 (2^6=64)
    in_dim = 64
    out_dim = 64
    
    # Random input and target (mimic learning a complex function)
    x = torch.randn(B, in_dim).to(device)
    # Target scaled to [0, 1] to fit Probability Readout range
    target = 0.5 * (torch.sin(x) + torch.cos(x*2)) + 0.5 
    
    # 1. Classical MLP (Teacher/Reference)
    classic_model = nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.SiLU(),
        nn.Linear(128, out_dim)
    ).to(device)
    opt_c = optim.Adam(classic_model.parameters(), lr=1e-3)
    
    # 2. Quantum MLP (Full Quantum)
    # Fair Params: n_qubits=2, n_groups=64, Depth=4
    # Total Params ~ 52k (33k Linear + ~20k Quantum/Other)
    quantum_model = QuantumMLP(in_dim, out_dim, n_qubits=2, q_depth=4,
                               encoding='amplitude', re_uploading=True,
                               output_mlp_ratio=0.0, n_groups=64, readout_mode='expectation').to(device)
    opt_q = optim.Adam(quantum_model.parameters(), lr=1e-2) 
    
    # 3. Quantum MLP (Linear Only / Depth=0)
    # Tests if proj_in is doing all the work.
    # Same structure but NO quantum variational layers.
    linear_only_model = QuantumMLP(in_dim, out_dim, n_qubits=2, q_depth=0,
                               encoding='amplitude', re_uploading=False, # No re-uploading since depth=0
                               output_mlp_ratio=0.0, n_groups=64, readout_mode='expectation').to(device)
    opt_l = optim.Adam(linear_only_model.parameters(), lr=1e-2)

    print(f"Classic Params: {count_parameters(classic_model)}")
    print(f"Quantum Params: {count_parameters(quantum_model)}")
    print(f"LinearOnly Params: {count_parameters(linear_only_model)}")
    
    print("\nTraining...")
    for i in range(steps):
        # Classic
        opt_c.zero_grad()
        loss_c = nn.MSELoss()(classic_model(x), target)
        loss_c.backward()
        opt_c.step()
        
        # Quantum
        opt_q.zero_grad()
        loss_q = nn.MSELoss()(quantum_model(x), target)
        loss_q.backward()
        opt_q.step()
        
        # Linear Only
        opt_l.zero_grad()
        loss_l = nn.MSELoss()(linear_only_model(x), target)
        loss_l.backward()
        opt_l.step()
        
        if i % 50 == 0 or i == steps-1:
            print(f"Step {i}: C={loss_c.item():.5f}, Q={loss_q.item():.5f}, L_Only={loss_l.item():.5f}")

    print(f"Final: C={loss_c.item():.5f}, Q={loss_q.item():.5f}, L_Only={loss_l.item():.5f}")

# --- Benchmark 2: Realistic Spatial Modulation (QCNN vs Classic) ---
class ClassicSpatialMod(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x, style):
        s = self.style_mlp(style).unsqueeze(-1).unsqueeze(-1)
        return self.conv(self.norm(x) * (1 + s))

def benchmark_spatial_modulation(steps=100, device='cpu'):
    print("\n=== Benchmark 2: Realistic Spatial Modulation (QCNN vs Classic) ===")
    set_seed(42)
    B = 4
    C = 128  # Realistic Dimension
    H, W = 16, 16 # Realistic Latent Resolution
    style_dim = 128
    
    x = torch.randn(B, C, H, W).to(device)
    style = torch.randn(B, style_dim).to(device)
    
    # Target: Complex modulation
    target_mod = torch.sigmoid(style[:, :C]).unsqueeze(-1).unsqueeze(-1)
    target = x * target_mod + torch.sin(x)
    
    # 1. Classic Student
    classic_model = ClassicSpatialMod(C, style_dim).to(device)
    opt_c = optim.Adam(classic_model.parameters(), lr=1e-3)
    
    # 2. Quantum QCNN (Realistic Config: No Bypass, Grouped)
    # C=128, n_groups=8 => 16 channels per group.
    # n_qubits_data=4 => 2^4 = 16 dimensions (Matched!)
    quantum_model = QuantumFrontEndQCNN(
        channels=C, 
        style_dim=style_dim,
        n_qubits_data=4, # 4 qubits -> 16 dim
        n_qubits_ancilla=2,
        n_layers=2, 
        n_groups=8, # 8 groups * 16 dim = 128 channels
        reupload_data=True,
        use_strong_bypass=False, # STRICTLY REMOVED as requested
        use_mlp_residual=False, # Removed to test pure quantum capacity
        encoding_type='tanh',
        stride=1
    ).to(device)
    opt_q = optim.Adam(quantum_model.parameters(), lr=1e-2) 
    
    print(f"Classic Params: {count_parameters(classic_model)}")
    print(f"Quantum Params: {count_parameters(quantum_model)}")
    
    print("\nTraining...")
    for i in range(steps):
        # Classic
        opt_c.zero_grad()
        loss_c = nn.MSELoss()(classic_model(x, style), target)
        loss_c.backward()
        opt_c.step()
        
        # Quantum
        opt_q.zero_grad()
        loss_q = nn.MSELoss()(quantum_model(x, style), target)
        loss_q.backward()
        opt_q.step()
        
        if i % 20 == 0 or i == steps-1:
            print(f"Step {i}: Loss C={loss_c.item():.5f}, Loss Q={loss_q.item():.5f}")
            
    print(f"Final Loss: C={loss_c.item():.5f}, Q={loss_q.item():.5f}")

# --- Benchmark 3: Full Hybrid Block (Classic Time/Affine + Quantum Spatial) ---
class ClassicBlock(nn.Module):
    def __init__(self, in_dim, channels):
        super().__init__()
        # Time Emb
        self.time_mlp = nn.Sequential(
            nn.Linear(in_dim, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels)
        )
        # Spatial
        self.block = ClassicSpatialMod(channels, channels) # style_dim = channels
        
    def forward(self, x, t_emb):
        style = self.time_mlp(t_emb)
        return self.block(x, style)

class HybridQuantumBlock(nn.Module):
    def __init__(self, in_dim, channels):
        super().__init__()
        # Classic Time Emb + Affine (As requested)
        self.time_mlp = nn.Sequential(
            nn.Linear(in_dim, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels)
        )
        # Quantum Spatial (No Bypass)
        self.qcnn = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=channels, # style comes from time_mlp
            n_qubits_data=4,
            n_qubits_ancilla=2,
            n_layers=2,
            n_groups=8,
            reupload_data=True,
            use_strong_bypass=False,
            use_mlp_residual=False,
            stride=1
        )
        
    def forward(self, x, t_emb):
        style = self.time_mlp(t_emb)
        return self.qcnn(x, style)

def benchmark_affine_modulation(steps=100, device='cpu'):
    print("\n=== Benchmark 3: Full Hybrid Block (Classic Time/Affine + Quantum Spatial) ===")
    set_seed(42)
    B = 4
    C = 128
    H, W = 16, 16
    time_dim = 128
    
    x = torch.randn(B, C, H, W).to(device)
    t_emb = torch.randn(B, time_dim).to(device) # Simulated Time Embedding
    
    # Target: Driven by time
    # complex non-linear interaction
    style_gt = torch.sin(t_emb)
    mod = style_gt[:, :C].unsqueeze(-1).unsqueeze(-1)
    target = x * mod + torch.cos(x * mod)
    
    # 1. Classic Block
    classic_model = ClassicBlock(time_dim, C).to(device)
    opt_c = optim.Adam(classic_model.parameters(), lr=1e-3)
    
    # 2. Hybrid Quantum Block
    quantum_model = HybridQuantumBlock(time_dim, C).to(device)
    opt_q = optim.Adam(quantum_model.parameters(), lr=1e-2)
    
    print(f"Classic Params: {count_parameters(classic_model)}")
    print(f"Quantum (Hybrid) Params: {count_parameters(quantum_model)}")
    
    print("\nTraining...")
    for i in range(steps):
        opt_c.zero_grad()
        loss_c = nn.MSELoss()(classic_model(x, t_emb), target)
        loss_c.backward()
        opt_c.step()
        
        opt_q.zero_grad()
        loss_q = nn.MSELoss()(quantum_model(x, t_emb), target)
        loss_q.backward()
        opt_q.step()
        
        if i % 20 == 0 or i == steps-1:
            print(f"Step {i}: Loss C={loss_c.item():.5f}, Loss Q={loss_q.item():.5f}")
    
    print(f"Final Loss: C={loss_c.item():.5f}, Q={loss_q.item():.5f}")

# --- Benchmark 5: Realistic U-Net Block Integration (The Sandwich Test) ---
class UNetBlockRef(nn.Module):
    """
    Reference Implementation of U-Net Block (Classic)
    Structure:
    1. Norm0 -> Silu -> Conv0
    2. Affine(emb) -> Scale/Shift
    3. Norm1 -> Scale/Shift -> Silu -> Dropout -> Conv1
    4. Skip Connection (Identity or Conv)
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm0 = nn.GroupNorm(32, channels)
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.affine = nn.Linear(emb_dim, channels * 2) # Scale + Shift
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.act = nn.SiLU()
        
    def forward(self, x, emb):
        orig = x
        # 1. First Block
        x = self.conv0(self.act(self.norm0(x)))
        
        # 2. Affine Injection
        params = self.affine(self.act(emb)) 
        scale, shift = params.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        
        # 3. Second Block (Modulated)
        x = self.norm1(x)
        x = x * (1 + scale) + shift
        x = self.act(x)
        x = self.conv1(x)
        
        # 4. Residual
        return x + orig

class UNetBlockQuantum(nn.Module):
    """
    Quantum Implementation of U-Net Block
    Structure:
    1. Norm0 -> Silu -> Conv0 (Classic)
    2. Norm1 (Crucial for stability)
    3. QuantumFrontEndQCNN (Replaces Modulate -> Conv1)
       - Input: Norm1(x)
       - Style: emb
       - Logic: RY(data + style)
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm0 = nn.GroupNorm(32, channels)
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        
        # Added Norm1 to stabilize input to QCNN (prevent tanh saturation)
        self.norm1 = nn.GroupNorm(32, channels)
        
        self.qcnn = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim, 
            n_qubits_data=4,
            n_qubits_ancilla=2,
            n_layers=2,
            n_groups=8,
            reupload_data=True,
            use_strong_bypass=False,
            use_mlp_residual=False,
            stride=1
        )
        
    def forward(self, x, emb):
        orig = x
        # 1. First Block
        x = self.conv0(self.act(self.norm0(x)))
        
        # 2. Quantum Block
        # Normalize first!
        x = self.norm1(x)
        x = self.qcnn(x, self.act(emb))
        
        # 4. Residual
        return x + orig

class UNetBlockSeparated(nn.Module):
    """
    Architecture 1: Separated Quantum Blocks
    QConv0 -> Measure -> Classic Affine -> QConv1 -> Measure
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        # QConv0 (Feature Extractor)
        self.qconv0 = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=1, # Dummy
            n_qubits_data=4,
            n_qubits_ancilla=2,
            n_layers=2,
            n_groups=8,
            stride=1,
            reupload_data=True,
            use_strong_bypass=False,
            use_mlp_residual=False
        )
        
        # Classic Middle (Norm -> Affine -> Act)
        self.norm_mid = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
        self.affine_proj = nn.Linear(emb_dim, channels * 2)
        
        # QConv1 (Feature Extractor)
        self.qconv1 = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=1, # Dummy
            n_qubits_data=4,
            n_qubits_ancilla=2,
            n_layers=2,
            n_groups=8,
            stride=1,
            reupload_data=True,
            use_strong_bypass=False,
            use_mlp_residual=False
        )
        
        self.norm0 = nn.GroupNorm(32, channels)

    def forward(self, x, emb):
        orig = x
        x = self.norm0(x)
        
        # 1. QConv0
        dummy_style = torch.zeros(x.shape[0], 1).to(x.device)
        x = self.qconv0(x, dummy_style)
        
        # 2. Classic Modulation
        x = self.norm_mid(x)
        
        style = self.affine_proj(self.act(emb))
        scale, shift = style.chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        x = self.act(x)
        
        # 3. QConv1
        x = self.qconv1(x, dummy_style)
        
        return x + orig

class UNetBlockIntegrated(nn.Module):
    """
    Architecture 2: Integrated Quantum Block
    Unitary(x) -> Modulate(emb) -> Unitary -> Measure
    Implemented via Deep QuantumFrontEndQCNN with Parameter Modulation
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.qcnn = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim, 
            n_qubits_data=4,
            n_qubits_ancilla=2,
            n_layers=4, # Double depth to match Separated capacity
            n_groups=8,
            stride=1,
            reupload_data=True,
            use_strong_bypass=False,
            use_mlp_residual=False
        )
        self.norm0 = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()

    def forward(self, x, emb):
        orig = x
        x = self.norm0(x)
        x = self.qcnn(x, self.act(emb))
        return x + orig

def benchmark_architectures(steps=200, device='cpu'):
    print("\n=== Benchmark 6: Quantum U-Net Architectures Comparison ===")
    set_seed(42)
    
    C = 128
    emb_dim = 512
    bsz = 4
    
    x = torch.randn(bsz, C, 16, 16).to(device)
    emb = torch.randn(bsz, emb_dim).to(device)
    
    # Target: Complex Modulation
    h = torch.sin(x) 
    style = torch.sigmoid(emb[:, :C]).unsqueeze(-1).unsqueeze(-1)
    target = torch.roll(h * style + torch.cos(h * style), shifts=1, dims=2) 
    
    model_ref = UNetBlockRef(C, emb_dim).to(device)
    model_sep = UNetBlockSeparated(C, emb_dim).to(device)
    model_int = UNetBlockIntegrated(C, emb_dim).to(device)
    
    print(f"Classic Ref Params: {count_parameters(model_ref)}")
    print(f"Separated Params: {count_parameters(model_sep)}")
    print(f"Integrated Params: {count_parameters(model_int)}")
    
    opt_ref = optim.Adam(model_ref.parameters(), lr=1e-3)
    opt_sep = optim.Adam(model_sep.parameters(), lr=2e-3)
    opt_int = optim.Adam(model_int.parameters(), lr=2e-3)
    
    print("\nTraining...")
    for i in range(steps):
        # Classic Ref
        opt_ref.zero_grad()
        out_ref = model_ref(x, emb)
        loss_ref = nn.MSELoss()(out_ref, target)
        loss_ref.backward()
        opt_ref.step()

        # Separated
        opt_sep.zero_grad()
        out_sep = model_sep(x, emb)
        loss_sep = nn.MSELoss()(out_sep, target)
        loss_sep.backward()
        opt_sep.step()
        
        # Integrated
        opt_int.zero_grad()
        out_int = model_int(x, emb)
        loss_int = nn.MSELoss()(out_int, target)
        loss_int.backward()
        opt_int.step()
        
        if i % 20 == 0 or i == steps - 1:
            print(f"Step {i}: Loss Ref={loss_ref.item():.5f}, Loss Sep={loss_sep.item():.5f}, Loss Int={loss_int.item():.5f}")
            
    print(f"Final Loss: Ref={loss_ref.item():.5f}, Sep={loss_sep.item():.5f}, Int={loss_int.item():.5f}")

def benchmark_unet_block(steps=200, device='cpu'):
    print("\n=== Benchmark 5: Realistic U-Net Block Integration (The Sandwich Test) ===")
    set_seed(42)
    B = 4
    C = 128
    H, W = 16, 16
    emb_dim = 512 
    
    x = torch.randn(B, C, H, W).to(device)
    emb = torch.randn(B, emb_dim).to(device)
    
    # Target: Synthetic Modulated Convolution
    # Mimics: Conv(Act(Norm(x) * Style + Shift))
    h = torch.sin(x) 
    style = torch.sigmoid(emb[:, :C]).unsqueeze(-1).unsqueeze(-1)
    target = torch.roll(h * style + torch.cos(h * style), shifts=1, dims=2) 
    # target = h * style # Pure Modulation Task
    
    classic_model = UNetBlockRef(C, emb_dim).to(device)
    quantum_model = UNetBlockQuantum(C, emb_dim).to(device)
    
    opt_c = optim.Adam(classic_model.parameters(), lr=1e-3)
    opt_q = optim.Adam(quantum_model.parameters(), lr=2e-3) # Lower LR for stability
    
    print(f"Classic Params: {count_parameters(classic_model)}")
    print(f"Quantum Params: {count_parameters(quantum_model)}")
    
    print("\nTraining...")
    for i in range(steps):
        opt_c.zero_grad()
        loss_c = nn.MSELoss()(classic_model(x, emb), target)
        loss_c.backward()
        opt_c.step()
        
        opt_q.zero_grad()
        loss_q = nn.MSELoss()(quantum_model(x, emb), target)
        loss_q.backward()
        opt_q.step()
        
        if i % 20 == 0 or i == steps-1:
            print(f"Step {i}: Loss C={loss_c.item():.5f}, Loss Q={loss_q.item():.5f}")
            
    print(f"Final Loss: C={loss_c.item():.5f}, Q={loss_q.item():.5f}")

# --- Benchmark 4: Progressive Training (Layer-wise) ---
def benchmark_progressive_training(steps=100, device='cpu'):
    print("\n=== Benchmark 4: Progressive Training (Layer-wise) ===")
    set_seed(42)
    B = 4
    C = 64
    H, W = 8, 8
    style_dim = 128
    
    x = torch.randn(B, C, H, W).to(device)
    style = torch.randn(B, style_dim).to(device)
    target = torch.randn(B, C, H, W).to(device)
    
    # Model A: End-to-End (Train all layers)
    model_e2e = QuantumFrontEndQCNN(C, style_dim, n_layers=2, n_groups=4, stride=1, reupload_data=True).to(device)
    opt_e2e = optim.Adam(model_e2e.parameters(), lr=1e-2)
    
    # Model B: Progressive (Layer 1 -> Layer 2)
    model_prog = QuantumFrontEndQCNN(C, style_dim, n_layers=2, n_groups=4, stride=1, reupload_data=True).to(device)
    opt_prog = optim.Adam(model_prog.parameters(), lr=1e-2)
    
    print("Training E2E vs Progressive...")
    
    loss_e2e_hist = []
    loss_prog_hist = []
    
    # Phase 1: Train Layer 1 (50 steps)
    model_prog.set_active_layers(1)
    for i in range(50):
        # E2E
        opt_e2e.zero_grad()
        loss_e2e = nn.MSELoss()(model_e2e(x, style), target)
        loss_e2e.backward()
        opt_e2e.step()
        loss_e2e_hist.append(loss_e2e.item())
        
        # Progressive
        opt_prog.zero_grad()
        loss_prog = nn.MSELoss()(model_prog(x, style), target)
        loss_prog.backward()
        opt_prog.step()
        loss_prog_hist.append(loss_prog.item())
        
    print(f"Phase 1 End (Step 50): E2E={loss_e2e.item():.5f}, Prog={loss_prog.item():.5f}")
    
    # Phase 2: Train Layer 2 (50 steps)
    model_prog.set_active_layers(2)
    # Re-initialize optimizer or keep momentum? Usually keep.
    # But new parameters (layer 2) were receiving 0 grad, so momentum is 0.
    
    for i in range(50):
        # E2E
        opt_e2e.zero_grad()
        loss_e2e = nn.MSELoss()(model_e2e(x, style), target)
        loss_e2e.backward()
        opt_e2e.step()
        loss_e2e_hist.append(loss_e2e.item())
        
        # Progressive
        opt_prog.zero_grad()
        loss_prog = nn.MSELoss()(model_prog(x, style), target)
        loss_prog.backward()
        opt_prog.step()
        loss_prog_hist.append(loss_prog.item())

    print(f"Phase 2 End (Step 100): E2E={loss_e2e.item():.5f}, Prog={loss_prog.item():.5f}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        benchmark_time_embedding(device=device)
        benchmark_spatial_modulation(device=device)
        benchmark_affine_modulation(device=device)
        benchmark_unet_block(device=device)
        benchmark_architectures(device=device)
        benchmark_progressive_training(device=device)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
