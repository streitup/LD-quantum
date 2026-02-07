
import sys
import os
import time
import torch
import torch.nn as nn
import math

# Add the directory containing quantum_transformer.py to sys.path
sys.path.append('/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion/training')

try:
    from quantum_transformer import QuantumAttentionAngle, QuantumAttention64, QuantumAttentionPatch
    import torchquantum as tq
    import torchquantum.functional as tqf
    print("Successfully imported QuantumAttentionAngle, QuantumAttention64, QuantumAttentionPatch and torchquantum")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class QuantumAttentionPure(QuantumAttentionAngle):
    """
    Pure Quantum Attention (No Classical Residuals).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove Classical Residual Projections
        del self.q_res_proj
        del self.k_res_proj
        del self.v_res_proj
        # Remove redundant input projection (optimization)
        if hasattr(self, 'inp_proj'):
            del self.inp_proj

    def _forward_impl(self, x_64: torch.Tensor, device_name: str) -> torch.Tensor:
        # 1. Prepare common state via Angle Encoding
        B, S, D = x_64.shape
        bsz = B * S
        x_bsz = x_64.reshape(bsz, D)
        
        # Skip inp_proj (Redundant)
        # x_bsz = self.inp_proj(x_bsz) 
        
        # Angle Encoding: 64 -> 12 -> Tanh -> Pi
        raw_out = self.angle_proj(x_bsz)
        angles = (torch.tanh(raw_out) + 1.0) * (torch.pi / 2.0)
        
        rx_angles = angles[:, :self.N_QUBITS]
        ry_angles = angles[:, self.N_QUBITS:]

        # Create common device
        qdev_common = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        
        # Apply Angle Encoding
        for i in range(self.N_QUBITS):
            tqf.rx(qdev_common, wires=i, params=rx_angles[:, i])
            tqf.ry(qdev_common, wires=i, params=ry_angles[:, i])
            
        # Apply Common PQC
        self._apply_pqc(qdev_common, self.enc_w)
        
        # Get common state
        if hasattr(qdev_common, 'get_states_1d'): 
            common_states_flat = qdev_common.get_states_1d()
        else: 
            common_states_flat = qdev_common.states.reshape(bsz, -1)
        
        target_shape = [bsz] + [2] * self.N_QUBITS
        common_states_reshaped = common_states_flat.reshape(target_shape)
        
        # Data Re-uploading
        reupload_angles = torch.tanh(self.reupload_proj(x_bsz)) * torch.pi

        # 2. Fork to Q/K/V branches (No Residuals)
        # Q Branch
        qdev_q = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_q.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_q, self.q_w, x_reupload=reupload_angles)
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_q, wires=i, params=self.meas_q_w[i].unsqueeze(0))
        probs_q = self._measure_multibasis(qdev_q)
        q = self.qk_ln(self.q_proj(probs_q)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        # No q_res addition
        
        # K Branch
        qdev_k = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_k.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_k, self.k_w, x_reupload=reupload_angles)
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_k, wires=i, params=self.meas_k_w[i].unsqueeze(0))
        probs_k = self._measure_multibasis(qdev_k)
        k = self.qk_ln(self.k_proj(probs_k)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        # No k_res addition
        
        # V Branch
        qdev_v = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_v.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_v, self.v_w, x_reupload=reupload_angles)
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_v, wires=i, params=self.meas_v_w[i].unsqueeze(0))
        probs_v = self._measure_multibasis(qdev_v) # Was _measure_probs in original, check usage
        # Original QuantumAttentionAngle used _measure_probs for V?
        # Let's check the code snippet again.
        # Line 466: probs_v = self._measure_probs(qdev_v)
        # But QuantumAttention64 used _measure_multibasis.
        # QuantumAttentionAngle inherited from 64.
        # I'll stick to _measure_multibasis for consistency or use what was there.
        # Wait, if I use _measure_multibasis, it returns [bsz, 64].
        # v_proj expects 64 input.
        v = self.v_proj(probs_v).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        # No v_res addition

        # 3. Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_out = (attn @ v).transpose(1, 2).reshape(B, S, self.inner_dim)
        x_out = self.out_proj(x_out)
        
        return x_out

class ClassicalAttention(nn.Module):
    """
    Standard Multi-Head Attention for comparison.
    Input: (B, S, D)
    Output: (B, S, D)
    """
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_convergence(model, x, target, steps=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()
    losses = []
    start_time = time.time()
    
    print(f"  Start training ({steps} steps)...")
    for i in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        if (i + 1) % 200 == 0:
            print(f"    Step {i+1}/{steps}: Loss = {loss.item():.6f}")
            
    end_time = time.time()
    return losses[-1], end_time - start_time

def benchmark():
    B, S, D = 4, 16, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. Instantiate Models
    # Reset seeds for fair comparison
    torch.manual_seed(42)
    classical_attn = ClassicalAttention(dim=D, num_heads=4).to(device)
    
    # torch.manual_seed(42)
    # quantum_hybrid = QuantumAttentionAngle(N_QUBITS=6, n_heads=4, device_name=device.type).to(device)
    
    # torch.manual_seed(42)
    # quantum_pure_angle = QuantumAttentionPure(N_QUBITS=6, n_heads=4, device_name=device.type).to(device)

    torch.manual_seed(42)
    # Patch size 2 means we group 2 tokens. 
    # With S=16, we have 8 patches. 
    # Group dim = 2 * 64 = 128.
    quantum_patch = QuantumAttentionPatch(dim=D, num_heads=4, patch_size=2, device_name=device.type).to(device)

    # 2. Count Parameters
    c_params = count_parameters(classical_attn)
    # qh_params = count_parameters(quantum_hybrid)
    # qpa_params = count_parameters(quantum_pure_angle)
    qp_patch_params = count_parameters(quantum_patch)

    print("-" * 50)
    print(f"Classical Attention Params: {c_params}")
    # print(f"Quantum Hybrid (Angle) Params: {qh_params} (Ratio: {qh_params/c_params:.2f}x)")
    # print(f"Quantum Pure (Angle) Params:   {qpa_params} (Ratio: {qpa_params/c_params:.2f}x)")
    print(f"Quantum Patch (New) Params:    {qp_patch_params} (Ratio: {qp_patch_params/c_params:.2f}x)")
    print("-" * 50)

    # 3. Convergence Test
    print("Convergence Test (1000 steps, fitting random target)")
    torch.manual_seed(42)
    x = torch.randn(B, S, D).to(device)
    target = torch.randn(B, S, D).to(device)

    print("\n[1] Training Classical Attention...")
    c_loss, c_time = train_convergence(classical_attn, x, target, steps=300)
    print(f"Classical Final Loss:      {c_loss:.6f} (Time: {c_time:.2f}s)")

    # print("\n[2] Training Quantum Hybrid (Angle)...")
    # qh_loss, qh_time = train_convergence(quantum_hybrid, x, target, steps=20)
    # print(f"Quantum Hybrid (Angle) Loss: {qh_loss:.6f} (Time: {qh_time:.2f}s)")
    
    # print("\n[3] Training Quantum Pure (Angle)...")
    # qpa_loss, qpa_time = train_convergence(quantum_pure_angle, x, target, steps=20)
    # print(f"Quantum Pure (Angle) Loss:   {qpa_loss:.6f} (Time: {qpa_time:.2f}s)")

    print("\n[2] Training Quantum Patch (New)...")
    qp_patch_loss, qp_patch_time = train_convergence(quantum_patch, x, target, steps=300)
    print(f"Quantum Patch (New) Loss:      {qp_patch_loss:.6f} (Time: {qp_patch_time:.2f}s)")

    print("-" * 50)

if __name__ == "__main__":
    benchmark()
