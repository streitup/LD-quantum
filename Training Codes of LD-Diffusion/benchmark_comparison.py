
import torch
import torch.nn as nn
import time
import numpy as np
import pennylane as qml
import sys
import os
import traceback

# Add training directory to path
sys.path.append(os.path.join(os.getcwd(), 'training'))

try:
    from training.quantum_transformer import QuantumAttentionPatch
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))
    from training.quantum_transformer import QuantumAttentionPatch

# ==========================================
# 1. Classical Attention (Baseline)
# ==========================================
class ClassicalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C) # [B, S, C]
        
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x = self.proj(x)
        
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]
        return x

# ==========================================
# 2. QSANN Implementation (Legacy)
# ==========================================
# Adapted from QSANN_pennylane.ipynb
import random

# Mock Module if torch not present (but we know it is)
class TorchLayer(nn.Module):
    def __init__(self, qnode, weights):
        super().__init__()
        self.qnode = qnode
        # self.qnode.interface = "torch" # Deprecated in newer PennyLane, handled by qnode definition
        self.qnode_weights = weights
        self.input_arg = "inputs"

    def forward(self, inputs):
        if len(inputs.shape) > 1:
            # Batch handling
            # Assuming inputs is [Batch, features]
            # PennyLane Torch interface handles batching if configured, but let's stick to the loop if unsure
            # or try direct passing
            return self._evaluate_qnode(inputs)
        return self._evaluate_qnode(inputs)

    def _evaluate_qnode(self, x):
        # Simplified for batch execution if supported by qnode
        # Otherwise, manual loop
        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)
        
        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)
        # If tuple of tensors
        return torch.hstack(res).type(x.dtype)

class QSAL_pennylane(nn.Module):
    def __init__(self, S, n, Denc, D):
        super().__init__()
        self.seq_num = S
        self.init_params_Q = nn.Parameter(torch.stack([(np.pi/4) * (2 * torch.randn(n*(D+2)) - 1) for _ in range(self.seq_num)]))
        self.init_params_K = nn.Parameter(torch.stack([(np.pi/4) * (2 * torch.randn(n*(D+2)) - 1) for _ in range(self.seq_num)]))
        self.init_params_V = nn.Parameter(torch.stack([(np.pi/4) * (2 * torch.randn(n*(D+2)) - 1) for _ in range(self.seq_num)]))
        self.num_q = n
        self.Denc = Denc
        self.D = D
        self.d = n*(Denc+2)
        
        # Use a lighter device if possible, or same device
        self.dev = qml.device("default.qubit", wires=self.num_q)
        
        # QNodes
        self.vqnod = qml.QNode(self.circuit_v, self.dev, interface="torch")
        self.qnod = qml.QNode(self.circuit_qk, self.dev, interface="torch")
        
        # We store weights in lists to mimic the notebook
        # But we need to be careful with gradients.
        # The original code creates list of TorchLayers.
        self.weight_v = [self.init_params_V[i] for i in range(self.seq_num)]
        self.weight_q = [self.init_params_Q[i] for i in range(self.seq_num)]
        self.weight_k = [self.init_params_K[i] for i in range(self.seq_num)]
        
    def random_op(self):
        # Deterministic for benchmark stability or random as per original?
        # Original is random per call? That seems chaotic for gradients.
        # But let's follow logic: it defines 'op' inside circuit.
        # Actually, random_op creates a fixed operator structure?
        # Wait, the notebook calls random_op() inside circuit_v/qk.
        # This means the measurement operator changes every time??
        # That would make training impossible.
        # Let's look closely at notebook.
        # "op=self.random_op()" is inside circuit_v.
        # This seems to be a specific feature of QSANN or a bug in the notebook for training.
        # For benchmarking, we will fix the operator to Identity or PauliZ to ensure stability, 
        # or use a fixed random seed.
        return qml.PauliZ(0) # Simplified for stability

    def circuit_v(self, inputs, weights):
        # op = self.random_op() # Simplify
        
        # Feature map
        indx = 0
        for j in range(self.num_q):
            qml.RX(inputs[indx], j)
            qml.RY(inputs[indx+1], j)
            indx += 2
        
        for i in range(self.Denc):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j+1)%self.num_q))
            for j in range(self.num_q):
                qml.RY(inputs[indx], j)
                indx += 1
                
        # Ansatz
        indx = 0
        for j in range(self.num_q):
            qml.RX(weights[indx], j)
            qml.RY(weights[indx+1], j)
            indx += 2
            
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j+1)%self.num_q))
            for j in range(self.num_q):
                qml.RY(weights[indx], j)
                indx += 1
        
        # Return expectations for each dimension of d?
        # The original returns [qml.expval(op) for i in range(self.d)]
        # But 'op' is single scalar expval.
        # How does it return 'd' values?
        # Original: return [qml.expval(op) for i in range(self.d)]
        # This returns 'd' copies of the SAME expectation value?
        # Unless 'op' changes?
        # In the original code, op is created once per circuit call.
        # So it returns d duplicates. This seems like a redundancy or placeholder.
        # We will replicate this behavior.
        return [qml.expval(qml.PauliZ(0)) for _ in range(self.d)]

    def circuit_qk(self, inputs, weights):
        # Feature map (Same as V)
        indx = 0
        for j in range(self.num_q):
            qml.RX(inputs[indx], j)
            qml.RY(inputs[indx+1], j)
            indx += 2
        for i in range(self.Denc):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j+1)%self.num_q))
            for j in range(self.num_q):
                qml.RY(inputs[indx], j)
                indx += 1
        
        # Ansatz
        indx = 0
        for j in range(self.num_q):
            qml.RX(weights[indx], j)
            qml.RY(weights[indx+1], j)
            indx += 2
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j+1)%self.num_q))
            for j in range(self.num_q):
                qml.RY(weights[indx], j)
                indx += 1
                
        return [qml.expval(qml.PauliZ(0))]

    def forward(self, x):
        # x: [B, C, H, W] -> Flatten to [B, S, C]
        # But this class expects [B, S, C] input as 'input' argument
        # We need an adapter.
        pass

# Adapter for QSANN to handle [B, C, H, W] and run the slow loop
class QSANNAdapter(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()
        self.S = height * width
        self.C = in_channels
        
        # Calculate n and Denc
        # d = n * (Denc + 2) = C
        # Try to find integer solution
        # For C=128:
        # n=8 -> Denc = 128/8 - 2 = 16 - 2 = 14
        self.n = 8
        self.Denc = 14
        self.D = 1 # Minimal ansatz depth
        
        # If C doesn't match exactly, we might need projection.
        # Assuming C=128 matches 8*(14+2).
        
        self.qsann = QSAL_pennylane(self.S, self.n, self.Denc, self.D)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C) # [B, S, C]
        
        # QSANN forward expects [B, S, C]
        
        # Ensure inputs match weight dtype (PennyLane sensitivity)
        dtype = self.qsann.init_params_Q.dtype
        if x_flat.dtype != dtype:
            x_flat = x_flat.to(dtype)
            
        input_seq = x_flat # [B, S, C]
        
        # Vectorized Implementation
        # We want to run B*S circuits in one go
        print(f"  Vectorized QSANN: Running {B}*{self.S}={B*self.S} circuits parallel...")
        bsz_total = B * self.S
        
        # 1. Prepare Inputs
        # input_seq: [B, S, C] -> [B*S, C] -> T -> [C, B*S]
        inputs_flat = input_seq.reshape(bsz_total, self.C).T
        
        # 2. Prepare Weights
        # Weights are [S, param_dim]
        # We need [B*S, param_dim] corresponding to flattened inputs
        # input_seq flattened is: (b0,s0), (b0,s1)... (b1,s0)...
        # So weights should be: w0, w1... wS, w0...
        
        # Q Weights
        # [S, D] -> repeat(B, 1) -> [B*S, D] -> T -> [D, B*S]
        w_q_flat = self.qsann.init_params_Q.repeat(B, 1).T 
        
        # K Weights
        w_k_flat = self.qsann.init_params_K.repeat(B, 1).T
        
        # V Weights
        w_v_flat = self.qsann.init_params_V.repeat(B, 1).T
        
        # 3. Run QNodes (Batched)
        
        # Q Branch
        q_res = self.qsann.qnod(inputs=inputs_flat, weights=w_q_flat)
        # q_res: [B*S] (if scalar return) or list
        if isinstance(q_res, list): q_res = q_res[0]
        if q_res.ndim == 1: q_res = q_res.unsqueeze(1) # [B*S, 1]
        
        # K Branch
        k_res = self.qsann.qnod(inputs=inputs_flat, weights=w_k_flat)
        if isinstance(k_res, list): k_res = k_res[0]
        if k_res.ndim == 1: k_res = k_res.unsqueeze(1) # [B*S, 1]
        
        # V Branch
        v_res = self.qsann.vqnod(inputs=inputs_flat, weights=w_v_flat)
        # v_res: list of [B*S], length d
        if isinstance(v_res, list):
            v_res = torch.stack(v_res, dim=1) # [B*S, d]
        
        # 4. Reshape back to [B, S, ...]
        Q_output = q_res.reshape(B, self.S, 1)
        K_output = k_res.reshape(B, self.S, 1)
        V_output = v_res.reshape(B, self.S, -1) # [B, S, d]
        
        # Attention Mechanism
        # Q: [B, S, 1], K: [B, S, 1]
        # Broadcast for pairwise
        Q_exp = Q_output.unsqueeze(2) # [B, S, 1, 1]
        K_exp = K_output.unsqueeze(1) # [B, 1, S, 1]
        
        # Alpha: [B, S, S]
        # alpha = exp(-(Q - K)**2)
        alpha = torch.exp(-(Q_exp - K_exp)**2).squeeze(-1) # [B, S, S]
        
        # Normalize
        row_sum = alpha.sum(dim=-1, keepdim=True)
        alpha_norm = alpha / (row_sum + 1e-8)
        
        # Output: alpha_norm @ V
        # [B, S, S] @ [B, S, d] -> [B, S, d]
        out = torch.bmm(alpha_norm, V_output)
        
        # Residual connection
        out = out + input_seq
        
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

# ==========================================
# 3. Benchmark Runner
# ==========================================
def run_benchmark():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 128, 16, 16 # Reduced Batch to 2 to accommodate slow QSANN
    
    print(f"Benchmarking on {device}")
    print(f"Input shape: [{B}, {C}, {H}, {W}]")
    
    # Models
    models = {}
    
    # 1. Classical
    models['Classical'] = ClassicalAttention(dim=C, num_heads=4).to(device)
    
    # 2. SOTA Quantum (Ours)
    # Using integration default params
    # n_qubits=7 (2^7=128), patch_size=1
    # Adapter for SOTA Quantum to handle [B, C, H, W]
    class SOTAQuantumAdapter(nn.Module):
        def __init__(self, dim, num_heads=4):
            super().__init__()
            # n_qubits=7 covers 128 dim
            self.model = QuantumAttentionPatch(
                dim=dim, num_heads=num_heads, patch_size=1, n_qubits=7, q_depth=4, lora_rank=16, device_name=str(device)
            )
            self.dim = dim
            
        def forward(self, x):
            B, C, H, W = x.shape
            # [B, C, H, W] -> [B, H*W, C] (Sequence format)
            x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            out = self.model(x_seq) # Returns [B, S, D]
            
            # [B, S, D] -> [B, C, H, W]
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return out

    models['SOTA_Quantum'] = SOTAQuantumAdapter(dim=C, num_heads=4).to(device)

    # 2b. SOTA Quantum Enhanced (More Expressive)
    class SOTAQuantumAdapterEnhanced(nn.Module):
        def __init__(self, dim, num_heads=4):
            super().__init__()
            # Increase q_depth to 12 and lora_rank to 64 for maximum expressivity
            self.model = QuantumAttentionPatch(
                dim=dim, num_heads=num_heads, patch_size=1, n_qubits=7, q_depth=12, lora_rank=64, device_name=str(device)
            )
            self.dim = dim
            
        def forward(self, x):
            B, C, H, W = x.shape
            x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            out = self.model(x_seq)
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return out

    models['SOTA_Quantum_Enhanced'] = SOTAQuantumAdapterEnhanced(dim=C, num_heads=4).to(device)
    
    # 3. QSANN (Legacy)
    # Note: QSANN is very slow on CPU/Simulator. We might run fewer steps or use very small H/W if needed.
    # But user asked for 16x16.
    # We will try. If it's too slow, we'll note it.
    print("Initializing QSANN... (this might take time)")
    # QSANN needs Double precision usually for PennyLane backprop stability or default.qubit
    models['QSANN'] = QSANNAdapter(in_channels=C, height=H, width=W).to(device)
    # Ensure QSANN is Double if needed, but let's try Float first with input casting.
    # If it failed with Float vs Double, it means some part generated Double.
    # PennyLane often generates Double.
    # So we force QSANN to Double.
    models['QSANN'] = models['QSANN'].double()

    # Data Generation Strategy: Real Image Features
    # Closer to UNet scenarios: Real Image -> Downsample -> Project to Channels -> Noise
    
    try:
        print("Loading real data from 100-shot-obama.zip...")
        # Add local path to find dnnlib and training
        sys.path.append(os.getcwd())
        from training.dataset import ImageFolderDataset
        
        # Path to dataset
        dataset_path = os.path.join(os.getcwd(), '100-shot-obama.zip')
        if not os.path.exists(dataset_path):
             dataset_path = os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion', '100-shot-obama.zip')
             
        ds = ImageFolderDataset(path=dataset_path, resolution=None)
        
        # Get B images
        images = []
        for i in range(B):
            img_np, _ = ds[i] # [3, H_orig, W_orig]
            images.append(torch.from_numpy(img_np))
            
        images = torch.stack(images).float() / 255.0 # [B, 3, H_orig, W_orig]
        images = images.to(device)
        
        # Simulate UNet Feature Extraction
        # 1. Downsample to 16x16
        features = torch.nn.functional.interpolate(images, size=(H, W), mode='bilinear', align_corners=False)
        
        # 2. Project 3 channels to C=128 channels (Simulate initial convs)
        # Use a fixed random projection to create consistent "features"
        projector = nn.Conv2d(3, C, kernel_size=1, bias=False).to(device)
        # Fix weights for reproducibility
        torch.manual_seed(42)
        projector.weight.data = torch.randn_like(projector.weight.data) * 0.1
        
        x_clean = projector(features) # [B, C, H, W]
        print("Successfully generated features from Real Images.")
        
    except Exception as e:
        print(f"Could not load real data: {e}. Falling back to Synthetic Features.")
        # Fallback: Synthetic Feature Maps (Blobs)
        # Generate random Gaussian blobs to simulate features
        x_clean = torch.zeros(B, C, H, W, device=device)
        for b in range(B):
            for c in range(C):
                 # Simple blob generation
                 cx, cy = torch.randint(0, W, (1,)), torch.randint(0, H, (1,))
                 grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                 dist = ((grid_x - cx)**2 + (grid_y - cy)**2).float()
                 x_clean[b, c] = torch.exp(-dist / 10.0) # sigma^2=10
        
        # Mix channels
        mixer = nn.Conv2d(C, C, kernel_size=1, bias=False).to(device)
        x_clean = mixer(x_clean)

    # Task: Denoising
    # Input = Clean Features + Noise
    # Target = Clean Features (predicting x_0)
    # This simulates the core task of a UNet block: preserving/restoring signal in a noisy flow.
    
    target = x_clean.detach()
    noise = torch.randn_like(target) * 0.5 # Add significant noise
    x = target + noise
    
    print(f"Task: Denoising Real-like Feature Maps (Signal+Noise -> Signal)")
    
    results = []
    
    for name, model in models.items():
        print(f"\nRunning {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Params
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Warmup
        try:
            # Short warmup
            y = model(x)
            
            # Ensure target matches y dtype (especially for Double precision QSANN)
            target_step = target.to(y.dtype)
            
            loss = criterion(y, target_step)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            print(f"Failed during warmup for {name}: {e}")
            traceback.print_exc()
            results.append({'name': name, 'params': params, 'time': 'N/A', 'loss': 'N/A', 'error': str(e)})
            continue

        # Measurement Loop
        steps = 100 # Increase steps to observe convergence
        start_time = time.time()
        final_loss = 0
        loss_history = []
        
        try:
            for step in range(steps):
                optimizer.zero_grad()
                y = model(x)
                
                # Ensure target matches y dtype
                target_step = target.to(y.dtype)
                
                loss = criterion(y, target_step)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
                loss_history.append(final_loss)
                
                if (step + 1) % 10 == 0:
                     print(f"  Step {step+1}/{steps}, Loss: {final_loss:.6f}")
                else:
                     print(f"  Step {step+1}/{steps}, Loss: {final_loss:.6f}", end='\r')
            
            total_time = time.time() - start_time
            avg_time = total_time / steps
            
            results.append({
                'name': name,
                'params': params,
                'time': avg_time,
                'loss': final_loss
            })
            print(f"\n  Done. Avg Time: {avg_time:.4f}s, Final Loss: {final_loss:.6f}, Params: {params}")
            
        except Exception as e:
            print(f"\n  Failed execution for {name}: {e}")
            results.append({'name': name, 'params': params, 'time': 'N/A', 'loss': 'N/A', 'error': str(e)})

    # Print Report
    print("\n" + "="*60)
    print(f"{'Model':<15} | {'Params':<10} | {'Time/Step (s)':<15} | {'Loss':<10}")
    print("-" * 60)
    for res in results:
        t = f"{res['time']:.4f}" if isinstance(res['time'], float) else res['time']
        l = f"{res['loss']:.6f}" if isinstance(res['loss'], float) else res['loss']
        print(f"{res['name']:<15} | {res['params']:<10} | {t:<15} | {l:<10}")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
