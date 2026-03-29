import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import sys
import os
import time
import math
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.quantum_transformer import QuantumFrontEndQCNNState, QuantumFrontEndSOTA
from training.dataset import ImageFolderDataset
from training.networks import QuantumFrontEndQCNN

# --- Models ---

class ClassicalBaseline(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x, emb):
        return x + self.conv(self.norm(x))

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Metrics Implementation (from benchmark_sota_ablation.py) ---

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class LPIPS(nn.Module):
    def __init__(self, device):
        super().__init__()
        try:
            vgg = models.vgg16(weights='DEFAULT').features.to(device)
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.layers = [3, 8, 15, 22, 29]
            self.eval()
        except Exception as e:
            print(f"Warning: Could not load VGG for LPIPS ({e}). LPIPS will be 0.")
            self.vgg = None
            
    def forward(self, img1, img2):
        if self.vgg is None:
            return torch.tensor(0.0).to(img1.device)
        if img1.shape[2] < 64:
            img1 = F.interpolate(img1, size=(64, 64), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(64, 64), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        img1 = (img1 - mean) / std
        img2 = (img2 - mean) / std
        loss = 0
        x = img1
        y = img2
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += F.mse_loss(x, y)
        return loss

# --- Data Preparation ---

def load_data(batch_size, device):
    dataset_path = "/home/zzn/qfl_tq/ffhq_workspace/100-shot-obama-128.zip"
    if not os.path.exists(dataset_path):
        print(f"Dataset zip not found at {dataset_path}")
        return None
    print(f"Loading dataset from {dataset_path}")
    ds = ImageFolderDataset(path=dataset_path, resolution=None) 
    class WrapperDataset(Dataset):
        def __init__(self, ds, target_res=16):
            self.ds = ds
            self.target_res = target_res
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            img = torch.from_numpy(img).float() / 255.0
            # Interpolate to 16x16
            img = F.interpolate(img.unsqueeze(0), size=(self.target_res, self.target_res), mode='bilinear', align_corners=False).squeeze(0)
            return img, label
    wrapped_ds = WrapperDataset(ds, target_res=16)
    loader = DataLoader(wrapped_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

# --- Time Embedding ---
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        # t: [B]
        half_dim = self.dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.mlp(emb)
        return emb

# --- Models ---

class ClassicalBaseline(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x, emb):
        return x + self.conv(self.norm(x))

class QCNN_Fused_State(nn.Module):
    """
    Original Fused-NoAttn: Amplitude Encoding -> Circuit -> State -> Abs^2 -> Proj
    (Linear Encoding + Linear Output + Manual Non-linearity)
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        self.state_dim = 64
        # Note: We use the fixed QCNNState class that handles Re-uploading correctly
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
        self.proj_out = nn.Linear(self.n_groups * self.state_dim, channels)

    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        state = self.fe(x, emb) # [B, L, G*D]
        features = (state.abs() ** 2).float() # Manual Measurement
        x_out = self.proj_out(features)
        x_out = x_out.transpose(1, 2).reshape(b, c, h, w)
        return x_out + resid

class QCNN_Amplitude_Meas(nn.Module):
    """
    Fused-NoAttn with Measurement Added Back properly?
    Actually, QCNN_Fused_State ALREADY does measurement (Abs^2).
    What if we use the Standard QCNN Projection logic (Per-Group Projection)?
    And ensure we are using the 'Measurement' flow.
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.channels = channels
        self.n_groups = 4
        # We subclass QCNNState but modify forward to behave like Standard QCNN
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
        # Standard QCNN uses a Linear layer per group output (probs)
        # 1<<N -> Channels_per_group
        self.out_proj = nn.Linear(64, channels // 4) 

    def forward(self, x, emb):
        resid = x
        b, c, h, w = x.shape
        # We need to hack FE to return probs inside or do it here
        state = self.fe(x, emb) # [B, L, G*D]
        
        # Reshape to groups
        state = state.reshape(b, -1, self.n_groups, 64) # [B, L, G, 64]
        probs = (state.abs() ** 2).float()
        
        # Apply Per-Group Projection (Standard QCNN style)
        # [B, L, G, 64] -> [B, L, G, C/G]
        out_g = self.out_proj(probs)
        
        # Flatten
        out_flat = out_g.reshape(b, -1, self.n_groups * (self.channels // 4))
        
        x_out = out_flat.transpose(1, 2).reshape(b, c, h, w)
        return x_out + resid

class QCNN_Angle_Meas(nn.Module):
    """
    Standard QCNN: Angle Encoding -> Circuit -> Measurement -> Proj
    (Non-linear Encoding + Non-linear Output)
    Using Base QuantumFrontEndQCNN (Angle mode is safe from the bug)
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        self.fe = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim,
            n_layers=1,
            n_groups=self.n_groups,
            encoding_type='tanh', # Angle Encoding
            n_qubits_ancilla=0,
            reupload_data=True,
            stride=1
        )
        # Base class handles projection internally

    def forward(self, x, emb):
        resid = x
        x_out = self.fe(x, emb)
        return x_out + resid

# --- Benchmark ---

class QCNN_Angle_Pure(nn.Module):
    """
    Pure QCNN (Angle Encoding) WITHOUT Classical Residual.
    Tests if the Quantum Circuit alone can learn the features.
    """
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.n_groups = 4
        # Standard QCNN uses Angle Encoding (tanh) and measures Pauli Z
        self.fe = QuantumFrontEndQCNN(
            channels=channels,
            style_dim=emb_dim,
            n_layers=1, 
            n_groups=self.n_groups,
            encoding_type='tanh', # Angle encoding
            n_qubits_ancilla=2,
            reupload_data=True,
            stride=1
        )
        # Note: QuantumFrontEndQCNN already includes output projection to 'channels'
        # But it usually expects a residual connection to be added outside.
        # Here we just use the QCNN output.

    def forward(self, x, emb):
        # No residual connection
        # x_out = self.fe(x, emb)
        # However, QuantumFrontEndQCNN returns [B, C, H, W] directly?
        # Let's check QuantumFrontEndQCNN in quantum_transformer.py
        # Yes, it returns [B, C, H_out, W_out]
        
        return self.fe(x, emb)

class QCNN_SOTA(nn.Module):
    """
    SOTA QCNN (Refined SOTA Patch):
    - Hybrid Encoding (Angle)
    - Grouped PQC
    - Data Re-uploading
    - Classical Residual
    - Uses ExperimentalQuantumFrontEnd with 'q_middle_freq' modulation
    """
    def __init__(self, channels, emb_dim, n_layers=2, n_groups=4, n_qubits=6, residual_mode='default', encoding_type='tanh'):
        super().__init__()
        # SOTA V3 (Optimal): Param-Free Residual Bypass (Center Pixel) + Basic Ansatz
        
        self.fe = QuantumFrontEndSOTA(
            channels=channels,
            style_dim=emb_dim,
            n_layers=n_layers,             
            n_groups=n_groups,
            n_qubits_data=n_qubits,        
            encoding_type=encoding_type, 
            stride=1,
            affine_mode='q_middle_freq', 
        )

    def forward(self, x, emb):
        # SOTA includes classical residual implicitly in some versions,
        # but ExperimentalQuantumFrontEnd adds 'res_proj' (MLP/Linear).
        # Let's trust the internal residual logic of ExperimentalQuantumFrontEnd.
        # But wait, ExperimentalQuantumFrontEnd inherits from QuantumFrontEndQCNN
        # which DOES add `out_res = self.res_proj(patches_flat)`.
        # So it is self-contained.
        return self.fe(x, emb)

def benchmark_qcnn_variants(epochs=50, device='cuda'):
    print("\n=== Benchmark: QCNN Variants Image Reconstruction ===")
    print("Comparing Classical, Hybrid, Fused, Pure QCNN, and SOTA architectures.")
    set_seed(42)
    
    # Load Real Data
    batch_size = 8
    loader = load_data(batch_size, device)
    if loader is None:
        print("Failed to load dataset.")
        return

    sample_img, _ = next(iter(loader))
    _, C_img, H, W = sample_img.shape
    print(f"Data Resolution: {C_img}x{H}x{W}")
    
    C_model = 64
    emb_dim = C_model * 4
    
    # Define Model Wrappers (Input Proj -> Block -> Output Proj)
    class ModelWrapper(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.input_proj = nn.Conv2d(C_img, C_model, kernel_size=3, padding=1)
            self.core = core_model
            self.output_proj = nn.Conv2d(C_model, C_img, kernel_size=3, padding=1)
            self.time_embed = TimeEmbedding(emb_dim)
            
        def forward(self, x, t):
            t_emb = self.time_embed(t)
            x = self.input_proj(x)
            x = self.core(x, t_emb)
            x = self.output_proj(x)
            return x
    
    models = {
        'Classical': ModelWrapper(ClassicalBaseline(C_model)).to(device),
        'SOTA (Base)': ModelWrapper(QCNN_SOTA(C_model, emb_dim, residual_mode='default')).to(device),
        'SOTA (Gaussian)': ModelWrapper(QCNN_SOTA(C_model, emb_dim, residual_mode='gaussian')).to(device),
        'SOTA (Laplacian)': ModelWrapper(QCNN_SOTA(C_model, emb_dim, residual_mode='laplacian')).to(device),
    }
    
    lpips_fn = LPIPS(device)
    results = {}
    
    print(f"{'Model':<15} | {'Params':<10}")
    print("-" * 30)
    for name, model in models.items():
        print(f"{name:<15} | {count_parameters(model):<10}")
    
    print("\nTraining for {} epochs...".format(epochs))
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        start_time = time.time()
        best_psnr = 0
        best_ssim = 0
        best_lpips = float('inf')
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            steps = 0
            
            for i, (imgs, _) in enumerate(loader):
                imgs = imgs.to(device)
                
                # Add noise
                noise = torch.randn_like(imgs) * 0.1
                noisy_imgs = imgs + noise
                
                t = torch.randint(0, 1000, (imgs.shape[0],), device=device).float()
                
                optimizer.zero_grad()
                pred_imgs = model(noisy_imgs, t)
                loss = F.mse_loss(pred_imgs, imgs)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
            
            scheduler.step()
            
            # Validation
            with torch.no_grad():
                # Reconstruct one batch
                model.eval()
                t_val = torch.zeros(imgs.shape[0], device=device).float()
                pred_val = model(imgs, t_val) # Clean input for reconstruction check
                
                pred_val = torch.clamp(pred_val, 0, 1)
                imgs_val = torch.clamp(imgs, 0, 1)
                
                val_psnr = psnr(pred_val, imgs_val).item()
                val_ssim = ssim(pred_val, imgs_val).item()
                val_lpips = lpips_fn(pred_val, imgs_val).item() * 1000
                
                if val_psnr > best_psnr: best_psnr = val_psnr
                if val_ssim > best_ssim: best_ssim = val_ssim
                if val_lpips < best_lpips: best_lpips = val_lpips
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/steps:.6f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

        end_time = time.time()
        training_time = end_time - start_time
        
        results[name] = {
            "Params": count_parameters(model),
            "PSNR": best_psnr,
            "SSIM": best_ssim,
            "LPIPS": best_lpips,
            "Time": training_time
        }
        
        print(f"  [Result] PSNR: {best_psnr:.2f} | SSIM: {best_ssim:.4f} | LPIPS: {best_lpips:.2f} | Time: {training_time:.2f}s")

    print("\n\n=== Final Benchmark Results (Converged) ===")
    print(f"{'Model':<15} | {'Params':<10} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8} | {'Time (s)':<8}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<15} | {res['Params']:<10} | {res['PSNR']:<8.2f} | {res['SSIM']:<8.4f} | {res['LPIPS']:<8.2f} | {res['Time']:<8.2f}")

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, t):
        return self.model(x, t)

# Multi-Resolution Benchmark
def run_benchmark_multi_res():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Multi-Resolution Benchmark on {device}")
    
    # Define resolutions and channel configurations (simulating U-Net stages)
    # Stage 1: 32x32, C=64
    # Stage 2: 16x16, C=128
    # Stage 3: 8x8,   C=256
    stages = [
        {'res': 32, 'channels': 64,  'emb_dim': 256},
        {'res': 16, 'channels': 128, 'emb_dim': 256},
        {'res': 8,  'channels': 256, 'emb_dim': 256},
    ]
    
    # Define hyperparameter grid to search
    # We want to find best (groups, layers, qubits) for each stage
    # Groups: 4, 8, 16 (scaling with channels)
    # Layers: 2, 4 (depth)
    # Qubits: 6 (fixed for efficiency, but maybe test 8 for deep layers?)
    
    param_grid = [
        {'n_groups': 4, 'n_layers': 2, 'n_qubits': 6}, # Baseline SOTA V3
        {'n_groups': 8, 'n_layers': 2, 'n_qubits': 6}, # Wide Groups
        {'n_groups': 16,'n_layers': 2, 'n_qubits': 6}, # Very Wide Groups
        {'n_groups': 4, 'n_layers': 4, 'n_qubits': 6}, # Deep
        # {'n_groups': 4, 'n_layers': 2, 'n_qubits': 8}, # Wide Qubits (Slow, skip for now unless needed)
    ]

    for stage in stages:
        res = stage['res']
        C = stage['channels']
        emb = stage['emb_dim']
        print(f"\n=== Benchmarking Stage: {res}x{res}, Channels={C} ===")
        
        # Create Dummy Data for this resolution
        # B=8 to be fast but stable
        x_train = torch.randn(32, C, res, res).to(device)
        t_train = torch.randn(32, emb).to(device)
        x_val = torch.randn(16, C, res, res).to(device)
        t_val = torch.randn(16, emb).to(device)
        
        # Classical Baseline for this stage
        classical = ModelWrapper(ClassicalBaseline(C)).to(device)
        opt = optim.Adam(classical.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
        start_t = time.time()
        for i in range(300): # Full convergence
            opt.zero_grad()
            out = classical(x_train, t_train)
            loss = F.mse_loss(out, x_train)
            loss.backward()
            opt.step()
            scheduler.step()
        train_t = time.time() - start_t
        
        # Eval Classical
        with torch.no_grad():
            rec = classical.model(x_val, t_val)
            p = psnr(x_val, rec)
            print(f"[Classical] PSNR: {p:.2f} dB | Time: {train_t:.2f}s | Params: {count_parameters(classical)}")

        # Search Grid
        best_cfg = None
        best_psnr = -1
        
        for cfg in param_grid:
            G = cfg['n_groups']
            L = cfg['n_layers']
            Q = cfg['n_qubits']
            
            # Skip invalid configs (e.g. groups > channels, though here C>=64 so ok)
            if G > C: continue
            
            # Skip if channels per group is too small (<4) or too large (>32)
            c_per_g = C // G
            # if c_per_g < 4: continue 
            
            model = ModelWrapper(QCNN_SOTA(C, emb, n_layers=L, n_groups=G, n_qubits=Q)).to(device)
            opt = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
            
            # Train
            start_t = time.time()
            for i in range(300):
                opt.zero_grad()
                out = model(x_train, t_train)
                loss = F.mse_loss(out, x_train)
                loss.backward()
                opt.step()
                scheduler.step()
            train_t = time.time() - start_t
            
            # Eval
            with torch.no_grad():
                rec = model.model(x_val, t_val)
                p = psnr(x_val, rec)
                
            print(f"[QCNN G{G} L{L} Q{Q}] PSNR: {p:.2f} dB | Time: {train_t:.2f}s | Params: {count_parameters(model)} | C/G: {c_per_g}")
            
            if p > best_psnr:
                best_psnr = p
                best_cfg = cfg
                
        print(f"--> Best Config for {res}x{res} (C={C}): Groups={best_cfg['n_groups']}, Layers={best_cfg['n_layers']} (C/G = {C // best_cfg['n_groups']})")

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x, t):
            return self.model(x, t)

# Overfitting Benchmark: Train vs Validation Performance
def run_benchmark_overfitting():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Overfitting Benchmark on {device}")
    
    # We focus on the "Shallow" stage where Classical showed >100dB PSNR
    res = 32
    C = 64
    emb = 256
    
    # Generate DISTINCT Train and Validation Sets
    # Train Set: Small enough to overfit (32 samples)
    # Val Set: Unseen samples (16 samples)
    x_train = torch.randn(32, C, res, res).to(device)
    t_train = torch.randn(32, emb).to(device)
    
    x_val = torch.randn(16, C, res, res).to(device)
    t_val = torch.randn(16, emb).to(device)
    
    print(f"\n=== Overfitting Test: 32x32, Channels={C} ===")
    
    # 1. Classical Model
    classical = ModelWrapper(ClassicalBaseline(C)).to(device)
    print(f"[Classical] Parameters: {count_parameters(classical)}")
    opt = optim.Adam(classical.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    
    print("Training Classical Model...")
    for i in range(300):
        opt.zero_grad()
        out = classical(x_train, t_train)
        loss = F.mse_loss(out, x_train)
        loss.backward()
        opt.step()
        scheduler.step()
        
    # Eval Classical
    with torch.no_grad():
        out_train = classical.model(x_train, t_train)
        psnr_train = psnr(x_train, out_train)
        
        out_val = classical.model(x_val, t_val)
        psnr_val = psnr(x_val, out_val)
        
    print(f"[Classical] Train PSNR: {psnr_train:.2f} dB | Val PSNR: {psnr_val:.2f} dB")
    print(f"--> Overfitting Gap: {psnr_train - psnr_val:.2f} dB")
    
    # 2. Quantum Model (Best Config for 32x32: G=4, L=4)
    qcnn = ModelWrapper(QCNN_SOTA(C, emb, n_layers=4, n_groups=4, n_qubits=6)).to(device)
    opt = optim.Adam(qcnn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    
    print("Training Quantum Model...")
    for i in range(300):
        opt.zero_grad()
        out = qcnn(x_train, t_train)
        loss = F.mse_loss(out, x_train)
        loss.backward()
        opt.step()
        scheduler.step()
        
    # Eval Quantum
    with torch.no_grad():
        out_train = qcnn.model(x_train, t_train)
        psnr_train = psnr(x_train, out_train)
        
        out_val = qcnn.model(x_val, t_val)
        psnr_val = psnr(x_val, out_val)
        
    print(f"[Quantum SOTA] Train PSNR: {psnr_train:.2f} dB | Val PSNR: {psnr_val:.2f} dB")
    print(f"--> Overfitting Gap: {psnr_train - psnr_val:.2f} dB")

# --- Real Dataset Loader (from benchmark_sota_ablation.py) ---
def load_real_data(batch_size, target_res, channels):
    # Path to 100-shot-obama dataset
    dataset_path = "/home/zzn/qfl_tq/Low-shot Datasets/100-shot-obama.zip"
    if not os.path.exists(dataset_path):
        # Try alternate path found in search
        dataset_path = "/home/zzn/qfl_tq/ffhq_workspace/100-shot-obama-128.zip"
        
    if not os.path.exists(dataset_path):
        print(f"Dataset zip not found at {dataset_path}")
        return None
    
    print(f"Loading dataset from {dataset_path}")
    ds = ImageFolderDataset(path=dataset_path, resolution=None) 
    
    class WrapperDataset(Dataset):
        def __init__(self, ds, target_res):
            self.ds = ds
            self.target_res = target_res
            
        def __len__(self):
            return len(self.ds)
            
        def __getitem__(self, idx):
            # ImageFolderDataset returns img as numpy [C,H,W] or [H,W,C]? Usually [C,H,W] normalized 0-255
            # Let's check ImageFolderDataset implementation or assume standard
            # Based on search results, it returns numpy array.
            # We need to convert to tensor and normalize to [-1, 1] or [0, 1]
            # Standard diffusion uses [-1, 1], but for this reconstruction task [0, 1] or normalized is fine.
            # Let's use [0, 1] for PSNR calc.
            
            img = self.ds[idx] # might be tuple (img, label)
            if isinstance(img, tuple):
                img = img[0]
                
            # Assume img is numpy array. Check shape.
            # If [H, W, C], transpose.
            if img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)
                
            img = torch.from_numpy(img).float() / 255.0
            
            # Interpolate to target_res
            img = F.interpolate(img.unsqueeze(0), size=(self.target_res, self.target_res), mode='bilinear', align_corners=False).squeeze(0)
            
            # If we need specific channels (e.g. 64), we might need to project or repeat?
            # But real images are RGB (3 channels).
            # Our benchmark assumes Latent space (64/128/256 channels).
            # To simulate this, we can project RGB -> C using a random linear layer (fixed) 
            # OR just repeat channels (but that's low rank).
            # OR use a 1x1 conv to project up.
            return img
            
    wrapped_ds = WrapperDataset(ds, target_res)
    loader = DataLoader(wrapped_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

# Real Image Denoising Benchmark
def run_benchmark_real_denoising():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Real Image Denoising Benchmark on {device}")
    
    # We will use 32x32 resolution as proxy for Latent features
    # But since we load RGB (3ch), we need to project to C=64 to match our architecture study
    res = 32
    C = 64
    emb = 256
    
    # Load Data
    loader = load_real_data(batch_size=32, target_res=res, channels=3)
    if loader is None: return

    # Fixed Projection RGB(3) -> Latent(C)
    # We use a random initialized Conv2d and freeze it to simulate an "Encoder"
    encoder = nn.Conv2d(3, C, kernel_size=1).to(device)
    encoder.requires_grad_(False)
    
    # Get a fixed batch for Train/Val split
    # We take first batch as Train, second as Val (if available)
    # 100-shot is small. 100 images.
    # Batch 32. 3 batches total.
    iter_loader = iter(loader)
    x_rgb_train = next(iter_loader).to(device) # [32, 3, 32, 32]
    x_rgb_val = next(iter_loader).to(device)   # [32, 3, 32, 32]
    
    # Encode to "Latent"
    with torch.no_grad():
        x_train = encoder(x_rgb_train) # [32, 64, 32, 32]
        x_val = encoder(x_rgb_val)     # [32, 64, 32, 32]
        
        # Normalize latents to have std=1 (approx)
        # x_train = (x_train - x_train.mean()) / x_train.std()
        # x_val = (x_val - x_val.mean()) / x_val.std()
        
        # Better normalization: Scale to [-1, 1] range which tanh expects
        # RGB is [0, 1]. After random Conv, it can be anything.
        # Let's enforce standard normalization
        x_train = F.normalize(x_train, p=2, dim=1) # Normalize channel vectors? No.
        
        # Instance Norm to mean=0, std=1 per sample
        x_train = F.instance_norm(x_train)
        x_val = F.instance_norm(x_val)
        
        # Scale down to avoid saturating tanh in QCNN immediately
        x_train = x_train * 0.5 
        x_val = x_val * 0.5
        
    t_train = torch.randn(32, emb).to(device)
    t_val = torch.randn(32, emb).to(device)
    
    print(f"\n=== Real Denoising Task: 32x32, Channels={C} ===")
    print("Task: Predict Noise epsilon added to x_0")
    print("Loss: MSE(Prediction, Noise)")
    print(f"Signal Mean/Std: {x_train.mean():.4f}/{x_train.std():.4f}")
    
    # Noise Injection (Diffusion Process Simulation)
    # x_t = sqrt(alpha)*x_0 + sqrt(1-alpha)*epsilon
    # We pick a fixed noise level for simplicity, e.g. t ~ 0.5 (SNR=1)
    noise_train = torch.randn_like(x_train)
    noise_val = torch.randn_like(x_val)
    print(f"Noise Mean/Std: {noise_train.mean():.4f}/{noise_train.std():.4f}")
    
    # Simple mix: x_noisy = 0.7*x + 0.7*noise (Approx SNR=1)
    # 0.7^2 + 0.7^2 = 0.49 + 0.49 = 0.98 ~ 1.0 (Variance preserving)
    x_train_noisy = 0.707 * x_train + 0.707 * noise_train
    x_val_noisy = 0.707 * x_val + 0.707 * noise_val
    print(f"Noisy Input Mean/Std: {x_train_noisy.mean():.4f}/{x_train_noisy.std():.4f}")
    
    # Models must predict 'noise_train' from 'x_train_noisy'
    
    # 1. Classical Model
    classical = ModelWrapper(ClassicalBaseline(C)).to(device)
    print(f"[Classical] Parameters: {count_parameters(classical)}")
    opt = optim.Adam(classical.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    
    print("Training Classical Model...")
    for i in range(300):
        opt.zero_grad()
        # Input: Noisy Image. Target: Noise.
        pred_noise = classical(x_train_noisy, t_train)
        loss = F.mse_loss(pred_noise, noise_train)
        loss.backward()
        opt.step()
        scheduler.step()
        
    # Eval Classical
    with torch.no_grad():
        pred_train = classical.model(x_train_noisy, t_train)
        # PSNR of Noise Prediction
        psnr_train = psnr(noise_train, pred_train)
        
        pred_val = classical.model(x_val_noisy, t_val)
        psnr_val = psnr(noise_val, pred_val)
        
    print(f"[Classical] Train PSNR: {psnr_train:.2f} dB | Val PSNR: {psnr_val:.2f} dB")
    print(f"--> Overfitting Gap: {psnr_train - psnr_val:.2f} dB")
    
    # 2. Quantum Model (SOTA V3)
    # Best for 32x32 C=64 is G=4, L=4
    # Trying L=8 for real denoising
    # Use arctan encoding to handle unbounded inputs better
    qcnn = ModelWrapper(QCNN_SOTA(C, emb, n_layers=8, n_groups=4, n_qubits=6, encoding_type='arctan')).to(device)
    opt = optim.Adam(qcnn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    
    print("Training Quantum Model...")
    for i in range(300):
        opt.zero_grad()
        pred_noise = qcnn(x_train_noisy, t_train)
        loss = F.mse_loss(pred_noise, noise_train)
        loss.backward()
        opt.step()
        scheduler.step()
        
    # Eval Quantum
    with torch.no_grad():
        pred_train = qcnn.model(x_train_noisy, t_train)
        psnr_train = psnr(noise_train, pred_train)
        
        pred_val = qcnn.model(x_val_noisy, t_val)
        psnr_val = psnr(noise_val, pred_val)
        
    print(f"[Quantum SOTA] Train PSNR: {psnr_train:.2f} dB | Val PSNR: {psnr_val:.2f} dB")
    print(f"--> Overfitting Gap: {psnr_train - psnr_val:.2f} dB")

# Real Image Reconstruction Benchmark
def run_benchmark_real_reconstruction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Real Image Reconstruction Benchmark on {device}")
    
    C = 64
    emb = 256
    
    # Define resolutions to test
    resolutions = [32, 16, 8]
    
    for res in resolutions:
        # Load Data
        loader = load_real_data(batch_size=32, target_res=res, channels=3)
        if loader is None: return
    
        # Fixed Projection RGB(3) -> Latent(C)
        encoder = nn.Conv2d(3, C, kernel_size=1).to(device)
        encoder.requires_grad_(False)
        
        iter_loader = iter(loader)
        x_rgb_train = next(iter_loader).to(device)
        x_rgb_val = next(iter_loader).to(device)
        
        # Encode to "Latent"
        with torch.no_grad():
            x_train = encoder(x_rgb_train)
            x_val = encoder(x_rgb_val)
            
            # Normalize
            x_train = F.normalize(x_train, p=2, dim=1) 
            x_train = F.instance_norm(x_train)
            x_val = F.instance_norm(x_val)
            
            # Scale down
            x_train = x_train * 0.5 
            x_val = x_val * 0.5
            
        t_train = torch.randn(32, emb).to(device)
        t_val = torch.randn(32, emb).to(device)
        
        print(f"\n=== Real Reconstruction Task: {res}x{res}, Channels={C} ===")
        print("Task: Reconstruct x_0 from x_0 (Autoencoding)")
        print(f"Input Mean/Std: {x_train.mean():.4f}/{x_train.std():.4f}")
        
        # 1. Classical Model
        classical = ModelWrapper(ClassicalBaseline(C)).to(device)
        print(f"[Classical] Parameters: {count_parameters(classical)}")
        opt = optim.Adam(classical.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
        
        print("Training Classical Model...")
        for i in range(300):
            opt.zero_grad()
            out = classical(x_train, t_train)
            loss = F.mse_loss(out, x_train)
            loss.backward()
            opt.step()
            scheduler.step()
            
        with torch.no_grad():
            out_train = classical.model(x_train, t_train)
            psnr_train = psnr(x_train, out_train)
            out_val = classical.model(x_val, t_val)
            psnr_val = psnr(x_val, out_val)
            
        print(f"[Classical] Train PSNR: {psnr_train:.2f} dB | Val PSNR: {psnr_val:.2f} dB")
        
        # 2. Quantum Model (SOTA V3)
        # L=8, G=4, Q=6, Arctan
        qcnn = ModelWrapper(QCNN_SOTA(C, emb, n_layers=8, n_groups=4, n_qubits=6, encoding_type='arctan')).to(device)
        print(f"[Quantum SOTA] Parameters: {count_parameters(qcnn)}")
        opt = optim.Adam(qcnn.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
        
        print("Training Quantum Model...")
        for i in range(300):
            opt.zero_grad()
            out = qcnn(x_train, t_train)
            loss = F.mse_loss(out, x_train)
            loss.backward()
            opt.step()
            scheduler.step()
            
        with torch.no_grad():
            out_train = qcnn.model(x_train, t_train)
            psnr_train = psnr(x_train, out_train)
            out_val = qcnn.model(x_val, t_val)
            psnr_val = psnr(x_val, out_val)
            
        print(f"[Quantum SOTA] Train PSNR: {psnr_train:.2f} dB | Val PSNR: {psnr_val:.2f} dB")

if __name__ == "__main__":
    # set_seed(42)
    # run_benchmark()
    # run_benchmark_multi_res()
    # run_benchmark_overfitting()
    # run_benchmark_real_denoising()
    run_benchmark_real_reconstruction()
