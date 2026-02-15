
import os
import sys
import time
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import math
import importlib

# Add training codes to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Training Codes of LD-Diffusion'))

try:
    # Import standard modules
    import training.networks
    from training.networks import UNetBlock
    from training.dataset import ImageFolderDataset
    # Import Quantum Classes including SOTA
    from training.quantum_transformer import (
        QuantumFrontEndQCNN, 
        QuantumAttention64, 
        QuantumAttentionHybridLite, 
        QuantumAttentionAngle, 
        QuantumAttentionAngleDense,
        QuantumAttentionPatch
    )
    
    # Try importing torchquantum for custom classes
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

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

# --- Metrics Implementation ---

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
            vgg = models.vgg16(pretrained=True).features.to(device)
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

# --- Experiment Runner ---

def run_experiment():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running SOTA Ablation Experiment on {device}")
    
    epochs = 50 # Long training
    batch_size = 8 # Decreased batch size to avoid OOM
    lr = 2e-4 # Slightly higher LR
    
    loader = load_data(batch_size, device)
    if loader is None:
        return

    sample_img, _ = next(iter(loader))
    _, C_img, H, W = sample_img.shape
    print(f"Data Resolution: {C_img}x{H}x{W}")
    
    # Use C=64 to match standard Quantum Amplitude Encoding (2^6=64)
    C_model = 64
    emb_dim = C_model * 4
    
    # Models Configuration
    # All use "use_qcnn_frontend": False (Classic Frontend for Speed)
    # Varying Attention
    models_config = {
        "0. Pure Classic (Classic Attn)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": False, # Classic Attention
                "attention": True,
                "num_heads": 4,
                "use_mlp_output": True,
            }
        },
        "1. QSANN (Amplitude, 4-Head)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                # Explicitly use QuantumAttention64 (Amplitude Encoding)
                "quantum_adapter": QuantumAttention64(
                    in_channels=C_model,
                    num_heads=4,
                    N_QUBITS=6,
                    Q_DEPTH=4,
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        },
        "2. QSANN (Amplitude, 1-Head)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                # Explicitly use QuantumAttention64 (Amplitude Encoding) with 1 Head
                "quantum_adapter": QuantumAttention64(
                    in_channels=C_model,
                    num_heads=1,
                    N_QUBITS=6,
                    Q_DEPTH=4,
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        },
        "3. SOTA (Dense Angle)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                # Use SOTA QuantumAttentionAngleDense
                "quantum_adapter": QuantumAttentionAngleDense(
                    in_channels=C_model,
                    num_heads=4,
                    N_QUBITS=6,
                    Q_DEPTH=12, # Deep Circuit as per SOTA specs (Increased to 12)
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        },
        "4. SOTA (Dense, 8 Qubits)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                # Use SOTA QuantumAttentionAngleDense with 8 Qubits
                "quantum_adapter": QuantumAttentionAngleDense(
                    in_channels=C_model,
                    num_heads=4,
                    N_QUBITS=8, # Increased Qubits
                    Q_DEPTH=12, 
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        },
        "5. SOTA (Hybrid Lite)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "num_heads": 4,
                "use_mlp_output": True,
                "quantum_adapter": QuantumAttentionHybridLite(
                    input_dim=C_model,
                    num_heads=4,
                    N_QUBITS=6, # Standard 6 Qubits for Hybrid
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        },
        "6. SOTA (Dense, Depth=4)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                # Use SOTA QuantumAttentionAngleDense with Reduced Depth
                "quantum_adapter": QuantumAttentionAngleDense(
                    in_channels=C_model,
                    num_heads=4,
                    N_QUBITS=6,
                    Q_DEPTH=4, # Reduced Depth (Baseline QSANN depth)
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        },
        "7. SOTA (Dense, No Grouped)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                # Use SOTA QuantumAttentionAngleDense with Standard Linear
                "quantum_adapter": QuantumAttentionAngleDense(
                    in_channels=C_model,
                    num_heads=4,
                    N_QUBITS=6,
                    Q_DEPTH=12,
                    use_grouped_linear=False, # Disable Grouped Linear
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        },
        "8. Refined SOTA (Patch)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Classic Frontend
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                # Use Refined QuantumAttentionPatch
                "quantum_adapter": QuantumAttentionPatch(
                    dim=C_model,
                    num_heads=4,
                    q_depth=2, 
                    n_qubits=7,
                    patch_size=2,
                    device_name=str(device).replace('cuda', 'cuda:0') if device.type == 'cuda' else 'cpu'
                )
            }
        }
    }
    
    results = {}
    
    lpips_fn = LPIPS(device)
    
    for name, config in models_config.items():
        print(f"\n--- Training Model: {name} ---")
        
        # Instantiate Model
        # Input Projection to C_model
        input_proj = nn.Conv2d(C_img, C_model, kernel_size=3, padding=1).to(device)
        
        # UNet Block (The Model Under Test)
        block = UNetBlock(
            in_channels=C_model,
            out_channels=C_model,
            emb_channels=emb_dim,
            **config['kwargs']
        ).to(device)
        
        # Output Projection to RGB
        output_proj = nn.Conv2d(C_model, C_img, kernel_size=3, padding=1).to(device)
        
        # Time Embedding
        time_embed = TimeEmbedding(emb_dim).to(device)
        
        optimizer = torch.optim.AdamW(
            list(input_proj.parameters()) + list(block.parameters()) + list(output_proj.parameters()) + list(time_embed.parameters()),
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Count Parameters
        total_params = sum(p.numel() for p in block.parameters())
        # Filter only quantum adapter params if possible to show specific overhead
        if hasattr(block, 'quantum_adapter') and block.quantum_adapter is not None:
             qa_params = sum(p.numel() for p in block.quantum_adapter.parameters())
             print(f"  Quantum Adapter Params: {qa_params}")
        print(f"  Total Block Params: {total_params}")
        
        # Training Loop
        loss_history = []
        start_time = time.time()
        
        best_loss = float('inf')
        patience = 50 # Increase patience for 50 epochs
        patience_counter = 0
        
        # Track best metrics
        best_psnr = 0
        best_ssim = 0
        best_lpips = float('inf')
        
        for epoch in range(epochs):
            input_proj.train()
            block.train()
            output_proj.train()
            time_embed.train()
            
            epoch_loss = 0
            steps = 0
            
            for i, (imgs, _) in enumerate(loader):
                imgs = imgs.to(device) # [B, 3, 16, 16]
                
                # Add noise
                noise = torch.randn_like(imgs) * 0.1
                noisy_imgs = imgs + noise
                
                # Time embedding (random t)
                t = torch.randint(0, 1000, (imgs.shape[0],), device=device).float()
                t_emb = time_embed(t)
                
                # Forward
                x = input_proj(noisy_imgs)
                x = block(x, t_emb)
                pred_imgs = output_proj(x)
                
                # Loss (Reconstruction)
                loss = F.mse_loss(pred_imgs, imgs)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
            
            avg_loss = epoch_loss / steps
            loss_history.append(avg_loss)
            scheduler.step()
            
            # Validation (On Training Set for Convergence Check)
            with torch.no_grad():
                # Reconstruct one batch (last batch)
                t_val = torch.zeros(imgs.shape[0], device=device).float()
                t_emb_val = time_embed(t_val)
                x_val = input_proj(imgs) # Clean input for pure reconstruction check
                x_val = block(x_val, t_emb_val)
                recon_val = output_proj(x_val)
                
                recon_val = torch.clamp(recon_val, 0, 1)
                imgs_val = torch.clamp(imgs, 0, 1)
                
                val_psnr = psnr(recon_val, imgs_val).item()
                val_ssim = ssim(recon_val, imgs_val).item()
                val_lpips = lpips_fn(recon_val, imgs_val).item() * 1000
                
                if val_psnr > best_psnr: best_psnr = val_psnr
                if val_ssim > best_ssim: best_ssim = val_ssim
                if val_lpips < best_lpips: best_lpips = val_lpips
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")
            
            # Early Stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
            
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"  [Result] PSNR: {best_psnr:.2f} | SSIM: {best_ssim:.4f} | LPIPS: {best_lpips:.2f}")
        print(f"  [Perf] Time: {training_time:.2f}s")
            
        results[name] = {
            "Params": total_params,
            "PSNR": best_psnr,
            "SSIM": best_ssim,
            "LPIPS": best_lpips,
            "Time": training_time
        }
        
        # Save images
        try:
            import torchvision.utils as vutils
            comparison = torch.cat([imgs_val, recon_val], dim=0)
            vutils.save_image(comparison, f"benchmark_{name.split(':')[0].strip().replace(' ', '_')}.png", nrow=batch_size)
        except:
            pass

    # Print Summary Table
    print("\n\n=== Final Benchmark Results (Converged) ===")
    print(f"{'Model':<25} | {'Params':<10} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8} | {'Time (s)':<8}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<25} | {res['Params']:<10} | {res['PSNR']:<8.2f} | {res['SSIM']:<8.4f} | {res['LPIPS']:<8.2f} | {res['Time']:<8.2f}")

if __name__ == "__main__":
    run_experiment()
