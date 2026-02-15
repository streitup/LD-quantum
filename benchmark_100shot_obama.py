
import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import math

# Add training codes to path
sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))

try:
    from training.networks import UNetBlock
    from training.dataset import ImageFolderDataset
    from training.quantum_transformer import QuantumFrontEndQCNNState, QuantumAttentionState
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Metrics Implementation ---

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(img1, img2, window_size=11, size_average=True):
    # Simplified SSIM implementation for PyTorch
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
        # Use VGG16 features as a proxy for LPIPS if lpips library is missing
        try:
            vgg = models.vgg16(pretrained=True).features.to(device)
            # Freeze parameters
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.layers = [3, 8, 15, 22, 29] # Relu layers
            self.eval()
        except Exception as e:
            print(f"Warning: Could not load VGG for LPIPS ({e}). LPIPS will be 0.")
            self.vgg = None

    def forward(self, img1, img2):
        if self.vgg is None:
            return torch.tensor(0.0).to(img1.device)
        
        # Upscale to 64x64 to avoid VGG pooling errors on 16x16 inputs
        if img1.shape[2] < 64:
            img1 = F.interpolate(img1, size=(64, 64), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Normalize to VGG input requirements roughly
        # Assuming img is [0, 1], VGG expects specific mean/std but for perceptual loss 
        # simple centering is often "good enough" for proxy or just passing as is.
        # Let's do simple normalization.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
        
        # If images are 1 channel, repeat to 3
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
    # Dataset Path
    dataset_path = "/home/zzn/qfl_tq/ffhq_workspace/100-shot-obama-128.zip"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset zip not found at {dataset_path}")
        return None
    
    print(f"Loading dataset from {dataset_path}")
    # Load original 128x128 images
    ds = ImageFolderDataset(path=dataset_path, resolution=None) 
    
    # Custom Wrapper to resize to 16x16 (Simulating VAE spatial compression to 3x16x16)
    class WrapperDataset(Dataset):
        def __init__(self, ds, target_res=16):
            self.ds = ds
            self.target_res = target_res
            
        def __len__(self):
            return len(self.ds)
            
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            img = torch.from_numpy(img).float() / 255.0
            # Resize to target_res (e.g., 16x16)
            # Input img is [3, H, W]
            img = F.interpolate(img.unsqueeze(0), size=(self.target_res, self.target_res), mode='bilinear', align_corners=False).squeeze(0)
            return img, label
            
    wrapped_ds = WrapperDataset(ds, target_res=16)
    loader = DataLoader(wrapped_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

# --- Custom Fused Model ---
class FusedQuantumBlock(nn.Module):
    def __init__(self, channels, emb_channels, device='cuda'):
        super().__init__()
        # 1. QCNN Frontend (State Output)
        # Set n_groups=1 to produce a single quantum state [B, L, 2^N] compatible with QAttention
        self.frontend = QuantumFrontEndQCNNState(
            channels=channels,
            style_dim=emb_channels,
            n_layers=8,
            n_groups=1,
            n_qubits_data=4,
            n_qubits_ancilla=2,
            device_name=device,
            reupload_data=True,
            stride=1 # Maintain resolution
        )
        
        # 2. QAttention (State Input)
        # Input State dim is 2^6 = 64.
        self.attn = QuantumAttentionState(
            N_QUBITS=6,   # Matches state dim 64
            qk_dim=16,
            n_heads=4,
            device_name=device
        )
        
        # 3. Output Projection
        # Attention output is [B, S, 64]. We need [B, S, channels].
        self.out_proj = nn.Linear(64, channels)
        
    def forward(self, x, emb):
        B, C, H, W = x.shape
        # x: [B, C, H, W]
        
        # 1. QCNN -> State [B, L, 2^6]
        # Note: QCNN output shape depends on implementation. 
        # QuantumFrontEndQCNNState returns [B, L, D_state]
        state = self.frontend(x, emb)
        
        # 2. Attention -> Classical [B, S, 64]
        # (Assuming S=L)
        attn_out = self.attn(state)
        
        # 3. Project & Reshape
        out = self.out_proj(attn_out) # [B, S, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        # 4. Residual Connection
        out = out + x
        
        return out

# --- Experiment Runner ---

def run_experiment():
    # 1. Configuration
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running strict experiment on {device}")
    
    epochs = 100
    # Reduce batch size to 4 to avoid OOM (Previous OOM at BS=16)
    # Maintain effective batch size (16*4 = 64) -> (4*16 = 64)
    batch_size = 4
    grad_accum = 16
    lr = 1e-4
    
    # 2. Data
    loader = load_data(batch_size, device)
    if loader is None:
        print("CRITICAL: Dataset not found. Aborting strict experiment.")
        return

    # 3. Models Setup
    # C=128, H=W=32 (Assuming Obama 100-shot is usually 32x32 or 64x64, let's adapt)
    # Get first batch to check resolution
    sample_img, _ = next(iter(loader))
    _, C_img, H, W = sample_img.shape
    print(f"Data Resolution: {C_img}x{H}x{W}")
    
    # UNetBlock expects specific channels. We project Image -> C=128
    C_model = 128
    emb_dim = C_model * 4
    
    # Define Models
    models_config = {
        "Classical Baseline": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False,
                "use_quantum_transformer": False,
                "attention": True
            }
        },
        "Quantum Hybrid": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,  # SOTA QCNN (Depth 8)
                "use_quantum_transformer": True, # SOTA QAttn
                "attention": True
            }
        },
        "Quantum Hybrid (MLP Enhanced)": {
             "type": "UNetBlock",
             "kwargs": {
                 "use_qcnn_frontend": True,
                 "use_quantum_transformer": True,
                 "attention": True,
                 "use_mlp_output": True # Enable MLP Enhancement
             }
        },
        "Quantum Fused (No Meas)": {
            "type": "FusedQuantumBlock",
            "kwargs": {
                "device": 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        }
    }
    
    # Input Projection (Shared/Fixed for fair comparison of Backbones)
    input_proj = nn.Conv2d(C_img, C_model, 1).to(device)
    # Output Projection (Back to Image)
    output_proj = nn.Conv2d(C_model, C_img, 1).to(device)
    
    # Metrics
    lpips_metric = LPIPS(device)
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}...")
        
        # Init Model
        if config["type"] == "UNetBlock":
            model = UNetBlock(
                in_channels=C_model,
                out_channels=C_model,
                emb_channels=emb_dim,
                num_heads=4,
                **config["kwargs"]
            ).to(device)
        elif config["type"] == "FusedQuantumBlock":
            model = FusedQuantumBlock(
                channels=C_model,
                emb_channels=emb_dim,
                **config["kwargs"]
            ).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(list(model.parameters()) + list(input_proj.parameters()) + list(output_proj.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Logs
        log_metrics = {"psnr": [], "ssim": [], "lpips": [], "loss": []}
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_psnr = []
            epoch_ssim = []
            epoch_lpips = []
            epoch_loss = []
            
            for i, (clean_img, _) in enumerate(loader):
                clean_img = clean_img.to(device)
                B = clean_img.shape[0]
                
                # Add Noise (Reproducible)
                noise = torch.randn_like(clean_img) * 0.1 # Gaussian Noise sigma=0.1
                noisy_img = clean_img + noise
                
                # Project to Model Space
                x_in = input_proj(noisy_img)
                emb = torch.zeros(B, emb_dim).to(device) # Dummy Time/Class embedding
                
                # Forward
                feat_out = model(x_in, emb)
                
                # Project back
                rec_img = output_proj(feat_out)
                
                # Loss (MSE)
                loss = F.mse_loss(rec_img, clean_img)
                # Keep loss for logging before scaling
                epoch_loss.append(loss.item())
                
                loss = loss / grad_accum
                loss.backward()
                
                if (i + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Metrics (Eval mode roughly)
                with torch.no_grad():
                    # Clamp for metrics
                    rec_clamped = torch.clamp(rec_img, 0, 1)
                    clean_clamped = torch.clamp(clean_img, 0, 1)
                    
                    p = psnr(rec_clamped, clean_clamped)
                    s = ssim(rec_clamped, clean_clamped)
                    l = lpips_metric(rec_clamped, clean_clamped)
                    
                    epoch_psnr.append(p.item())
                    epoch_ssim.append(s.item())
                    epoch_lpips.append(l.item())
            
            scheduler.step()
            
            # Avg Metrics
            avg_psnr = np.mean(epoch_psnr)
            avg_ssim = np.mean(epoch_ssim)
            avg_lpips = np.mean(epoch_lpips)
            avg_loss = np.mean(epoch_loss)
            
            log_metrics["psnr"].append(avg_psnr)
            log_metrics["ssim"].append(avg_ssim)
            log_metrics["lpips"].append(avg_lpips)
            log_metrics["loss"].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
        
        # Save Final
        results[model_name] = {
            "final_psnr": log_metrics["psnr"][-1],
            "final_ssim": log_metrics["ssim"][-1],
            "final_lpips": log_metrics["lpips"][-1],
            "final_loss": log_metrics["loss"][-1],
            "params": sum(p.numel() for p in model.parameters())
        }
        
        del model
        torch.cuda.empty_cache()

    # Comparison Report
    print("\n" + "="*80)
    print("COMPARISON REPORT")
    print("="*80)
    print(f"{'Metric':<15} | {'Classical':<15} | {'Q-Hybrid':<15} | {'Q-MLP':<15} | {'Q-Fused'}")
    print("-" * 80)
    
    c_res = results["Classical Baseline"]
    q_res = results["Quantum Hybrid"]
    qm_res = results["Quantum Hybrid (MLP Enhanced)"]
    f_res = results["Quantum Fused (No Meas)"]
    
    print(f"{'PSNR (dB)':<15} | {c_res['final_psnr']:<15.2f} | {q_res['final_psnr']:<15.2f} | {qm_res['final_psnr']:<15.2f} | {f_res['final_psnr']:<15.2f}")
    print(f"{'SSIM':<15} | {c_res['final_ssim']:<15.4f} | {q_res['final_ssim']:<15.4f} | {qm_res['final_ssim']:<15.4f} | {f_res['final_ssim']:<15.4f}")
    print(f"{'LPIPS':<15} | {c_res['final_lpips']:<15.4f} | {q_res['final_lpips']:<15.4f} | {qm_res['final_lpips']:<15.4f} | {f_res['final_lpips']:<15.4f}")
    print(f"{'Loss (MSE)':<15} | {c_res['final_loss']:<15.6f} | {q_res['final_loss']:<15.6f} | {qm_res['final_loss']:<15.6f} | {f_res['final_loss']:<15.6f}")
    print(f"{'Params':<15} | {c_res['params']:<15} | {q_res['params']:<15} | {qm_res['params']:<15} | {f_res['params']:<15}")
    print("-" * 80)
    
    # Conclusion
    best_psnr = max(c_res['final_psnr'], q_res['final_psnr'], f_res['final_psnr'])
    best_model = [k for k, v in results.items() if v['final_psnr'] == best_psnr][0]
    print(f"CONCLUSION: Best performing model is {best_model}.")

if __name__ == "__main__":
    run_experiment()
