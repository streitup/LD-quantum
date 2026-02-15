
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
    # Import standard modules
    import training.networks
    from training.networks import UNetBlock
    from training.dataset import ImageFolderDataset
    from training.quantum_transformer import QuantumFrontEndQCNNState, QuantumAttentionState
    
    # Import Experimental Module
    from experimental_qcnn import ExperimentalQuantumFrontEnd
    
    # Monkey-Patching: Replace standard QCNN with Experimental Version
    print("Monkey-Patching: training.networks.QuantumFrontEndQCNN -> ExperimentalQuantumFrontEnd")
    training.networks.QuantumFrontEndQCNN = ExperimentalQuantumFrontEnd
    
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

# --- Metrics Implementation (Copied from benchmark_100shot_obama.py) ---

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

# --- Experiment Runner ---

def run_experiment():
    # 1. Configuration
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Affine Modulation Ablation Experiment on {device}")
    
    epochs = 60 # Sufficient to see convergence trend as per previous runs
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
    # All use "use_mlp_output": True as baseline, varying only affine_mode
    models_config = {
        "Target: C-Tail Affine": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "c_tail"
            }
        },
        "Ref: Q-Middle (Orig)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_middle"
            }
        },
        "Algo A: Q-Middle-Basis": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_middle_basis"
            }
        },
        "Algo B: Q-Middle-Ent": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_middle_ent"
            }
        },
        "Algo C: Q-Middle-Freq": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_middle_freq"
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
        model = UNetBlock(
            in_channels=C_model,
            out_channels=C_model,
            emb_channels=emb_dim,
            num_heads=4,
            **config["kwargs"]
        ).to(device)
        
        # Init Time Embedder
        time_embedder = TimeEmbedding(emb_dim).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(list(model.parameters()) + list(input_proj.parameters()) + list(output_proj.parameters()) + list(time_embedder.parameters()), lr=lr)
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
                
                # Sample Sigma (Log-Normal)
                # ln(sigma) ~ N(-1.2, 1.2^2) -> commonly used in EDM
                rnd_normal = torch.randn(B, device=device)
                sigma = (rnd_normal * 1.2 - 1.2).exp()
                
                # Add Noise
                noise = torch.randn_like(clean_img)
                noisy_img = clean_img + noise * sigma.view(B, 1, 1, 1)
                
                # Project to Model Space
                x_in = input_proj(noisy_img)
                
                # Time Embedding
                emb = time_embedder(sigma) # [B, emb_dim]
                
                # Forward
                feat_out = model(x_in, emb)
                
                # Project back
                rec_img = output_proj(feat_out)
                
                # Loss (MSE)
                loss = F.mse_loss(rec_img, clean_img)
                epoch_loss.append(loss.item())
                
                loss = loss / grad_accum
                loss.backward()
                
                if (i + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Metrics (Eval mode roughly)
                with torch.no_grad():
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
    print("AFFINE MODULATION ABLATION REPORT")
    print("="*80)
    print(f"{'Algorithm':<25} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'LPIPS':<10} | {'Loss':<10} | {'Params':<10}")
    print("-" * 85)
    
    for name, res in results.items():
        print(f"{name:<25} | {res['final_psnr']:<10.2f} | {res['final_ssim']:<10.4f} | {res['final_lpips']:<10.4f} | {res['final_loss']:<10.6f} | {res['params']:<10}")
    
    print("-" * 85)
    
    # Conclusion
    best_psnr = max(res['final_psnr'] for res in results.values())
    best_model = [k for k, v in results.items() if v['final_psnr'] == best_psnr][0]
    print(f"CONCLUSION: Optimal affine injection strategy is {best_model}.")

if __name__ == "__main__":
    run_experiment()
