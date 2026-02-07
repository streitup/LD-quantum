
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Add training codes to path
sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))

# Import UNetBlock and other necessary modules
try:
    from training.networks import UNetBlock
    from training.dataset import ImageFolderDataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_real_features(B, C, H, W, device):
    """
    Load real images, downsample to HxW, and project to C channels.
    Simulates the input features to a UNet Block.
    """
    try:
        # Try to find the dataset zip
        dataset_path = os.path.join(os.getcwd(), '100-shot-obama.zip')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion', '100-shot-obama.zip')
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Dataset zip not found")

        ds = ImageFolderDataset(path=dataset_path, resolution=None)
        
        # Load B images
        images = []
        for i in range(B):
            idx = i % len(ds)
            img_np, _ = ds[idx]
            images.append(torch.from_numpy(img_np))
        
        images = torch.stack(images).float() / 255.0 # [B, 3, H_orig, W_orig]
        images = images.to(device)
        
        # Downsample to HxW
        features = torch.nn.functional.interpolate(images, size=(H, W), mode='bilinear', align_corners=False)
        
        # Project to C channels (Random but fixed projection to simulate previous layers)
        projector = nn.Conv2d(3, C, kernel_size=1, bias=False).to(device)
        torch.manual_seed(42) # Deterministic projection
        projector.weight.data = torch.randn_like(projector.weight.data) * 0.1
        
        x_clean = projector(features).detach() # [B, C, H, W]
        
        # Create Embedding (Time/Class)
        # UNetBlock expects emb_channels
        emb_channels = C * 4 # Common ratio
        emb = torch.randn(B, emb_channels).to(device)
        
        print("Successfully loaded Real Image Features.")
        return x_clean, emb
        
    except Exception as e:
        print(f"Warning: Could not load real data ({e}). Using Synthetic Features.")
        # Fallback
        x_clean = torch.randn(B, C, H, W, device=device)
        emb = torch.randn(B, C * 4, device=device)
        return x_clean, emb

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Configuration
    B = 2
    C = 128
    H = 16
    W = 16
    emb_channels = C * 4
    
    # Prepare Data
    x_clean, emb = get_real_features(B, C, H, W, device)
    
    # Add Noise for Denoising Task
    noise = torch.randn_like(x_clean) * 0.1
    x_input = x_clean + noise
    target = x_clean # Goal: Recover clean features (Denoising)
    
    # Define 4 Configurations
    configs = [
        {
            "name": "1. Enhanced QCNN (8 Layers) + SOTA Q-Attn",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "quantum_adapter": None, # Use default SOTA
                "quantum_adapter_kwargs": {"lora_rank": 4}, # Optimize params
                "attention": True
            }
        },
        {
            "name": "2. Enhanced QCNN (8 Layers) + Classic Attn",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": False,
                "attention": True
            }
        },
        {
            "name": "3. Classic Conv + SOTA Q-Attn",
            "kwargs": {
                "use_qcnn_frontend": False,
                "use_quantum_transformer": True,
                "quantum_adapter": None, # Use default SOTA
                "quantum_adapter_kwargs": {"lora_rank": 4}, # Optimize params
                "attention": True
            }
        },
        {
            "name": "4. Classic Conv + Classic Attn",
            "kwargs": {
                "use_qcnn_frontend": False,
                "use_quantum_transformer": False,
                "attention": True
            }
        }
    ]
    
    results = []
    
    print("\nStarting UNetBlock Benchmark...")
    print(f"Input: [{B}, {C}, {H}, {W}]")
    
    criterion = nn.MSELoss()
    
    for config in configs:
        print(f"\nEvaluating: {config['name']}")
        try:
            # Instantiate UNetBlock
            model = UNetBlock(
                in_channels=C,
                out_channels=C,
                emb_channels=emb_channels,
                num_heads=4,
                **config['kwargs']
            ).to(device)
            
            # Count Params
            params = count_parameters(model)
            
            # Warmup
            _ = model(x_input, emb)
            
            # Benchmark Loop
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            steps = 100
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            final_loss = 0.0
            
            for step in range(steps):
                optimizer.zero_grad()
                out = model(x_input, emb)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
                
                if step % 5 == 0:
                    print(f"  Step {step}: Loss {final_loss:.6f}")
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / steps
            
            results.append({
                "name": config['name'],
                "params": params,
                "loss": final_loss,
                "time": avg_time
            })
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            
    # Print Summary
    print("\n" + "="*85)
    print(f"{'Model':<35} | {'Params':<10} | {'Time/Step (s)':<15} | {'Loss (MSE)':<10}")
    print("-" * 85)
    for res in results:
        print(f"{res['name']:<35} | {res['params']:<10} | {res['time']:<15.4f} | {res['loss']:<10.6f}")
    print("=" * 85)

if __name__ == "__main__":
    run_benchmark()
