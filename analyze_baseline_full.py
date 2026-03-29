import os
import sys
import torch
import torch.nn as nn

# [FIX] Add 'Training Codes of LD-Diffusion' to path
sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))

from training.networks import SongUNet
from experimental_qcnn import ExperimentalQuantumFrontEnd

# Monkey-Patching
import training.networks
training.networks.QuantumFrontEndQCNN = ExperimentalQuantumFrontEnd

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def run_baseline_analysis():
    # Config for Full 32x32 Latent Diffusion Model
    img_resolution = 32
    in_channels = 4
    out_channels = 4
    model_channels = 128
    channel_mult = [1, 2, 2]
    num_blocks = 4
    attn_resolutions = [16]
    label_dim = 0
    
    print(f"Configuration: Baseline (w/o G-QCNN & w/o Q-SA) | 32x32 Latent | Base: {model_channels}")
    
    # Define Model: Baseline (Pure Classical)
    config = {
        "use_qcnn_frontend": False, # Disabled
        "use_quantum_transformer": False, # Disabled
    }
    
    model = SongUNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        label_dim=label_dim,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        **config
    )
    
    params = count_params(model)
    size_mb = params * 4 / (1024 * 1024)
    
    print("\n" + "="*60)
    print("BASELINE MODEL ANALYSIS (w/o G-QCNN & w/o Q-SA)")
    print("="*60)
    print(f"{'Algorithm':<25} | {'Params':<10} | {'Size (MB)':<10}")
    print("-" * 60)
    print(f"{'Baseline (Pure Classic)':<25} | {params:<10} | {size_mb:<10.2f}")

if __name__ == "__main__":
    run_baseline_analysis()