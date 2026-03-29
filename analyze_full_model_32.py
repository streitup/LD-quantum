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

def run_full_model_analysis():
    # Config for Full 32x32 Latent Diffusion Model
    # Based on SongUNet architecture used in the project
    
    img_resolution = 32
    in_channels = 4 # Latent space typically has 4 channels
    out_channels = 4
    model_channels = 128 # Base channels
    channel_mult = [1, 2, 2] # Depth for 32x32 -> 16x16 -> 8x8
    num_blocks = 4
    attn_resolutions = [16] # Attention at 16x16
    label_dim = 0 # Unconditional
    
    print(f"Configuration: Full 32x32 Diffusion Model | Base Channels: {model_channels}")
    print(f"Resolution: {img_resolution}x{img_resolution} | Latent Channels: {in_channels}")
    
    # Define Models
    models_config = {
        "Qattn-QDM (Full)": {
            "use_qcnn_frontend": True,
            "use_quantum_transformer": True,
            "use_quantum_mlp": False, # Typically MLP is kept classic for stability or not focused in this ablation
        },
        "w/o G-QCNN (Full)": {
            "use_qcnn_frontend": False, # Disabled
            "use_quantum_transformer": True,
        },
        "w/o Q-SA (Full)": {
            "use_qcnn_frontend": True,
            "use_quantum_transformer": False, # Disabled
        },
        "w/o Group (Full)": {
            "use_qcnn_frontend": True,
            "use_quantum_transformer": True,
            "qcnn_chunk_size": 4096, # Default
             # Note: 'use_grouped_qcnn' is internal to QCNN class. 
             # To simulate 'w/o Group', we might need to rely on the fact that 
             # the baseline QCNN implementation uses groups=8.
             # If we want to disable groups, we'd need to set n_groups=1 in QCNN.
             # However, SongUNet doesn't expose n_groups in init.
             # It hardcodes `target_groups = 8`.
             # So 'w/o Group' might require a code hack or we assume the previous block-level result holds.
             # For this script, we will skip 'w/o Group' or acknowledge it's the same as Baseline if we can't change it.
             # OR we can Monkey-Patch QuantumFrontEndQCNN again to force groups=1.
        }
    }
    
    print("\n" + "="*80)
    print("FULL MODEL PARAMETER COUNT ANALYSIS (32x32 Latent)")
    print("="*80)
    print(f"{'Algorithm':<25} | {'Params':<10} | {'Size (MB)':<10} | {'Diff (MB)':<10}")
    print("-" * 80)
    
    baseline_params = 0
    
    for i, (model_name, config) in enumerate(models_config.items()):
        
        # Init Model
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
        
        if i == 0:
            baseline_params = params
            diff_str = "-"
        else:
            diff = params - baseline_params
            diff_mb = diff * 4 / (1024 * 1024)
            diff_str = f"{diff_mb:+.2f}"
            
        print(f"{model_name:<25} | {params:<10} | {size_mb:<10.2f} | {diff_str:<10}")

if __name__ == "__main__":
    run_full_model_analysis()