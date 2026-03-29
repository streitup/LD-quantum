import os
import sys
import torch
import torch.nn as nn

# [FIX] Add 'Training Codes of LD-Diffusion' to path
sys.path.append(os.path.join(os.getcwd(), 'Training Codes of LD-Diffusion'))

from training.networks import UNetBlock
from experimental_qcnn import ExperimentalQuantumFrontEnd

# Monkey-Patching
import training.networks
training.networks.QuantumFrontEndQCNN = ExperimentalQuantumFrontEnd

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def run_param_analysis_latent():
    # Config for Latent Diffusion (32x32 Latent Space)
    # Typically Latent Diffusion (LDM) uses 4 channels (from VAE)
    # But inside the UNet, channels grow. 
    # Let's assume a typical middle-block channel count for 32x32 resolution.
    # If base is 64, layer 2 might be 128 or 256.
    # Let's use 256 to simulate a heavy workload in latent space.
    C_model = 256 
    emb_dim = C_model * 4
    device = 'cpu' 
    
    print(f"Configuration: Latent Space 32x32 | Channels: {C_model} | Embedding: {emb_dim}")
    
    # Define Models
    models_config = {
        "Qattn-QDM (Latent)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_head" 
            }
        },
        "w/o G-QCNN (Latent)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": False, # Disabled
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_head"
            }
        },
        "w/o Q-SA (Latent)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": False, # Disabled
                "attention": True, # Fallback to Classic Attention
                "use_mlp_output": True,
                "affine_mode": "q_head"
            }
        },
        "w/o Group (Latent)": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_head",
                "use_grouped_qcnn": False 
            }
        }
    }
    
    print("\n" + "="*80)
    print("PARAMETER COUNT ANALYSIS (Latent Diffusion 32x32)")
    print("="*80)
    print(f"{'Algorithm':<25} | {'Params':<10} | {'Size (MB)':<10} | {'Diff (MB)':<10}")
    print("-" * 80)
    
    baseline_params = 0
    baseline_size = 0
    
    for i, (model_name, config) in enumerate(models_config.items()):
        # Init Model
        # Note: We simulate input resolution implicitly by the block structure, 
        # but param count is resolution-agnostic for Conv/Attention weights.
        model = UNetBlock(
            in_channels=C_model,
            out_channels=C_model,
            emb_channels=emb_dim,
            num_heads=4,
            **config["kwargs"]
        ).to(device)
        
        params = count_params(model)
        size_mb = params * 4 / (1024 * 1024)
        
        if i == 0:
            baseline_params = params
            baseline_size = size_mb
            diff_mb = 0
            diff_str = "-"
        else:
            diff = params - baseline_params
            diff_mb = diff * 4 / (1024 * 1024)
            diff_str = f"{diff_mb:+.2f}"
            
        print(f"{model_name:<25} | {params:<10} | {size_mb:<10.2f} | {diff_str:<10}")

if __name__ == "__main__":
    run_param_analysis_latent()