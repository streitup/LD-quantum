
import os
import sys
import torch
import torch.nn as nn
from training.networks import UNetBlock
from experimental_qcnn import ExperimentalQuantumFrontEnd

# Monkey-Patching
import training.networks
training.networks.QuantumFrontEndQCNN = ExperimentalQuantumFrontEnd

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def run_param_analysis():
    # Config
    C_model = 128
    emb_dim = C_model * 4
    device = 'cpu' # Sufficient for param counting
    
    # Define Models
    models_config = {
        "Algo 1: Q-Head Affine": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_head"
            }
        },
        "Algo 2: Q-Middle Affine": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_middle"
            }
        },
        "Algo 3: Q-Tail Affine": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "q_tail"
            }
        },
        "Algo 4: C-Head Affine": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "c_head"
            }
        },
        "Algo 5: C-Tail Affine": {
            "type": "UNetBlock",
            "kwargs": {
                "use_qcnn_frontend": True,
                "use_quantum_transformer": True,
                "attention": True,
                "use_mlp_output": True,
                "affine_mode": "c_tail"
            }
        }
    }
    
    print("\n" + "="*60)
    print("PARAMETER COUNT ANALYSIS (Affine Modulation Strategies)")
    print("="*60)
    print(f"{'Algorithm':<25} | {'Total Params':<15} | {'Diff vs Q-Head':<15}")
    print("-" * 60)
    
    baseline_params = 0
    
    for i, (model_name, config) in enumerate(models_config.items()):
        # Init Model
        model = UNetBlock(
            in_channels=C_model,
            out_channels=C_model,
            emb_channels=emb_dim,
            num_heads=4,
            **config["kwargs"]
        ).to(device)
        
        params = count_params(model)
        
        if i == 0:
            baseline_params = params
            diff = 0
        else:
            diff = params - baseline_params
            
        print(f"{model_name:<25} | {params:<15} | {diff:<+15}")
        
    print("-" * 60)

if __name__ == "__main__":
    run_param_analysis()
