import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_attn_iso import QuantumFusedBlock

def check_gradients():
    print("Checking gradients for QuantumFusedBlock...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = QuantumFusedBlock(channels=128, emb_dim=128).to(device)
    except Exception as e:
        print(f"Failed to init model: {e}")
        return

    model.fe.reupload_data = True
    
    x = torch.randn(2, 128, 16, 16).to(device)
    emb = torch.randn(2, 128).to(device)
    
    try:
        out = model(x, emb)
        loss = out.mean()
        loss.backward()
    except Exception as e:
        print(f"Forward/Backward failed: {e}")
        return
    
    print("\n--- Gradient Report ---")
    
    has_grad = False
    for name, param in model.fe.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(f"QCNN {name}: Grad Norm = {param.grad.norm().item():.6f}")
                has_grad = True
            else:
                print(f"QCNN {name}: Grad is None!")
    
    if not has_grad:
        print("WARNING: No gradients in QCNN part!")

    for name, param in model.qattn.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if 'enc_w' in name or 'q_w' in name:
                    print(f"Attn {name}: Grad Norm = {param.grad.norm().item():.6f}")
            else:
                 # Ignore some params that might not be used
                 pass

if __name__ == "__main__":
    check_gradients()
