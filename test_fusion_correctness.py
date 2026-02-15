
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import sys
import os

sys.path.append('/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion')
from training.quantum_transformer import QuantumFrontEndQCNN

def test_fusion_correctness():
    # Setup
    B = 2
    n_qubits_data = 4
    n_qubits_ancilla = 2
    total_qubits = n_qubits_data + n_qubits_ancilla
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create Model
    model = QuantumFrontEndQCNN(
        channels=8, 
        style_dim=8, 
        n_qubits_data=n_qubits_data, 
        n_qubits_ancilla=n_qubits_ancilla,
        n_layers=1,
        n_groups=1,
        device_name=str(device)
    ).to(device)
    
    # Mock Data
    qdev = tq.QuantumDevice(n_wires=total_qubits, bsz=B, device=device)
    
    # Random Params
    sub_da = torch.randn(B, n_qubits_data, device=device)
    sub_sa = torch.randn(B, n_qubits_data, device=device)
    
    # QCNN Rot Params: [B, n_layers, n_qubits, 2, 1] or [n_layers, n_qubits, 2, 1]
    # Let's use batch params to test broadcasting
    qcnn_rot_params = torch.randn(B, 1, n_qubits_data, 2, 1, device=device)
    
    # Ancilla Interaction
    interaction_wires = list(range(n_qubits_data, total_qubits)) # [4, 5]
    data_wires = list(range(n_qubits_data)) # [0, 1, 2, 3]
    mod_params = torch.randn(B, 1, n_qubits_data, 3, device=device) # [B, layers, data, 3]
    
    # --- 1. Run Sequential (Golden Reference) ---
    
    # Reset state
    qdev.reset_states(B)
    
    # Sequential Logic
    # Init (RY)
    init_params = sub_da + sub_sa
    for i in range(n_qubits_data):
        tqf.ry(qdev, wires=i, params=init_params[:, i])
        
    # Entanglement
    for i in range(n_qubits_data):
        ancilla_idx = i % n_qubits_ancilla
        ctl = interaction_wires[ancilla_idx]
        tgt = data_wires[i]
        
        strength = mod_params[:, 0, i, 0]
        
        if ancilla_idx % 2 == 0:
            tqf.crx(qdev, wires=[ctl, tgt], params=strength)
        else:
            tqf.crz(qdev, wires=[ctl, tgt], params=strength)
            
    # Backbone
    ry_params = qcnn_rot_params[:, 0, :, 0, 0]
    rz_params = qcnn_rot_params[:, 0, :, 1, 0]
    
    # Rot Layer
    for i in range(n_qubits_data):
        tqf.ry(qdev, wires=i, params=ry_params[:, i])
        tqf.rz(qdev, wires=i, params=rz_params[:, i])
        
    # CNOT Layer
    for i in range(n_qubits_data):
        tqf.cnot(qdev, wires=[i, (i + 1) % n_qubits_data])
        
    state_seq = qdev.get_states_1d().clone()
    
    # --- 2. Run Fused ---
    qdev.reset_states(B)
    
    model._apply_fusion_circuit(
        qdev=qdev,
        sub_bsz=B,
        sub_da=sub_da,
        sub_sa=sub_sa,
        interaction_wires=interaction_wires,
        data_wires=data_wires,
        mod_params=mod_params,
        qcnn_rot_params=qcnn_rot_params,
        n_qubits_data=n_qubits_data,
        n_qubits_ancilla=n_qubits_ancilla,
        active_layers=1,
        use_strided_cnot=False,
        reupload_data=False,
        encoding_type='angle'
    )
    
    state_fused = qdev.get_states_1d().clone()
    
    # Compare
    # Fidelity
    # fidelity = tq.utils.pqc_fidelity(state_seq, state_fused)
    # Manual fidelity: |<psi|phi>|^2
    inner_prod = (state_seq.conj() * state_fused).sum(dim=-1)
    fidelity = inner_prod.abs() ** 2
    print(f"Fidelity: {fidelity.mean().item()}")
    
    if not torch.allclose(state_seq, state_fused, atol=1e-5):
        print("Mismatch found!")
        print("Seq:", state_seq[0, :10])
        print("Fused:", state_fused[0, :10])
        diff = (state_seq - state_fused).abs().max()
        print(f"Max Diff: {diff}")
    else:
        print("Verification Passed!")

if __name__ == "__main__":
    try:
        test_fusion_correctness()
    except Exception as e:
        import traceback
        traceback.print_exc()
