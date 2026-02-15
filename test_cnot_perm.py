
import torch
import sys
import os
sys.path.append('/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion')
from training.quantum_transformer import QuantumFrontEndQCNN

class MockQDev:
    def __init__(self, bsz, n_wires):
        self.bsz = bsz
        self.n_wires = n_wires
        dim = 2**n_wires
        # [B, 2**N]
        self.states = torch.randn(bsz, dim, dtype=torch.cfloat, device='cuda')

def test_cnot_perm_fix():
    print("Testing CNOT Permutation Fix...")
    
    # Setup
    bsz = 2
    n_data = 2
    n_ancilla = 1
    n_wires = n_data + n_ancilla
    dim_data = 2**n_data
    dim_ancilla = 2**n_ancilla
    
    qdev = MockQDev(bsz, n_wires)
    print(f"Original states shape: {qdev.states.shape}")
    
    # Instantiate QCNN (dummy args)
    qcnn = QuantumFrontEndQCNN(channels=4, style_dim=4, n_qubits_data=n_data, n_qubits_ancilla=n_ancilla, device_name='cuda')
    qcnn.to('cuda')
    
    # Try calling _fast_cnot_layer
    try:
        qcnn._fast_cnot_layer(qdev, n_data, use_strided=False)
        print("Success! _fast_cnot_layer passed.")
    except Exception as e:
        print(f"Failed! Error: {e}")
        # Expected to fail if not fixed because of reshape(bsz, dim) where dim=2**n_data != 2**n_wires

if __name__ == "__main__":
    test_cnot_perm_fix()
