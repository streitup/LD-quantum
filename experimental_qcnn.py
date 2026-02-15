
import math
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from training.quantum_transformer import QuantumFrontEndQCNN

class ExperimentalQuantumFrontEnd(QuantumFrontEndQCNN):
    """
    Experimental subclass of QuantumFrontEndQCNN to test 5 affine modulation strategies.
    Supports affine_mode: ['q_head', 'q_middle', 'q_tail', 'c_head', 'c_tail']
    """
    def __init__(self, channels: int, style_dim: int, affine_mode: str = 'q_head', **kwargs):
        # Initialize parent with all kwargs
        super().__init__(channels=channels, style_dim=style_dim, **kwargs)
        self.affine_mode = affine_mode
        
        # Classical Affine Layers for C-Head/C-Tail
        if self.affine_mode == 'c_head':
            self.c_head_scale = nn.Linear(style_dim, channels)
            self.c_head_shift = nn.Linear(style_dim, channels)
            # Initialize close to Identity
            nn.init.zeros_(self.c_head_scale.weight)
            nn.init.zeros_(self.c_head_scale.bias)
            nn.init.zeros_(self.c_head_shift.weight)
            nn.init.zeros_(self.c_head_shift.bias)
            
        if self.affine_mode == 'c_tail':
            self.c_tail_scale = nn.Linear(style_dim, channels)
            self.c_tail_shift = nn.Linear(style_dim, channels)
            # Initialize close to Identity
            nn.init.zeros_(self.c_tail_scale.weight)
            nn.init.zeros_(self.c_tail_scale.bias)
            nn.init.zeros_(self.c_tail_shift.weight)
            nn.init.zeros_(self.c_tail_shift.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Algorithm 4: C-Head Affine (Before QCNN)
        # x: [B, C, H, W], style: [B, S]
        if self.affine_mode == 'c_head':
            # Project style to [B, C]
            scale = self.c_head_scale(style).unsqueeze(-1).unsqueeze(-1) + 1.0 # 1 + scale
            shift = self.c_head_shift(style).unsqueeze(-1).unsqueeze(-1)
            x = x * scale + shift
            
        # Run QCNN (Parent Logic)
        out = super().forward(x, style)
        
        # Algorithm 5: C-Tail Affine (After QCNN, before Attention)
        if self.affine_mode == 'c_tail':
            # Project style to [B, C]
            # out has same shape as x: [B, C, H, W]
            scale = self.c_tail_scale(style).unsqueeze(-1).unsqueeze(-1) + 1.0
            shift = self.c_tail_shift(style).unsqueeze(-1).unsqueeze(-1)
            out = out * scale + shift
            
        return out

    def _apply_fusion_circuit(self, qdev, sub_bsz, sub_da, sub_sa, interaction_wires, data_wires, 
                              mod_params, qcnn_rot_params, 
                              n_qubits_data: int, n_qubits_ancilla: int, active_layers: int, 
                              use_strided_cnot: bool, reupload_data: bool, encoding_type: str):
        """
        Modified circuit application to support Q-Head, Q-Middle, Q-Tail injection.
        And enhanced Middle strategies: Q-Middle-Basis, Q-Middle-Ent, Q-Middle-Freq.
        Optimized with parallel gate operations (vectorized tqf calls).
        """
        
        # 1. Determine Injection Points
        inject_head = (self.affine_mode == 'q_head')
        inject_middle = (self.affine_mode == 'q_middle')
        inject_tail = (self.affine_mode == 'q_tail')
        
        # Enhanced Middle Strategies
        inject_middle_basis = (self.affine_mode == 'q_middle_basis')
        inject_middle_ent = (self.affine_mode == 'q_middle_ent')
        inject_middle_freq = (self.affine_mode == 'q_middle_freq')
        
        # Any middle injection
        any_middle_injection = inject_middle or inject_middle_basis or inject_middle_ent or inject_middle_freq
        
        # 2. Encode Data (RY/Amplitude)
        if encoding_type == 'amplitude':
            if inject_head and sub_sa is not None:
                for i in range(n_qubits_data):
                    tqf.ry(qdev, wires=i, params=sub_sa[:, i])
        else:
            # Angle Encoding (RY)
            if inject_head and sub_sa is not None:
                 init_params = sub_da + sub_sa
            else:
                 init_params = sub_da
                 
            for i in range(n_qubits_data):
                tqf.ry(qdev, wires=i, params=init_params[:, i])
        
        # 3. Entanglement (Ancilla -> Data) - Skipped for this ablation (pass)
        
        # 4. Spatial QCNN Backbone
        middle_layer = active_layers // 2
        
        for l in range(active_layers):
            # Q-Middle Injections
            if l == middle_layer and any_middle_injection and sub_sa is not None:
                
                # Standard Q-Middle: RZ Rotation (Additive)
                if inject_middle:
                    for i in range(n_qubits_data):
                        tqf.rz(qdev, wires=i, params=sub_sa[:, i])
                
                # Q-Middle-Basis: Strong Rotation (RY+RZ)
                # Changes the basis significantly
                if inject_middle_basis:
                    for i in range(n_qubits_data):
                        # Use sub_sa for RY and RZ (split or reuse?)
                        # Reuse sub_sa for both for strong effect, or assume sub_sa is enough
                        tqf.ry(qdev, wires=i, params=sub_sa[:, i])
                        tqf.rz(qdev, wires=i, params=sub_sa[:, i])
                
                # Q-Middle-Ent: Controlled Entanglement
                # Use sub_sa to control CRZ gates between neighbors
                if inject_middle_ent:
                    for i in range(n_qubits_data):
                        # Use sub_sa as control parameter for entanglement
                        # CRZ(control=i, target=i+1, theta=sub_sa[i])
                        ctl = i
                        tgt = (i + 1) % n_qubits_data
                        tqf.crz(qdev, wires=[ctl, tgt], params=sub_sa[:, i])

            # Layer Operations
            # Pre-fetch params for this layer
            if qcnn_rot_params.ndim == 5 and qcnn_rot_params.shape[0] == sub_bsz:
                layer_ry = qcnn_rot_params[:, l, :, 0, 0] # [B, N]
                layer_rz = qcnn_rot_params[:, l, :, 1, 0] # [B, N]
            else:
                layer_ry = qcnn_rot_params[l, :, 0, 0].expand(sub_bsz, -1) # [B, N]
                layer_rz = qcnn_rot_params[l, :, 1, 0].expand(sub_bsz, -1) # [B, N]
            
            for i in range(n_qubits_data):
                tqf.ry(qdev, wires=i, params=layer_ry[:, i])
                tqf.rz(qdev, wires=i, params=layer_rz[:, i])
            
            # Ring CNOTs
            for i in range(n_qubits_data):
                tqf.cnot(qdev, wires=[i, (i + 1) % n_qubits_data])
            
            if use_strided_cnot and n_qubits_data >= 4:
                for i in range(n_qubits_data):
                    tqf.cnot(qdev, wires=[i, (i + 2) % n_qubits_data])
            
            if reupload_data and (l < active_layers - 1):
                # Data Re-uploading
                
                # Special Case: Q-Middle-Freq
                # If we are at middle layer and freq mode is on, use Multiplicative re-uploading
                # Note: Re-uploading happens *after* layer l operations.
                # If l == middle_layer - 1, this is the re-uploading *into* the middle layer?
                # Or if l == middle_layer, this is re-uploading *after* middle layer?
                # Let's apply it exactly at the middle cut.
                
                is_middle_reupload = (l == middle_layer)
                
                if inject_middle_freq and is_middle_reupload:
                    # Multiplicative Scaling: Data * Style
                    # sub_da: [B, N], sub_sa: [B, N]
                    # We use sub_sa as a scaling factor. 
                    # Ideally style is centered around 0, so 1+style or exp(style) might be better for scale.
                    # Let's try simple multiplication: params = sub_da * (1 + sub_sa)
                    # This mimics scale * x
                    reup_params = sub_da * (1.0 + sub_sa)
                else:
                    # Standard Additive
                    if inject_head and sub_sa is not None:
                        reup_params = sub_da + sub_sa
                    else:
                        reup_params = sub_da
                    
                for i in range(n_qubits_data):
                    tqf.rz(qdev, wires=i, params=reup_params[:, i])

        # Q-Tail Injection
        if inject_tail and sub_sa is not None:
             for i in range(n_qubits_data):
                    tqf.rz(qdev, wires=i, params=sub_sa[:, i])

