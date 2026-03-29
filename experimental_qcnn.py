
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from training.quantum_transformer import QuantumFrontEndQCNN

class ExperimentalQuantumFrontEnd(QuantumFrontEndQCNN):
    """
    Experimental subclass of QuantumFrontEndQCNN to test 5 affine modulation strategies.
    Supports affine_mode: ['q_head', 'q_middle', 'q_tail', 'c_head', 'c_tail']
    """
    def __init__(self, channels: int, style_dim: int, affine_mode: str = 'q_head', encoding_type: str = 'angle', use_parameter_free_residual: bool = False, use_lightweight_residual: bool = False, use_trainable_reupload: bool = False, ansatz_type: str = 'basic', use_u3_gates: bool = False, residual_mode: str = 'default', **kwargs):
        # Initialize parent with all kwargs
        super().__init__(channels=channels, style_dim=style_dim, **kwargs)
        self.affine_mode = affine_mode
        self.encoding_type = encoding_type
        self.ansatz_type = ansatz_type 
        self.use_u3_gates = use_u3_gates
        self.residual_mode = residual_mode # 'default', 'global', 'scalar', 'gaussian', 'laplacian'
        
        # --- NEW: Learnable Scalar Residual ---
        if self.residual_mode == 'scalar':
             self.res_scale = nn.Parameter(torch.ones(1) * 1.0) 
        
        # --- NEW: Fixed Filter Definitions ---
        if self.residual_mode == 'gaussian':
             # 3x3 Gaussian Blur Filter
             kernel = torch.tensor([[1., 2., 1.],
                                    [2., 4., 2.],
                                    [1., 2., 1.]]) / 16.0
             # Expand to [C, 1, 3, 3] for grouped conv (depthwise)
             self.register_buffer('gaussian_kernel', kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
             
        elif self.residual_mode == 'laplacian':
             # 3x3 Laplacian Filter (Edge Detection)
             # Center 8, neighbors -1 (Standard Laplacian) or 4/-1
             kernel = torch.tensor([[-1., -1., -1.],
                                    [-1., 8., -1.],
                                    [-1., -1., -1.]])
             self.register_buffer('laplacian_kernel', kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        
        # --- NEW: Trainable Re-uploading Scaling ---
        if use_trainable_reupload:
            # Add trainable scale/bias for re-uploading layers
            # Shape: [n_layers, n_qubits_data, 2] (RY, RZ) or just [n_layers, n_qubits_data] if only RZ
            # Assuming active_layers is known or max layers
            self.reupload_weights = nn.Parameter(torch.ones(self.n_layers, self.n_qubits_data, 2))
            self.reupload_bias = nn.Parameter(torch.zeros(self.n_layers, self.n_qubits_data, 2))
            self.use_trainable_reupload = True
        else:
            self.use_trainable_reupload = False
            
        # --- NEW: U3 Gate Parameters ---
        if self.use_u3_gates:
             # U3(theta, phi, lam) requires 3 parameters per qubit per layer.
             # qcnn_rot_params provides [layers, qubits, 2, 3] usually (RY, RZ params).
             # But here we are generating params from a linear layer self.rot_proj.
             # Parent class init defines self.rot_proj outputting `n_layers * n_qubits * 2 * 3`?
             # Let's check parent class.
             pass 
             # We will handle U3 parameter extraction in _apply_fusion_circuit.
             # Assuming standard rot_proj is sufficient (it outputs a large vector).
             # We might need to ensure rot_proj dimension is compatible.
             # Standard rot_proj: n_layers * n_qubits * 2 (params) * 3 (colors?) No, usually 2 params.
             # Actually QuantumFrontEndQCNN defines n_rotations = n_layers * n_qubits * 2.
             # If U3 needs 3 params, we need 50% more parameters.
             # This requires modifying the parent class or overwriting self.rot_proj.
             
             # Re-define rot_proj for U3 (3 params per gate)
             # Parent: self.n_rotations = self.n_layers * self.n_qubits_data * 2
             # U3: self.n_rotations = self.n_layers * self.n_qubits_data * 3
             
             self.n_rotations_u3 = self.n_layers * self.n_qubits_data * 3
             self.rot_proj = nn.Linear(self.patch_dim, self.n_rotations_u3)
            
        # --- NEW: Residual Options ---
        if use_lightweight_residual:
              # Lightweight: Global Avg Pool (Patch) + Linear (C->C)
              # Parameters: C*C (much less than C*K*K*C)
              self.res_proj = nn.Linear(self.channels, self.channels)
              self.use_lightweight_residual = True
              self.use_param_free_bypass = False
        elif use_parameter_free_residual:
              # Instead of a fixed linear layer, we'll use a custom parameter-free bypass in forward()
              # We disable the trainable res_proj
              self.res_proj = nn.Identity() # Placeholder, not used in logic below
              self.use_param_free_bypass = True
              self.use_lightweight_residual = False
        else:
              self.use_param_free_bypass = False
              self.use_lightweight_residual = False
              
        # Amplitude Encoding Projection (for grouped input)
        if self.encoding_type == 'amplitude':
            # Map patch_dim_per_group -> 2^n_qubits_data
            self.amp_proj = nn.Linear(self.patch_dim_per_group, 2 ** self.n_qubits_data)
        
        # Override out_proj to match n_qubits_data (ignoring ancilla wires if any)
        # This ensures compatibility with the "No Ancilla" logic requested by user.
        self.out_proj = nn.Linear(2 ** self.n_qubits_data, self.channels_per_group)
        
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
        B, C, H, W = x.shape
        
        # Algorithm 4: C-Head Affine (Before QCNN)
        if self.affine_mode == 'c_head':
            scale = self.c_head_scale(style).unsqueeze(-1).unsqueeze(-1) + 1.0
            shift = self.c_head_shift(style).unsqueeze(-1).unsqueeze(-1)
            x = x * scale + shift

        # 1. Unfold & Reshape
        patches = self.unfold(x) # [B, patch_dim, L]
        L = patches.shape[-1]
        patches_flat = patches.transpose(1, 2).reshape(-1, self.patch_dim) # [B*L, patch_dim]
        
        # Group Logic
        bsz_total = patches_flat.shape[0]
        # [B*L, groups, patch_dim_per_group]
        sub_patches = patches_flat.reshape(bsz_total, self.n_groups, self.patch_dim_per_group)
        # [B*L*groups, patch_dim_per_group]
        sub_patches_flat = sub_patches.reshape(-1, self.patch_dim_per_group)
        sub_bsz = sub_patches_flat.shape[0]
        
        # 3. Generate Rotation Parameters
        if self.use_u3_gates:
            # Input-Dependent U3 (Dynamic)
            rot_params_flat = self.rot_proj(patches_flat)
            # U3: [B*L, 1, layers, qubits, 3] -> Expand groups -> [B*L*G, layers, qubits, 3]
            rot_params = rot_params_flat.view(bsz_total, 1, self.n_layers, self.n_qubits_data, 3)
            rot_params = rot_params.expand(-1, self.n_groups, -1, -1, -1)
            rot_params = rot_params.reshape(sub_bsz, self.n_layers, self.n_qubits_data, 3)
        else:
            # Static RY/RZ (Parent Logic)
            # Use the trainable parameter directly
            rot_params = self.qcnn_rot_params
 
         # 2. Expand Style
        # style: [B, style_dim] -> [B*L, style_dim] -> [B*L*groups, style_dim]
        style_expanded = style.unsqueeze(1).expand(B, L, -1).reshape(bsz_total, -1)
        style_grouped = style_expanded.unsqueeze(1).expand(bsz_total, self.n_groups, -1).reshape(sub_bsz, -1)
        
        # 3. Classical Pre-processing
        # Map inputs to rotation angles (for Re-uploading or Angle Encoding)
        # sub_patches_flat is [sub_bsz, patch_dim_per_group]
        data_angles = torch.tanh(self.data_proj(sub_patches_flat)) * math.pi # [-pi, pi]
        
        # Style projection
        style_angles = None
        if hasattr(self, 'style_to_data'):
             style_angles = torch.tanh(self.style_to_data(style_grouped)) * math.pi
        
        # Amplitude Features (if needed)
        amp_features = None
        if self.encoding_type == 'amplitude':
            # Map patch_dim_per_group -> 2^N
            amp_features = self.amp_proj(sub_patches_flat)
            # Normalize
            norm = amp_features.norm(p=2, dim=1, keepdim=True) + 1e-9
            amp_features = amp_features / norm
        
        # 4. Quantum Simulation
        bsz = sub_bsz
        dev = x.device
        device_name = self.device_name or ('cuda' if dev.type == 'cuda' else 'cpu')
        qdev = tq.QuantumDevice(n_wires=self.n_qubits_data, bsz=bsz, device=device_name)
        
        # Call Fusion Circuit (The Fix)
        self._apply_fusion_circuit(
            qdev=qdev,
            sub_bsz=bsz,
            sub_da=data_angles,       # Always used for re-uploading
            sub_sa=style_angles,
            amp_features=amp_features, # New Argument
            interaction_wires=None,    # Not used
            data_wires=None,           # Not used
            mod_params=self.mod_params,
            qcnn_rot_params=rot_params,
            n_qubits_data=self.n_qubits_data,
            active_layers=self.active_layers,
            use_strided_cnot=True,
            reupload_data=True,
            encoding_type=self.encoding_type
        )
            
        # 5. Trainable Measurement Basis
        # measure_params: [n_groups, n_qubits, 3] or [n_qubits, 3]
        if self.measure_params.ndim == 3 and self.measure_params.shape[0] == self.n_groups: 
            # Expand to [B*L, groups, n_qubits, 3]
            meas_expanded = self.measure_params.unsqueeze(0).expand(bsz_total, -1, -1, -1)
            # Flatten to [sub_bsz, n_qubits, 3]
            meas_flat = meas_expanded.reshape(sub_bsz, self.n_qubits_data, 3)
            
            for i in range(self.n_qubits_data):
                tqf.u3(qdev, wires=i, params=meas_flat[:, i])
        else:
            # Fallback for non-grouped params (if any)
            for i in range(self.n_qubits_data):
                params_expanded = self.measure_params[i].unsqueeze(0).expand(bsz, -1)
                tqf.u3(qdev, wires=i, params=params_expanded)
            
        # 6. Measurement
        if hasattr(qdev, 'get_states_1d'): states = qdev.get_states_1d()
        else: states = qdev.states
        # Avoid sqrt
        probs = states.real**2 + states.imag**2
        quant_out = probs # [sub_bsz, 2^N]
        
        # 7. Post-processing & Residual
        # out_proj expects [sub_bsz, 2^N] -> [sub_bsz, channels_per_group]
        out_quant = self.out_proj(quant_out)
        
        # Reshape back to groups and merge
        # [B*L*groups, channels_per_group] -> [B*L, groups, channels_per_group]
        out_quant_grouped = out_quant.reshape(bsz_total, self.n_groups, self.channels_per_group)
        # [B*L, channels]
        out_quant_merged = out_quant_grouped.reshape(bsz_total, self.channels)
        
        # Residual Calculation
        out_res = None
        
        # Mode 1: Global Residual (Skip Patching) - Implemented at image level?
        # But here we are at patch level.
        # If we want Global Residual: x + Q(x), we need x in the same shape.
        # patches_flat is [B*L, C*K*K].
        # We need [B*L, C].
        
        if getattr(self, 'use_lightweight_residual', False):
             # Lightweight: AvgPool + Linear
             K = self.kernel_size 
             patches_reshaped = patches_flat.reshape(bsz_total, self.channels, K, K)
             patches_mean = patches_reshaped.mean(dim=(2, 3)) 
             out_res = self.res_proj(patches_mean)
             
        elif getattr(self, 'use_param_free_bypass', False):
             # Parameter-Free Bypass: Center Crop
             K = self.kernel_size 
             patches_reshaped = patches_flat.reshape(bsz_total, self.channels, K, K)
             center = K // 2
             out_res = patches_reshaped[:, :, center, center] 
             
        else:
             # Default: Trainable Linear
             out_res = self.res_proj(patches_flat)
             
        out_flat = out_quant_merged + out_res
        
        # 8. Reshape back
        out = out_flat.view(B, H, W, self.channels).permute(0, 3, 1, 2)
        
        # Algorithm 5: C-Tail Affine
        if self.affine_mode == 'c_tail':
            scale = self.c_tail_scale(style).unsqueeze(-1).unsqueeze(-1) + 1.0
            shift = self.c_tail_shift(style).unsqueeze(-1).unsqueeze(-1)
            out = out * scale + shift
            
        return out

    def _apply_fusion_circuit(self, qdev, sub_bsz, sub_da, sub_sa, amp_features, interaction_wires, data_wires, 
                              mod_params, qcnn_rot_params, 
                              n_qubits_data: int, active_layers: int, 
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
        
        # Trainable Re-uploading Logic
        use_trainable_reupload = getattr(self, 'use_trainable_reupload', False)
        
        # 2. Encode Data (RY/Amplitude)
        if encoding_type == 'amplitude':
            # Initialize Data Qubits with Amplitude Encoding
            if amp_features is not None:
                # Assuming amp_features is already normalized and complex-ready (or float)
                states_data = amp_features.to(torch.cfloat)
                
                # Directly set states for data qubits (No Ancilla)
                if hasattr(qdev, 'set_states'):
                    qdev.set_states(states_data)
                else:
                    qdev.states = states_data
            
            # Style Injection at Head (Rotation on top of Amplitude)
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
        
        # 4. Spatial QCNN Backbone
        middle_layer = active_layers // 2
        
        for l in range(active_layers):
            # Q-Middle Injections (Before Layer Operations)
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
            # qcnn_rot_params: [groups, layers, n, 2/3, 3?] 
            # If using U3, rot_proj output is [B, L*N*3]. We need to reshape.
            # But qcnn_rot_params is passed from forward() which reshapes it.
            # We need to ensure forward() reshapes correctly for U3.
            
            # Let's fix the reshaping logic here, assuming qcnn_rot_params is raw flat or partially structured.
            # Wait, forward() calls self.rot_proj(patches_flat) -> [B*L, n_rotations]
            # Then reshapes to [groups, layers, n, 2, 3] usually?
            # Actually forward() passes `qcnn_rot_params` as [B*L, n_rotations] or similar.
            
            # If U3 is used, we expect 3 params per qubit.
            if self.use_u3_gates:
                 # Extract params for U3
                 # qcnn_rot_params should be reshaped to [sub_bsz, layers, n_qubits, 3]
                 # But it comes in as whatever forward() produces.
                 # Let's assume we handle raw `mod_params` or `qcnn_rot_params` directly if needed.
                 # Actually `qcnn_rot_params` in `_apply_fusion_circuit` arg list is what we use.
                 
                 # If we redefined rot_proj, forward() logic might break if it hardcodes reshape.
                 # We need to check forward() in parent class.
                 # Parent forward: `rot_params = self.rot_proj(patches_flat).view(-1, self.n_groups, self.n_layers, self.n_qubits_data, 2)`
                 # It hardcodes `2`. So we MUST override forward() or handle the flat vector here?
                 # Overriding forward is cleaner but more code.
                 # Let's just use `mod_params` (which is `rot_params`?) No, `mod_params` is style.
                 
                 # We can't easily change parent forward reshape.
                 # So we will IGNORE `qcnn_rot_params` argument here and re-compute it from `rot_proj`?
                 # No, `rot_proj` input `patches_flat` is not passed to `_apply_fusion_circuit`.
                 
                 # Hack: If we use U3, we must have overridden `rot_proj`.
                 # But `forward` will crash on `.view(..., 2)`.
                 # So we MUST override `forward` in `ExperimentalQuantumFrontEnd`.
                 pass

            # Assuming forward() is fixed (we will fix it next),
            # qcnn_rot_params comes in as [sub_bsz, layers, n, 3] (if simple) or [sub_bsz, G, L, N, 3]
            
            if self.use_u3_gates:
                 # U3 Logic
                 # qcnn_rot_params: [sub_bsz, n_layers, n_qubits, 3] (Dynamic) or [G, L, N, 3] (Static)
                 
                 if qcnn_rot_params.shape[0] == sub_bsz:
                      # Input-Dependent (Dynamic): [B, L, N, 3]
                      u3_theta = qcnn_rot_params[:, l, :, 0]
                      u3_phi = qcnn_rot_params[:, l, :, 1]
                      u3_lam = qcnn_rot_params[:, l, :, 2]
                 else:
                      # Static Parameters: [G, L, N, 3] (if expanded)
                      n_groups = qcnn_rot_params.shape[0]
                      params_l = qcnn_rot_params[:, l, :, :] # [G, N, 3]
                      params_expanded = params_l.unsqueeze(0).expand(sub_bsz // n_groups, -1, -1, -1)
                      params_flat = params_expanded.reshape(sub_bsz, n_qubits_data, 3)
                      u3_theta = params_flat[:, :, 0]
                      u3_phi = params_flat[:, :, 1]
                      u3_lam = params_flat[:, :, 2]
                 
                 for i in range(n_qubits_data):
                      # Stack params to [B, 3]
                      params_i = torch.stack([u3_theta[:, i], u3_phi[:, i], u3_lam[:, i]], dim=-1)
                      tqf.u3(qdev, wires=i, params=params_i)
            
            else:
                # Standard RY/RZ
                # Pre-fetch params for this layer
                # qcnn_rot_params: [groups, layers, n, 2, 3]
                if qcnn_rot_params.ndim == 5:
                     n_groups = qcnn_rot_params.shape[0]
                     params_l = qcnn_rot_params[:, l, :, :, :] 
                     params_expanded = params_l.unsqueeze(0).expand(sub_bsz // n_groups, -1, -1, -1, -1)
                     params_flat = params_expanded.reshape(sub_bsz, n_qubits_data, 2, 3)
                     layer_ry = params_flat[:, :, 0, 0]
                     layer_rz = params_flat[:, :, 1, 0]
                elif qcnn_rot_params.shape[0] == sub_bsz:
                    layer_ry = qcnn_rot_params[:, l, :, 0, 0] # [B, N]
                    layer_rz = qcnn_rot_params[:, l, :, 1, 0] # [B, N]
                else:
                    layer_ry = qcnn_rot_params[l, :, 0, 0].expand(sub_bsz, -1) # [B, N]
                    layer_rz = qcnn_rot_params[l, :, 1, 0].expand(sub_bsz, -1) # [B, N]
                
                for i in range(n_qubits_data):
                    tqf.ry(qdev, wires=i, params=layer_ry[:, i])
                    tqf.rz(qdev, wires=i, params=layer_rz[:, i])
            
            # --- Entanglement Layers (Ansatz) ---
            ansatz_type = getattr(self, 'ansatz_type', 'basic')
            
            if ansatz_type == 'hea':
                # Hardware-Efficient Ansatz: CNOT Ring + Extra RY
                for i in range(n_qubits_data):
                    tqf.cnot(qdev, wires=[i, (i + 1) % n_qubits_data])
                # Extra rotation after CNOTs (simulating more complex ansatz)
                # We reuse layer_rz for simplicity, or we should have extra params.
                # Since we don't have extra params, we skip extra rotation or reuse.
                # HEA typically is Rot-Ent-Rot. We have Rot-Ent.
                # Let's add strided CNOTs by default for HEA
                if n_qubits_data >= 4:
                     for i in range(n_qubits_data):
                        tqf.cnot(qdev, wires=[i, (i + 2) % n_qubits_data])
                        
            elif ansatz_type == 'sea':
                # Strong Entanglement Ansatz: All-to-All or multiple layers of CNOTs
                # Here we implement a "Block" entanglement
                # Odd-Even CNOTs
                # Layer 1: (0,1), (2,3), ...
                for i in range(0, n_qubits_data - 1, 2):
                    tqf.cnot(qdev, wires=[i, i+1])
                # Layer 2: (1,2), (3,4), ...
                for i in range(1, n_qubits_data - 1, 2):
                    tqf.cnot(qdev, wires=[i, i+1])
                # Layer 3: Ring closure
                tqf.cnot(qdev, wires=[n_qubits_data-1, 0])
                
            else:
                # Basic / Default
                # Ring CNOTs
                for i in range(n_qubits_data):
                    tqf.cnot(qdev, wires=[i, (i + 1) % n_qubits_data])
                
                if use_strided_cnot and n_qubits_data >= 4:
                    for i in range(n_qubits_data):
                        tqf.cnot(qdev, wires=[i, (i + 2) % n_qubits_data])
            
            # Data Re-uploading Logic (After Layer Operations)
            if reupload_data and (l < active_layers - 1):
                
                is_middle_reupload = (l == middle_layer)
                
                # Flag to handle standard re-uploading loop
                skip_default_loop = False
                reup_params = None 
                
                if inject_middle_freq and is_middle_reupload:
                    # Multiplicative Scaling: Data * Style
                    # sub_da: [B, N], sub_sa: [B, N]
                    reup_params = sub_da * (1.0 + sub_sa)
                    reup_params = torch.atan(reup_params) * 2.0
                    
                elif self.affine_mode == 'q_layer_wise_reupload':
                    # Layer-wise Dual-Axis Re-uploading (RY + RZ)
                    if inject_head and sub_sa is not None:
                        reup_params = sub_da + sub_sa
                    else:
                        reup_params = sub_da
                        
                    for i in range(n_qubits_data):
                        tqf.rz(qdev, wires=i, params=reup_params[:, i])
                        tqf.ry(qdev, wires=i, params=reup_params[:, i]) 
                    
                    skip_default_loop = True
                
                elif use_trainable_reupload:
                    # Trainable Re-uploading: params = data * scale + bias
                    # Weights: [L, N, 2], Bias: [L, N, 2]
                    
                    # Get weights for this layer
                    scale_l = self.reupload_weights[l].unsqueeze(0) # [1, N, 2]
                    bias_l = self.reupload_bias[l].unsqueeze(0)     # [1, N, 2]
                    
                    # Prepare input [B, N, 1]
                    da_expanded = sub_da.unsqueeze(-1) # [B, N, 1]
                    
                    # Apply scale and bias
                    # [B, N, 1] * [1, N, 2] -> [B, N, 2]
                    reup_vals = da_expanded * scale_l + bias_l
                    
                    # Apply RZ and RY
                    rz_vals = reup_vals[:, :, 1] # [B, N]
                    ry_vals = reup_vals[:, :, 0] # [B, N]
                    
                    # Add style if needed
                    if inject_head and sub_sa is not None:
                        rz_vals = rz_vals + sub_sa
                        ry_vals = ry_vals + sub_sa
                        
                    for i in range(n_qubits_data):
                        tqf.ry(qdev, wires=i, params=ry_vals[:, i])
                        tqf.rz(qdev, wires=i, params=rz_vals[:, i])
                    
                    skip_default_loop = True
                        
                else:
                    # Standard Additive Re-uploading
                    if inject_head and sub_sa is not None:
                        reup_params = sub_da + sub_sa
                    else:
                        reup_params = sub_da
                
                if not skip_default_loop:
                    for i in range(n_qubits_data):
                        tqf.rz(qdev, wires=i, params=reup_params[:, i])

        # Q-Tail Injection
        if inject_tail and sub_sa is not None:
             for i in range(n_qubits_data):
                    tqf.rz(qdev, wires=i, params=sub_sa[:, i])

