import math
import os
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint as checkpoint_utils # Rename to avoid conflict if any
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast as _autocast
    _AUTOCAST_SUPPORTS_DEVICE_TYPE = True
except Exception:
    try:
        from torch import autocast as _autocast
        _AUTOCAST_SUPPORTS_DEVICE_TYPE = True
    except Exception:
        from torch.cuda.amp import autocast as _autocast
        _AUTOCAST_SUPPORTS_DEVICE_TYPE = False

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    from torchquantum.functional.gate_wrapper import apply_unitary_einsum, apply_unitary_bmm
    _TQ_AVAILABLE = True
except Exception:
    _TQ_AVAILABLE = False
    apply_unitary_einsum = None
    apply_unitary_bmm = None

from torch_utils import persistence


class _AutocastOff:
    """Utility context manager to force FP32 ops when requested."""
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._ctx = None

    def __enter__(self):
        if not self.enabled:
            return None
        self._ctx = _autocast(device_type='cuda', enabled=False) if _AUTOCAST_SUPPORTS_DEVICE_TYPE else _autocast(enabled=False)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


class GroupedLinear(nn.Module):
    """
    Helper module for Grouped Linear Layer using Conv1d.
    Reduces parameters by using independent groups.
    """
    def __init__(self, in_features, out_features, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, groups=groups)
    
    def forward(self, x):
        # x: [..., in_features]
        # Reshape to [Batch, in_features, 1] for Conv1d
        shape = x.shape
        x_reshaped = x.view(-1, shape[-1], 1)
        out = self.conv(x_reshaped)
        # Reshape back to [..., out_features]
        return out.view(*shape[:-1], -1)


class QuantumAttention64(nn.Module):
    """
    QSANN attention strictly aligned with tq_qsann_min_train.py (TQ_QSANN):
      - amplitude encoding with N_QUBITS=6 (2**6=64), Q/K/V all via the same input 64-d vector.
      - Q/K: enc + {q_w,k_w} -> Z expectation per qubit -> Linear proj to qk_dim -> LayerNorm (optional).
      - V: enc + v_w -> state probabilities (64) with energy normalization.
      - RBF attention weights: alpha = exp(-||q - k||^2 / tau), tau positive via softplus if trainable.
      - Output: dropout(attn_out_64), shape (B, S, 64). Residual and gating are handled by AdaLN-Zero in the outer block.

    forward(x_64, has_cls=False) -> (B, S, 64)
    """

    def __init__(self,
                 N_QUBITS: int = 6,
                 Q_DEPTH: int = 8, # Optimized: Depth 8 (Full Quantum Deep) provides superior expressibility
                 qk_dim: int = 16, # Optimized: Head Dimension (16 * 4 = 64)
                 num_heads: int = 4, # Optimized: 4 Heads
                 tau: float = 0.5,
                 tau_trainable: bool = True,
                 attn_gate_init: float = 0.5,
                 attn_dropout: float = 0.1,
                 qk_norm: str = 'layernorm',
                 force_fp32_attention: bool = True,
                 device_name: Optional[str] = None,
                 enable_reupload: bool = True,
                 **kwargs):
        super().__init__()
        if not _TQ_AVAILABLE:
            raise ImportError("TorchQuantum 未安装或不可用：QuantumAttention64 依赖 torchquantum。请先安装 'torchquantum'.")
        
        # Handle input dimension (support 'in_channels' or 'input_dim')
        self.in_channels = kwargs.get('in_channels', kwargs.get('input_dim', 64))
        
        # assert N_QUBITS == 6, "本实现固定使用 N_QUBITS=6（2^6=64）以匹配 64 维幅度编码。"
        # [Optimized] Only warn if using strict Amplitude Encoding (64 dims).
        # Angle Encoding supports variable qubits.
        is_angle_encoding = 'Angle' in self.__class__.__name__
        if N_QUBITS != 6 and not is_angle_encoding:
            print(f"Warning: N_QUBITS={N_QUBITS} (Expected 6 for standard 64-dim amp encoding). Ensure dimensions match.")
        
        assert qk_norm in ('none', 'layernorm')

        self.N_QUBITS = int(N_QUBITS)
        self.Q_DEPTH = int(Q_DEPTH)
        self.qk_dim = int(qk_dim)
        self.num_heads = int(num_heads)
        self.inner_dim = self.num_heads * self.qk_dim # 64
        
        self.force_fp32_attention = bool(force_fp32_attention)
        self.device_name = device_name
        
        # Trainable PQC parameters (enc + branch-specific)
        self.enc_w = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.q_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.k_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.v_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))

        # Z measurement and q/k projections
        # Optimized: Multi-Head Projection (64 -> H*D)
        # Multi-Basis Update: Input is 2**N_QUBITS (Probabilities)
        self.input_dim = 2 ** self.N_QUBITS
        # [Optimized] Use GroupedLinear (groups=2) to reduce classical parameters
        self.q_proj = GroupedLinear(self.input_dim, self.inner_dim, groups=2) 
        self.k_proj = GroupedLinear(self.input_dim, self.inner_dim, groups=2) 
        self.v_proj = GroupedLinear(self.input_dim, self.inner_dim, groups=2) 
        self.qk_ln  = nn.LayerNorm(self.inner_dim) if qk_norm == 'layernorm' else nn.Identity()
        
        # Output Projection (to match standard MultiheadAttention)
        # [Optimized] Use GroupedLinear (groups=2)
        self.out_proj = GroupedLinear(self.inner_dim, self.in_channels, groups=2)
        
        # SOTA Improvement: Zero-Initialization
        # Force output projection weights and bias to zero for stable training start
        nn.init.zeros_(self.out_proj.conv.weight)
        nn.init.zeros_(self.out_proj.conv.bias)

        # Attention dropout (residual gating moved to AdaLN-Zero in the block)
        self.attn_drop = nn.Dropout(p=float(attn_dropout))

        # Temperature (tau)
        self.tau_trainable = bool(tau_trainable)
        init_tau = float(tau)
        if self.tau_trainable:
            # raw parameter for softplus
            self.raw_tau = nn.Parameter(torch.tensor(math.log(math.exp(init_tau) - 1.0), dtype=torch.float32))
        else:
            self.register_buffer('tau_value', torch.tensor(init_tau, dtype=torch.float32))

        # Learnable Input Scaling (pre-encoding) replaced by Full Projection
        # This matches Classical Attention's ability to mix features before processing
        # Use self.in_channels (default 64)
        self.inp_proj = nn.Linear(self.in_channels, 64)
        
        # Data Re-uploading Projector (in_channels -> 6)
        self.enable_reupload = enable_reupload
        if self.enable_reupload:
            self.reupload_proj = nn.Linear(self.in_channels, self.N_QUBITS)
        else:
            self.reupload_proj = None

        # Trainable Measurement Basis (U3 before measurement)
        # For Q, K (Expectation based) and V (Prob based)
        self.meas_q_w = nn.Parameter(0.1 * torch.randn(self.N_QUBITS, 3))
        self.meas_k_w = nn.Parameter(0.1 * torch.randn(self.N_QUBITS, 3))
        self.meas_v_w = nn.Parameter(0.1 * torch.randn(self.N_QUBITS, 3))

        # numerical stability epsilon
        self.eps = 1e-9
        self._printed_exec = False
        
        # [Optimization] Device Reuse to speed up training
        self.reuse_device = True
        self._qdev_cached = None
        self._qdev_cached_bsz = None
        self._qdev_common_cached = None
        self._qdev_common_cached_bsz = None


    # --- internal helpers ---
    def _apply_pqc(self, qdev: 'tq.QuantumDevice', weights: torch.Tensor, x_reupload: Optional[torch.Tensor] = None):
        """RX+RY -> CNOT chain -> RY with weights [depth, N_QUBITS, 3]. Supports Data Re-uploading."""
        depth = weights.shape[0]
        # Calculate split point for re-uploading
        reupload_idx = depth // 2
        
        for l in range(depth):
            # Apply Data Re-uploading at the middle
            if x_reupload is not None and l == reupload_idx:
                 # x_reupload shape: [bsz, N_QUBITS] (angles)
                for i in range(self.N_QUBITS):
                    tqf.rx(qdev, wires=i, params=x_reupload[:, i])
                    
            rx_params = weights[l, :, 0]
            ry_params = weights[l, :, 1]
            ent_ry_params = weights[l, :, 2]
            # local rotations
            for i in range(self.N_QUBITS):
                tqf.rx(qdev, wires=i, params=rx_params[i])
                tqf.ry(qdev, wires=i, params=ry_params[i])
            # linear CNOT chain
            for i in range(self.N_QUBITS - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            # post-entanglement rotations
            for i in range(self.N_QUBITS):
                tqf.ry(qdev, wires=i, params=ent_ry_params[i])

    def _amplitude_encode(self, qdev: 'tq.QuantumDevice', x_state: torch.Tensor):
        """Set amplitude state vector from 64-dim input: L2-normalize, set complex state."""
        states = x_state / (x_state.norm(p=2, dim=1, keepdim=True) + self.eps)
        states = states.to(torch.cfloat)
        # TorchQuantum API versions differ; try common ones.
        if hasattr(qdev, 'set_states'):
            qdev.set_states(states)
        elif hasattr(qdev, 'set_states_1d'):
            qdev.set_states_1d(states)
        else:
            qdev.states = states

    def _measure_probs(self, qdev: 'tq.QuantumDevice') -> torch.Tensor:
        """Return computational basis probabilities from device state vector."""
        if hasattr(qdev, 'get_states'):
            states = qdev.get_states()
        elif hasattr(qdev, 'get_states_1d'):
            states = qdev.get_states_1d()
        else:
            states = qdev.states
        # Optimized: Avoid sqrt in abs() since we square it anyway
        # probs = (states.abs() ** 2)
        probs = states.real**2 + states.imag**2
        return probs

    # --- branches ---
    def _q_branch(self, x: torch.Tensor, device_name: str) -> torch.Tensor:
        B, S, D = x.shape
        assert D == 64, "QuantumAttention64 期望输入最后一维为 64（幅度编码）"
        bsz = B * S
        qdev_q = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        x_bsz = x.reshape(bsz, D)
        self._amplitude_encode(qdev_q, x_bsz)
        self._apply_pqc(qdev_q, self.enc_w)
        self._apply_pqc(qdev_q, self.q_w)
        z_q = self.measure_z(qdev_q)  # (bsz, 6)
        q_vec = self.q_proj(z_q)
        q_vec = self.qk_ln(q_vec)
        return q_vec.reshape(B, S, self.qk_dim)

    def _k_branch(self, x: torch.Tensor, device_name: str) -> torch.Tensor:
        B, S, D = x.shape
        bsz = B * S
        qdev_k = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        x_bsz = x.reshape(bsz, D)
        self._amplitude_encode(qdev_k, x_bsz)
        self._apply_pqc(qdev_k, self.enc_w)
        self._apply_pqc(qdev_k, self.k_w)
        z_k = self.measure_z(qdev_k)
        k_vec = self.k_proj(z_k)
        k_vec = self.qk_ln(k_vec)
        return k_vec.reshape(B, S, self.qk_dim)

    def _v_branch(self, x: torch.Tensor, device_name: str) -> torch.Tensor:
        B, S, D = x.shape
        bsz = B * S
        qdev_v = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        x_bsz = x.reshape(bsz, D)
        self._amplitude_encode(qdev_v, x_bsz)
        self._apply_pqc(qdev_v, self.enc_w)
        self._apply_pqc(qdev_v, self.v_w)
        probs = self._measure_probs(qdev_v)   # (bsz, 64)
        v = probs.reshape(B, S, D)
        # energy normalization for stability
        v = F.layer_norm(v, normalized_shape=(D,))
        return v

    def forward(self, x_64: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
        """x_64: (B, S, 64) -> returns (B, S, 64) as gate * attn_out (no residual)."""
        if not self._printed_exec:
            self._printed_exec = True
        dev = x_64.device
        device_name = self.device_name or dev.type
        # Force attention in FP32 if requested to avoid AMP instabilities.
        if self.force_fp32_attention and dev.type == 'cuda':
            with _AutocastOff(enabled=True):
                out = self._forward_impl(x_64.float(), device_name)
                return out.to(x_64.device, dtype=x_64.dtype)
        else:
            out = self._forward_impl(x_64, device_name)
            return out.to(x_64.device, dtype=x_64.dtype)

    def _measure_multibasis(self, qdev: 'tq.QuantumDevice') -> torch.Tensor:
        """
        Optimized Measurement: Z-Basis Probability Distribution.
        Reverted from Multi-Basis due to performance degradation (0.19 vs 0.07).
        Returns: [bsz, 64]
        """
        # 1. Z-Basis Measurement (Standard)
        probs_z = self._measure_probs(qdev) # [bsz, 64]
        return probs_z

    def _apply_pqc_batched(self, qdev: 'tq.QuantumDevice', weights_q: torch.Tensor, weights_k: torch.Tensor, weights_v: torch.Tensor, bsz: int, x_reupload: Optional[torch.Tensor] = None):
        """
        Apply PQC to a unified device containing [Q_batch, K_batch, V_batch].
        weights_*: [Depth, N_QUBITS, 3]
        bsz: Batch size for ONE branch (total device bsz = 3 * bsz)
        x_reupload: [bsz, N_QUBITS] (angles) or None. If provided, replicated for 3 branches.
        """
        depth = weights_q.shape[0]
        reupload_idx = depth // 2
        
        # Pre-expand reupload angles if needed: [bsz, N] -> [3*bsz, N]
        if x_reupload is not None:
            x_reupload_all = x_reupload.repeat(3, 1) # [3*bsz, N]
        
        for l in range(depth):
            # Apply Data Re-uploading at the middle
            if x_reupload is not None and l == reupload_idx:
                for i in range(self.N_QUBITS):
                    # x_reupload_all[:, i] is [3*bsz]
                    tqf.rx(qdev, wires=i, params=x_reupload_all[:, i])
            
            # Prepare batched parameters for this layer
            # Each weight[l, i, :] is (3,) -> Rx, Ry, Ent_Ry
            # We need to construct params [3*bsz] for each rotation
            
            # 1. Rx
            # [N_QUBITS]
            rx_q = weights_q[l, :, 0]
            rx_k = weights_k[l, :, 0]
            rx_v = weights_v[l, :, 0]
            
            # 2. Ry
            ry_q = weights_q[l, :, 1]
            ry_k = weights_k[l, :, 1]
            ry_v = weights_v[l, :, 1]
            
            # 3. Ent_Ry
            ery_q = weights_q[l, :, 2]
            ery_k = weights_k[l, :, 2]
            ery_v = weights_v[l, :, 2]
            
            # Local Rotations
            for i in range(self.N_QUBITS):
                # Construct params: [bsz_q, bsz_k, bsz_v] -> [3*bsz]
                # scalar .expand(bsz) -> [bsz]
                p_rx = torch.cat([
                    rx_q[i].expand(bsz),
                    rx_k[i].expand(bsz),
                    rx_v[i].expand(bsz)
                ], dim=0)
                tqf.rx(qdev, wires=i, params=p_rx)
                
                p_ry = torch.cat([
                    ry_q[i].expand(bsz),
                    ry_k[i].expand(bsz),
                    ry_v[i].expand(bsz)
                ], dim=0)
                tqf.ry(qdev, wires=i, params=p_ry)
                
            # Linear CNOT chain (entanglement is topology-dependent, not param-dependent, so just run it)
            for i in range(self.N_QUBITS - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
                
            # Post-entanglement Rotations
            for i in range(self.N_QUBITS):
                p_ery = torch.cat([
                    ery_q[i].expand(bsz),
                    ery_k[i].expand(bsz),
                    ery_v[i].expand(bsz)
                ], dim=0)
                tqf.ry(qdev, wires=i, params=p_ery)

    def _forward_impl(self, x_64: torch.Tensor, device_name: str) -> torch.Tensor:
        # Optimized implementation: Common Encoding Fork + Batch Parallel Execution
        # 1. Prepare common state |psi_enc> = U_enc(AmplitudeEncode(x))
        B, S, D = x_64.shape
        bsz = B * S
        x_bsz = x_64.reshape(bsz, D)
        
        # Apply Input Projection (Linear Mix) before Amplitude Encoding
        # [Fix] Keep original for reupload_proj
        x_bsz_proj = self.inp_proj(x_bsz)
        
        # Create common device
        qdev_common = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        self._amplitude_encode(qdev_common, x_bsz_proj)
        self._apply_pqc(qdev_common, self.enc_w)
        
        # Get common state (Flattened)
        if hasattr(qdev_common, 'get_states_1d'): 
            common_states_flat = qdev_common.get_states_1d()
        else: 
            # Flatten if it is [B, 2, 2, ...]
            common_states_flat = qdev_common.states.reshape(bsz, -1)
        
        # Prepare state for injection: [B] + [2]*N
        # We need to replicate this state 3 times for Q, K, V branches
        # common_states_flat: [bsz, 2^N]
        # target: [3*bsz, 2^N]
        common_states_3x = common_states_flat.repeat(3, 1)
        target_shape_3x = [3 * bsz] + [2] * self.N_QUBITS
        common_states_reshaped = common_states_3x.reshape(target_shape_3x)
        
        # Prepare Data Re-uploading Angles (Tanh -> Pi)
        # x_bsz: [bsz, 64] -> [bsz, 18] -> Tanh -> * Pi
        if self.enable_reupload:
            reupload_angles = torch.tanh(self.reupload_proj(x_bsz)) * torch.pi
        else:
            reupload_angles = None
            
        # 2. Unified Q/K/V Execution
        # Create ONE device for 3*bsz
        qdev_all = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=3*bsz, device=device_name)
        qdev_all.states = common_states_reshaped.clone() # Direct set
        
        # Apply Batched PQC (Q, K, V weights in parallel)
        self._apply_pqc_batched(qdev_all, self.q_w, self.k_w, self.v_w, bsz, x_reupload=reupload_angles)
        
        # Apply Batched Measurement Basis
        # meas_q_w: [N, 3]
        for i in range(self.N_QUBITS):
            # Construct [3*bsz, 3] params for u3
            # q_params: [bsz, 3]
            p_q = self.meas_q_w[i].unsqueeze(0).expand(bsz, -1)
            p_k = self.meas_k_w[i].unsqueeze(0).expand(bsz, -1)
            p_v = self.meas_v_w[i].unsqueeze(0).expand(bsz, -1)
            
            p_all = torch.cat([p_q, p_k, p_v], dim=0) # [3*bsz, 3]
            tqf.u3(qdev_all, wires=i, params=p_all)
            
        # Unified Measurement
        probs_all = self._measure_multibasis(qdev_all) # [3*bsz, 64]
        
        # Split results
        probs_q, probs_k, probs_v = torch.chunk(probs_all, 3, dim=0) # Each [bsz, 64]
        
        # 3. Post-Processing (Project & Attention)
        
        # Q Branch
        q = self.qk_ln(self.q_proj(probs_q)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # K Branch
        k = self.qk_ln(self.k_proj(probs_k)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # V Branch
        v = self.v_proj(probs_v).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # Dot-Product Attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qk_dim)
        alpha = torch.softmax(attn_score, dim=-1)
        alpha = self.attn_drop(alpha)
        attn_out = torch.matmul(alpha, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.inner_dim)
        attn_out = self.out_proj(attn_out)
        
        # Debug Prints (Only once)
        if not self._printed_exec:
            self._printed_exec = True
            print(f"\n[QuantumAttention64] Debug Exec (Optimized Batch-Parallel Q/K/V):")
            print(f"  Input Shape: {x_64.shape}")
            print(f"  Total Batch Size: {3*bsz} (3 branches x {bsz})")
            print(f"  Output Shape: {attn_out.shape}")
            print(f"  Param Count: {sum(p.numel() for p in self.parameters())}")
        
        return attn_out


class QuantumAttentionAngle(QuantumAttention64):
    """
    Angle Encoding (Tanh) version of Quantum Attention.
    Replaces Amplitude Encoding with Rx/Ry rotations driven by Tanh-scaled inputs.
    Features:
    - Input Projection: 64 -> N_QUBITS * 2 (Rx, Ry params)
    - Encoding: Rx(tanh(x)*pi), Ry(tanh(x)*pi)
    - No Amplitude Encoding (starts from |0>)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Projector for Angle Encoding (in_channels -> 12 for 6 qubits Rx+Ry)
        # [Optimized] Single Linear Projection (matches SOTA Dense simplification)
        self.angle_proj = nn.Linear(self.in_channels, self.N_QUBITS * 2)
        
        # Re-initialize to ensure fresh weights
        nn.init.xavier_uniform_(self.angle_proj.weight)
        nn.init.zeros_(self.angle_proj.bias)
        
        # [Optimized] Remove Classical Residuals to match SOTA and reduce params
        # This ensures a fair "Quantum vs Quantum" comparison
        self.q_res_proj = None
        self.k_res_proj = None
        self.v_res_proj = None

        # [Optimized] Remove inp_proj from parent (Angle Encoding uses angle_proj directly)
        self.inp_proj = None

        # Learnable Head-wise Temperature (Scale)
        # Initialize to 1/sqrt(qk_dim)
        self.attn_scale = nn.Parameter(torch.full((self.num_heads, 1, 1), self.qk_dim ** -0.5))

    def _forward_impl(self, x_64: torch.Tensor, device_name: str) -> torch.Tensor:
        # 1. Prepare common state via Angle Encoding (No Amplitude Encoding)
        B, S, D = x_64.shape
        bsz = B * S
        x_bsz = x_64.reshape(bsz, D)
        
        # [Optimized] No Input Projection (Direct Angle Encoding)
        # x_bsz = self.inp_proj(x_bsz) -> Removed
        
        # [Optimized] No Classical Residuals
        # q_res, k_res, v_res -> Removed
        
        # Angle Encoding: in_channels -> 12 -> Tanh -> Pi
        # Shifted Mapping: [-1, 1] -> [0, pi]
        raw_out = self.angle_proj(x_bsz)
        angles = (torch.tanh(raw_out) + 1.0) * (torch.pi / 2.0)
        
        rx_angles = angles[:, :self.N_QUBITS]
        ry_angles = angles[:, self.N_QUBITS:]

        # Create common device (starts at |0>)
        qdev_common = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        
        # Apply Angle Encoding
        for i in range(self.N_QUBITS):
            tqf.rx(qdev_common, wires=i, params=rx_angles[:, i])
            tqf.ry(qdev_common, wires=i, params=ry_angles[:, i])
            
        # Apply Common PQC
        self._apply_pqc(qdev_common, self.enc_w)
        
        # Get common state (Flattened)
        if hasattr(qdev_common, 'get_states_1d'): 
            common_states_flat = qdev_common.get_states_1d()
        else: 
            common_states_flat = qdev_common.states.reshape(bsz, -1)
        
        # Prepare state for injection
        target_shape = [bsz] + [2] * self.N_QUBITS
        common_states_reshaped = common_states_flat.reshape(target_shape)
        
        # Data Re-uploading for branches
        # For consistency with QuantumAttention64, we reuse the reupload_proj (64->6) logic
        # Note: reupload_proj expects in_channels input. x_bsz is [bsz, 64] (actually in_channels)
        if self.enable_reupload:
            reupload_angles = torch.tanh(self.reupload_proj(x_bsz)) * torch.pi
        else:
            reupload_angles = None

        # 2. Fork to Q/K/V branches (Optimized Batch-Parallel)
        # Prepare Batch: [Q_batch, K_batch, V_batch]
        # common_states_reshaped: [bsz, 2, 2, ...] -> [3*bsz, 2, 2, ...]
        # Flatten first:
        common_states_flat = common_states_reshaped.reshape(bsz, -1)
        common_states_3x = common_states_flat.repeat(3, 1)
        target_shape_3x = [3 * bsz] + [2] * self.N_QUBITS
        common_states_all = common_states_3x.reshape(target_shape_3x)
        
        # Create Unified Device
        qdev_all = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=3*bsz, device=device_name)
        qdev_all.states = common_states_all.clone()
        
        # Apply Batched PQC
        self._apply_pqc_batched(qdev_all, self.q_w, self.k_w, self.v_w, bsz, x_reupload=reupload_angles)
        
        # Apply Batched Measurement Basis
        for i in range(self.N_QUBITS):
            # Construct [3*bsz, 3] params for u3
            p_q = self.meas_q_w[i].unsqueeze(0).expand(bsz, -1)
            p_k = self.meas_k_w[i].unsqueeze(0).expand(bsz, -1)
            p_v = self.meas_v_w[i].unsqueeze(0).expand(bsz, -1)
            
            p_all = torch.cat([p_q, p_k, p_v], dim=0) # [3*bsz, 3]
            tqf.u3(qdev_all, wires=i, params=p_all)
            
        # Unified Measurement
        probs_all = self._measure_multibasis(qdev_all) # [3*bsz, 64]
        
        # Split results
        probs_q, probs_k, probs_v = torch.chunk(probs_all, 3, dim=0)
        
        # 3. Post-Processing (Project & Attention)
        
        # Q Branch (No Residual)
        q = self.qk_ln(self.q_proj(probs_q)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # K Branch (No Residual)
        k = self.qk_ln(self.k_proj(probs_k)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # V Branch (No Residual)
        v = self.v_proj(probs_v).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)

        # 3. Attention
        # Optimized: Use Learnable Head-wise Scale
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_out = (attn @ v).transpose(1, 2).reshape(B, S, self.inner_dim)
        x_out = self.out_proj(x_out)
        
        # Debug Prints
        if not self._printed_exec:
            self._printed_exec = True
            print(f"\n[QuantumAttentionAngle] Debug Exec (Angle Encoding Rx+Ry):")
            print(f"  Input Shape: {x_64.shape}")
            print(f"  Probs Q Shape: {probs_q.shape}")
            print(f"  Output Shape: {x_out.shape}")
            print(f"  Learnable Scale Mean: {self.attn_scale.mean().item():.4f}")
        
        return x_out


class QuantumAttentionAngleDense(QuantumAttentionAngle):
    """
    Dense Angle Encoding version of Quantum Attention.
    Splits the 64-dim input into chunks and injects them layer-by-layer (Dense Encoding).
    Features:
    - Input Slicing: 64 features -> Split into N_LAYERS chunks
    - Dense Injection: Each PQC layer receives a fresh chunk of data via Rx/Ry rotations
    - Enhanced Capacity: Utilizes circuit depth to encode more information
    """
    def __init__(self, *args, use_grouped_linear=True, chunk_size=2048, use_checkpoint=True, **kwargs):
        self.use_grouped_linear = use_grouped_linear
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        super().__init__(*args, **kwargs)
        # print("DEBUG: QuantumAttentionAngleDense instantiated!")
        
        # Determine number of chunks based on depth and qubits
        # We want to inject data at multiple layers. 
        # Strategy: Inject every layer (or every K layers).
        # For Q_DEPTH=4, we can inject at layers 0, 1, 2, 3.
        # Max capacity = N_QUBITS * 2 (params) * Q_DEPTH
        
        # [Optimized] Dense Angle Projection (Classic Parameter Reduction)
        # Replaced MLP (Linear->GELU->Linear) with Single Linear Projection
        # Reducing redundant classical params while keeping quantum capacity
        self.dense_angle_proj = nn.Linear(self.in_channels, self.Q_DEPTH * self.N_QUBITS * 2)
        
        # Re-initialize
        nn.init.xavier_uniform_(self.dense_angle_proj.weight)
        nn.init.zeros_(self.dense_angle_proj.bias)
        
        # [Optimized] Q/K/V/Out Projections (Grouped Linear vs Standard)
        if self.use_grouped_linear:
            # Use groups=2 to halve parameters for these classical mappings
            self.q_proj = GroupedLinear(self.input_dim, self.inner_dim, groups=2)
            self.k_proj = GroupedLinear(self.input_dim, self.inner_dim, groups=2)
            self.v_proj = GroupedLinear(self.input_dim, self.inner_dim, groups=2)
            self.out_proj = GroupedLinear(self.inner_dim, self.in_channels, groups=2)
        else:
            # Standard Linear (Ablation Baseline)
            self.q_proj = nn.Linear(self.input_dim, self.inner_dim)
            self.k_proj = nn.Linear(self.input_dim, self.inner_dim)
            self.v_proj = nn.Linear(self.input_dim, self.inner_dim)
            self.out_proj = nn.Linear(self.inner_dim, self.in_channels)
        
        # Zero-Init Output (for conv layer inside GroupedLinear or standard Linear)
        if hasattr(self.out_proj, 'conv'):
            nn.init.zeros_(self.out_proj.conv.weight)
            nn.init.zeros_(self.out_proj.conv.bias)
        else:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        
        # Disable standard angle_proj from parent to save params (optional, but cleaner)
        self.angle_proj = None

        # [Optimized] Remove unused inp_proj from parent
        self.inp_proj = None
        
        # [Optimized] Disable Classical Residuals by default to ensure fair comparison
        # User Feedback: "Baseline parameters should be more than Quantum model."
        # Removing these 3 projections saves ~12k parameters.
        self.q_res_proj = None
        self.k_res_proj = None
        self.v_res_proj = None

    def _process_qkv_chunk(self, common_states_chunk, reupload_angles_chunk, device_name):
        """
        Process a single chunk of Q/K/V generation on the quantum device.
        Designed to be used with torch.utils.checkpoint.
        """
        bsz_chunk = common_states_chunk.shape[0]
        actual_qdev_bsz = 3 * bsz_chunk
        
        # [Optimization] Device Reuse
        qdev_all = None
        if self.reuse_device and self._qdev_cached is not None and self._qdev_cached_bsz == actual_qdev_bsz:
             qdev_all = self._qdev_cached
             # We will overwrite states anyway, so just ensure bsz matches
        else:
             qdev_all = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=actual_qdev_bsz, device=device_name)
             if self.reuse_device:
                 self._qdev_cached = qdev_all
                 self._qdev_cached_bsz = actual_qdev_bsz
        
        # Prepare Batch: [Q_chunk, K_chunk, V_chunk]
        # common_states_chunk: [bsz_chunk, 2^N_QUBITS] -> [3*bsz_chunk, 2, 2, ...]
        
        common_states_3x = common_states_chunk.repeat(3, 1)
        target_shape_3x = [3 * bsz_chunk] + [2] * self.N_QUBITS
        common_states_all = common_states_3x.reshape(target_shape_3x)
        
        # Assign states directly
        qdev_all.states = common_states_all.clone()
        
        # Apply Batched PQC
        self._apply_pqc_batched(qdev_all, self.q_w, self.k_w, self.v_w, bsz_chunk, x_reupload=reupload_angles_chunk)
        
        # Apply Batched Measurement Basis
        for i in range(self.N_QUBITS):
            # Construct params for u3
            p_q = self.meas_q_w[i].unsqueeze(0).expand(bsz_chunk, -1)
            p_k = self.meas_k_w[i].unsqueeze(0).expand(bsz_chunk, -1)
            p_v = self.meas_v_w[i].unsqueeze(0).expand(bsz_chunk, -1)
            p_all = torch.cat([p_q, p_k, p_v], dim=0)
            tqf.u3(qdev_all, wires=i, params=p_all)
            
        # Unified Measurement
        probs_all = self._measure_multibasis(qdev_all) # [3*bsz_chunk, 64]
        
        # Split results
        probs_q, probs_k, probs_v = torch.chunk(probs_all, 3, dim=0)
        return probs_q, probs_k, probs_v

    def _forward_impl(self, x_64: torch.Tensor, device_name: str) -> torch.Tensor:
        # 1. Prepare common state via Dense Angle Encoding
        B, S, D = x_64.shape
        bsz = B * S
        x_bsz = x_64.reshape(bsz, D)
        
        # Note: We work directly with input features (D = in_channels)
        # inp_proj is skipped as we use Dense Angle Encoding directly on inputs
        
        # Calculate Classical Residuals (Hybrid-Residual Architecture)
        # [Optimized] Disabled for Parameter Efficiency
        if self.q_res_proj is not None:
            q_res = self.q_res_proj(x_bsz).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
            k_res = self.k_res_proj(x_bsz).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
            v_res = self.v_res_proj(x_bsz).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        else:
            q_res = 0
            k_res = 0
            v_res = 0
        
        # Dense Angle Encoding: D -> Q_DEPTH * N_QUBITS * 2
        raw_out = self.dense_angle_proj(x_bsz)
        # Shape: [bsz, Q_DEPTH, N_QUBITS, 2]
        # Improved Mapping: Center around 0 (Identity) instead of pi/2
        # Range: [-pi, pi]
        angles = torch.tanh(raw_out) * torch.pi
        angles = angles.reshape(bsz, self.Q_DEPTH, self.N_QUBITS, 2)
        
        rx_angles = angles[:, :, :, 0] # [bsz, depth, n_qubits]
        ry_angles = angles[:, :, :, 1] # [bsz, depth, n_qubits]

        # Create common device (starts at |0>)
        # Optimization: Device Reuse
        qdev_common = None
        if self.reuse_device and self._qdev_common_cached is not None and self._qdev_common_cached_bsz == bsz:
             qdev_common = self._qdev_common_cached
             if hasattr(qdev_common, 'reset_states'):
                 qdev_common.reset_states(bsz=bsz)
             else:
                 qdev_common = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        else:
             qdev_common = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
             if self.reuse_device:
                 self._qdev_common_cached = qdev_common
                 self._qdev_common_cached_bsz = bsz
        
        # Apply Dense Encoding (Layer-wise injection)
        # enc_w shape: [Q_DEPTH, N_QUBITS, 3]
        for d in range(self.Q_DEPTH):
            # 1. Data Injection (Rx, Ry)
            for i in range(self.N_QUBITS):
                tqf.rx(qdev_common, wires=i, params=rx_angles[:, d, i])
                tqf.ry(qdev_common, wires=i, params=ry_angles[:, d, i])
            
            # 2. Trainable Unitary (PQC Layer)
            # Apply rotations from enc_w[d]
            for i in range(self.N_QUBITS):
                tqf.u3(qdev_common, wires=i, params=self.enc_w[d, i].unsqueeze(0))
            
            # 3. Entanglement (CNOT Ring)
            for i in range(self.N_QUBITS):
                tqf.cnot(qdev_common, wires=[i, (i + 1) % self.N_QUBITS])
            
        # Get common state (Flattened)
        if hasattr(qdev_common, 'get_states_1d'): 
            common_states_flat = qdev_common.get_states_1d()
        else: 
            common_states_flat = qdev_common.states.reshape(bsz, -1)
        
        # Data Re-uploading for branches (Standard 64->6 reupload)
        # Note: We could use Dense here too, but let's keep branches lightweight
        reupload_angles = torch.tanh(self.reupload_proj(x_bsz)) * torch.pi

        # 2. Fork to Q/K/V branches (Optimized Batch-Parallel with Chunking)
        q_list, k_list, v_list = [], [], []
        
        # Checkpoint-friendly Loop
        for i in range(0, bsz, self.chunk_size):
            end = min(i + self.chunk_size, bsz)
            chunk_states = common_states_flat[i:end]
            chunk_reupload = reupload_angles[i:end]
            
            if self.use_checkpoint and self.training:
                 # Checkpointing saves memory by re-computing forward pass during backward
                 chunk_q, chunk_k, chunk_v = checkpoint_utils.checkpoint(
                     self._process_qkv_chunk,
                     chunk_states,
                     chunk_reupload,
                     device_name,
                     use_reentrant=False
                 )
            else:
                 chunk_q, chunk_k, chunk_v = self._process_qkv_chunk(
                     chunk_states, 
                     chunk_reupload, 
                     device_name
                 )
            
            q_list.append(chunk_q)
            k_list.append(chunk_k)
            v_list.append(chunk_v)
            
        probs_q = torch.cat(q_list, dim=0)
        probs_k = torch.cat(k_list, dim=0)
        probs_v = torch.cat(v_list, dim=0)
        
        # 3. Projections (Batched execution results)
        q = self.qk_ln(self.q_proj(probs_q)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        q = q + q_res
        
        k = self.qk_ln(self.k_proj(probs_k)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        k = k + k_res
        
        v = self.v_proj(probs_v).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        v = v + v_res
        
        # 3. Attention Mechanism (Standard)
        # Manually implement attention logic since we can't easily call _attention_core from parent if it's not exposed
        # Actually _attention_core is not defined in QuantumAttentionAngle, it's just logic inside forward.
        # Let's copy the attention logic.
        
        # Optimized: Use Learnable Head-wise Scale
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_out = (attn @ v).transpose(1, 2).reshape(B, S, self.inner_dim)
        x_out = self.out_proj(x_out)
        
        return x_out

class QSANN_Angle_Dense_Pure(QuantumAttentionAngleDense):
    """
    Pure Quantum version of Dense Angle Encoding (No Classical Residuals).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-initialize Output Projection (Crucial for Pure Quantum)
        # Since we removed the classical residual path, we cannot start with zero output.
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Disable Classical Residuals
        for proj in [self.q_res_proj, self.k_res_proj, self.v_res_proj]:
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
            proj.weight.requires_grad = False
            if proj.bias is not None:
                proj.bias.requires_grad = False

class QuantumAttentionPatch(nn.Module):
    """
    Patch-based Grouped Quantum Attention.
    Implements: Patching -> Grouped Tokenization -> Quantum PQC for Q/K/V -> Classical Attention -> Patch Merging.
    """
    def __init__(self, dim, num_heads=4, q_depth=2, n_qubits=6, patch_size=2, device_name='cuda'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.q_dim = 2 ** n_qubits # 64
        self.device_name = device_name
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 1. Patch Embedding & Group Projection
        # Input: [B, S, D] -> Reshape [B, S/P, P*D] -> Project to [B, S/P, Q_DIM]
        # Assumes S is divisible by patch_size (P)
        self.group_dim = patch_size * dim
        # Single linear layer to map group to quantum dimension (User request: "只需要一个线性层")
        self.patch_proj = nn.Linear(self.group_dim, self.q_dim)
        
        # 2. Quantum Circuits for Q/K/V
        # Using simple Angle Encoding (Rx) + Basic Entanglement + Measurement
        # We share the encoding part, but have separate variational weights for Q/K/V
        self.enc_w = nn.Parameter(torch.randn(q_depth, n_qubits))
        self.q_w = nn.Parameter(torch.randn(q_depth, n_qubits))
        self.k_w = nn.Parameter(torch.randn(q_depth, n_qubits))
        self.v_w = nn.Parameter(torch.randn(q_depth, n_qubits))
        
        # 3. Output Projection
        # Quantum Output (64) -> Original Group Dimension -> Reshape back
        self.out_proj = nn.Linear(self.q_dim, self.group_dim)
        
        self._printed_exec = False

    def _apply_pqc(self, qdev, weights):
        # Simple PQC: Ry rotations + CNOT ring
        # weights: [depth, n_qubits]
        for d in range(weights.shape[0]):
            for i in range(self.n_qubits):
                tqf.ry(qdev, wires=i, params=weights[d][i].unsqueeze(0).repeat(qdev.bsz))
            # Ring CNOT
            for i in range(self.n_qubits):
                tqf.cnot(qdev, wires=[i, (i + 1) % self.n_qubits])

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        P = self.patch_size
        assert S % P == 0, f"Sequence length {S} must be divisible by patch size {P}"
        num_patches = S // P
        
        # 1. Patching & Projection
        # [B, S, D] -> [B, S/P, P, D] -> [B, S/P, P*D]
        x_patched = x.reshape(B, num_patches, P * D)
        # Project to Quantum Dim (64)
        x_q_in = torch.tanh(self.patch_proj(x_patched)) * torch.pi # Scale to [-pi, pi] for angle encoding
        
        bsz_q = B * num_patches
        x_q_flat = x_q_in.reshape(bsz_q, self.q_dim)
        
        # 2. Quantum Execution
        # We process Q, K, V in one go or separate devices. 
        # For efficiency, let's use one device per branch.
        
        # Common Device Init
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz_q, device=self.device_name)
        
        # Encoding (Rx) - Angle Encoding
        # Since input is 64-dim and we have 6 qubits, we can't map 1-to-1 easily if we want 6 rotations.
        # But wait, 6 qubits = 6 rotations. x_q_in is 64 dim.
        # We need to map 64 -> 6 for rotation parameters.
        # To strictly follow "minimal parameters", let's assume we slice or average, 
        # OR we just project to 6 in the first place?
        # User said: "只需要一个线性层将分组的特征图块进行维度映射"
        # So patch_proj should map to n_qubits (6) or n_qubits*depth?
        # Let's map to n_qubits for simple encoding.
        # RE-DEFINITION: patch_proj maps to n_qubits (6)
        
        # Correcting logic based on "minimal params":
        # self.patch_proj should be Linear(group_dim, n_qubits)
        # But wait, Q/K/V usually need high dim features. 6 dims is too small for attention key?
        # Let's stick to the user's "Quantum Attention Calculation". 
        # Usually Q-Attention replaces the matrix multiplication. 
        # But here user says "进行qkv线路的测量，最后进行经典的注意力机制的计算".
        # This implies Quantum is used to GENERATE Q, K, V vectors.
        # So: Input -> Quantum Circuit -> Measure -> Q, K, V vectors -> Classical Attention (Softmax...)
        
        # Re-adjusting Projection:
        # We need enough capacity. Let's map to n_qubits (6) for encoding angles.
        pass # Placeholder for the re-init in __init__

class QuantumAttentionPatch(nn.Module):
    """
    Patch-based Grouped Quantum Attention (Refined).
    Features:
    1. Amplitude Encoding (State Preparation).
    2. Data Re-uploading (In PQC).
    3. Multi-head Quantum Attention.
    """
    def __init__(self, dim, num_heads=4, q_depth=2, n_qubits=7, patch_size=2, device_name='cuda', lora_rank=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.device_name = device_name
        self.head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5)) # Learnable scale
        
        self.group_dim = patch_size * dim
        
        # Ensure n_qubits is sufficient for amplitude encoding
        # We need 2^n_qubits >= group_dim to encode group_dim amplitudes
        target_dim = 2 ** self.n_qubits
        assert target_dim >= self.group_dim, \
            f"n_qubits={self.n_qubits} (dim={target_dim}) insufficient for group_dim={self.group_dim}"
        
        # 1. Data Re-uploading Projection
        # Maps patch features to rotation angles for re-uploading in PQC
        # We use 3 params per qubit (U3 gate) for maximum expressivity
        self.reupload_proj = nn.Linear(self.group_dim, n_qubits * 3)
        
        # 2. Variational Circuits (Weights for U3 gates: theta, phi, lam)
        # Shape: [q_depth, n_qubits, 3]
        self.enc_w = nn.Parameter(torch.randn(q_depth, n_qubits, 3))
        self.q_w = nn.Parameter(torch.randn(q_depth, n_qubits, 3))
        self.k_w = nn.Parameter(torch.randn(q_depth, n_qubits, 3))
        self.v_w = nn.Parameter(torch.randn(q_depth, n_qubits, 3))
        
        # 3. Trainable Measurement Basis
        self.meas_w = nn.Parameter(torch.randn(n_qubits, 3))
        
        # 4. Measurement Projection
        # Measure expectations (n_qubits) -> group_dim -> reshape
        self.q_out = nn.Linear(n_qubits, self.group_dim)
        self.k_out = nn.Linear(n_qubits, self.group_dim)
        self.v_out = nn.Linear(n_qubits, self.group_dim)
        
        # 5. Hybrid Classical Projections (Residuals)
        # Replaced with Low-Rank Adapters (LoRA) for parameter efficiency
        # Rank=lora_rank reduces params. If lora_rank <= 0, disable classical residuals.
        self.use_classical_residual = lora_rank > 0
        if self.use_classical_residual:
            self.q_classical = nn.Sequential(
                nn.Linear(self.group_dim, lora_rank, bias=False),
                nn.Linear(lora_rank, self.group_dim, bias=False)
            )
            self.k_classical = nn.Sequential(
                nn.Linear(self.group_dim, lora_rank, bias=False),
                nn.Linear(lora_rank, self.group_dim, bias=False)
            )
            self.v_classical = nn.Sequential(
                nn.Linear(self.group_dim, lora_rank, bias=False),
                nn.Linear(lora_rank, self.group_dim, bias=False)
            )
        else:
            self.q_classical = None
            self.k_classical = None
            self.v_classical = None
        
        self.out_proj = nn.Linear(dim, dim)
        self._printed_exec = False

    def _apply_pqc(self, qdev, weights, x_reupload):
        # weights: [q_depth, n_qubits, 3] OR [B, q_depth, n_qubits, 3]
        # x_reupload: [bsz, n_qubits * 3]
        
        bsz = qdev.bsz
        # Reshape reupload for easier indexing: [bsz, n_qubits, 3]
        x_reup = x_reupload.reshape(bsz, self.n_qubits, 3)
        
        # Determine depth based on weights shape
        if weights.dim() == 4:
            depth = weights.shape[1]
        else:
            depth = weights.shape[0]
            
        for d in range(depth):
            # 1. Trainable Layer (U3)
            for i in range(self.n_qubits):
                # Expand weights to batch
                # If weights have batch dim (for parallel branches), use them directly
                if weights.dim() == 4: # [B, q_depth, n_qubits, 3]
                     params = weights[:, d, i, :] # [B, 3]
                else:
                     params = weights[d][i].unsqueeze(0).repeat(bsz, 1)
                
                # Explicit unpacking to avoid ambiguity in tqf.u3
                theta = params[:, 0]
                phi = params[:, 1]
                lam = params[:, 2]
                # Pass as list to ensure unpacking works correctly in tqf.u3
                tqf.u3(qdev, wires=i, params=torch.stack([theta, phi, lam], dim=-1))
            
            # 2. Data Re-uploading Layer (U3)
            # Inject input information into the circuit
            for i in range(self.n_qubits):
                params = x_reup[:, i, :]
                theta = params[:, 0]
            phi = params[:, 1]
            lam = params[:, 2]
            tqf.u3(qdev, wires=i, params=torch.stack([theta, phi, lam], dim=-1))
                
            # 3. Entanglement (CNOT Ring)
            for i in range(self.n_qubits):
                tqf.cnot(qdev, wires=[i, (i + 1) % self.n_qubits])

    def _get_expectations(self, qdev):
        # Trainable Measurement Basis
        bsz = qdev.bsz
        for i in range(self.n_qubits):
             params = self.meas_w[i].unsqueeze(0).repeat(bsz, 1)
             theta = params[:, 0]
             phi = params[:, 1]
             lam = params[:, 2]
             tqf.u3(qdev, wires=i, params=torch.stack([theta, phi, lam], dim=-1))

        # Calculate PauliZ expectation for each qubit
        meas = tq.MeasureAll(tq.PauliZ)
        return meas(qdev)

    def forward(self, x):
        B, S, D = x.shape
        P = self.patch_size
        num_patches = S // P
        
        # 1. Patching & Amplitude Encoding Prep
        # [B, S, D] -> [B, num_patches, P*D]
        x_patched = x.reshape(B, num_patches, P * D)
        bsz_q = B * num_patches
        
        # Normalize for Amplitude Encoding
        # Pad if necessary (though 128 == 2^7, so no padding needed if n_qubits=7)
        target_dim = 2 ** self.n_qubits
        if self.group_dim < target_dim:
            # Zero pad
            padding = torch.zeros(B, num_patches, target_dim - self.group_dim, device=x.device)
            x_amp = torch.cat([x_patched, padding], dim=-1)
        else:
            x_amp = x_patched
            
        # L2 Normalize state vector
        x_amp = F.normalize(x_amp, p=2, dim=-1)
        x_amp_flat = x_amp.reshape(bsz_q, target_dim)
        
        # 2. Re-uploading Angles
        # Map input features to rotation angles
        reupload_angles = torch.tanh(self.reupload_proj(x_patched)) * torch.pi # [B, num_patches, n_qubits*3]
        reupload_flat = reupload_angles.reshape(bsz_q, -1)
        
        # 3. Quantum Forward (Parallel Q/K/V)
        # We process Q, K, V in a single batch to maximize GPU parallelism
        
        # Prepare Batch: [Q_batch, K_batch, V_batch]
        bsz_total = 3 * bsz_q
        
        # Repeat states and reupload angles for 3 branches
        x_amp_total = x_amp_flat.repeat(3, 1) # [3*bsz_q, target_dim]
        reupload_total = reupload_flat.repeat(3, 1) # [3*bsz_q, -1]
        
        # Init Device for Total Batch
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz_total, device=self.device_name)
        target_shape = [bsz_total] + [2] * self.n_qubits
        qdev.states = x_amp_total.reshape(target_shape).type(torch.complex64)
        
        # Prepare Weights for Parallel Execution
        # enc_w is shared across Q, K, V -> Repeat 3 times
        # q_w, k_w, v_w are specific -> Concatenate
        
        # Expand enc_w to [3*bsz_q, q_depth, n_qubits, 3]
        enc_w_expanded = self.enc_w.unsqueeze(0).repeat(bsz_total, 1, 1, 1)
        
        # Construct branch weights
        # q_w: [q_depth, n_qubits, 3] -> [bsz_q, ...]
        q_w_expanded = self.q_w.unsqueeze(0).repeat(bsz_q, 1, 1, 1)
        k_w_expanded = self.k_w.unsqueeze(0).repeat(bsz_q, 1, 1, 1)
        v_w_expanded = self.v_w.unsqueeze(0).repeat(bsz_q, 1, 1, 1)
        
        branch_w_total = torch.cat([q_w_expanded, k_w_expanded, v_w_expanded], dim=0) # [3*bsz_q, ...]
        
        # Apply Shared Encoder PQC (Parallel)
        self._apply_pqc(qdev, enc_w_expanded, reupload_total)
        
        # Apply Specific Branch PQC (Parallel)
        self._apply_pqc(qdev, branch_w_total, reupload_total)
        
        # Measure All
        exp_total = self._get_expectations(qdev) # [3*bsz_q, n_qubits]
        
        # Split Results
        exp_q, exp_k, exp_v = torch.chunk(exp_total, 3, dim=0) # Each [bsz_q, n_qubits]
        
        # 4. Projection & Hybrid Residual
        # Quantum Projection
        q_quant = self.q_out(exp_q).reshape(B, num_patches, P*D)
        k_quant = self.k_out(exp_k).reshape(B, num_patches, P*D)
        v_quant = self.v_out(exp_v).reshape(B, num_patches, P*D)
        
        # Classical Projection (Residual)
        if self.use_classical_residual:
            q_class = self.q_classical(x_patched)
            k_class = self.k_classical(x_patched)
            v_class = self.v_classical(x_patched)
            
            # Combine
            q = (q_quant + q_class).reshape(B, S, D)
            k = (k_quant + k_class).reshape(B, S, D)
            v = (v_quant + v_class).reshape(B, S, D)
        else:
            q = q_quant.reshape(B, S, D)
            k = k_quant.reshape(B, S, D)
            v = v_quant.reshape(B, S, D)
        
        # 5. Classical Multi-Head Attention
        # Split heads
        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        
        x_out = self.out_proj(x_out)
        
        if not self._printed_exec:
            self._printed_exec = True
            print(f"QuantumAttentionPatch Exec: Patches={num_patches}, Qubits={self.n_qubits} (AmpEnc+ReUpload+ParallelQKV+HybridResidual)")
            
        return x_out


class ClassicAttention64(nn.Module):
    """
    标准多头自注意力（MHSA），作用在 64 维 token 上，返回形状 [B, S, 64]。
    """
    def __init__(self, num_heads: int = 8, attn_dropout: float = 0.0, force_fp32_attention: bool = True):
        super().__init__()
        assert 64 % int(num_heads) == 0
        self.num_heads = int(num_heads)
        self.head_dim = 64 // self.num_heads
        self.inner_dim = self.num_heads * self.head_dim  # =64
        self.scale = self.head_dim ** -0.5
        self.force_fp32_attention = bool(force_fp32_attention)

        self.to_q = nn.Linear(64, self.inner_dim, bias=False)
        self.to_k = nn.Linear(64, self.inner_dim, bias=False)
        self.to_v = nn.Linear(64, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, 64, bias=True)
        self.attn_drop = nn.Dropout(p=float(attn_dropout))

    def forward(self, x_64: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
        # x_64: [B, S, 64]
        B, S, D = x_64.shape
        assert D == 64
        dev = x_64.device

        def _forward_impl(inp: torch.Tensor) -> torch.Tensor:
            q = self.to_q(inp)  # [B,S,64]
            k = self.to_k(inp)
            v = self.to_v(inp)
            q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,S,hd]
            k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,S,hd]
            v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,S,hd]

            attn_logits = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # [B,H,S,S]
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
            out = torch.einsum('bhij,bhjd->bhid', attn_weights, v)  # [B,H,S,hd]
            out = out.transpose(1, 2).reshape(B, S, self.inner_dim)  # [B,S,64]
            out = self.to_out(out)  # [B,S,64]
            out = self.attn_drop(out)
            return out

        if self.force_fp32_attention and dev.type == 'cuda':
            with _AutocastOff(enabled=True):
                return _forward_impl(x_64.float()).to(x_64.dtype)
        else:
            return _forward_impl(x_64)


class PatchEmbed2D(nn.Module):
    """
    2D patch embedder with dual outputs:
      - tokens_384: DiT-style Conv2d(kernel=p, stride=p) -> (B, L, model_dim)
      - tokens_64: nn.Unfold(p, stride=p) -> reshape (B, L, 64), L2 normalized for amplitude encoding
    Assumes input latent tensor x: [B, C_in, H, W], with H=W divisible by p and C_in expected 4.
    """

    def __init__(self, in_channels: int, model_dim: int, patch_size: int = 4, eps: float = 1e-9, projection_type: str = 'linear'):
        super().__init__()
        assert patch_size > 0 and isinstance(patch_size, int)
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.eps = float(eps)
        self.projection_type = projection_type

        self.conv = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # Ensure unfolded tokens map to 64-d for quantum attention.
        D_unfold = in_channels * patch_size * patch_size
        
        if self.projection_type == 'mlp':
            # MLP Projection: Linear -> SiLU -> Linear (Better compression)
            hidden_dim = max(D_unfold, 64 * 4)
            self.unfold_proj64 = nn.Sequential(
                nn.Linear(D_unfold, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 64)
            )
        elif D_unfold == 64:
            self.unfold_proj64 = nn.Identity()
        else:
            self.unfold_proj64 = nn.Linear(D_unfold, 64)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "输入尺寸必须能被 patch_size 整除"
        # DiT-style tokens: [B, model_dim, H/p, W/p] -> [B, L, model_dim]
        t384 = self.conv(x)  # [B, model_dim, H/p, W/p]
        L = (H // self.patch_size) * (W // self.patch_size)
        t384 = t384.flatten(2).transpose(1, 2)  # [B, L, model_dim]

        # QSANN tokens: Unfold -> [B, C*p*p, L] -> [B, L, C*p*p] -> Linear to 64 -> L2 norm
        t64 = self.unfold(x)                    # [B, C*p*p, L]
        t64 = t64.transpose(1, 2)              # [B, L, C*p*p]
        # Map to 64-d if needed
        t64 = self.unfold_proj64(t64)          # [B, L, 64]
        # L2 normalize for amplitude encoding stability
        t64 = t64 / (t64.norm(p=2, dim=-1, keepdim=True) + self.eps)
        return t384, t64


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class QuantumMLP(nn.Module):
    """
    Quantum MLP that maps input -> quantum state -> PQC -> Measurement -> output.
    Can also be used as a pure circuit generator for other modules (e.g. QCNN).
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int = None,
                 n_qubits: int = 6, q_depth: int = 2, device_name: Optional[str] = None,
                 encoding: str = 'amplitude', re_uploading: bool = True, output_mlp_ratio: float = 0.0,
                 n_groups: int = 1, readout_mode: str = 'linear'):
        super().__init__()
        if not _TQ_AVAILABLE:
            raise ImportError("TorchQuantum not installed.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features or in_features 
        self.n_qubits = int(n_qubits)
        self.q_dim = 2 ** self.n_qubits # 64
        self.device_name = device_name
        self.encoding = encoding # 'amplitude' or 'angle'
        self.re_uploading = re_uploading
        self.eps = 1e-9
        self._printed_exec = False
        self.output_mlp_ratio = output_mlp_ratio
        self.n_groups = n_groups
        self.readout_mode = readout_mode
        
        # Ensure divisibility
        assert in_features % n_groups == 0, f"Input features {in_features} must be divisible by n_groups {n_groups}"
        self.in_features_per_group = in_features // n_groups
        
        # 1. Classical projection
        if self.encoding == 'amplitude':
            self.proj_in = nn.Linear(in_features, self.q_dim * self.n_groups)
            self.norm_in = nn.LayerNorm(self.q_dim)
            self.reupload_proj = nn.Linear(in_features, self.n_qubits * self.n_groups)
        else:
            self.proj_in = nn.Conv1d(
                in_channels=in_features,
                out_channels=self.n_qubits * self.n_groups,
                kernel_size=1,
                groups=self.n_groups
            )
        
        # 2. Quantum Circuit Parameters
        # [n_groups, depth, n_qubits, 3]
        self.q_weights = nn.Parameter(0.1 * torch.randn(self.n_groups, q_depth, self.n_qubits, 3))
        
        if self.re_uploading:
            self.upload_scales = nn.Parameter(torch.ones(self.n_groups, q_depth, self.n_qubits))

        # Learnable Input Scaling
        self.inp_scale = nn.Parameter(torch.ones(in_features))
        
        # Trainable Measurement Basis
        # [n_groups, n_qubits, 3]
        self.meas_w = nn.Parameter(0.1 * torch.randn(self.n_groups, self.n_qubits, 3))

        # 3. Output projection
        # Input to projection is [B, n_groups * 2^n_qubits]
        total_q_dim = self.n_groups * self.q_dim
        if self.readout_mode == 'linear':
            if self.output_mlp_ratio > 0:
                hidden_dim = int(total_q_dim * self.output_mlp_ratio)
                self.proj_out = nn.Sequential(
                    nn.Linear(total_q_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, out_features)
                )
            else:
                self.proj_out = nn.Linear(total_q_dim, out_features)
        elif self.readout_mode == 'expectation':
             self.measure_z = tq.MeasureAll(tq.PauliZ)
             self.proj_out = None
        else:
            self.proj_out = None

    def apply_circuit(self, qdev, wires, inputs):
        """
        Apply the MLP circuit (Encoding + PQC) to the given wires on qdev.
        Used for direct integration with QCNN.
        inputs: Raw input tensor [B, in_features]
        """
        # This method assumes SINGLE GROUP usage (legacy integration)
        # or we need to adapt it. For now, we assume simple usage.
        # 1. Pre-process inputs
        inputs = inputs * self.inp_scale
        if self.encoding == 'amplitude':
             # ... (existing amplitude logic)
             pass
        else:
            # Angle Encoding
            # Inputs: [B, in_features]
            # Proj: Conv1d [B, in, 1] -> [B, out, 1]
            # If n_groups > 1, this produces [B, n_groups*n_qubits]
            # But apply_circuit typically expects simple behavior.
            # We'll assume this is called only when n_groups=1 for legacy support
            # OR we implement full grouped logic here too?
            # QCNN usually calls this for TIME EMBEDDING which might not be grouped.
            pass

    def _apply_pqc(self, qdev: 'tq.QuantumDevice', weights: torch.Tensor):
        pass 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features]
        if not self._printed_exec:
            self._printed_exec = True

        B = x.shape[0]
        dev = x.device
        device_name = self.device_name or x.device.type
        
        # Input Scaling
        x = x * self.inp_scale
        
        # 1. Encoding
        if self.encoding == 'amplitude':
            # Support grouped amplitude encoding
            # [B, in] -> [B, n_groups * q_dim]
            if self.proj_in is None:
                x_q = x
            else:
                x_q = self.proj_in(x)
            
            # Reshape to [B, n_groups, q_dim]
            x_q = x_q.view(B, self.n_groups, self.q_dim)
            
            # Normalize per group state
            x_q = x_q / (x_q.norm(p=2, dim=-1, keepdim=True) + self.eps)
            
            # Flatten for batch processing: [B*n_groups, q_dim]
            states = x_q.reshape(B * self.n_groups, self.q_dim).to(torch.cfloat)

            # Re-uploading angles: [B, in] -> [B, n_groups * n_qubits]
            angles = torch.tanh(self.reupload_proj(x)) * math.pi
            # Flatten: [B*n_groups, n_qubits]
            angles = angles.view(B * self.n_groups, self.n_qubits)
            
            bsz_total = B * self.n_groups
            depth = self.q_weights.shape[1]
            
            # Expand weights: [n_groups, D, Q, 3] -> [B, n_groups, D, Q, 3] -> [B*n_groups, D, Q, 3]
            q_weights_flat = self.q_weights.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(bsz_total, -1, self.n_qubits, 3)
            
            if self.re_uploading:
                # [n_groups, D, Q] -> [B*n_groups, D, Q]
                upload_scales_flat = self.upload_scales.unsqueeze(0).expand(B, -1, -1, -1).reshape(bsz_total, -1, self.n_qubits)
            
            # [n_groups, Q, 3] -> [B*n_groups, Q, 3]
            meas_w_flat = self.meas_w.unsqueeze(0).expand(B, -1, -1, -1).reshape(bsz_total, self.n_qubits, 3)

            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz_total, device=device_name)
            if hasattr(qdev, 'set_states'):
                qdev.set_states(states)
            elif hasattr(qdev, 'set_states_1d'):
                qdev.set_states_1d(states)
            else:
                qdev.states = states

            for l in range(depth):
                if self.re_uploading:
                    for i in range(self.n_qubits):
                        scaled_angle = angles[:, i] * upload_scales_flat[:, l, i]
                        tqf.ry(qdev, wires=i, params=scaled_angle)
                rx_params = q_weights_flat[:, l, :, 0]
                ry_params = q_weights_flat[:, l, :, 1]
                ent_ry_params = q_weights_flat[:, l, :, 2]
                for i in range(self.n_qubits):
                    tqf.rx(qdev, wires=i, params=rx_params[:, i])
                    tqf.ry(qdev, wires=i, params=ry_params[:, i])
                n = self.n_qubits
                if n > 1:
                    for i in range(n):
                        tqf.cnot(qdev, wires=[i, (i + 1) % n])
                for i in range(self.n_qubits):
                    tqf.ry(qdev, wires=i, params=ent_ry_params[:, i])

            for i in range(self.n_qubits):
                tqf.u3(qdev, wires=i, params=meas_w_flat[:, i])

            if self.readout_mode == 'expectation':
                # [B*n_groups, n_qubits]
                expval = self.measure_z(qdev)
                # Reshape back: [B, n_groups, n_qubits]
                expval_grouped = expval.view(B, self.n_groups, self.n_qubits)
                # Flatten: [B, n_groups * n_qubits]
                output_flat = expval_grouped.view(B, -1)
            else:
                if hasattr(qdev, 'get_states'): states_out = qdev.get_states()
                elif hasattr(qdev, 'get_states_1d'): states_out = qdev.get_states_1d()
                else: states_out = qdev.states
                probs = (states_out.abs() ** 2)
                
                # Reshape back: [B, n_groups, 2^n_qubits]
                probs_grouped = probs.view(B, self.n_groups, self.q_dim)
                output_flat = probs_grouped.view(B, -1)

            if self.readout_mode == 'linear':
                if isinstance(self.proj_out, nn.Sequential):
                    p = self.proj_out[0].weight
                else:
                    p = self.proj_out.weight
                proj_dev = p.device
                proj_dtype = p.dtype
                out = self.proj_out(output_flat.to(proj_dev, dtype=proj_dtype))
                return out.to(x.device, dtype=x.dtype)
            else:
                td = output_flat.shape[-1]
                if td == self.out_features:
                    return output_flat.to(x.device, dtype=x.dtype)
                elif td > self.out_features:
                    return output_flat[:, :self.out_features].to(x.device, dtype=x.dtype)
                else:
                    pad = torch.zeros(B, self.out_features - td, device=output_flat.device, dtype=output_flat.dtype)
                    return torch.cat([output_flat, pad], dim=-1).to(x.device, dtype=x.dtype)
        else:
            # Angle Encoding via Grouped Conv1d
            # [B, C] -> [B, C, 1]
            x_in = x.unsqueeze(-1)
            # [B, n_groups * n_qubits, 1]
            x_enc = self.proj_in(x_in).squeeze(-1)
            
            # Tanh activation
            angles = torch.tanh(x_enc) * math.pi # [B, n_groups * n_qubits]
            
            # Reshape for Batch Grouping
            # [B, n_groups, n_qubits]
            angles_grouped = angles.view(B, self.n_groups, self.n_qubits)
            # Flatten to [B*n_groups, n_qubits]
            angles_flat = angles_grouped.view(-1, self.n_qubits)
            
            # Expand Weights for Batch Grouping
            # [n_groups, depth, n_qubits, 3] -> [B, n_groups, ...] -> [B*n_groups, ...]
            bsz_total = B * self.n_groups
            q_weights_flat = self.q_weights.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(bsz_total, -1, self.n_qubits, 3)
            
            if self.re_uploading:
                 upload_scales_flat = self.upload_scales.unsqueeze(0).expand(B, -1, -1, -1).reshape(bsz_total, -1, self.n_qubits)
                 
            meas_w_flat = self.meas_w.unsqueeze(0).expand(B, -1, -1, -1).reshape(bsz_total, self.n_qubits, 3)
            
            # Create Quantum Device for [B*n_groups]
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz_total, device=device_name)
            
            # Circuit Execution
            # Initial Encoding
            for i in range(self.n_qubits):
                tqf.ry(qdev, wires=i, params=angles_flat[:, i])

            # PQC
            depth = self.q_weights.shape[1]
            for l in range(depth):
                # Re-uploading
                if self.re_uploading and self.encoding == 'angle':
                    for i in range(self.n_qubits):
                        # Inject data again
                        scaled_angle = angles_flat[:, i] * upload_scales_flat[:, l, i]
                        tqf.ry(qdev, wires=i, params=scaled_angle)

                # Variational Layers
                rx_params = q_weights_flat[:, l, :, 0]
                ry_params = q_weights_flat[:, l, :, 1]
                ent_ry_params = q_weights_flat[:, l, :, 2]
                
                for i in range(self.n_qubits):
                    tqf.rx(qdev, wires=i, params=rx_params[:, i])
                    tqf.ry(qdev, wires=i, params=ry_params[:, i])
                
                # Entanglement (Ring)
                n = self.n_qubits
                if n > 1:
                    for i in range(n):
                        tqf.cnot(qdev, wires=[i, (i + 1) % n])
                
                for i in range(self.n_qubits):
                    tqf.ry(qdev, wires=i, params=ent_ry_params[:, i])
            
            # Trainable Measurement Basis
            for i in range(self.n_qubits):
                tqf.u3(qdev, wires=i, params=meas_w_flat[:, i])
            
            # 4. Measure
            if self.readout_mode == 'expectation':
                # [B*n_groups, n_qubits]
                expval = self.measure_z(qdev)
                # Reshape back: [B, n_groups, n_qubits]
                expval_grouped = expval.view(B, self.n_groups, self.n_qubits)
                # Flatten: [B, n_groups * n_qubits]
                output_flat = expval_grouped.view(B, -1)
            else:
                if hasattr(qdev, 'get_states'): states_out = qdev.get_states()
                elif hasattr(qdev, 'get_states_1d'): states_out = qdev.get_states_1d()
                else: states_out = qdev.states
                
                # [B*n_groups, 2^n_qubits]
                probs = (states_out.abs() ** 2)
                
                # Reshape back: [B, n_groups, 2^n_qubits]
                probs_grouped = probs.view(B, self.n_groups, self.q_dim)
                
                # Flatten: [B, n_groups * 2^n_qubits]
                output_flat = probs_grouped.view(B, -1)
            
            if self.readout_mode == 'linear':
                if isinstance(self.proj_out, nn.Sequential):
                    p = self.proj_out[0].weight
                else:
                    p = self.proj_out.weight
                proj_dev = p.device
                proj_dtype = p.dtype
                out = self.proj_out(output_flat.to(proj_dev, dtype=proj_dtype))
                return out.to(x.device, dtype=x.dtype)
            else:
                td = output_flat.shape[-1]
                if td == self.out_features:
                    return output_flat.to(x.device, dtype=x.dtype)
                elif td > self.out_features:
                    return output_flat[:, :self.out_features].to(x.device, dtype=x.dtype)
                else:
                    pad = torch.zeros(B, self.out_features - td, device=output_flat.device, dtype=output_flat.dtype)
                    return torch.cat([output_flat, pad], dim=-1).to(x.device, dtype=x.dtype)


class QuantumAdaGN(nn.Module):
    """
    Quantum Adaptive Group Normalization (Injection Layer).
    Replaces classic AdaGN: x * (1+scale) + shift
    Logic:
      1. Encode x into quantum state.
      2. Use 'style' (from Affine) to control rotation gates on the state.
      3. Measure to get modulated output.
    """
    def __init__(self, channels: int, style_dim: int, n_qubits: int = 6, q_depth: int = 1, device_name: Optional[str] = None):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.device_name = device_name
        self.eps = 1e-9

        # Project input channels to quantum dimension (64)
        self.q_dim = 2 ** self.n_qubits
        self.in_proj = nn.Linear(channels, self.q_dim)
        
        # Mapper from style to rotation parameters [B, depth, n_qubits, 3]
        self.style_mapper = nn.Linear(style_dim, q_depth * n_qubits * 3)
        
        # PQC weights (trainable base parameters)
        self.weights = nn.Parameter(0.1 * torch.randn(q_depth, n_qubits, 3))
        
        # Output projection back to channels
        self.out_proj = nn.Linear(self.q_dim, channels)
        
        if _TQ_AVAILABLE:
            self.measure_z = tq.MeasureAll(tq.PauliZ)

    # @torch.jit.script # TQ functional calls are not scriptable due to global dict lookups
    # @torch.compile(mode="max-autotune") # We will use torch.compile externally or on the wrapper
    def _apply_fusion_circuit(self, qdev, sub_bsz, sub_da, interaction_wires, data_wires, 
                              mod_params, qcnn_rot_params, 
                              n_qubits_data: int, n_qubits_ancilla: int, active_layers: int, 
                              use_strided_cnot: bool, reupload_data: bool):
        # 2. Encode Data (RY)
        # mod_params: [1, n_data, 3] or [n_layers, n_data, 3]? 
        # In forward, we pass `chunk_mod_params` which is [sub_bsz, n_layers, n_data, 3].
        # But here mod_params is passed as `self.mod_params` in one branch, or `chunk_mod_params` in another.
        # Let's standardize on passing TENSORS (parameters) to this function, not `self.xxx`.
        # The `forward` method calls this with `self.mod_params` (global) or chunked.
        
        # NOTE: TorchQuantum functional ops (tqf.ry, tqf.cnot) operate on qdev.
        # They are Python loops invoking underlying kernels.
        # To make this compilable, we need to minimize Python logic or ensure it traces well.
        # TQ operations are generally traceable if they use standard PyTorch ops internally.
        
        for i in range(n_qubits_data):
            tqf.ry(qdev, wires=i, params=sub_da[:, i])
        
        # 3. Entanglement (Ancilla -> Data) with Split Control
        # This loop logic is static (structure doesn't change), so it should trace fine.
        for i in range(n_qubits_data):
            ancilla_idx = i % n_qubits_ancilla
            ctl = interaction_wires[ancilla_idx]
            tgt = data_wires[i]
            # mod_params shape handling:
            # If mod_params is [sub_bsz, n_layers, n_data, 3], we need to slice it.
            # If mod_params is [n_groups, n_layers, n_data, 3], we need expansion.
            # The caller `forward` ensures params are ready.
            # Here we assume mod_params is [sub_bsz, n_layers, n_data, 3].
            # Wait, `forward` passes `chunk_mod_params` which is [sub_bsz, n_layers, n_data, 3].
            
            # strength = mod_params[0, i, 0].expand(sub_bsz) -> This assumed mod_params was [layers, data, 3] global param.
            # But in `forward`, we now use `chunk_mod_params` which includes batch dimension if we are in Grouped mode?
            # In Grouped mode, we expanded params to [sub_bsz, ...].
            # Let's verify `forward` logic.
            
            # In `forward`:
            # chunk_mod_params = mod_params_expanded[s:e] -> [sub_bsz, n_layers, n_data, 3]
            # So `mod_params` here IS [sub_bsz, n_layers, n_data, 3].
            
            strength = mod_params[:, 0, i, 0] # [sub_bsz]
            
            if ancilla_idx % 2 == 0:
                tqf.crx(qdev, wires=[ctl, tgt], params=strength)
            else:
                tqf.crz(qdev, wires=[ctl, tgt], params=strength)
        
        # 4. Spatial QCNN Backbone
        for l in range(active_layers):
            for i in range(n_qubits_data):
                ry_params = qcnn_rot_params[:, l, i, 0, 0] # [sub_bsz]
                rz_params = qcnn_rot_params[:, l, i, 1, 0]
                tqf.ry(qdev, wires=i, params=ry_params)
                tqf.rz(qdev, wires=i, params=rz_params)
            for i in range(n_qubits_data):
                tqf.cnot(qdev, wires=[i, (i + 1) % n_qubits_data])
            if use_strided_cnot and n_qubits_data >= 4:
                for i in range(n_qubits_data):
                    tqf.cnot(qdev, wires=[i, (i + 2) % n_qubits_data])
            if reupload_data and (l < active_layers - 1):
                for i in range(n_qubits_data):
                    tqf.rz(qdev, wires=i, params=sub_da[:, i])

    # @torch.jit.script # TQ functional calls are not scriptable due to global dict lookups
    def _fast_ry_layer(self, qdev, params):
        # params: [B, N]
        # Construct and apply full layer unitary
        bsz = params.shape[0]
        n_qubits = params.shape[1]
        dim = 2 ** n_qubits
        
        # 1. Construct Rot Matrices [B, N, 2, 2]
        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        # [B, N, 2, 2]
        mats = torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1)
        ], dim=-2).to(qdev.states.dtype)
        
        # 2. Batch Kron
        # Start with [B, 2, 2]
        res = mats[:, 0]
        for i in range(1, n_qubits):
            # einsum: bik, bjl -> bijkl -> reshape
            # [B, dim_curr, dim_curr] x [B, 2, 2]
            res = torch.einsum('bik,bjl->bijkl', res, mats[:, i])
            new_dim = res.shape[1] * 2
            res = res.reshape(bsz, new_dim, new_dim)
                
            # 3. Apply
        # qdev.states: [B, 2, 2...]
        # Handle potential ancilla qubits (extra dimensions)
        original_shape = qdev.states.shape
        flat_states = qdev.states.reshape(bsz, dim, -1)
        new_states = torch.bmm(res, flat_states)
        qdev.states = new_states.reshape(original_shape)

    def _fast_rz_layer(self, qdev, params):
        # params: [B, N]
        # Diagonal fusion
        bsz = params.shape[0]
        n_qubits = params.shape[1]
        dim = 2 ** n_qubits
        device = params.device
        
        # Precompute signs if not cached
        cache_key = (n_qubits, device)
        if not hasattr(self, '_rz_signs_cache') or self._rz_signs_cache_key != cache_key:
            arange = torch.arange(dim, device=device)
            # bits: [dim, n] - extract bits. Wire 0 is MSB or LSB?
            # TQ uses standard tensor product order: q0 (x) q1 ...
            # So index 0..2^N-1.
            # q0 is MSB (stride 2^(N-1))
            # bits: [dim, n]
            # (arange >> shift) & 1
            shifts = torch.arange(n_qubits - 1, -1, -1, device=device)
            bits = (arange.unsqueeze(1) >> shifts) & 1
            # 0 -> -0.5, 1 -> 0.5
            # exp(-i * theta/2) for 0
            # exp(i * theta/2) for 1
            # coeff: if 0 -> -0.5, if 1 -> 0.5
            signs = (bits.float() - 0.5) 
            self._rz_signs_cache = signs
            self._rz_signs_cache_key = cache_key
        
        signs = self._rz_signs_cache # [dim, n]
        
        # phases: [B, dim]
        # sum_j (signs[k, j] * params[b, j])
        phases = torch.matmul(params, signs.T)
        
        # rot: exp(i * phases)
        rot_diag = torch.complex(torch.cos(phases), torch.sin(phases))
        
        # Apply
        original_shape = qdev.states.shape
        flat_states = qdev.states.reshape(bsz, dim, -1)
        rot_diag_expanded = rot_diag.unsqueeze(2)
        flat_states = flat_states * rot_diag_expanded
        qdev.states = flat_states.reshape(original_shape)

    def _fast_cnot_layer(self, qdev, n_qubits, use_strided):
        # Permutation fusion
        bsz = qdev.bsz
        dim = 2 ** n_qubits
        device = qdev.states.device
        
        cache_key = (n_qubits, use_strided, device)
        if not hasattr(self, '_cnot_perm_cache') or self._cnot_perm_cache_key != cache_key:
            # Construct permutation
            # Apply CNOTs to basis states indices
            indices = torch.arange(dim, device=device)
            
            # Helper to flip bit
            def flip_bit(inds, target):
                # target is 0..N-1 (0 is MSB)
                # bit mask: 1 << (N - 1 - target)
                mask = 1 << (n_qubits - 1 - target)
                return inds ^ mask
                
            # Helper to check control
            def check_bit(inds, control):
                mask = 1 << (n_qubits - 1 - control)
                return (inds & mask) != 0
            
            # Apply ring CNOTs
            for i in range(n_qubits):
                ctl = i
                tgt = (i + 1) % n_qubits
                # If bit ctl is 1, flip tgt
                mask_ctl = 1 << (n_qubits - 1 - ctl)
                mask_tgt = 1 << (n_qubits - 1 - tgt)
                
                # Where ctl is 1
                should_flip = (indices & mask_ctl) != 0
                # Flip tgt where should_flip
                indices = torch.where(should_flip, indices ^ mask_tgt, indices)
                
            if use_strided and n_qubits >= 4:
                for i in range(n_qubits):
                    ctl = i
                    tgt = (i + 2) % n_qubits
                    mask_ctl = 1 << (n_qubits - 1 - ctl)
                    mask_tgt = 1 << (n_qubits - 1 - tgt)
                    should_flip = (indices & mask_ctl) != 0
                    indices = torch.where(should_flip, indices ^ mask_tgt, indices)
            
            self._cnot_perm_cache = torch.argsort(indices)
            self._cnot_perm_cache_key = cache_key
            
        perm = self._cnot_perm_cache
        
        # Apply permutation
        # [B, dim]
        flat_states = qdev.states.reshape(bsz, dim)
        flat_states = flat_states[:, perm]
        qdev.states = flat_states.reshape([bsz] + [2]*n_qubits)

    def _fast_ry(self, qdev, wires, params):
        # params: [B]
        if apply_unitary_bmm is None:
            tqf.ry(qdev, wires=wires, params=params)
            return
            
        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        # [B, 2, 2]
        matrix = torch.stack([
            torch.stack([c, -s], dim=1),
            torch.stack([s, c], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, [wires] if isinstance(wires, int) else wires)

    def _fast_rz(self, qdev, wires, params):
        if apply_unitary_bmm is None:
            tqf.rz(qdev, wires=wires, params=params)
            return
            
        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        e_neg = torch.complex(c, -s)
        e_pos = torch.complex(c, s)
        
        matrix = torch.stack([
            torch.stack([e_neg, torch.zeros_like(e_neg)], dim=1),
            torch.stack([torch.zeros_like(e_pos), e_pos], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, [wires] if isinstance(wires, int) else wires)

    def _fast_cnot(self, qdev, wires):
        if apply_unitary_bmm is None:
            tqf.cnot(qdev, wires=wires)
            return
            
        if not hasattr(self, '_cnot_mat') or self._cnot_mat.device != qdev.states.device:
             m = torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=qdev.states.dtype, device=qdev.states.device)
             self._cnot_mat = m
             
        qdev.states = apply_unitary_bmm(qdev.states, self._cnot_mat, wires)

    def _fast_crx(self, qdev, wires, params):
        if apply_unitary_bmm is None:
            tqf.crx(qdev, wires=wires, params=params)
            return

        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        
        # CRX = diag(I, RX(theta))
        # RX = [[c, -is], [-is, c]]
        # top left I: [[1, 0], [0, 1]]
        # [B, 4, 4]
        
        bsz = theta.shape[0]
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        
        # Construct via stacking
        # Row 0: 1, 0, 0, 0
        # Row 1: 0, 1, 0, 0
        # Row 2: 0, 0, c, -is
        # Row 3: 0, 0, -is, c
        
        # complex -is
        neg_is = torch.complex(zeros, -s)
        c_complex = torch.complex(c, zeros)
        one_complex = torch.complex(ones, zeros)
        zero_complex = torch.complex(zeros, zeros)
        
        matrix = torch.stack([
            torch.stack([one_complex, zero_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, one_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, zero_complex, c_complex, neg_is], dim=1),
            torch.stack([zero_complex, zero_complex, neg_is, c_complex], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, wires)

    def _fast_crz(self, qdev, wires, params):
        if apply_unitary_bmm is None:
            tqf.crz(qdev, wires=wires, params=params)
            return
            
        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        e_neg = torch.complex(c, -s)
        e_pos = torch.complex(c, s)
        
        # CRZ = diag(1, 1, e_neg, e_pos)
        
        bsz = theta.shape[0]
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        one_complex = torch.complex(ones, zeros)
        zero_complex = torch.complex(zeros, zeros)
        
        matrix = torch.stack([
            torch.stack([one_complex, zero_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, one_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, zero_complex, e_neg, zero_complex], dim=1),
            torch.stack([zero_complex, zero_complex, zero_complex, e_pos], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, wires)

    def _apply_fusion_circuit(self, qdev, sub_bsz, sub_da, sub_sa, interaction_wires, data_wires, 
                              mod_params, qcnn_rot_params, 
                              n_qubits_data: int, n_qubits_ancilla: int, active_layers: int, 
                              use_strided_cnot: bool, reupload_data: bool):
        # 2. Encode Data (RY)
        for i in range(n_qubits_data):
            tqf.ry(qdev, wires=i, params=(sub_da[:, i] + sub_sa[:, i]))
        
        # 3. Entanglement (Ancilla -> Data) with Split Control
        for i in range(n_qubits_data):
            ancilla_idx = i % n_qubits_ancilla
            ctl = interaction_wires[ancilla_idx]
            tgt = data_wires[i]
            # mod_params: [n_layers, n_data, 3]
            strength = mod_params[0, i, 0].expand(sub_bsz)
            if ancilla_idx % 2 == 0:
                tqf.crx(qdev, wires=[ctl, tgt], params=strength)
            else:
                tqf.crz(qdev, wires=[ctl, tgt], params=strength)
        
        # 4. Spatial QCNN Backbone
        for l in range(active_layers):
            for i in range(n_qubits_data):
                ry_params = qcnn_rot_params[l, i, 0, 0].expand(sub_bsz)
                rz_params = qcnn_rot_params[l, i, 1, 0].expand(sub_bsz)
                tqf.ry(qdev, wires=i, params=ry_params)
                tqf.rz(qdev, wires=i, params=rz_params)
            for i in range(n_qubits_data):
                tqf.cnot(qdev, wires=[i, (i + 1) % n_qubits_data])
            if use_strided_cnot and n_qubits_data >= 4:
                for i in range(n_qubits_data):
                    tqf.cnot(qdev, wires=[i, (i + 2) % n_qubits_data])
            if reupload_data and (l < active_layers - 1):
                for i in range(n_qubits_data):
                    # Internal Time Embedding: Re-upload both Data and Style at each layer
                    tqf.rz(qdev, wires=i, params=(sub_da[:, i] + sub_sa[:, i]))

    def _batch_kron(self, mat_list, sub_bsz):
        res = mat_list[0]
        for m in mat_list[1:]:
            res = torch.einsum('bik,bjl->bijkl', res, m).reshape(sub_bsz, res.shape[1]*m.shape[1], res.shape[2]*m.shape[2])
        return res

    def _fast_batch_rot_layer(self, qdev, n_qubits, ry_params, rz_params, sub_bsz):
        # ry_params, rz_params: [B, N]
        if n_qubits > 8: # Fallback for large systems
            for i in range(n_qubits):
                self._fast_ry(qdev, i, ry_params[:, i])
                self._fast_rz(qdev, i, rz_params[:, i])
            return

        mats = []
        dtype = qdev.states.dtype
        device = qdev.states.device
        
        for i in range(n_qubits):
            # RY
            theta = ry_params[:, i]
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            m_ry = torch.stack([torch.stack([c, -s], 1), torch.stack([s, c], 1)], 1).to(dtype)
            
            # RZ
            theta = rz_params[:, i]
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            e_neg = torch.complex(c, -s)
            e_pos = torch.complex(c, s)
            z = torch.zeros_like(c)
            m_rz = torch.stack([torch.stack([e_neg, z], 1), torch.stack([z, e_pos], 1)], 1).to(dtype)
            
            mats.append(torch.bmm(m_rz, m_ry))
            
        full_op = self._batch_kron(mats, sub_bsz)
        qdev.states = apply_unitary_bmm(qdev.states, full_op, list(range(n_qubits)))

    def _fast_batch_cnot_layer(self, qdev, n_qubits, stride, sub_bsz):
        if n_qubits > 8:
            for i in range(n_qubits):
                self._fast_cnot(qdev, wires=[i, (i + stride) % n_qubits])
            return
            
        # Group disjoint pairs
        # Simple greedy grouping
        pairs = [(i, (i + stride) % n_qubits) for i in range(n_qubits)]
        
        # Split into layers where no qubit is used twice
        layers = []
        while pairs:
            current_layer = []
            used_qubits = set()
            remaining_pairs = []
            for u, v in pairs:
                if u not in used_qubits and v not in used_qubits:
                    current_layer.append((u, v))
                    used_qubits.add(u)
                    used_qubits.add(v)
                else:
                    remaining_pairs.append((u, v))
            layers.append(current_layer)
            pairs = remaining_pairs
            
        # Apply each layer
        dtype = qdev.states.dtype
        device = qdev.states.device
        
        # CNOT matrix
        cnot_mat = torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=dtype, device=device)
        cnot_mat = cnot_mat.unsqueeze(0).expand(sub_bsz, -1, -1)
        
        # Identity matrix
        eye_mat = torch.eye(2, dtype=dtype, device=device).unsqueeze(0).expand(sub_bsz, -1, -1)
        
        for layer_pairs in layers:
            # Construct full operator for this layer
            # Map qubit -> op
            qubit_ops = {}
            for u, v in layer_pairs:
                qubit_ops[u] = ('cnot_ctl', v)
                qubit_ops[v] = ('cnot_tgt', u)
            
            # Iterate 0..N-1
            mats = []
            i = 0
            while i < n_qubits:
                if i in qubit_ops:
                    op_type, other = qubit_ops[i]
                    if op_type == 'cnot_ctl':
                        # Found a pair (i, other)
                        # Depending on order, might need SWAP logic if we construct linearly
                        # BUT batch_kron assumes order 0, 1, 2...
                        # If pair is (i, i+1), we append CNOT.
                        # If pair is (i, i+k), we can't easily use batch_kron unless they are adjacent in list?
                        # Wait, batch_kron constructs Tensor Product M0 (x) M1 ...
                        # This corresponds to applying M0 on wire 0, M1 on wire 1...
                        # CNOT(0, 2) CANNOT be represented as M0 (x) M1 (x) M2 directly if wires are permuted?
                        # Actually, if we use apply_unitary_bmm with wires=[0...N], the matrix MUST represent the operator on those wires.
                        # Operator on 0,1,2: CNOT(0,2) (x) I(1).
                        # This is NOT CNOT (x) I. It involves permutation.
                        # SWAP gates are needed to bring them adjacent.
                        
                        # So fusion of non-adjacent CNOTs into a single large matrix is TRICKY.
                        # We need to construct the matrix elements.
                        pass
                    pass
                i += 1
                
        # Actually, simpler approach for CNOTs:
        # Just use sequential application for now, OR only fuse adjacent CNOTs.
        # Ring CNOTs: (0,1), (1,2)... are adjacent.
        # (N-1, 0) is not.
        
        # Let's fallback to sequential for CNOTs for now to avoid bugs, 
        # but optimize the RY/RZ part which is 2/3 of the operations.
        # The RY/RZ part is strictly local (1-qubit), so fusion is trivial.
        
        for i in range(n_qubits):
             self._fast_cnot(qdev, wires=[i, (i + stride) % n_qubits])

    # @torch.jit.script # TQ functional calls are not scriptable due to global dict lookups
    def _apply_fusion_circuit(self, qdev, sub_bsz, sub_da, sub_sa, interaction_wires, data_wires, 
                              mod_params, qcnn_rot_params, 
                              n_qubits_data: int, n_qubits_ancilla: int, active_layers: int, 
                              use_strided_cnot: bool, reupload_data: bool, encoding_type: str):
        # 2. Encode Data
        if encoding_type == 'amplitude':
            # Amplitude Encoding: Data is already encoded in state vector.
            # We only apply Style Modulation here (if any)
            if sub_sa is not None:
                for i in range(n_qubits_data):
                    tqf.ry(qdev, wires=i, params=sub_sa[:, i])
        else:
            # Angle Encoding (RY)
            # Integrated Fusion: Data + Style
            if sub_sa is not None:
                 init_params = sub_da + sub_sa
            else:
                 init_params = sub_da
                 
            for i in range(n_qubits_data):
                tqf.ry(qdev, wires=i, params=init_params[:, i])
        
        # 3. Entanglement (Ancilla -> Data) with Split Control
        # If interaction_wires is provided (Ancilla Mode)
        if interaction_wires is not None and data_wires is not None:
            for i in range(n_qubits_data):
                ancilla_idx = i % n_qubits_ancilla
                ctl = interaction_wires[ancilla_idx]
                tgt = data_wires[i]
                
                # mod_params: [n_layers, n_data, 3] OR [B, n_layers, n_data, 3]
                if mod_params.ndim == 4 and mod_params.shape[0] == sub_bsz:
                     strength = mod_params[:, 0, i, 0]
                else:
                     strength = mod_params[0, i, 0].expand(sub_bsz)
                     
                if ancilla_idx % 2 == 0:
                    tqf.crx(qdev, wires=[ctl, tgt], params=strength)
                else:
                    tqf.crz(qdev, wires=[ctl, tgt], params=strength)
        
        # 4. Spatial QCNN Backbone
        for l in range(active_layers):
            for i in range(n_qubits_data):
                # qcnn_rot_params: [L, N, 2, 3] OR [B, L, N, 2, 3]
                # Debug info
                if l == 0 and i == 0 and not self._printed_exec:
                    print(f"DEBUG: qcnn_rot_params shape: {qcnn_rot_params.shape}, ndim: {qcnn_rot_params.ndim}, sub_bsz: {sub_bsz}")
                    self._printed_exec = True

                if qcnn_rot_params.ndim == 5 and qcnn_rot_params.shape[0] == sub_bsz:
                    ry_params = qcnn_rot_params[:, l, i, 0, 0]
                    rz_params = qcnn_rot_params[:, l, i, 1, 0]
                else:
                    ry_params = qcnn_rot_params[l, i, 0, 0].expand(sub_bsz)
                    rz_params = qcnn_rot_params[l, i, 1, 0].expand(sub_bsz)
                    
                tqf.ry(qdev, wires=i, params=ry_params)
                tqf.rz(qdev, wires=i, params=rz_params)
            for i in range(n_qubits_data):
                tqf.cnot(qdev, wires=[i, (i + 1) % n_qubits_data])
            if use_strided_cnot and n_qubits_data >= 4:
                for i in range(n_qubits_data):
                    tqf.cnot(qdev, wires=[i, (i + 2) % n_qubits_data])
            if reupload_data and (l < active_layers - 1):
                # [SOTA Update] Frequency Modulation Re-uploading (Q-Middle-Freq)
                # At the middle layer, we use multiplicative scaling for effective noise intensity modulation.
                middle_layer = active_layers // 2
                is_middle_reupload = (l == middle_layer)
                
                for i in range(n_qubits_data):
                    # Fusion Re-uploading
                    if sub_sa is not None:
                        if is_middle_reupload:
                            # Multiplicative Scaling: Data * (1 + Style)
                            # Simulates Frequency Modulation (Scale) rather than Phase Shift (Shift)
                            reup_params = sub_da[:, i] * (1.0 + sub_sa[:, i])
                            tqf.rz(qdev, wires=i, params=reup_params)
                        else:
                            # Standard Additive for other layers
                            tqf.rz(qdev, wires=i, params=(sub_da[:, i] + sub_sa[:, i]))
                    else:
                        tqf.rz(qdev, wires=i, params=sub_da[:, i])

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> need to treat each pixel as a token or channel-wise?
        # Standard AdaGN works on channels. Let's treat (H,W) as batch/sequence for quantum processing
        # Reshape: [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C) # [N_pixels, C]
        
        # Style needs to be expanded to pixels: [B, style_dim] -> [B, 1, 1, style_dim] -> [B, H, W, style_dim] -> [N_pixels, style_dim]
        style_flat = style.view(B, 1, 1, -1).expand(B, H, W, -1).reshape(-1, self.style_dim)
        
        # 1. Map to Quantum Dim
        x_q = self.in_proj(x_flat) # [N, 64]
        
        # 2. Amplitude Encode
        x_q = x_q / (x_q.norm(p=2, dim=-1, keepdim=True) + self.eps)
        states = x_q.to(torch.cfloat)
        
        # 3. Style to Params
        # params: [N, depth * n_qubits * 3]
        params = self.style_mapper(style_flat).reshape(-1, self.q_depth, self.n_qubits, 3)
        
        # Combined params: base weights + style modulation
        # This implements the "injection"
        total_params = self.weights.unsqueeze(0) + params
        
        # 4. Quantum Simulation
        # Since N_pixels is large, we might need chunking or verify if TQ handles large batch
        # For efficiency in this demo, we assume it fits or rely on TQ's batch handling.
        bsz = x_flat.shape[0]
        dev = x.device
        device_name = self.device_name or x.device.type
        
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=device_name)
        
        if hasattr(qdev, 'set_states'): qdev.set_states(states)
        elif hasattr(qdev, 'set_states_1d'): qdev.set_states_1d(states)
        else: qdev.states = states
        
        # Apply PQC with style-modulated parameters
        for l in range(self.q_depth):
            for i in range(self.n_qubits):
                tqf.rx(qdev, wires=i, params=total_params[:, l, i, 0])
                tqf.ry(qdev, wires=i, params=total_params[:, l, i, 1])
            for i in range(self.n_qubits - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            for i in range(self.n_qubits):
                tqf.ry(qdev, wires=i, params=total_params[:, l, i, 2])
                
        # 5. Measure
        if hasattr(qdev, 'get_states'): states_out = qdev.get_states()
        elif hasattr(qdev, 'get_states_1d'): states_out = qdev.get_states_1d()
        else: states_out = qdev.states
        probs = (states_out.abs() ** 2)
        
        # 6. Project back
        out_flat = self.out_proj(probs) # [N, C]
        
        # Reshape back to [B, C, H, W]
        out = out_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        return out.to(x.device, dtype=x.dtype)


class QuantumConv2d(nn.Module):
    """
    Quantum Convolution Layer.
    Replaces classic Conv2d(3x3).
    Uses Unfold -> Quantum Processing -> Fold (or Reshape).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, 
                 n_qubits: int = 6, q_depth: int = 2, device_name: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.device_name = device_name
        self.eps = 1e-9
        
        # Unfold extracts patches. Dimension of a patch: in_channels * kernel * kernel
        self.patch_dim = in_channels * kernel_size * kernel_size
        self.q_dim = 2 ** self.n_qubits
        
        # Input Scaling
        self.inp_scale = nn.Parameter(torch.ones(self.patch_dim))
        # Trainable Measurement
        self.meas_w = nn.Parameter(0.1 * torch.randn(n_qubits, 3))
        
        # Project patch to quantum dim
        self.in_proj = nn.Linear(self.patch_dim, self.q_dim)
        
        # PQC weights
        self.weights = nn.Parameter(0.1 * torch.randn(q_depth, n_qubits, 3))
        
        # Output projection
        self.out_proj = nn.Linear(self.q_dim, out_channels)
        
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1. Unfold: [B, patch_dim, L] where L = H_out * W_out
        patches = self.unfold(x) 
        L = patches.shape[-1]
        
        # Transpose to [B*L, patch_dim] for batch processing
        patches_flat = patches.transpose(1, 2).reshape(-1, self.patch_dim)
        
        # Apply Input Scale
        patches_flat = patches_flat * self.inp_scale
        
        # 2. Map to Quantum Dim
        x_q = self.in_proj(patches_flat)
        
        # 3. Amplitude Encode
        x_q = x_q / (x_q.norm(p=2, dim=-1, keepdim=True) + self.eps)
        states = x_q.to(torch.cfloat)
        
        # 4. Quantum Simulation
        bsz = x_q.shape[0]
        dev = x.device
        device_name = self.device_name or x.device.type
        
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=device_name)
        
        if hasattr(qdev, 'set_states'): qdev.set_states(states)
        elif hasattr(qdev, 'set_states_1d'): qdev.set_states_1d(states)
        else: qdev.states = states
        
        # Apply static PQC (convolution kernel)
        for l in range(self.q_depth):
            rx_params = self.weights[l, :, 0]
            ry_params = self.weights[l, :, 1]
            ent_ry_params = self.weights[l, :, 2]
            for i in range(self.n_qubits):
                tqf.rx(qdev, wires=i, params=rx_params[i])
                tqf.ry(qdev, wires=i, params=ry_params[i])
            for i in range(self.n_qubits - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            for i in range(self.n_qubits):
                tqf.ry(qdev, wires=i, params=ent_ry_params[i])
        
        # Trainable Measurement
        for i in range(self.n_qubits):
            tqf.u3(qdev, wires=i, params=self.meas_w[i])
                
        # 5. Measure
        if hasattr(qdev, 'get_states'): states_out = qdev.get_states()
        elif hasattr(qdev, 'get_states_1d'): states_out = qdev.get_states_1d()
        else: states_out = qdev.states
        probs = (states_out.abs() ** 2)
        
        # 6. Project to out_channels
        out_flat = self.out_proj(probs.to(self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)) # [B*L, out_channels]
        
        # 7. Reshape to image [B, out_channels, H_out, W_out]
        # Calculate output height/width
        H_out = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        W_out = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
        
        out = out_flat.reshape(B, L, self.out_channels).transpose(1, 2).reshape(B, self.out_channels, H_out, W_out)
        return out.to(x.device, dtype=x.dtype)


class QuantumFrontEndQCNN(nn.Module):
    """
    Advanced Quantum FrontEnd using QCNN architecture for Latent Space.
    Features:
      - Ancilla-based Time Modulation (Entanglement)
      - Hardware-Efficient Ansatz (HEA) with Ring CNOTs
      - Hybrid Encoding (RY init + RZ re-uploading)
      - Trainable Measurement Basis
      - Layer-wise Training Interface
      - Classical Residual Connection
    """
    # @torch.jit.script # TQ functional calls are not scriptable due to global dict lookups
    def _fast_ry(self, qdev, wires, params):
        # params: [B]
        if apply_unitary_bmm is None:
            tqf.ry(qdev, wires=wires, params=params)
            return
            
        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        # [B, 2, 2]
        matrix = torch.stack([
            torch.stack([c, -s], dim=1),
            torch.stack([s, c], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, [wires] if isinstance(wires, int) else wires)

    def _fast_rz(self, qdev, wires, params):
        if apply_unitary_bmm is None:
            tqf.rz(qdev, wires=wires, params=params)
            return
            
        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        e_neg = torch.complex(c, -s)
        e_pos = torch.complex(c, s)
        
        matrix = torch.stack([
            torch.stack([e_neg, torch.zeros_like(e_neg)], dim=1),
            torch.stack([torch.zeros_like(e_pos), e_pos], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, [wires] if isinstance(wires, int) else wires)

    def _fast_cnot(self, qdev, wires):
        if apply_unitary_bmm is None:
            tqf.cnot(qdev, wires=wires)
            return
            
        if not hasattr(self, '_cnot_mat') or self._cnot_mat.device != qdev.states.device:
             m = torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=qdev.states.dtype, device=qdev.states.device)
             self._cnot_mat = m
             
        qdev.states = apply_unitary_bmm(qdev.states, self._cnot_mat, wires)

    def _fast_crx(self, qdev, wires, params):
        if apply_unitary_bmm is None:
            tqf.crx(qdev, wires=wires, params=params)
            return

        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        
        # CRX = diag(I, RX(theta))
        # RX = [[c, -is], [-is, c]]
        # top left I: [[1, 0], [0, 1]]
        # [B, 4, 4]
        
        bsz = theta.shape[0]
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        
        # Construct via stacking
        # Row 0: 1, 0, 0, 0
        # Row 1: 0, 1, 0, 0
        # Row 2: 0, 0, c, -is
        # Row 3: 0, 0, -is, c
        
        # complex -is
        neg_is = torch.complex(zeros, -s)
        c_complex = torch.complex(c, zeros)
        one_complex = torch.complex(ones, zeros)
        zero_complex = torch.complex(zeros, zeros)
        
        matrix = torch.stack([
            torch.stack([one_complex, zero_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, one_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, zero_complex, c_complex, neg_is], dim=1),
            torch.stack([zero_complex, zero_complex, neg_is, c_complex], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, wires)

    def _fast_crz(self, qdev, wires, params):
        if apply_unitary_bmm is None:
            tqf.crz(qdev, wires=wires, params=params)
            return
            
        theta = params
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        e_neg = torch.complex(c, -s)
        e_pos = torch.complex(c, s)
        
        # CRZ = diag(1, 1, e_neg, e_pos)
        
        bsz = theta.shape[0]
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        one_complex = torch.complex(ones, zeros)
        zero_complex = torch.complex(zeros, zeros)
        
        matrix = torch.stack([
            torch.stack([one_complex, zero_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, one_complex, zero_complex, zero_complex], dim=1),
            torch.stack([zero_complex, zero_complex, e_neg, zero_complex], dim=1),
            torch.stack([zero_complex, zero_complex, zero_complex, e_pos], dim=1)
        ], dim=1).to(qdev.states.dtype)
        
        qdev.states = apply_unitary_bmm(qdev.states, matrix, wires)

    def _batch_kron(self, mat_list, sub_bsz):
        res = mat_list[0]
        for m in mat_list[1:]:
            res = torch.einsum('bik,bjl->bijkl', res, m).reshape(sub_bsz, res.shape[1]*m.shape[1], res.shape[2]*m.shape[2])
        return res

    def _get_batch_rot_matrix(self, n_qubits, ry_params, rz_params, sub_bsz, device, dtype):
        mats = []
        for i in range(n_qubits):
            # RY
            theta = ry_params[:, i]
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            m_ry = torch.stack([torch.stack([c, -s], 1), torch.stack([s, c], 1)], 1).to(dtype)
            
            # RZ
            theta = rz_params[:, i]
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            e_neg = torch.complex(c, -s)
            e_pos = torch.complex(c, s)
            z = torch.zeros_like(c)
            m_rz = torch.stack([torch.stack([e_neg, z], 1), torch.stack([z, e_pos], 1)], 1).to(dtype)
            
            mats.append(torch.bmm(m_rz, m_ry))
            
        return self._batch_kron(mats, sub_bsz)

    def _get_cnot_layer_matrix(self, n_qubits, use_strided, device, dtype):
        # Check cache
        cache_key = (n_qubits, use_strided, device, dtype)
        if hasattr(self, '_cnot_layer_cache') and self._cnot_layer_cache.get('key') == cache_key:
            return self._cnot_layer_cache['mat']
        
        dim = 2 ** n_qubits
        # Apply U to Identity to get U^T (columns of U)
        state = torch.eye(dim, device=device, dtype=dtype).reshape([dim] + [2]*n_qubits)
        cnot_mat_2q = torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=dtype, device=device)
        
        for i in range(n_qubits):
            wires = [i, (i + 1) % n_qubits]
            state = apply_unitary_bmm(state, cnot_mat_2q, wires)
            
        if use_strided and n_qubits >= 4:
            for i in range(n_qubits):
                wires = [i, (i + 2) % n_qubits]
                state = apply_unitary_bmm(state, cnot_mat_2q, wires)
        
        # Transpose to get U
        mat = state.reshape(dim, dim).T
        self._cnot_layer_cache = {'key': cache_key, 'mat': mat}
        return mat

    def _fast_ry_layer(self, qdev, params):
        # params: [B, N]
        # Sequential application using apply_unitary_bmm for efficiency
        # Benchmarks show sequential (2.5ms) is ~5x faster than full matrix fusion (12.7ms) for N=6
        import math
        bsz = params.shape[0]
        n_qubits = params.shape[1]
        
        # Ensure states are in tensor form [B, 2, 2, ..., 2] for apply_unitary_bmm
        original_shape = qdev.states.shape
        
        # Reshape to tensor form if needed
        if qdev.states.ndim == 2:
            # [B, dim_total]
            n_wires_total = int(math.log2(qdev.states.shape[1]))
            qdev.states = qdev.states.reshape([bsz] + [2] * n_wires_total)
        elif qdev.states.ndim == 3 and qdev.states.shape[2] == 1:
            # [B, dim_total, 1]
            n_wires_total = int(math.log2(qdev.states.shape[1]))
            qdev.states = qdev.states.reshape([bsz] + [2] * n_wires_total)
            
        for i in range(n_qubits):
            theta = params[:, i]
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            # Matrix [B, 2, 2]
            mat = torch.stack([
                torch.stack([c, -s], dim=-1),
                torch.stack([s, c], dim=-1)
            ], dim=-2).to(qdev.states.dtype)
            
            # Apply to wire i using apply_unitary_bmm
            qdev.states = apply_unitary_bmm(qdev.states, mat, [i])
            
        # Restore original shape if it was different
        if qdev.states.shape != original_shape:
            qdev.states = qdev.states.reshape(original_shape)

    def _fast_rz_layer(self, qdev, params):
        # params: [B, N]
        # Diagonal fusion
        bsz = params.shape[0]
        n_qubits = params.shape[1]
        dim = 2 ** n_qubits
        device = params.device
        
        # Precompute signs if not cached
        cache_key = (n_qubits, device)
        if not hasattr(self, '_rz_signs_cache') or getattr(self, '_rz_signs_cache_key', None) != cache_key:
            arange = torch.arange(dim, device=device)
            shifts = torch.arange(n_qubits - 1, -1, -1, device=device)
            bits = (arange.unsqueeze(1) >> shifts) & 1
            signs = (bits.float() - 0.5) 
            self._rz_signs_cache = signs
            self._rz_signs_cache_key = cache_key
        
        signs = self._rz_signs_cache # [dim, n]
        
        # phases: [B, dim]
        phases = torch.matmul(params, signs.T)
        
        # rot: exp(i * phases)
        rot_diag = torch.complex(torch.cos(phases), torch.sin(phases))
        
        # Apply
        original_shape = qdev.states.shape
        flat_states = qdev.states.reshape(bsz, dim, -1)
        rot_diag_expanded = rot_diag.unsqueeze(2)
        flat_states = flat_states * rot_diag_expanded
        qdev.states = flat_states.reshape(original_shape)

    def _fast_cnot_layer(self, qdev, n_qubits, use_strided):
        # Permutation fusion
        bsz = qdev.bsz
        dim = 2 ** n_qubits
        device = qdev.states.device
        
        cache_key = (n_qubits, use_strided, device)
        if not hasattr(self, '_cnot_perm_cache') or getattr(self, '_cnot_perm_cache_key', None) != cache_key:
            indices = torch.arange(dim, device=device)
            
            # Apply ring CNOTs
            for i in range(n_qubits):
                ctl = i
                tgt = (i + 1) % n_qubits
                mask_ctl = 1 << (n_qubits - 1 - ctl)
                mask_tgt = 1 << (n_qubits - 1 - tgt)
                
                should_flip = (indices & mask_ctl) != 0
                indices = torch.where(should_flip, indices ^ mask_tgt, indices)
                
            if use_strided and n_qubits >= 4:
                for i in range(n_qubits):
                    ctl = i
                    tgt = (i + 2) % n_qubits
                    mask_ctl = 1 << (n_qubits - 1 - ctl)
                    mask_tgt = 1 << (n_qubits - 1 - tgt)
                    should_flip = (indices & mask_ctl) != 0
                    indices = torch.where(should_flip, indices ^ mask_tgt, indices)
            
            self._cnot_perm_cache = torch.argsort(indices)
            self._cnot_perm_cache_key = cache_key
            
        perm = self._cnot_perm_cache
        
        # Apply permutation
        original_shape = qdev.states.shape
        flat_states = qdev.states.reshape(bsz, dim, -1)
        flat_states = flat_states[:, perm, :]
        qdev.states = flat_states.reshape(original_shape)

    def _get_fused_controlled_matrix(self, params_list, gate_type, device, dtype):
        # params_list: list of tensors [B], length k
        # gate_type: 'rx' or 'rz'
        # Returns [B, 2^(k+1), 2^(k+1)]
        
        mats = []
        bsz = params_list[0].shape[0]
        
        for theta in params_list:
            c = torch.cos(theta / 2)
            s = torch.sin(theta / 2)
            if gate_type == 'rx':
                # RX = [[c, -is], [-is, c]]
                zeros = torch.zeros_like(c)
                neg_is = torch.complex(zeros, -s)
                c_complex = torch.complex(c, zeros)
                m = torch.stack([torch.stack([c_complex, neg_is], 1), torch.stack([neg_is, c_complex], 1)], 1).to(dtype)
            else: # rz
                # RZ = [[e_neg, 0], [0, e_pos]]
                e_neg = torch.complex(c, -s)
                e_pos = torch.complex(c, s)
                z = torch.zeros_like(c)
                m = torch.stack([torch.stack([e_neg, z], 1), torch.stack([z, e_pos], 1)], 1).to(dtype)
            mats.append(m)
            
        # U_targets = U1 x U2 x ...
        u_targets = self._batch_kron(mats, bsz) # [B, 2^k, 2^k]
        
        # Construct Controlled Matrix: [[I, 0], [0, U_targets]]
        dim_targets = u_targets.shape[1]
        
        # Identity for control=0 branch
        eye = torch.eye(dim_targets, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
        zeros = torch.zeros(bsz, dim_targets, dim_targets, device=device, dtype=dtype)
        
        row0 = torch.cat([eye, zeros], dim=2)
        row1 = torch.cat([zeros, u_targets], dim=2)
        matrix = torch.cat([row0, row1], dim=1)
        
        return matrix

    def _apply_fusion_circuit(self, qdev, sub_bsz, sub_da, sub_sa, interaction_wires, data_wires, 
                              mod_params, qcnn_rot_params, 
                              n_qubits_data: int, n_qubits_ancilla: int, active_layers: int, 
                              use_strided_cnot: bool, reupload_data: bool, encoding_type: str):
        
        device = qdev.states.device
        dtype = qdev.states.dtype

        # 2. Encode Data (Fused)
        if encoding_type == 'amplitude':
            # Amplitude Encoding: Data is already encoded in state vector.
            # We only apply Style Modulation here (if any)
            if sub_sa is not None:
                # Fused RY layer
                # zeros = torch.zeros_like(sub_sa)
                # rot_mat = self._get_batch_rot_matrix(n_qubits_data, sub_sa, zeros, sub_bsz, device, dtype)
                # qdev.states = apply_unitary_bmm(qdev.states, rot_mat, list(range(n_qubits_data)))
                self._fast_ry_layer(qdev, sub_sa)
        else:
            # Angle Encoding (RY)
            # Integrated Fusion: Data + Style
            if sub_sa is not None:
                 init_params = sub_da + sub_sa
            else:
                 init_params = sub_da
            
            # Fused RY layer
            # zeros = torch.zeros_like(init_params)
            # rot_mat = self._get_batch_rot_matrix(n_qubits_data, init_params, zeros, sub_bsz, device, dtype)
            # qdev.states = apply_unitary_bmm(qdev.states, rot_mat, list(range(n_qubits_data)))
            self._fast_ry_layer(qdev, init_params)
        
        # 3. Entanglement (Ancilla -> Data) with Split Control (Fused per Ancilla)
        # If interaction_wires is provided (Ancilla Mode)
        if interaction_wires is not None and data_wires is not None:
            # Group data wires by ancilla
            # Assuming standard pattern: data_wire i controlled by ancilla i % n_ancilla
            
            for a_idx in range(n_qubits_ancilla):
                ctl = interaction_wires[a_idx]
                tgt_indices = [i for i in range(n_qubits_data) if i % n_qubits_ancilla == a_idx]
                
                if not tgt_indices:
                    continue
                
                # Collect params and real data wires
                params_list = []
                real_tgt_wires = []
                
                for i in tgt_indices:
                    real_tgt_wires.append(data_wires[i])
                    # mod_params logic
                    if mod_params.ndim == 4 and mod_params.shape[0] == sub_bsz:
                         strength = mod_params[:, 0, i, 0]
                    else:
                         strength = mod_params[0, i, 0].expand(sub_bsz)
                    params_list.append(strength)
                
                gate_type = 'rx' if a_idx % 2 == 0 else 'rz'
                
                # Get fused matrix
                # Wires: [ctl, tgt1, tgt2, ...]
                mat = self._get_fused_controlled_matrix(params_list, gate_type, device, dtype)
                
                apply_wires = [ctl] + real_tgt_wires
                qdev.states = apply_unitary_bmm(qdev.states, mat, apply_wires)

        
        # 4. Spatial QCNN Backbone
        for l in range(active_layers):
            # Optimized: Fused Rotation Layer + CNOTs
            if qcnn_rot_params.ndim == 5 and qcnn_rot_params.shape[0] == sub_bsz:
                ry_params = qcnn_rot_params[:, l, :, 0, 0] # [B, N]
                rz_params = qcnn_rot_params[:, l, :, 1, 0] # [B, N]
            else:
                ry_params = qcnn_rot_params[l, :, 0, 0].expand(sub_bsz, -1) # [B, N]
                rz_params = qcnn_rot_params[l, :, 1, 0].expand(sub_bsz, -1) # [B, N]
            
            # 1. Get Batch Rotation Matrix [B, Dim, Dim]
            # rot_mat = self._get_batch_rot_matrix(n_qubits_data, ry_params, rz_params, sub_bsz, device, dtype)
            
            # 2. Get CNOT Layer Matrix [Dim, Dim]
            # cnot_mat = self._get_cnot_layer_matrix(n_qubits_data, use_strided_cnot, device, dtype)
            
            # 3. Fuse: Full_Op = CNOT * Rot
            # matmul supports broadcasting: [D, D] * [B, D, D] -> [B, D, D]
            # full_op = torch.matmul(cnot_mat, rot_mat)
            
            # 4. Apply
            # qdev.states = apply_unitary_bmm(qdev.states, full_op, list(range(n_qubits_data)))
            
            # Optimized Fusion: Layer-wise (O(dim) for RZ/CNOT)
            # 1. RY (Batch Matrix Apply)
            self._fast_ry_layer(qdev, ry_params)
            
            # 2. RZ (Diagonal Apply)
            self._fast_rz_layer(qdev, rz_params)
            
            # 3. CNOT (Permutation Apply)
            self._fast_cnot_layer(qdev, n_qubits_data, use_strided_cnot)
            
            # Re-uploading (Fused)
            if reupload_data and (l < active_layers - 1):
                # Prepare params
                if sub_sa is not None:
                    rz_reupload = sub_da + sub_sa
                else:
                    rz_reupload = sub_da
                
                # zeros = torch.zeros_like(rz_reupload)
                # Apply RZ (RY=0)
                # Note: _get_batch_rot_matrix applies RZ * RY. 
                # If RY params are 0, RY=I. So it applies RZ * I = RZ. Correct.
                
                # reupload_mat = self._get_batch_rot_matrix(n_qubits_data, zeros, rz_reupload, sub_bsz, device, dtype)
                # qdev.states = apply_unitary_bmm(qdev.states, reupload_mat, list(range(n_qubits_data)))
                self._fast_rz_layer(qdev, rz_reupload)

    def __init__(self, channels: int, style_dim: int, n_qubits_data: int = 6, n_qubits_ancilla: int = 2, 
                 n_layers: int = 2, freeze_qcnn: bool = False, device_name: Optional[str] = None,
                 time_emb_module: Optional[nn.Module] = None,
                 use_strided_cnot: bool = False,
                 reupload_data: bool = False,
                 max_qdev_bsz: int = 4096,
                 encoding_type: str = 'tanh',
                 use_mlp_residual: bool = False,
                 n_groups: int = 1,  # Grouped QCNN
                 use_strong_bypass: bool = False, # Strong Classical Bypass
                 stride: int = 2,
                 injection_mode: str = 'simple', # 'simple' or 'rich'
                 projection_type: str = 'linear', # 'linear', 'mlp'
                 use_mlp_output: bool = False,
                 use_checkpoint: bool = True):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.n_groups = int(n_groups)
        self.use_strong_bypass = bool(use_strong_bypass)
        self.injection_mode = injection_mode
        self.projection_type = projection_type
        self.use_mlp_output = use_mlp_output
        self.use_checkpoint = use_checkpoint
        
        assert channels % self.n_groups == 0, f"Channels {channels} must be divisible by n_groups {n_groups}"
        self.channels_per_group = channels // self.n_groups
        self.n_qubits_data = n_qubits_data
        
        # If integrated with QuantumMLP, we allow dimension mismatch (Adaptation via Measurement)
        if time_emb_module is not None and hasattr(time_emb_module, 'n_qubits'):
             self.n_qubits_qmlp = time_emb_module.n_qubits
        else:
             self.n_qubits_qmlp = 0

        self.n_qubits_ancilla = n_qubits_ancilla
        # Update total wires to include the larger of ancilla count or QMLP requirement
        self.n_wires_ancilla = max(n_qubits_ancilla, self.n_qubits_qmlp)
        self.n_wires = n_qubits_data + self.n_wires_ancilla
        self.n_layers = n_layers
        self.freeze_qcnn = freeze_qcnn
        self.device_name = device_name
        self.time_emb_module = time_emb_module
        self.use_strided_cnot = bool(use_strided_cnot)
        self.reupload_data = bool(reupload_data)
        self.encoding_type = encoding_type
        self.use_mlp_residual = use_mlp_residual
        self.eps = 1e-9
        self._printed_exec = False
        self.active_layers = n_layers # For layer-wise training
        self.max_qdev_bsz = int(max_qdev_bsz)
        self.reuse_device = True
        self.cache_device = False # Default to False to avoid graph retention issues
        self._qdev_cached = None
        self._qdev_cached_bsz = 0
        self._qdev_cached_devname = None

        # Pre-processing: Patch extraction via Unfold + Dimension Reduction
        # Assume 3x3 kernel for local context
        self.kernel_size = 3
        self.padding = 1
        self.stride = stride
        
        # Input Patch Dimension per Group
        # Original: channels * k * k
        # Grouped: (channels/groups) * k * k
        self.patch_dim_per_group = self.channels_per_group * self.kernel_size * self.kernel_size
        self.patch_dim = channels * self.kernel_size * self.kernel_size # Total patch dim
        
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        # Input Scaling (per group or global? Global is easier, but per-group is more flexible)
        # Let's keep it per patch_dim (global across groups but sliced)
        self.inp_scale = nn.Parameter(torch.ones(self.patch_dim))
        
        # Project Patch to Data Qubits dimension (for encoding)
        # We use Conv1d with kernel_size=1 to handle grouping if needed, or just reshape
        # But here we need to map [B, groups, patch_dim_per_group] -> [B, groups, n_qubits]
        # Or flatten groups into batch: [B*groups, patch_dim_per_group] -> [B*groups, n_qubits]
        # This is cleaner.
        if encoding_type == 'amplitude':
            # For Amplitude Encoding, we project to the State Vector size (2^N)
            # This allows capturing more information than Angle Encoding (N)
            self.data_proj_dim = 2 ** n_qubits_data
        else:
            # For Angle Encoding, we project to the number of Rotation Gates (N)
            self.data_proj_dim = n_qubits_data
            
        if self.projection_type == 'mlp':
            # MLP Projection: Linear -> SiLU -> Linear
            # Increases capacity of the classical encoder
            hidden_dim = max(self.patch_dim_per_group, self.data_proj_dim * 4)
            self.data_proj = nn.Sequential(
                nn.Linear(self.patch_dim_per_group, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.data_proj_dim)
            )
        else:
            # Standard Linear Projection
            self.data_proj = nn.Linear(self.patch_dim_per_group, self.data_proj_dim)
        
        self.style_proj = nn.Linear(style_dim, n_qubits_ancilla)
        self.style_to_data = nn.Linear(style_dim, n_qubits_data) # Style shared across groups for now
        
        if self.injection_mode == 'rich':
            # Rich Injection: Map style to ALL rotation parameters in the QCNN backbone
            # Target: [n_groups, n_layers, n_qubits_data, 2 (RY, RZ)]
            # We map to flat vector and then reshape
            self.rich_param_dim = self.n_groups * n_layers * n_qubits_data * 2
            self.style_rich_proj = nn.Linear(style_dim, self.rich_param_dim)
        
        # Refactored Parameters for Fine-grained Control and Strict Adherence to Document
        
        # 1. Ancilla Evolution Params (U_mlp): Applied to Ancilla qubits to generate |psi_time>
        # Shared across groups
        self.ancilla_params = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits_ancilla, 3))
        
        # 2. Modulation Params (Ancilla-Data Interface): CRX/CZ weights
        # Independent per group: [groups, layers, data, 3]
        self.mod_params = nn.Parameter(0.1 * torch.randn(self.n_groups, n_layers, n_qubits_data, 3)) 
        
        # 3. Spatial QCNN Params (The Backbone):
        # Data Rotations - Independent per group: [groups, layers, data, 2, 3]
        self.qcnn_rot_params = nn.Parameter(0.1 * torch.randn(self.n_groups, n_layers, n_qubits_data, 2, 3)) 
        
        if self.freeze_qcnn:
            self.qcnn_rot_params.requires_grad = False
            self.ancilla_params.requires_grad = False
        
        # Trainable Measurement Basis: Single layer of U3 rotations on Data Qubits
        # Independent per group: [groups, data, 3]
        self.measure_params = nn.Parameter(0.1 * torch.randn(self.n_groups, n_qubits_data, 3))
        
        # Output Projection (Using Probabilities: 2^N_wires -> Channels_per_group)
        # Independent per group (via Linear since we process groups in batch)
        self.out_proj = nn.Linear(1 << self.n_wires, self.channels_per_group)

        # Output MLP Enhancement
        if self.use_mlp_output:
            self.out_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.channels_per_group, self.channels_per_group)
            )
        
        # Classical Residual
        if self.use_mlp_residual:
            # Hybrid Architecture: Replace linear residual with MLP
            hidden_dim = self.patch_dim * 2
            self.res_proj = nn.Sequential(
                nn.Linear(self.patch_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, channels)
            )
        else:
            self.res_proj = nn.Linear(self.patch_dim, channels)
            
        # Strong Classical Bypass (Parallel Conv Branch)
        if self.use_strong_bypass:
            # Must match QCNN downsampling (stride=self.stride)
            self.strong_bypass = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, stride=self.stride), # Downsample
                nn.GroupNorm(min(32, channels), channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1)
            )
        else:
            self.strong_bypass = None
        
        if _TQ_AVAILABLE:
            self.measure_z = tq.MeasureAll(tq.PauliZ)
        try:
            _rd_env = os.getenv('QCNN_REUSE_DEVICE', '').strip().lower()
            if _rd_env in ('0', 'false', 'no', 'off'):
                self.reuse_device = False
            elif _rd_env in ('1', 'true', 'yes', 'on'):
                self.reuse_device = True
            _cd_env = os.getenv('QCNN_CACHE_DEVICE', '').strip().lower()
            if _cd_env in ('0', 'false', 'no', 'off'):
                self.cache_device = False
            elif _cd_env in ('1', 'true', 'yes', 'on'):
                self.cache_device = True
            _mb_env = os.getenv('QCNN_MAX_QDEV_BSZ', '').strip()
            if _mb_env:
                self.max_qdev_bsz = int(_mb_env)
        except Exception:
            pass

    def set_active_layers(self, n: int):
        self.active_layers = min(max(1, n), self.n_layers)

    def _process_chunk(self, chunk_patches, chunk_style, chunk_mod_params, chunk_rot_params, chunk_meas_params):
        """
        Process a single chunk of data on the quantum device.
        Designed to be used with torch.utils.checkpoint.
        """
        actual_chunk_size = chunk_patches.shape[0]
        
        # Logic for qdev retrieval/creation
        qdev_chunk = None
        if self.reuse_device and self._qdev_cached is not None and self._qdev_cached_bsz >= actual_chunk_size:
             qdev_chunk = self._qdev_cached
             if hasattr(qdev_chunk, 'reset_states'):
                 qdev_chunk.reset_states(bsz=actual_chunk_size)
             else:
                 # Re-init is safer if reset not available
                 qdev_chunk = tq.QuantumDevice(n_wires=self.n_wires, bsz=actual_chunk_size, device=self.device_name)
        else:
             qdev_chunk = tq.QuantumDevice(n_wires=self.n_wires, bsz=actual_chunk_size, device=self.device_name)
             if self.reuse_device:
                 self._qdev_cached = qdev_chunk
                 self._qdev_cached_bsz = actual_chunk_size

        # For Amplitude Encoding, we will set states explicitly later
        if self.encoding_type != 'amplitude' and hasattr(qdev_chunk, 'reset_states'):
            qdev_chunk.reset_states(actual_chunk_size)
        
        # Data Encoding (Common)
        dtype = next(self.data_proj.parameters()).dtype
        if self.encoding_type == 'linear':
            chunk_da = self.data_proj(chunk_patches.to(dtype))
        elif self.encoding_type == 'amplitude':
            chunk_da = self.data_proj(chunk_patches.to(dtype))
        else:
            chunk_da = torch.tanh(self.data_proj(chunk_patches.to(dtype))) * math.pi
        
        chunk_sa = torch.tanh(self.style_to_data(chunk_style.to(self.style_to_data.weight.dtype))) * math.pi
        
        # State Preparation for Amplitude Encoding
        if self.encoding_type == 'amplitude':
            # 1. Normalize Data (L2)
            norm = torch.norm(chunk_da, p=2, dim=1, keepdim=True) + 1e-8
            chunk_da_norm = chunk_da / norm
            
            # 2. Pad to 2^n_qubits_data
            target_dim = 2 ** self.n_qubits_data
            curr_dim = chunk_da_norm.shape[1]
            
            if curr_dim < target_dim:
                padding = torch.zeros(actual_chunk_size, target_dim - curr_dim, device=chunk_da.device, dtype=chunk_da.dtype)
                data_state = torch.cat([chunk_da_norm, padding], dim=1)
            elif curr_dim > target_dim:
                data_state = chunk_da_norm[:, :target_dim]
                norm = torch.norm(data_state, p=2, dim=1, keepdim=True) + 1e-8
                data_state = data_state / norm
            else:
                data_state = chunk_da_norm
            
            # 3. Handle Ancilla (Tensor Product)
            if self.n_wires_ancilla > 0:
                ancilla_dim = 2 ** self.n_wires_ancilla
                ancilla_state = torch.zeros(actual_chunk_size, ancilla_dim, device=chunk_da.device, dtype=chunk_da.dtype)
                ancilla_state[:, 0] = 1.0
                full_state_real = torch.einsum('bi,bj->bij', data_state, ancilla_state).reshape(actual_chunk_size, -1)
            else:
                full_state_real = data_state

            # 4. Set States (Complex)
            flat_state = torch.complex(full_state_real, torch.zeros_like(full_state_real))
            state_shape = [actual_chunk_size] + [2] * self.n_wires
            qdev_chunk.states = flat_state.reshape(state_shape)

        # Apply Circuit (Integrated Fusion)
        self._apply_fusion_circuit(
            qdev_chunk, actual_chunk_size, chunk_da, chunk_sa, 
            None, # interaction_wires
            None, # data_wires
            chunk_mod_params, chunk_rot_params,
            self.n_qubits_data, self.n_qubits_ancilla, self.active_layers,
            self.use_strided_cnot, self.reupload_data, self.encoding_type
        )
        
        # Trainable Measurement Basis
        for i in range(self.n_qubits_data):
            tqf.u3(qdev_chunk, wires=i, params=chunk_meas_params[:, i])
            
        # Measurement (Probabilities)
        if hasattr(qdev_chunk, 'get_states_1d'): 
            states = qdev_chunk.get_states_1d()
        elif hasattr(qdev_chunk, 'get_states'): 
            states = qdev_chunk.get_states()
        else: 
            states = qdev_chunk.states
        
        # [Optimization] Avoid sqrt in abs() since we square it anyway
        # probs = (states.abs() ** 2)
        probs = states.real**2 + states.imag**2
        return probs

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # style can be 'emb' (raw input) if time_emb_module is used
        if not self._printed_exec:
            self._printed_exec = True

        B, C, H, W = x.shape
        
        # 0. Strong Classical Bypass (Parallel Branch)
        bypass_out = 0
        if self.use_strong_bypass:
            bypass_out = self.strong_bypass(x)
        
        # 1. Unfold & Reshape
        patches = self.unfold(x)
        L = patches.shape[-1]
        
        # Flatten patches: [B, C*K*K, L] -> [B, L, C*K*K] -> [B*L, C*K*K]
        patches_flat = patches.transpose(1, 2).reshape(-1, self.patch_dim)
        
        # Apply Input Scale (Global scaling)
        patches_flat = patches_flat * self.inp_scale
        
        # Grouped QCNN Reshape
        # [B*L, channels*K*K] -> [B*L, groups, channels_per_group*K*K]
        bsz_total = patches_flat.shape[0]
        sub_patches = patches_flat.reshape(bsz_total, self.n_groups, self.patch_dim_per_group)
        
        # Flatten groups into batch dimension for parallel processing
        # [B*L*groups, patch_dim_per_group]
        sub_patches_flat = sub_patches.reshape(-1, self.patch_dim_per_group)
        sub_bsz = sub_patches_flat.shape[0] # B * L * groups
        
        # 2. Build parameters (Classical or Quantum Injection)
        use_fusion = (self.n_qubits_ancilla > 0) and (self.time_emb_module is not None)
        
        # [Optimization] Pre-compute QMLP states for Fusion Scheme
        # This avoids re-running the same QMLP circuit for every patch (B*L times -> B times)
        qmlp_states_expanded = None
        if use_fusion:
            # Assume style is [B, qmlp_qubits] (measurements) or we access qdev inside time_emb
            # The current integration passes 'style' as the time embedding vector.
            # If we want to fuse, we need the QUANTUM STATE of the QMLP.
            # ... (Fusion logic remains similar, but adapted for groups)
            pass

        # Prepare Style for Grouped Processing
        # style: [B, style_dim] -> [B*L*groups, style_dim]
        # First expand to L: [B, L, style_dim] -> [B*L, style_dim]
        style_expanded = style.unsqueeze(1).expand(-1, L, -1).reshape(bsz_total, -1)
        # Then expand to groups: [B*L, 1, style_dim] -> [B*L, groups, style_dim] -> [B*L*groups, style_dim]
        sub_style = style_expanded.unsqueeze(1).expand(-1, self.n_groups, -1).reshape(sub_bsz, -1)

        # 3. Batch Processing
        # Optimized: Try to process full batch at once if possible to avoid Python loop overhead
        # If sub_bsz is too large, we still chunk, but we reuse the device.
        
        # Expand Group Parameters to Batch Level
        mod_params_expanded = self.mod_params.unsqueeze(0).expand(bsz_total, -1, -1, -1, -1).reshape(sub_bsz, self.n_layers, self.n_qubits_data, 3)
        qcnn_rot_params_expanded = self.qcnn_rot_params.unsqueeze(0).expand(bsz_total, -1, -1, -1, -1, -1).reshape(sub_bsz, self.n_layers, self.n_qubits_data, 2, 3)
        measure_params_expanded = self.measure_params.unsqueeze(0).expand(bsz_total, -1, -1, -1).reshape(sub_bsz, self.n_qubits_data, 3)

        # [Rich Injection]
        if self.injection_mode == 'rich':
            # 1. Project: [B, groups*layers*qubits*2]
            delta = self.style_rich_proj(style)
            # 2. Reshape: [B, groups, layers, qubits, 2]
            delta = delta.reshape(B, self.n_groups, self.n_layers, self.n_qubits_data, 2)
            # 3. Expand L: [B, L, groups, layers, qubits, 2]
            delta = delta.unsqueeze(1).expand(-1, L, -1, -1, -1, -1)
            # 4. Flatten: [sub_bsz, layers, qubits, 2]
            delta_flat = delta.reshape(sub_bsz, self.n_layers, self.n_qubits_data, 2)
            
            # 5. Apply to parameters (broadcast to last dim 3)
            # qcnn_rot_params_expanded is [sub_bsz, L, N, 2, 3]
            # We add to index [..., 0]
            
            # Create a zero tensor of same shape
            # We can't do in-place modification on expanded tensor easily if it's a view.
            # But here we just add.
            
            # We need to construct the adder [sub_bsz, L, N, 2, 3]
            adder = torch.zeros_like(qcnn_rot_params_expanded)
            # RY: delta_flat[..., 0] -> adder[..., 0, 0]
            adder[:, :, :, 0, 0] = delta_flat[:, :, :, 0]
            # RZ: delta_flat[..., 1] -> adder[..., 1, 0]
            adder[:, :, :, 1, 0] = delta_flat[:, :, :, 1]
            
            qcnn_rot_params_expanded = qcnn_rot_params_expanded + adder

        # [Optimization] Reuse QuantumDevice
        # We check if we have a cached device or create a new one for the max required size
        # Ideally, we process everything in one go if sub_bsz fits in memory.
        # 4 qubits -> state vector is 16 complex64 (128 bytes).
        # sub_bsz = 4 * 256 * 8 = 8192.
        # Memory = 8192 * 128 bytes ~= 1 MB. extremely small.
        # Even with 10 qubits (1KB), 8192 is ~8MB.
        # So we can SAFELY process the entire batch at once for small qubit counts.
        
        chunk_size_limit = self.max_qdev_bsz
        if sub_bsz <= chunk_size_limit * 2: # heuristic: if close enough, just do it once
            chunk_size_limit = sub_bsz

        outs = []
        
        # Process chunks with optional checkpointing
        for s in range(0, sub_bsz, chunk_size_limit):
            e = min(s + chunk_size_limit, sub_bsz)
            
            chunk_patches = sub_patches_flat[s:e]
            chunk_style = sub_style[s:e]
            
            # Slice params
            chunk_mod_params = mod_params_expanded[s:e]
            chunk_rot_params = qcnn_rot_params_expanded[s:e]
            chunk_meas_params = measure_params_expanded[s:e]
            
            if self.use_checkpoint and self.training:
                # Use checkpointing to save memory
                # Note: inputs must be tensors. All args are tensors here.
                # checkpoint_utils.checkpoint handles the backward pass by re-running the function
                probs = checkpoint_utils.checkpoint(
                    self._process_chunk,
                    chunk_patches,
                    chunk_style,
                    chunk_mod_params,
                    chunk_rot_params,
                    chunk_meas_params,
                    use_reentrant=False # Recommended for newer PyTorch
                )
            else:
                probs = self._process_chunk(
                    chunk_patches,
                    chunk_style,
                    chunk_mod_params,
                    chunk_rot_params,
                    chunk_meas_params
                )
            outs.append(probs)

        # Concat chunks
        quant_out_flat = torch.cat(outs, dim=0).to(x.device)
        
        # Project Output
        # [sub_bsz, out_channels_per_group]
        quant_proj = self.out_proj(quant_out_flat.to(self.out_proj.weight.dtype))
        
        # Add Activation and Linear Layer (Optional Enhancement)
        # Check if we should apply this enhancement (default to False to preserve old behavior unless flag is set?)
        # User requested to add it, so let's add it. 
        # To be safe, we can add it as a new attribute if it doesn't exist, or just modify the architecture.
        # Given "current hybrid architecture", and "optimize model effect", I will add it if self.use_mlp_output is True.
        if getattr(self, 'use_mlp_output', False):
             quant_proj = self.out_mlp(quant_proj)

        # Reshape back to [B*L, groups, channels_per_group]
        quant_grouped = quant_proj.view(bsz_total, self.n_groups, self.channels_per_group)
        
        # Flatten groups: [B*L, channels]
        quant_final = quant_grouped.view(bsz_total, self.channels)
        
        # Classical Residual (on original flattened patches)
        if isinstance(self.res_proj, nn.Sequential):
            p = self.res_proj[0].weight
        else:
            p = self.res_proj.weight
            
        res_out = self.res_proj(patches_flat.to(p.dtype))

        # Combine
        out_flat = quant_final + res_out
        
        # Reshape to Image
        # Calculate output height/width
        H_out = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        W_out = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
        
        out = out_flat.reshape(B, L, self.channels).transpose(1, 2).reshape(B, self.channels, H_out, W_out)
        
        if self.use_strong_bypass:
            out = out + bypass_out
            
        return out.to(x.device, dtype=x.dtype)
        if use_fusion:
             # style is [B, style_dim]. Run QMLP once per batch item.
             # Create a temporary qdev for QMLP
             qdev_qmlp = tq.QuantumDevice(n_wires=self.n_qubits_qmlp, bsz=B, device=x.device.type)
             # Apply circuit (assume wires 0..n_qmlp-1)
             self.time_emb_module.apply_circuit(qdev_qmlp, wires=list(range(self.n_qubits_qmlp)), inputs=style)
             # Get states: [B, 2^n_qmlp]
             if hasattr(qdev_qmlp, 'get_states_1d'): states_ancilla = qdev_qmlp.get_states_1d()
             else: states_ancilla = qdev_qmlp.states
             
             # Expand to [B*L, 2^n_qmlp]
             H_out = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
             W_out = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
             L_patches = H_out * W_out
             # Use repeat_interleave to broadcast: [S1, S2, ...] -> [S1, S1, ..., S2, S2, ...]
             # This matches patches_flat which is [P1_1, P1_2... P2_1...]
             qmlp_states_expanded = states_ancilla.repeat_interleave(L_patches, dim=0)
             # Free memory
             del qdev_qmlp

        if use_fusion:
            # Fusion Scheme: style is raw input (e.g. sigma)
            # Expand raw style to match spatial dimensions [B, style_dim] -> [B*L, style_dim]
            # Note: style_dim must match QMLP input dim
            # NOTE: sub_style is used in the loop. Even with optimization, we keep this structure for compatibility
            # if we fall back to per-patch circuit. But with state injection, we might skip sub_style usage for QMLP.
            H_out = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
            W_out = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
            style_flat = style.view(B, 1, 1, -1).expand(B, H_out, W_out, -1).reshape(-1, self.style_dim)
        else:
            # Original Scheme: style is processed to classical params
            if self.time_emb_module is not None:
                style_base = self.time_emb_module(style)  # [B, style_dim_out]
            else:
                style_base = style  # [B, style_dim]
            H_out = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
            W_out = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
            style_flat = style_base.view(B, 1, 1, -1).expand(B, H_out, W_out, -1).reshape(-1, self.style_dim)

        bsz = patches_flat.shape[0]
        device_name = self.device_name or x.device.type
        step_dyn = B * L
        step = bsz if self.max_qdev_bsz <= 0 else min(step_dyn, self.max_qdev_bsz)
        outs = []
        start = 0
        while start < bsz:
            end = min(start + step, bsz)
            sub_patches = patches_flat[start:end]
            sub_style   = style_flat[start:end]
            sub_bsz = end - start
            
            # Data Encoding (Common)
            if self.encoding_type == 'linear':
                # Linear encoding: direct mapping without tanh limit, scaled by pi/2 or just pi
                # Using pi as scale to cover rotation range
                sub_da = self.data_proj(sub_patches.to(self.data_proj.weight.dtype))
            else:
                # Tanh encoding (default)
                sub_da = torch.tanh(self.data_proj(sub_patches.to(self.data_proj.weight.dtype))) * math.pi
            
            # Device Init
            if self.reuse_device and self.cache_device and self._qdev_cached is not None and self._qdev_cached_bsz == sub_bsz and self._qdev_cached_devname == device_name:
                 qdev = self._qdev_cached
                 # Important: For Fusion optimization, we are setting states directly.
                 # If reusing device, we must ensure previous states are cleared or overwritten.
                 # set_states overwrites, so it's safe.
            else:
                 qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=sub_bsz, device=device_name)
                 if self.cache_device:
                     self._qdev_cached = qdev
                     self._qdev_cached_bsz = sub_bsz
                     self._qdev_cached_devname = device_name
            
            if use_fusion:
                # --- FUSION SCHEME (OPTIMIZED) ---
                data_wires = list(range(self.n_qubits_data))
                qmlp_wires = list(range(self.n_qubits_data, self.n_qubits_data + self.n_qubits_qmlp))
                interaction_wires = qmlp_wires[:self.n_qubits_ancilla]
                
                # 1. State Injection (Broadcasting Optimization)
                sub_anc_states = qmlp_states_expanded[start:end] # [sub_bsz, 2^n_qmlp]
                zero_state_data = torch.zeros(sub_bsz, 1 << self.n_qubits_data, dtype=sub_anc_states.dtype, device=sub_anc_states.device)
                zero_state_data[:, 0] = 1.0
                full_states = (zero_state_data[:, :, None] * sub_anc_states[:, None, :]).reshape(sub_bsz, -1)
                
                if hasattr(qdev, 'set_states'): qdev.set_states(full_states)
                elif hasattr(qdev, 'set_states_1d'): qdev.set_states_1d(full_states)
                else: qdev.states = full_states
                
                # Apply Circuit (extracted for JIT potential)
                # We can compile this call if PyTorch supports it on `self`.
                # To enable compilation, we can wrap this in a static function or use `self._apply_fusion_circuit`.
                # For now, we call it directly.
                self._apply_fusion_circuit(qdev, sub_bsz, sub_da, interaction_wires, data_wires,
                                           chunk_mod_params, chunk_rot_params, # Pass chunked params!
                                           self.n_qubits_data, self.n_qubits_ancilla, self.active_layers,
                                           self.use_strided_cnot, self.reupload_data)

            else:
                # --- ORIGINAL SCHEME ---
                sub_sa = torch.tanh(self.style_to_data(sub_style.to(self.style_to_data.weight.dtype))) * math.pi
                for i in range(self.n_qubits_data):
                    tqf.ry(qdev, wires=i, params=(sub_da[:, i] + sub_sa[:, i]))
                for l in range(self.active_layers):
                    for i in range(self.n_qubits_data):
                        ry_params = self.qcnn_rot_params[l, i, 0, 0].unsqueeze(0).expand(sub_bsz)
                        rz_params = self.qcnn_rot_params[l, i, 1, 0].unsqueeze(0).expand(sub_bsz)
                        tqf.ry(qdev, wires=i, params=ry_params)
                        tqf.rz(qdev, wires=i, params=rz_params)
                    for i in range(self.n_qubits_data):
                        tqf.cnot(qdev, wires=[i, (i + 1) % self.n_qubits_data])
                    if self.use_strided_cnot and self.n_qubits_data >= 4:
                        for i in range(self.n_qubits_data):
                            tqf.cnot(qdev, wires=[i, (i + 2) % self.n_qubits_data])
                    # 3. Data Re-uploading (Original)
                if self.reupload_data and (l < self.active_layers - 1):
                    for i in range(self.n_qubits_data):
                        tqf.rz(qdev, wires=i, params=sub_da[:, i])

                # 4. Mid-Circuit Style Injection (NEW)
                # Inject style information coherently in the middle layers
                if l < self.active_layers - 1:
                     for i in range(self.n_qubits_data):
                         # Use RX for style to be orthogonal to RZ (data re-upload) and RY (ansatz)
                         tqf.rx(qdev, wires=i, params=sub_sa[:, i])
            
            # Trainable Measurement (U3) on probabilities? No, U3 must be before measurement.
            # Re-adding Trainable Measurement (U3)
            # Apply to ALL wires because we measure all wires (or subset if fusion)
            # But self.measure_params is shape [n_data, 3].
            # So we only apply to data wires.
            for i in range(self.n_qubits_data):
                tqf.u3(qdev, wires=i, params=self.measure_params.unsqueeze(0).expand(sub_bsz, -1, -1)[:, i])
                
            # Measurement (Probabilities)
            if hasattr(qdev, 'get_states'): states = qdev.get_states()
            elif hasattr(qdev, 'get_states_1d'): states = qdev.get_states_1d()
            else: states = qdev.states
            
            # states: [sub_bsz, 2^n_wires]
            probs = (states.abs() ** 2)
            outs.append(probs)
            start = end
        quant_out = torch.cat(outs, dim=0)
        
        # 7. Post-processing & Residual
        # Cast quant_out to out_proj weight dtype (fp16/fp32)
        oq = []
        orp = []
        step = step
        for s in range(0, bsz, step):
            e = min(s + step, bsz)
            sub_q = quant_out[s:e].to(self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
            oq.append(self.out_proj(sub_q))
            
            # Determine device/dtype from first parameter of res_proj (if Sequential)
            res_param = next(self.res_proj.parameters())
            sub_p = patches_flat[s:e].to(res_param.device, dtype=res_param.dtype)
            orp.append(self.res_proj(sub_p))
        out_quant = torch.cat(oq, dim=0)
        out_res = torch.cat(orp, dim=0)
        out_flat = out_quant + out_res
        
        # 8. Reshape back
        out = out_flat.view(B, H_out, W_out, self.channels).permute(0, 3, 1, 2)
        out = F.interpolate(out, size=(H, W), mode='nearest')
        return out.to(x.device, dtype=x.dtype)


class TransformerBlock64(nn.Module):
    """
    通用 64 维 TransformerBlock（AdaLN-Zero），可接入任意返回 [B,L,64] 的注意力模块。
    """
    def __init__(self, attention: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.attention = attention
        self.norm1_64 = nn.LayerNorm(64)
        self.norm2_64 = nn.LayerNorm(64)
        self.mlp64 = MLP(dim=64, mlp_ratio=4.0, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 6 * 64)
        )
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

        
    def forward(self, tokens_64: torch.Tensor, cond64: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
        B, L, D = tokens_64.shape
        params = self.adaln(cond64)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params.chunk(6, dim=-1)
        shift_a = shift_a.unsqueeze(1)
        scale_a = scale_a.unsqueeze(1)
        gate_a  = gate_a.unsqueeze(1)
        shift_m = shift_m.unsqueeze(1)
        scale_m = scale_m.unsqueeze(1)
        gate_m  = gate_m.unsqueeze(1)

        y1 = self.norm1_64(tokens_64)
        y1 = y1 * (1.0 + scale_a) + shift_a
        attn_out = self.attention(y1, has_cls=has_cls)  # [B,L,64]
        attn_out = self.drop(attn_out)
        tokens_64 = tokens_64 + gate_a * attn_out

        y2 = self.norm2_64(tokens_64)
        y2 = y2 * (1.0 + scale_m) + shift_m
        y2 = self.mlp64(y2)
        y2 = self.drop(y2)
        tokens_64 = tokens_64 + gate_m * y2
        return tokens_64


@persistence.persistent_class
class QuantumTransformerDenoiser(nn.Module):
    """
    DiT-style denoiser backbone that uses QuantumAttention64 to modulate classical tokens.

    Init args follow DhariwalUNet signature so that training.networks.*Precond can construct it:
      - img_resolution, in_channels, out_channels, label_dim, plus model kwargs.
    """

    def __init__(self,
                 img_resolution: int,
                 in_channels: int,
                 out_channels: int,
                 label_dim: int = 0,
                 # Classical backbone params
                 model_dim: int = 384,
                 num_heads: int = 8,  # not used by quantum attention, kept for interface compatibility
                 layers: int = 4,
                 patch_size: int = 4,
                 dropout: float = 0.0,
                 pos_embed: str = 'sincos',
                 # Quantum params
                 quantum_n_qubits: int = 6,
                 quantum_q_depth: int = 2,
                 quantum_qk_dim: int = 4,
                 quantum_tau: float = 0.5,
                 quantum_attn_dropout: float = 0.1,
                 quantum_attn_gate_init: float = 0.5,
                 quantum_qk_norm: str = 'layernorm',
                 force_fp32_attention: bool = True,
                 attn_type: str = 'quantum',
                 projection_type: str = 'linear',
                 **kwargs):
        super().__init__()
        assert pos_embed in ('none', 'sincos')
        assert attn_type in ('quantum', 'classic')
        self.img_resolution = int(img_resolution)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.label_dim = int(label_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.layers = int(layers)
        self.patch_size = int(patch_size)
        self.dropout = float(dropout)
        self.pos_embed = pos_embed
        self.attn_type = attn_type
        self.projection_type = projection_type

        # Patch embedding with dual outputs (only tokens_64 will be used)
        self.patch_embed = PatchEmbed2D(in_channels=self.in_channels, model_dim=self.model_dim, patch_size=self.patch_size, projection_type=self.projection_type)

        # Positional embeddings for quantum (64-d) tokens
        H_p = W_p = self.img_resolution // self.patch_size
        self.num_tokens = H_p * W_p
        if self.pos_embed == 'sincos':
            self.pos_table_64 = self._build_sincos_table(self.num_tokens, 64)
        else:
            self.register_buffer('pos_table_64', None)

        if self.attn_type == 'classic':
            attn_mod = ClassicAttention64(num_heads=self.num_heads,
                                          attn_dropout=quantum_attn_dropout,
                                          force_fp32_attention=force_fp32_attention)
        else:
            attn_mod = QuantumAttention64(N_QUBITS=quantum_n_qubits,
                                          Q_DEPTH=quantum_q_depth,
                                          qk_dim=quantum_qk_dim,
                                          tau=quantum_tau,
                                          tau_trainable=True,
                                          attn_gate_init=quantum_attn_gate_init,
                                          attn_dropout=quantum_attn_dropout,
                                          qk_norm=quantum_qk_norm,
                                          force_fp32_attention=force_fp32_attention)

        # Transformer blocks (quantum or classic)
        self.blocks = nn.ModuleList([
            TransformerBlock64(attention=attn_mod, dropout=self.dropout)
            for _ in range(self.layers)
        ])

        # Output projection from 64-d tokens to patch pixels then fold back to image
        self.out_patch_proj = nn.Linear(64, self.out_channels * self.patch_size * self.patch_size)
        self.fold = nn.Fold(output_size=(self.img_resolution, self.img_resolution),
                            kernel_size=self.patch_size, stride=self.patch_size)

        # Noise and label embedding for 64-d quantum tokens (additive conditioning)
        self.map_noise64_0 = nn.Linear(1, 64)
        self.map_noise64_1 = nn.Linear(64, 64)
        self.map_label64 = nn.Linear(self.label_dim, 64, bias=False) if self.label_dim > 0 else None

    def _build_sincos_table(self, L: int, D: int) -> torch.Tensor:
        pos = torch.arange(L).unsqueeze(1)        # [L,1]
        div_term = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))  # [D/2]
        pe = torch.zeros(L, D)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe  # [L,D]

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor, class_labels: Optional[torch.Tensor] = None, augment_labels: Optional[torch.Tensor] = None):
        # x: [B, C_in, H, W], noise_labels: [B]
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.img_resolution and W == self.img_resolution

        # Patch embedding (only 64-d tokens are used)
        _, tokens_64 = self.patch_embed(x)  # [B,L,64]

        # Add positional embeddings to 64-d tokens
        if self.pos_embed == 'sincos' and self.pos_table_64 is not None:
            pos64 = self.pos_table_64.to(tokens_64.device, dtype=tokens_64.dtype)  # [L,64]
            tokens_64 = tokens_64 + pos64.unsqueeze(0)

        # Build time (noise) conditioning embedding; will be used for AdaLN-Zero, not added to tokens
        nl = noise_labels.reshape(B, 1).to(tokens_64.dtype)
        cond64 = F.silu(self.map_noise64_0(nl))
        cond64 = self.map_noise64_1(cond64)
        if self.map_label64 is not None and class_labels is not None:
            cond64 = cond64 + self.map_label64(class_labels.to(tokens_64.dtype))

        # Re-normalize tokens_64 for amplitude encoding stability
        tokens_64 = tokens_64 / (tokens_64.norm(p=2, dim=-1, keepdim=True) + 1e-9)

        # Quantum-only transformer blocks
        for blk in self.blocks:
            tokens_64 = blk(tokens_64, cond64.squeeze(1), has_cls=False)

        # Project back to pixels per patch and fold to image (from 64-d tokens)
        patches = self.out_patch_proj(tokens_64)  # [B,L, out*C*p*p]
        patches = patches.transpose(1, 2)          # [B, out*C*p*p, L]
        x_out = self.fold(patches)                 # [B, out, H, W]
        return x_out

class LoRALayer(nn.Module):
    """Low-Rank Adaptation Layer for Lightweight Residuals"""
    def __init__(self, in_features, out_features, rank=4, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B) # Initialize to 0 to start as identity/zero contribution

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling


class QuantumAttentionLight(QuantumAttention64):
    """
    Lightweight Quantum Attention:
    1. Scaling Encoding (No MLP)
    2. Shared Q/K/V Projection
    3. LoRA Residuals
    4. Enhanced Data Re-uploading
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. Lightweight Encoding (Linear 64->12)
        # Replaces complex encoding or MLP
        self.enc_proj = nn.Linear(64, self.N_QUBITS * 2)
        
        # 2. Shared Measurement Projection
        # Replace independent q/k/v projs
        if hasattr(self, 'q_proj'): del self.q_proj
        if hasattr(self, 'k_proj'): del self.k_proj
        if hasattr(self, 'v_proj'): del self.v_proj
        self.shared_proj = nn.Linear(64, self.inner_dim)
        
        # 3. LoRA Residuals
        # These replace the full rank residuals
        self.q_res_lora = LoRALayer(64, self.inner_dim, rank=4)
        self.k_res_lora = LoRALayer(64, self.inner_dim, rank=4)
        self.v_res_lora = LoRALayer(64, self.inner_dim, rank=4)
        
        # Learnable Scale
        self.attn_scale = nn.Parameter(torch.full((self.num_heads, 1, 1), self.qk_dim ** -0.5))
        
        # Remove unused
        if hasattr(self, 'inp_proj'): del self.inp_proj
        if hasattr(self, 'reupload_proj'): del self.reupload_proj

    def _forward_impl(self, x_64: torch.Tensor, device_name: str) -> torch.Tensor:
        B, S, D = x_64.shape
        bsz = B * S
        x_bsz = x_64.reshape(bsz, D)
        
        # Encoding Angles: [BSZ, 12] -> split to [BSZ, 6] (Rx) and [BSZ, 6] (Ry)
        # We use Tanh * pi to bound angles
        angles = torch.tanh(self.enc_proj(x_bsz)) * torch.pi
        
        # Split into Rx and Ry params
        rx_params = angles[:, :self.N_QUBITS]
        ry_params = angles[:, self.N_QUBITS:]
        
        # Common State Preparation (No Amplitude Encoding, start from |0>)
        qdev_common = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        
        # Apply Initial Rotations (Encoding)
        for i in range(self.N_QUBITS):
            tqf.rx(qdev_common, wires=i, params=rx_params[:, i])
            tqf.ry(qdev_common, wires=i, params=ry_params[:, i])
            
        # Apply PQC (Entanglement)
        self._apply_pqc(qdev_common, self.enc_w)
        
        # Get common state
        if hasattr(qdev_common, 'get_states_1d'): 
            common_states_flat = qdev_common.get_states_1d()
        else: 
            common_states_flat = qdev_common.states.reshape(bsz, -1)
        target_shape = [bsz] + [2] * self.N_QUBITS
        common_states_reshaped = common_states_flat.reshape(target_shape)
        
        # Re-uploading angles
        reupload_angles = rx_params
        
        # Branches
        # Q Branch
        qdev_q = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_q.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_q, self.q_w, x_reupload=reupload_angles)
        for i in range(self.N_QUBITS): tqf.u3(qdev_q, wires=i, params=self.meas_q_w[i].unsqueeze(0))
        probs_q = self._measure_multibasis(qdev_q)
        # Hybrid Q: Shared Proj + LoRA Residual
        q_quant = self.shared_proj(probs_q)
        q_res = self.q_res_lora(x_bsz)
        q = self.qk_ln(q_quant + q_res).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # K Branch
        qdev_k = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_k.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_k, self.k_w, x_reupload=reupload_angles)
        for i in range(self.N_QUBITS): tqf.u3(qdev_k, wires=i, params=self.meas_k_w[i].unsqueeze(0))
        probs_k = self._measure_multibasis(qdev_k)
        # Hybrid K
        k_quant = self.shared_proj(probs_k)
        k_res = self.k_res_lora(x_bsz)
        k = self.qk_ln(k_quant + k_res).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # V Branch
        qdev_v = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_v.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_v, self.v_w, x_reupload=reupload_angles)
        for i in range(self.N_QUBITS): tqf.u3(qdev_v, wires=i, params=self.meas_v_w[i].unsqueeze(0))
        probs_v = self._measure_multibasis(qdev_v)
        # Hybrid V
        v_quant = self.shared_proj(probs_v)
        v_res = self.v_res_lora(x_bsz)
        v = (v_quant + v_res).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # Attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        alpha = torch.softmax(attn_score, dim=-1)
        alpha = self.attn_drop(alpha)
        attn_out = torch.matmul(alpha, v)
        
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.inner_dim)
        attn_out = self.out_proj(attn_out)
        
        # Debug Prints
        if not self._printed_exec:
            self._printed_exec = True
            print(f"\n[QuantumAttentionLight] Debug Exec:")
            print(f"  Input Shape: {x_64.shape}")
            print(f"  Probs Q Shape: {probs_q.shape}")
            print(f"  Output Shape: {attn_out.shape}")
            print(f"  Learnable Scale Mean: {self.attn_scale.mean().item():.4f}")
        
        return attn_out


class QuantumAttentionDeep(QuantumAttentionAngle):
    """
    Deep Quantum Attention:
    1. Increased Depth (8 layers)
    2. Circular Entanglement (CNOTs between 0-1, 1-2, ..., N-0)
    3. Denser Rotation Gates
    """
    def __init__(self, *args, **kwargs):
        # Force depth 8 if not specified, but respect kwargs if they want more
        if 'Q_DEPTH' not in kwargs:
            kwargs['Q_DEPTH'] = 8
        super().__init__(*args, **kwargs)

    def _apply_pqc(self, qdev: 'tq.QuantumDevice', weights: torch.Tensor, x_reupload: Optional[torch.Tensor] = None):
        """Deep PQC with Circular Entanglement"""
        depth = weights.shape[0]
        reupload_idx = depth // 2
        
        for l in range(depth):
            # Re-uploading
            if x_reupload is not None and l == reupload_idx:
                for i in range(self.N_QUBITS):
                    tqf.rx(qdev, wires=i, params=x_reupload[:, i])
            
            rx_params = weights[l, :, 0]
            ry_params = weights[l, :, 1]
            rz_params = weights[l, :, 2] # Use 3rd param as RZ instead of post-ent RY
            
            # Pre-rotations (Rx, Ry, Rz) - denser
            for i in range(self.N_QUBITS):
                tqf.rx(qdev, wires=i, params=rx_params[i])
                tqf.ry(qdev, wires=i, params=ry_params[i])
                tqf.rz(qdev, wires=i, params=rz_params[i])
                
            # Circular CNOT Entanglement
            # 0->1, 1->2, ..., 5->0
            for i in range(self.N_QUBITS):
                tqf.cnot(qdev, wires=[i, (i + 1) % self.N_QUBITS])
                
            # No post-rotations in this layer block, we used 3 params already.
            # This is a different ansatz structure than Base.


class QuantumAttentionHybrid(QuantumAttentionAngle):
    """
    Hybrid Quantum Attention:
    1. Classical Q/K Projection (Standard Linear) -> Stable Attention Map
    2. Quantum V Projection (Angle Encoding + PQC) -> Quantum Feature Transformation
    3. Fuses best of both worlds: Geometric stability of classical attention + High dimensional mapping of Quantum
    """
    def __init__(self, input_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Classical Q and K Projections
        # Override/Shadow the quantum ones if needed, or just define new ones
        self.input_dim_val = int(input_dim)
        self.q_proj_class = nn.Linear(self.input_dim_val, self.inner_dim)
        self.k_proj_class = nn.Linear(self.input_dim_val, self.inner_dim)
        
        # If input_dim is not 64, we need to adapt the quantum path projections as well
        if self.input_dim_val != 64:
            # Re-initialize input projection for quantum path
            self.inp_proj = nn.Linear(self.input_dim_val, 64)
            # Re-initialize residual projections
            self.q_res_proj = nn.Linear(self.input_dim_val, self.inner_dim)
            self.k_res_proj = nn.Linear(self.input_dim_val, self.inner_dim)
            self.v_res_proj = nn.Linear(self.input_dim_val, self.inner_dim)
            # Re-uploading projector
            self.reupload_proj = nn.Linear(self.input_dim_val, self.N_QUBITS)
            # Angle projection
            self.angle_proj = nn.Linear(self.input_dim_val, self.N_QUBITS * 2)
            
            # Re-initialize Output Projection to match input dimension
            self.out_proj = nn.Linear(self.inner_dim, self.input_dim_val)

    def _forward_impl(self, x_64: torch.Tensor, device_name: str) -> torch.Tensor:
        B, S, D = x_64.shape
        bsz = B * S
        x_bsz = x_64.reshape(bsz, D)
        
        # 1. Classical Path for Q/K
        # Input proj for classical path (optional, but let's use the one from super or raw)
        # Let's use raw x_bsz or self.inp_proj(x_bsz). 
        # super() has self.inp_proj. Let's use it for consistency.
        # However, if input_dim != 64, inp_proj maps to 64, which is for quantum path.
        # For classical path, we want to project from input_dim directly if possible, 
        # or use the projected 64-dim feature?
        # To maintain full information for classical Q/K, we should use x_bsz directly.
        # But q_proj_class expects self.input_dim_val.
        
        x_proj = x_bsz # Direct input usage

        q = self.qk_ln(self.q_proj_class(x_proj)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        k = self.qk_ln(self.k_proj_class(x_proj)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # Apply Residuals (Classical)
        q_res = self.q_res_proj(x_bsz).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        k_res = self.k_res_proj(x_bsz).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        q = q + q_res
        k = k + k_res

        # 2. Quantum Path for V
        # Prepare Common State (Angle Encoding)
        # Code borrowed from QuantumAttentionAngle._forward_impl
        raw_out = self.angle_proj(x_bsz)
        angles = (torch.tanh(raw_out) + 1.0) * (torch.pi / 2.0)
        rx_angles = angles[:, :self.N_QUBITS]
        ry_angles = angles[:, self.N_QUBITS:]
        
        qdev_v = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        
        # Encode
        for i in range(self.N_QUBITS):
            tqf.rx(qdev_v, wires=i, params=rx_angles[:, i])
            tqf.ry(qdev_v, wires=i, params=ry_angles[:, i])
            
        # PQC (Encoder)
        self._apply_pqc(qdev_v, self.enc_w)
        
        # Re-uploading angles
        reupload_angles = torch.tanh(self.reupload_proj(x_bsz)) * torch.pi
        
        # V PQC
        self._apply_pqc(qdev_v, self.v_w, x_reupload=reupload_angles)
        
        # Measure V
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_v, wires=i, params=self.meas_v_w[i].unsqueeze(0))
        probs_v = self._measure_multibasis(qdev_v)
        
        # Project V
        # Note: parent has self.v_proj (Linear 64->Inner)
        # We reuse it.
        v_quant = self.v_proj(probs_v).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # V Residual
        v_res = self.v_res_proj(x_bsz).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        v = v_quant + v_res
        
        # 3. Attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        alpha = torch.softmax(attn_score, dim=-1)
        alpha = self.attn_drop(alpha)
        
        attn_out = torch.matmul(alpha, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.inner_dim)
        attn_out = self.out_proj(attn_out)
        
        if not self._printed_exec:
            self._printed_exec = True
            print(f"\n[QuantumAttentionHybridLite] Debug Exec:")
            print(f"  Mode: Lite Classical Q/K (Groups=2) + Quantum V")
            print(f"  Params pruned: Unused Quantum Q/K weights deleted")
            
        return attn_out
        
        # 3. Attention Mechanism
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        alpha = torch.softmax(attn_score, dim=-1)
        alpha = self.attn_drop(alpha)
        
        attn_out = torch.matmul(alpha, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.inner_dim)
        attn_out = self.out_proj(attn_out)
        
        # Debug Prints
        if not self._printed_exec:
            self._printed_exec = True
            print(f"\n[QuantumAttentionHybrid] Debug Exec:")
            print(f"  Mode: Classical Q/K + Quantum V")
            print(f"  V Probs Shape: {probs_v.shape}")
            
        return attn_out


class QuantumAttentionHybridLite(QuantumAttentionHybrid):
    """
    Lite Hybrid Quantum Attention:
    1. Removes unused Quantum Q/K parameters (Big saving).
    2. Uses Grouped Linear (Conv1d groups=2) for Classical Q/K (Parameter saving).
    3. Retains Quantum V for feature extraction.
    """
    def __init__(self, input_dim=64, *args, **kwargs):
        super().__init__(input_dim=input_dim, *args, **kwargs)
        
        # 1. Delete unused Quantum Q/K parameters and projections from base class
        # These are generated by QuantumAttentionAngle (grandparent) but not used in Hybrid
        to_delete = ['q_w', 'k_w', 'meas_q_w', 'meas_k_w', 'q_proj', 'k_proj']
        for name in to_delete:
            if hasattr(self, name):
                delattr(self, name)
                
        # 2. Replace Standard Classical Projections (from Hybrid) with Lite Grouped versions
        # Hybrid created self.q_proj_class and self.k_proj_class (Linear)
        # We replace them with Grouped Conv1d (equivalent to Grouped Linear)
        
        # Note: Conv1d expects [B, C, L]. We will reshape in forward.
        # Groups=2 reduces parameters by half.
        self.q_proj_lite = nn.Conv1d(self.input_dim_val, self.inner_dim, kernel_size=1, groups=2)
        self.k_proj_lite = nn.Conv1d(self.input_dim_val, self.inner_dim, kernel_size=1, groups=2)
        
        # Delete the standard linear ones from Hybrid to save params
        if hasattr(self, 'q_proj_class'): del self.q_proj_class
        if hasattr(self, 'k_proj_class'): del self.k_proj_class
        
        # 3. Lite Residuals Optimization (SOTA V2)
        # Removed q_res_lite and k_res_lite as they are redundant in Attention mechanism
        # (Standard attention usually applies residuals after the block, not inside Q/K)
        if hasattr(self, 'q_res_proj'): del self.q_res_proj
        if hasattr(self, 'k_res_proj'): del self.k_res_proj
        
        # 4. Lite V Residual and Out Projection
        # Replace Linear with Grouped Conv1d (groups=2) for parameter efficiency
        if hasattr(self, 'v_res_proj'): del self.v_res_proj
        if hasattr(self, 'out_proj'): del self.out_proj
        
        self.v_res_lite = nn.Conv1d(self.input_dim_val, self.inner_dim, kernel_size=1, groups=2)
        self.out_proj_lite = nn.Conv1d(self.inner_dim, self.input_dim_val, kernel_size=1, groups=2)

        # We keep V quantum parts as is (v_w, meas_v_w)
        # Revert: Use Probability Measurement to preserve high-dimensional features (2^N)
        # instead of Expectation Measurement (N).
        # self.measure_all = tq.MeasureAll(tq.PauliZ)
        
        # Re-initialize v_proj to match 2**N_QUBITS output from Probability Measurement
        self.v_proj = nn.Linear(2**self.N_QUBITS, self.inner_dim)
        
        # Zero-Init Output Projection to act as Identity initially (Stability)
        nn.init.zeros_(self.out_proj_lite.weight)
        nn.init.zeros_(self.out_proj_lite.bias)

    def _forward_impl(self, x_in: torch.Tensor, device_name: str) -> torch.Tensor:
        # x_in is [B, S, D]
        B, S, D = x_in.shape
        bsz = B * S
        x_bsz = x_in.reshape(bsz, D)
        
        # Prepare for Conv1d: [N, C, L] -> here [BSZ, D, 1]
        x_conv = x_bsz.unsqueeze(-1) # [BSZ, D, 1]
        
        # 1. Classical Path for Q/K (Lite)
        q_raw = self.q_proj_lite(x_conv).squeeze(-1) # [BSZ, Inner]
        k_raw = self.k_proj_lite(x_conv).squeeze(-1)
        
        q = self.qk_ln(q_raw).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        k = self.qk_ln(k_raw).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # Removed redundant Q/K residuals

        # 2. Quantum Path for V (Same as Standard Hybrid)
        # Use logic from Hybrid/Angle
        # Note: Hybrid uses self.angle_proj(x_bsz) and self.reupload_proj(x_bsz)
        # These are set up by Hybrid.__init__ using input_dim_val, so they are correct.
        
        raw_out = self.angle_proj(x_bsz)
        angles = (torch.tanh(raw_out) + 1.0) * (torch.pi / 2.0)
        rx_angles = angles[:, :self.N_QUBITS]
        ry_angles = angles[:, self.N_QUBITS:]
        
        qdev_v = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        for i in range(self.N_QUBITS):
            tqf.rx(qdev_v, wires=i, params=rx_angles[:, i])
            tqf.ry(qdev_v, wires=i, params=ry_angles[:, i])
            
        self._apply_pqc(qdev_v, self.enc_w)
        reupload_angles = torch.tanh(self.reupload_proj(x_bsz)) * torch.pi
        self._apply_pqc(qdev_v, self.v_w, x_reupload=reupload_angles)
        
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_v, wires=i, params=self.meas_v_w[i].unsqueeze(0))
        
        # Revert: Use Probability Measurement -> [B*S, 2^N]
        # This preserves the high-dimensional feature space of the quantum state.
        meas_v = self._measure_probs(qdev_v)
        
        v_quant = self.v_proj(meas_v).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # Lite V Residual
        v_res = self.v_res_lite(x_conv).squeeze(-1).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        v = v_quant + v_res
        
        # 3. Attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        alpha = torch.softmax(attn_score, dim=-1)
        alpha = self.attn_drop(alpha)
        
        attn_out = torch.matmul(alpha, v)
        # attn_out: [B, H, S, D_head] -> permute -> [B, S, H, D_head] -> reshape -> [B, S, Inner]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.inner_dim)
        
        # Lite Out Projection (Conv1d groups=2)
        # Input to Conv1d must be [Batch, Channels, Length]
        # Here we treat 'S' as Batch or just reshape?
        # Standard: [B*S, Inner, 1]
        attn_out_bsz = attn_out.reshape(bsz, self.inner_dim).unsqueeze(-1) # [BSZ, Inner, 1]
        attn_out = self.out_proj_lite(attn_out_bsz).squeeze(-1) # [BSZ, OutDim]
        attn_out = attn_out.reshape(B, S, self.input_dim_val)
        
        if not self._printed_exec:
            self._printed_exec = True
            print(f"\n[QuantumAttentionHybridLite] Debug Exec:")
            print(f"  Mode: Lite Classical Q/K/V_res/Out (Groups=2) + Quantum V")
            print(f"  Input Dim: {D}")
            
        return attn_out


class QuantumFrontEndQCNNState(QuantumFrontEndQCNN):
    """
    QCNN Front-End that returns the Quantum State Vector instead of measured values.
    Used for 'No Measurement' architecture where QCNN state feeds directly into Quantum Attention.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fix for Re-uploading in Amplitude Encoding mode
        # We need a separate projection to generate angles for re-uploading layers
        # because the main data_proj generates the state vector (2^N) directly.
        if self.reupload_data and self.encoding_type == 'amplitude':
             self.reupload_proj = nn.Linear(self.patch_dim_per_group, self.n_qubits_data)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Simplified forward based on parent class, but returns state.
        # Assumes n_groups=1 for simplicity in this benchmark mode.
        
        B, C, H, W = x.shape
        
        # 1. Unfold & Reshape
        patches = self.unfold(x)
        # Flatten patches: [B, C*K*K, L] -> [B*L, C*K*K]
        patches_flat = patches.transpose(1, 2).reshape(-1, self.patch_dim)
        patches_flat = patches_flat * self.inp_scale
        
        bsz_total = patches_flat.shape[0]
        # Reshape to groups (assuming n_groups=1 or we flatten groups into batch)
        # [B*L, groups, patch_dim_per_group]
        sub_patches = patches_flat.reshape(bsz_total, self.n_groups, self.patch_dim_per_group)
        sub_patches_flat = sub_patches.reshape(-1, self.patch_dim_per_group)
        sub_bsz = sub_patches_flat.shape[0]

        # Prepare Style
        L = patches.shape[-1]
        style_expanded = style.unsqueeze(1).expand(-1, L, -1).reshape(bsz_total, -1)
        sub_style = style_expanded.unsqueeze(1).expand(-1, self.n_groups, -1).reshape(sub_bsz, -1)
        
        # Expand Params
        mod_params_expanded = self.mod_params.unsqueeze(0).expand(bsz_total, -1, -1, -1, -1).reshape(sub_bsz, self.n_layers, self.n_qubits_data, 3)
        qcnn_rot_params_expanded = self.qcnn_rot_params.unsqueeze(0).expand(bsz_total, -1, -1, -1, -1, -1).reshape(sub_bsz, self.n_layers, self.n_qubits_data, 2, 3)
        
        # Batch Processing
        chunk_size_limit = self.max_qdev_bsz
        if sub_bsz <= chunk_size_limit * 2: 
            chunk_size_limit = sub_bsz
            
        outs_state = []
        
        # Device creation
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=min(sub_bsz, chunk_size_limit), device=self.device_name)
        
        for s in range(0, sub_bsz, chunk_size_limit):
            e = min(s + chunk_size_limit, sub_bsz)
            actual_chunk_size = e - s
            
            if actual_chunk_size != qdev.bsz:
                qdev_chunk = tq.QuantumDevice(n_wires=self.n_wires, bsz=actual_chunk_size, device=self.device_name)
            else:
                qdev_chunk = qdev
                if hasattr(qdev_chunk, 'reset_states'): qdev_chunk.reset_states(actual_chunk_size)
            
            chunk_patches = sub_patches_flat[s:e]
            chunk_style = sub_style[s:e]
            chunk_mod_params = mod_params_expanded[s:e]
            chunk_rot_params = qcnn_rot_params_expanded[s:e]
            
            # Data Encoding (Linear/Amplitude)
            dtype = next(self.data_proj.parameters()).dtype
            chunk_da_reupload = None

            if self.encoding_type == 'amplitude':
                chunk_da = self.data_proj(chunk_patches.to(dtype))
                # Normalize and Pad logic (Simplified from parent)
                norm = torch.norm(chunk_da, p=2, dim=1, keepdim=True) + 1e-8
                chunk_da_norm = chunk_da / norm
                target_dim = 2 ** self.n_qubits_data
                curr_dim = chunk_da_norm.shape[1]
                if curr_dim < target_dim:
                    padding = torch.zeros(actual_chunk_size, target_dim - curr_dim, device=chunk_da.device, dtype=chunk_da.dtype)
                    data_state = torch.cat([chunk_da_norm, padding], dim=1)
                else:
                    data_state = chunk_da_norm[:, :target_dim]
                    data_state = data_state / (torch.norm(data_state, p=2, dim=1, keepdim=True) + 1e-8)
                
                # Ancilla logic: Pad with |0> for ancilla qubits
                # Assume Data on Wires 0..N_data-1, Ancilla on N_data..N_wires-1
                n_ancilla = self.n_wires - self.n_qubits_data
                
                # Reshape data to [B, 2, ..., 2] (N_data times)
                data_view = data_state.reshape([actual_chunk_size] + [2] * self.n_qubits_data)
                data_complex = torch.complex(data_view, torch.zeros_like(data_view))
                
                if n_ancilla > 0:
                    # Initialize full state with zeros
                    full_state = torch.zeros([actual_chunk_size] + [2] * self.n_wires, dtype=torch.cfloat, device=chunk_da.device)
                    
                    # Assign data to the slice where ancilla are |0>
                    # Slicing: [:, :, ..., :, 0, 0, ..., 0]
                    # Create a tuple of slices
                    # dim 0 is batch (slice(None))
                    # dims 1..N_data are data (slice(None))
                    # dims N_data+1..end are ancilla (index 0)
                    idx = [slice(None)] * (1 + self.n_qubits_data) + [0] * n_ancilla
                    full_state[tuple(idx)] = data_complex
                    
                    qdev_chunk.states = full_state
                else:
                    qdev_chunk.states = data_complex
                
                # Calculate Re-uploading Angles if enabled
                if self.reupload_data and hasattr(self, 'reupload_proj'):
                    chunk_da_reupload = torch.tanh(self.reupload_proj(chunk_patches.to(dtype))) * math.pi
            else:
                 # Angle encoding
                 chunk_da = torch.tanh(self.data_proj(chunk_patches.to(dtype))) * math.pi
                 chunk_da_reupload = chunk_da
            
            chunk_sa = torch.tanh(self.style_to_data(chunk_style.to(self.style_to_data.weight.dtype))) * math.pi
            
            # Apply Circuit
            self._apply_fusion_circuit(
                qdev_chunk, actual_chunk_size, chunk_da_reupload, 
                chunk_sa, None, None, chunk_mod_params, chunk_rot_params,
                self.n_qubits_data, self.n_qubits_ancilla, self.active_layers,
                self.use_strided_cnot, self.reupload_data, self.encoding_type
            )
            
            # NO MEASUREMENT - Get State
            if hasattr(qdev_chunk, 'get_states_1d'): 
                states = qdev_chunk.get_states_1d() # [chunk, 2^N]
            else: 
                states = qdev_chunk.states.reshape(actual_chunk_size, -1)
            
            outs_state.append(states)
            
        # Concat all states
        all_states = torch.cat(outs_state, dim=0) # [B*L*groups, 2^N]
        
        # Reshape to [B, L, groups, 2^N] -> [B, L, 2^N] (assuming groups=1)
        # Or just return [B, L, D_state]
        return all_states.reshape(B, L, -1)


class QuantumAttentionState(QuantumAttention64):
    """
    Quantum Attention that accepts a Quantum State Vector as input.
    Skips encoding step.
    """
    def forward(self, x_state: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
        """
        x_state: [B, S, 2^N_QUBITS] (Complex64/128)
        """
        dev = x_state.device
        device_name = self.device_name or dev.type
        
        # We need to ensure input is complex
        if not x_state.is_complex():
             # Should not happen if coming from QCNNState
             pass
             
        out = self._forward_impl_state(x_state, device_name)
        return out

    def _forward_impl_state(self, x_state: torch.Tensor, device_name: str) -> torch.Tensor:
        B, S, StateDim = x_state.shape
        bsz = B * S
        x_flat = x_state.reshape(bsz, StateDim) # [bsz, 2^N]
        
        # Prepare shape for TQ [bsz, 2, 2, ...]
        target_shape = [bsz] + [2] * self.N_QUBITS
        common_states_reshaped = x_flat.reshape(target_shape)
        
        # Since we have the state, we can compute "reupload angles" from the state?
        # No, re-upload usually requires classical data. 
        # For this "No Measurement" mode, we might skip re-upload or use a dummy.
        # Or we can project the state probabilities to classical to get reupload parameters?
        # For simplicity, we SKIP re-upload or set it to zero.
        x_reupload = None 
        
        # 2. Fork to Q/K/V branches (same as parent but start from state)
        # Q Branch
        qdev_q = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_q.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_q, self.q_w, x_reupload=x_reupload)
        for i in range(self.N_QUBITS): tqf.u3(qdev_q, wires=i, params=self.meas_q_w[i].unsqueeze(0))
        probs_q = self._measure_multibasis(qdev_q)
        q = self.qk_ln(self.q_proj(probs_q)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # K Branch
        qdev_k = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_k.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_k, self.k_w, x_reupload=x_reupload)
        for i in range(self.N_QUBITS): tqf.u3(qdev_k, wires=i, params=self.meas_k_w[i].unsqueeze(0))
        probs_k = self._measure_multibasis(qdev_k)
        k = self.qk_ln(self.k_proj(probs_k)).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # V Branch
        qdev_v = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_v.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_v, self.v_w, x_reupload=x_reupload)
        for i in range(self.N_QUBITS): tqf.u3(qdev_v, wires=i, params=self.meas_v_w[i].unsqueeze(0))
        probs_v = self._measure_multibasis(qdev_v)
        v = self.v_proj(probs_v).reshape(B, S, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        
        # Attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qk_dim)
        alpha = torch.softmax(attn_score, dim=-1)
        alpha = self.attn_drop(alpha)
        attn_out = torch.matmul(alpha, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.inner_dim)
        attn_out = self.out_proj(attn_out)
        
        return attn_out


class QuantumAdapterHybridLite(nn.Module):
    """
    Adapter class to wrap QuantumAttentionHybridLite for integration into UNetBlock.
    Handles shape transformation from [B, C, H, W] to [B, S, C] and back.
    """
    def __init__(self, in_channels, num_heads=4, device_name='cuda', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # Calculate qk_dim based on input channels and heads
        if in_channels % num_heads != 0:
            qk_dim = in_channels // num_heads
        else:
            qk_dim = in_channels // num_heads
            
        self.attn = QuantumAttentionHybridLite(
            input_dim=in_channels,
            N_QUBITS=6,
            qk_dim=qk_dim,
            num_heads=num_heads,
            device_name=device_name,
            **kwargs
        )

    def forward(self, x: torch.Tensor, num_heads: Optional[int] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        out_flat = self.attn(x_flat)
        out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

class QuantumAdapterHybrid(nn.Module):
    """
    Adapter class to wrap QuantumAttentionHybrid for integration into UNetBlock.
    Handles shape transformation from [B, C, H, W] to [B, S, C] and back.
    """
    def __init__(self, in_channels, num_heads=4, device_name='cuda', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # Calculate qk_dim based on input channels and heads
        # Ensure divisible
        if in_channels % num_heads != 0:
            # Fallback or error? For now, we assume it's divisible as per UNet logic
            qk_dim = in_channels // num_heads
        else:
            qk_dim = in_channels // num_heads
            
        self.attn = QuantumAttentionHybrid(
            input_dim=in_channels,
            N_QUBITS=6, # Fixed for now
            qk_dim=qk_dim,
            num_heads=num_heads, # Passed to QuantumAttention64 base
            device_name=device_name,
            **kwargs
        )

    def forward(self, x: torch.Tensor, num_heads: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
            num_heads: Optional override (ignored for now as init fixes it)
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        # Reshape to [B, S, C] where S = H*W
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Forward pass through Quantum Attention
        # Note: QuantumAttentionHybrid expects [B, S, D]
        out_flat = self.attn(x_flat)
        
        # Reshape back to [B, C, H, W]
        out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return out
