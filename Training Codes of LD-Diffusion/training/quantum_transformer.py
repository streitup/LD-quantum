import math
import os
from typing import Optional, Tuple

import torch
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
    _TQ_AVAILABLE = True
except Exception:
    _TQ_AVAILABLE = False

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
                 Q_DEPTH: int = 2,
                 qk_dim: int = 4,
                 tau: float = 0.5,
                 tau_trainable: bool = True,
                 attn_gate_init: float = 0.5,  # kept for API compatibility (ignored; gating via AdaLN)
                 attn_dropout: float = 0.1,
                 qk_norm: str = 'layernorm',
                 force_fp32_attention: bool = True,
                 device_name: Optional[str] = None):
        super().__init__()
        if not _TQ_AVAILABLE:
            raise ImportError("TorchQuantum 未安装或不可用：QuantumAttention64 依赖 torchquantum。请先安装 'torchquantum'.")
        assert N_QUBITS == 6, "本实现固定使用 N_QUBITS=6（2^6=64）以匹配 64 维幅度编码。"
        assert qk_norm in ('none', 'layernorm')

        self.N_QUBITS = int(N_QUBITS)
        self.Q_DEPTH = int(Q_DEPTH)
        self.qk_dim = int(qk_dim)
        self.force_fp32_attention = bool(force_fp32_attention)
        self.device_name = device_name

        # Trainable PQC parameters (enc + branch-specific)
        self.enc_w = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.q_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.k_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.v_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))

        # Z measurement and q/k projections
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.q_proj = nn.Linear(self.N_QUBITS, self.qk_dim)
        self.k_proj = nn.Linear(self.N_QUBITS, self.qk_dim)
        self.qk_ln  = nn.LayerNorm(self.qk_dim) if qk_norm == 'layernorm' else nn.Identity()

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

        # Learnable Input Scaling (pre-encoding)
        self.inp_scale = nn.Parameter(torch.ones(64))

        # Trainable Measurement Basis (U3 before measurement)
        # For Q, K (Expectation based) and V (Prob based)
        self.meas_q_w = nn.Parameter(0.1 * torch.randn(self.N_QUBITS, 3))
        self.meas_k_w = nn.Parameter(0.1 * torch.randn(self.N_QUBITS, 3))
        self.meas_v_w = nn.Parameter(0.1 * torch.randn(self.N_QUBITS, 3))

        # numerical stability epsilon
        self.eps = 1e-9
        self._printed_exec = False

    # --- internal helpers ---
    def _apply_pqc(self, qdev: 'tq.QuantumDevice', weights: torch.Tensor):
        """RX+RY -> CNOT chain -> RY with weights [depth, N_QUBITS, 3]."""
        depth = weights.shape[0]
        for l in range(depth):
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
        probs = (states.abs() ** 2)  # (bsz, 64)
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

    def _forward_impl(self, x_64: torch.Tensor, device_name: str) -> torch.Tensor:
        # Optimized implementation: Common Encoding Fork
        # 1. Prepare common state |psi_enc> = U_enc(AmplitudeEncode(x))
        B, S, D = x_64.shape
        bsz = B * S
        x_bsz = x_64.reshape(bsz, D)
        
        # Apply Input Scaling
        x_bsz = x_bsz * self.inp_scale
        
        # Create common device
        qdev_common = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        self._amplitude_encode(qdev_common, x_bsz)
        self._apply_pqc(qdev_common, self.enc_w)
        
        # Get common state (Flattened)
        if hasattr(qdev_common, 'get_states_1d'): 
            common_states_flat = qdev_common.get_states_1d()
        else: 
            # Flatten if it is [B, 2, 2, ...]
            common_states_flat = qdev_common.states.reshape(bsz, -1)
        
        # Prepare state for injection: [B] + [2]*N
        target_shape = [bsz] + [2] * self.N_QUBITS
        common_states_reshaped = common_states_flat.reshape(target_shape)

        # 2. Fork to Q/K/V branches
        # Q Branch
        qdev_q = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_q.states = common_states_reshaped.clone() # Direct set
        self._apply_pqc(qdev_q, self.q_w)
        # Trainable Measurement Basis
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_q, wires=i, params=self.meas_q_w[i])
        z_q = self.measure_z(qdev_q)
        q = self.qk_ln(self.q_proj(z_q)).reshape(B, S, self.qk_dim)
        
        # K Branch
        qdev_k = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_k.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_k, self.k_w)
        # Trainable Measurement Basis
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_k, wires=i, params=self.meas_k_w[i])
        z_k = self.measure_z(qdev_k)
        k = self.qk_ln(self.k_proj(z_k)).reshape(B, S, self.qk_dim)
        
        # V Branch
        qdev_v = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_v.states = common_states_reshaped.clone()
        self._apply_pqc(qdev_v, self.v_w)
        # Trainable Measurement Basis
        for i in range(self.N_QUBITS):
            tqf.u3(qdev_v, wires=i, params=self.meas_v_w[i])
        probs = self._measure_probs(qdev_v)
        v = F.layer_norm(probs.reshape(B, S, D), normalized_shape=(D,))

        # RBF attention weights
        dist_sq = torch.cdist(q, k, p=2) ** 2   # (B, S, S)
        tau_eff = (F.softplus(self.raw_tau) + self.eps) if self.tau_trainable else (self.tau_value + self.eps)
        alpha = torch.exp(-dist_sq / tau_eff)   # (B, S, S)
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + self.eps)
        attn_out = torch.einsum('bsj,bjd->bsd', alpha, v)  # (B,S,64)
        attn_out = self.attn_drop(attn_out)
        return attn_out


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

    def __init__(self, in_channels: int, model_dim: int, patch_size: int = 4, eps: float = 1e-9):
        super().__init__()
        assert patch_size > 0 and isinstance(patch_size, int)
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.eps = float(eps)

        self.conv = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # Ensure unfolded tokens map to 64-d for quantum attention.
        D_unfold = in_channels * patch_size * patch_size
        if D_unfold == 64:
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
                for i in range(n_qubits_data):
                    # Fusion Re-uploading
                    if sub_sa is not None:
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
                for i in range(n_qubits_data):
                    # Fusion Re-uploading
                    if sub_sa is not None:
                        tqf.rz(qdev, wires=i, params=(sub_da[:, i] + sub_sa[:, i]))
                    else:
                        tqf.rz(qdev, wires=i, params=sub_da[:, i])

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
                 stride: int = 2):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.n_groups = int(n_groups)
        self.use_strong_bypass = bool(use_strong_bypass)
        
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
        self._qdev_cached_bsz = None
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
            
        self.data_proj = nn.Linear(self.patch_dim_per_group, self.data_proj_dim)
        
        self.style_proj = nn.Linear(style_dim, n_qubits_ancilla)
        self.style_to_data = nn.Linear(style_dim, n_qubits_data) # Style shared across groups for now
        
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
        
        # Create Device Once
        # Note: tq.QuantumDevice(bsz=N) allocates memory. 
        # We create a device large enough for the chunk.
        current_chunk_size = min(sub_bsz, chunk_size_limit)
        
        # Reuse strategy: Use a persistent device if enabled
        if self.reuse_device and self._qdev_cached is not None and self._qdev_cached_bsz >= current_chunk_size:
             qdev = self._qdev_cached
             # Reset state is done implicitly by new encoding usually, or we force reset
             if hasattr(qdev, 'reset_states'):
                 qdev.reset_states(bsz=current_chunk_size)
             else:
                 # Re-init is safer if reset not available, but slower
                 qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=current_chunk_size, device=self.device_name)
        else:
             qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=current_chunk_size, device=self.device_name)
             if self.reuse_device:
                 self._qdev_cached = qdev
                 self._qdev_cached_bsz = current_chunk_size
        
        for s in range(0, sub_bsz, chunk_size_limit):
            e = min(s + chunk_size_limit, sub_bsz)
            actual_chunk_size = e - s
            
            # If the last chunk is smaller, we might need a smaller device or just pad/slice
            # TQ requires bsz match exactly usually.
            if actual_chunk_size != qdev.bsz:
                # Resize or create temp
                qdev_chunk = tq.QuantumDevice(n_wires=self.n_wires, bsz=actual_chunk_size, device=self.device_name)
            else:
                qdev_chunk = qdev
                # Important: Reset states to |0>
            # For Amplitude Encoding, we will set states explicitly later
            if self.encoding_type != 'amplitude' and hasattr(qdev_chunk, 'reset_states'):
                qdev_chunk.reset_states(actual_chunk_size)
            
            chunk_patches = sub_patches_flat[s:e]
            chunk_style = sub_style[s:e]
            
            # Slice params
            chunk_mod_params = mod_params_expanded[s:e]
            chunk_rot_params = qcnn_rot_params_expanded[s:e]
            chunk_meas_params = measure_params_expanded[s:e]
            
            # Data Encoding (Common)
            if self.encoding_type == 'linear':
                chunk_da = self.data_proj(chunk_patches.to(self.data_proj.weight.dtype))
            elif self.encoding_type == 'amplitude':
                # Amplitude Encoding: Use projected features directly (Linear)
                # They will be normalized and mapped to states below
                chunk_da = self.data_proj(chunk_patches.to(self.data_proj.weight.dtype))
            else:
                chunk_da = torch.tanh(self.data_proj(chunk_patches.to(self.data_proj.weight.dtype))) * math.pi
            
            chunk_sa = torch.tanh(self.style_to_data(chunk_style.to(self.style_to_data.weight.dtype))) * math.pi
            
            # State Preparation for Amplitude Encoding
            if self.encoding_type == 'amplitude':
                # 1. Normalize Data (L2)
                # chunk_da: [B, D]
                # Avoid div by zero
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
                    # Re-normalize after truncation? Yes
                    norm = torch.norm(data_state, p=2, dim=1, keepdim=True) + 1e-8
                    data_state = data_state / norm
                else:
                    data_state = chunk_da_norm
                
                # 3. Handle Ancilla (Tensor Product)
                # State = |Data> (x) |Ancilla=0>
                # Ancilla state is [1, 0, ..., 0] of size 2^n_ancilla
                if self.n_wires_ancilla > 0:
                    ancilla_dim = 2 ** self.n_wires_ancilla
                    # Construct full state via Kronecker product
                    # Optimized: Since ancilla is |0...0>, we just pad zeros in the interleaved manner?
                    # No, A (x) [1, 0...] = [a0, 0..., a1, 0...]
                    # This is equivalent to inserting (ancilla_dim - 1) zeros after each element of data_state
                    # Or simpler: torch.kron
                    
                    # Create Ancilla State [1, 0...]
                    ancilla_state = torch.zeros(actual_chunk_size, ancilla_dim, device=chunk_da.device, dtype=chunk_da.dtype)
                    ancilla_state[:, 0] = 1.0
                    
                    # Batch Kron: A [B, N], B [B, M] -> [B, N*M]
                    # torch.einsum 'bi,bj->bij' -> flatten
                    full_state_real = torch.einsum('bi,bj->bij', data_state, ancilla_state).reshape(actual_chunk_size, -1)
                else:
                    full_state_real = data_state

                # 4. Set States (Complex)
                flat_state = torch.complex(full_state_real, torch.zeros_like(full_state_real))
                # Reshape to [B, 2, 2, ..., 2] required by TQ gate operations
                state_shape = [actual_chunk_size] + [2] * self.n_wires
                qdev_chunk.states = flat_state.reshape(state_shape)

            # Apply Circuit (Integrated Fusion)
            self._apply_fusion_circuit(
                qdev_chunk, actual_chunk_size, chunk_da, chunk_sa, 
                None, # interaction_wires (not used in simplified grouped mode)
                None, # data_wires (implicit 0..n)
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
            
            # Probabilities: |psi|^2
            probs = (states.abs() ** 2)
            outs.append(probs)

        # Concat chunks
        quant_out_flat = torch.cat(outs, dim=0).to(x.device)
        
        # Project Output
        # [sub_bsz, out_channels_per_group]
        quant_proj = self.out_proj(quant_out_flat.to(self.out_proj.weight.dtype))
        
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

        # Patch embedding with dual outputs (only tokens_64 will be used)
        self.patch_embed = PatchEmbed2D(in_channels=self.in_channels, model_dim=self.model_dim, patch_size=self.patch_size)

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
