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
        B, S, D = x_64.shape
        q = self._q_branch(x_64, device_name)   # (B,S,qk_dim)
        k = self._k_branch(x_64, device_name)   # (B,S,qk_dim)
        v = self._v_branch(x_64, device_name)   # (B,S,64)

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
                 encoding: str = 'angle', re_uploading: bool = True):
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
        
        # 1. Classical projection
        if self.encoding == 'amplitude':
            self.proj_in = nn.Linear(in_features, self.q_dim)
            self.norm_in = nn.LayerNorm(self.q_dim)
        else:
            # For angle encoding, project to n_qubits angles
            self.proj_in = nn.Linear(in_features, self.n_qubits)
        
        # 2. Quantum Circuit Parameters
        self.q_weights = nn.Parameter(0.1 * torch.randn(q_depth, self.n_qubits, 3))
        
        # Re-uploading weights (scales for input injection)
        if self.re_uploading and self.encoding == 'angle':
             self.upload_scales = nn.Parameter(torch.ones(q_depth, self.n_qubits))

        # 3. Output projection
        self.proj_out = nn.Linear(self.q_dim, out_features)

    def apply_circuit(self, qdev, wires, inputs):
        """
        Apply the MLP circuit (Encoding + PQC) to the given wires on qdev.
        Used for direct integration with QCNN.
        inputs: Raw input tensor [B, in_features]
        """
        # 1. Pre-process inputs
        if self.encoding == 'amplitude':
            x_q = self.proj_in(inputs)
            x_q = self.norm_in(x_q)
            x_q = x_q / (x_q.norm(p=2, dim=-1, keepdim=True) + self.eps)
            states = x_q.to(torch.cfloat)
            # Amplitude encoding sets state, difficult to apply to specific wires if entangled?
            # TorchQuantum set_states applies to ALL wires usually.
            # If used as subsystem, we might need a workaround or assume it's initialization.
            # Here we assume it's initialization of these wires.
            # CAUTION: set_states on qdev sets the whole state.
            # If wires != all wires of qdev, this is invalid for set_states.
            # For integration, we prefer Angle Encoding.
            if hasattr(qdev, 'set_states'): qdev.set_states(states) # Only works if qdev has matching n_wires
            else: pass # Fallback or error?
        else:
            # Angle Encoding
            angles = torch.tanh(self.proj_in(inputs)) * math.pi # [-pi, pi]
            
            # Initial Encoding
            for i, wire in enumerate(wires):
                tqf.ry(qdev, wires=wire, params=angles[:, i])

        # 2. PQC
        depth = self.q_weights.shape[0]
        for l in range(depth):
            # Re-uploading
            if self.re_uploading and self.encoding == 'angle':
                for i, wire in enumerate(wires):
                    # Inject data again
                    scaled_angle = angles[:, i] * self.upload_scales[l, i]
                    tqf.ry(qdev, wires=wire, params=scaled_angle)

            # Variational Layers
            rx_params = self.q_weights[l, :, 0]
            ry_params = self.q_weights[l, :, 1]
            ent_ry_params = self.q_weights[l, :, 2]
            
            for i, wire in enumerate(wires):
                tqf.rx(qdev, wires=wire, params=rx_params[i])
                tqf.ry(qdev, wires=wire, params=ry_params[i])
            
            # Entanglement (Ring)
            n = len(wires)
            if n > 1:
                for i in range(n):
                    tqf.cnot(qdev, wires=[wires[i], wires[(i + 1) % n]])
            
            for i, wire in enumerate(wires):
                tqf.ry(qdev, wires=wire, params=ent_ry_params[i])

    def _apply_pqc(self, qdev: 'tq.QuantumDevice', weights: torch.Tensor):
        # Legacy internal method, redirected to apply_circuit if possible
        # But apply_circuit needs 'inputs'.
        # For backward compatibility of _apply_pqc signature, we keep it simple or deprecated.
        pass 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features]
        if not self._printed_exec:
            self._printed_exec = True

        B = x.shape[0]
        dev = x.device
        device_name = self.device_name or x.device.type
        
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=B, device=device_name)
        
        # Use apply_circuit logic
        # wires are 0..n_qubits-1
        wires = list(range(self.n_qubits))
        self.apply_circuit(qdev, wires, x)
        
        # 4. Measure
        if hasattr(qdev, 'get_states'): states_out = qdev.get_states()
        elif hasattr(qdev, 'get_states_1d'): states_out = qdev.get_states_1d()
        else: states_out = qdev.states
        probs = (states_out.abs() ** 2)
        
        # 5. Project to output
        proj_dev = self.proj_out.weight.device
        proj_dtype = self.proj_out.weight.dtype
        out = self.proj_out(probs.to(proj_dev, dtype=proj_dtype))
        return out.to(x.device, dtype=x.dtype)


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
    def __init__(self, channels: int, style_dim: int, n_qubits_data: int = 6, n_qubits_ancilla: int = 2, 
                 n_layers: int = 2, freeze_qcnn: bool = False, device_name: Optional[str] = None,
                 time_emb_module: Optional[nn.Module] = None,
                 use_strided_cnot: bool = False,
                 reupload_data: bool = False):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.n_qubits_data = n_qubits_data
        
        # If integrated with QuantumMLP, we allow dimension mismatch (Adaptation via Measurement)
        if time_emb_module is not None and hasattr(time_emb_module, 'n_qubits'):
             self.n_qubits_qmlp = time_emb_module.n_qubits
        else:
             self.n_qubits_qmlp = 0

        self.n_qubits_ancilla = n_qubits_ancilla
        self.n_wires = n_qubits_data
        self.n_layers = n_layers
        self.freeze_qcnn = freeze_qcnn
        self.device_name = device_name
        self.time_emb_module = time_emb_module
        self.use_strided_cnot = bool(use_strided_cnot)
        self.reupload_data = bool(reupload_data)
        self.eps = 1e-9
        self._printed_exec = False
        self.active_layers = n_layers # For layer-wise training

        # Pre-processing: Patch extraction via Unfold + Dimension Reduction
        # Assume 3x3 kernel for local context
        self.kernel_size = 3
        self.padding = 1
        self.patch_dim = channels * self.kernel_size * self.kernel_size
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=1)
        
        # Project Patch to Data Qubits dimension (for encoding)
        self.data_proj = nn.Linear(self.patch_dim, n_qubits_data)
        
        self.style_proj = nn.Linear(style_dim, n_qubits_ancilla)
        self.style_to_data = nn.Linear(style_dim, n_qubits_data)
        
        # Refactored Parameters for Fine-grained Control and Strict Adherence to Document
        
        # 1. Ancilla Evolution Params (U_mlp): Applied to Ancilla qubits to generate |psi_time>
        # We model this as layer-wise rotations on ancilla qubits
        self.ancilla_params = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits_ancilla, 3))
        
        # 2. Modulation Params (Ancilla-Data Interface): CRX/CZ weights
        # These should be TRAINABLE even when freeze_qcnn is True
        self.mod_params = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits_data, 3)) # One param per interaction pair
        
        # 3. Spatial QCNN Params (The Backbone):
        # Includes Local Rotations (RY/RZ) and Entanglement weights (if parameterized)
        # We separate them into two sets for clarity, but they belong to the backbone.
        # Data Rotations
        self.qcnn_rot_params = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits_data, 2, 3)) # 2 ops (RY, RZ) per qubit
        # Entanglement Rotations (if using parameterized entanglement or post-entanglement rotations)
        # Assuming HEA with fixed CNOTs but maybe some rotations. 
        # The previous code had "Stage 3" which was used for local rotations.
        # Let's keep it simple: qcnn_rot_params covers the HEA local operations.
        
        if self.freeze_qcnn:
            # Freeze QCNN Backbone (Spatial Features)
            self.qcnn_rot_params.requires_grad = False
            # We also freeze Ancilla Evolution? The document says "Only train Time Modulation ... and Measurement Basis".
            # This implies Ancilla Evolution (which is part of the "Generator") might also be frozen if it's considered "Backbone"?
            # Or maybe it should be trained?
            # "Random initialize QCNN backbone parameters... only train Time Modulation (Ancilla-Data Interface) and Measurement Basis."
            # This strongly suggests freezing the internal evolution of both Data and Ancilla, and only training the INTERFACE.
            self.ancilla_params.requires_grad = False
        
        # Trainable Measurement Basis: Single layer of U3 rotations on Data Qubits
        self.measure_params = nn.Parameter(0.1 * torch.randn(n_qubits_data, 3))
        
        # Output Projection
        self.out_proj = nn.Linear(n_qubits_data, channels)
        
        # Classical Residual
        self.res_proj = nn.Linear(self.patch_dim, channels)
        
        if _TQ_AVAILABLE:
            self.measure_z = tq.MeasureAll(tq.PauliZ)

    def set_active_layers(self, n: int):
        self.active_layers = min(max(1, n), self.n_layers)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # style can be 'emb' (raw input) if time_emb_module is used
        if not self._printed_exec:
            self._printed_exec = True

        B, C, H, W = x.shape
        
        # 1. Unfold & Reshape
        patches = self.unfold(x) # [B, patch_dim, L], L=H*W
        L = patches.shape[-1]
        patches_flat = patches.transpose(1, 2).reshape(-1, self.patch_dim) # [N_samples, patch_dim]
        
        # 2. Build classical style parameters via QMLP (once per batch) or pass-through
        # QMLP outputs [B, style_dim] as classical parameters; expand to pixels
        if self.time_emb_module is not None:
            style_base = self.time_emb_module(style)  # [B, style_dim]
        else:
            style_base = style  # [B, style_dim]
        style_flat = style_base.view(B, 1, 1, -1).expand(B, H, W, -1).reshape(-1, self.style_dim)
        data_angles = torch.tanh(self.data_proj(patches_flat.to(self.data_proj.weight.dtype))) * math.pi
        style_data_angles = torch.tanh(self.style_to_data(style_flat.to(self.style_to_data.weight.dtype))) * math.pi
        bsz = patches_flat.shape[0]
        dev = x.device
        device_name = self.device_name or x.device.type
        qdev = tq.QuantumDevice(n_wires=self.n_qubits_data, bsz=bsz, device=device_name)
        if device_name == 'cpu':
            data_angles = data_angles.to('cpu')
            style_data_angles = style_data_angles.to('cpu')
        for i in range(self.n_qubits_data):
            tqf.ry(qdev, wires=i, params=(data_angles[:, i] + style_data_angles[:, i]))
        for l in range(self.active_layers):
            # 1. Local Rotations (RY/RZ) on Data Qubits
            for i in range(self.n_qubits_data):
                # Expand params to [bsz]
                ry_params = self.qcnn_rot_params[l, i, 0, 0].unsqueeze(0).expand(bsz)
                rz_params = self.qcnn_rot_params[l, i, 1, 0].unsqueeze(0).expand(bsz)
                if device_name == 'cpu':
                    ry_params = ry_params.to('cpu')
                    rz_params = rz_params.to('cpu')
                tqf.ry(qdev, wires=i, params=ry_params)
                tqf.rz(qdev, wires=i, params=rz_params)
                
            # 2. Multi-scale Entanglement (Ring CNOTs + Strided CNOTs)
            # Nearest Neighbor
            for i in range(self.n_qubits_data):
                tqf.cnot(qdev, wires=[i, (i + 1) % self.n_qubits_data])
            
            # Optional strided entanglement for additional mixing
            if self.use_strided_cnot and self.n_qubits_data >= 4:
                for i in range(self.n_qubits_data):
                    tqf.cnot(qdev, wires=[i, (i + 2) % self.n_qubits_data])
                
            # --- C. Data Re-uploading ---
            # Optional: Re-encode data using RZ between layers
            if self.reupload_data and (l < self.active_layers - 1):
                for i in range(self.n_qubits_data):
                    tqf.rz(qdev, wires=i, params=data_angles[:, i])
                    
        # 5. Trainable Measurement Basis
        # Apply U3(theta, phi, lam) on Data Qubits
        for i in range(self.n_qubits_data):
            params_expanded = self.measure_params[i].unsqueeze(0).expand(bsz, -1)
            if device_name == 'cpu':
                params_expanded = params_expanded.to('cpu')
            tqf.u3(qdev, wires=i, params=params_expanded)
            
        # 6. Measurement (Z-basis on Data Qubits only)
        # Use MeasureAll to get expectation values for all wires, then slice
        all_expvals = self.measure_z(qdev) # [N_samples, n_wires]
        quant_out = all_expvals[:, :self.n_qubits_data] # [N_samples, n_qubits_data]
        
        # 7. Post-processing & Residual
        # Cast quant_out to out_proj weight dtype (fp16/fp32)
        out_quant = self.out_proj(quant_out.to(self.out_proj.weight.device, dtype=self.out_proj.weight.dtype))
        # Cast patches_flat to res_proj device/dtype
        out_res = self.res_proj(patches_flat.to(self.res_proj.weight.device, dtype=self.res_proj.weight.dtype))
        
        out_flat = out_quant + out_res
        
        # 8. Reshape back
        out = out_flat.view(B, H, W, self.channels).permute(0, 3, 1, 2)
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
