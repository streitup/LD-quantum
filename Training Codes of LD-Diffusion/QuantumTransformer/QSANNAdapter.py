import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# Prefer torch.amp.autocast; fall back to torch.autocast or torch.cuda.amp.autocast for compatibility
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


class QSANNAdapter(nn.Module):
    """TorchQuantum QSANN 自注意力适配器（兼容原有接口）。

    主接口：adapter(x_norm, num_heads=H) -> attn_out
      - x_norm: [B, C, H, W]
      - attn_out: [B, C, H, W]（仅注意力输出，不含残差相加；残差在 UNetBlock 中由 proj + add 完成）

    兼容接口：adapter(q, k, v)
      - q, k, v: [B*H, C_head, HW]
      - 返回 a: [B*H, C_head, HW]

    配置：
      - N_QUBITS: 量子比特数；当使用严格幅度编码时，态空间维度为 2**N_QUBITS
      - Q_DEPTH: PQC 混合层深度
      - qk_dim: 注意力中 q/k 的嵌入维度（<= 2**N_QUBITS），用于 RBF 注意力
      - encoding: 'angle' 或 'amplitude'
        * angle：通过 RY 旋转进行角度编码（旧实现）
        * amplitude：严格的“幅度编码到量子态”（将输入向量经线性映射到 2**N_QUBITS 维并归一化为态幅，调用 qdev.set_states 直接设定初态）
      - eps: 数值稳定性 epsilon
      - tau: RBF 温度；可训练或固定
      - tau_trainable: 是否将温度设为可训练参数（softplus 保正）
      - attn_dropout: 注意力输出的 dropout
      - qk_norm: 'none' 或 'layernorm'，用于 q/k 的投影归一化
      - prefer_x_interface: 是否偏好 x-only 接口
      - force_fp32_attention: 在混合精度下强制注意力计算使用 FP32
      - device_name: TorchQuantum 设备名（'cuda' 或 'cpu'）
    """

    def __init__(self, N_QUBITS: int = 8, Q_DEPTH: int = 2, qk_dim: int = 4,
                 encoding: str = 'angle', eps: float = 1e-6,
                 tau: float = None, tau_trainable: bool = True,
                 attn_dropout: float = 0.0, qk_norm: str = 'layernorm',
                 prefer_x_interface: bool = True,
                 force_fp32_attention: bool = True,
                 device_name: str = None):
        super().__init__()
        assert encoding in ('amplitude', 'angle'), "encoding must be 'amplitude' or 'angle'"
        assert qk_norm in ('none', 'layernorm')
        self.N_QUBITS = int(N_QUBITS)
        self.Q_DEPTH = int(Q_DEPTH)
        self.qk_dim = int(qk_dim)
        self.encoding = encoding
        self.eps = float(eps)
        self.prefer_x_interface = bool(prefer_x_interface)
        self.force_fp32_attention = bool(force_fp32_attention)
        # 设备名：默认与 x 张量的 device 保持一致；若提供则优先使用
        self.device_name = device_name

        # 强制仅使用 TorchQuantum，实现量子注意力；不再提供经典回退
        if not _TQ_AVAILABLE:
            raise ImportError(
                "TorchQuantum 未安装或不可用：本适配器仅支持 TorchQuantum 实现的量子注意力。请先安装 'torchquantum' 并确保可导入。"
            )

        # --- Debug controls (for testing) ---
        # Enable debug prints via env: QTRANSFORMER_DEBUG=1 (or true/yes/on)
        dbg_env = os.getenv('QTRANSFORMER_DEBUG', '').strip().lower()
        self._dbg_enabled = dbg_env in ('1', 'true', 'yes', 'on')
        # Limit number of prints per path to avoid flooding logs.
        try:
            self._dbg_max_calls = int(os.getenv('QTRANSFORMER_DEBUG_STEPS', '3'))
        except Exception:
            self._dbg_max_calls = 3
        # Print only on rank 0 by default (torchrun sets RANK env).
        dbg_rank0_only_env = os.getenv('QTRANSFORMER_DEBUG_RANK0_ONLY', '1').strip().lower()
        self._dbg_rank0_only = dbg_rank0_only_env in ('1', 'true', 'yes', 'on')
        try:
            self._rank = int(os.getenv('RANK', '0'))
        except Exception:
            self._rank = 0
        self._dbg_calls_x = 0
        self._dbg_calls_qkv = 0

        # RBF 温度配置（与 TQ_QSANN 对齐）
        if tau is None:
            tau = 1.0 if encoding == 'angle' else 0.5
        self.tau_trainable = bool(tau_trainable)
        init_tau = float(tau)
        if self.tau_trainable:
            self.raw_tau = nn.Parameter(torch.tensor(math.log(math.exp(init_tau) - 1.0), dtype=torch.float32))
        else:
            self.register_buffer('tau_value', torch.tensor(init_tau, dtype=torch.float32))

        # PQC 权重（与 tq_qsann_min_train 中一致：RX+RY -> CNOT 链 -> RY）
        self.enc_w = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.q_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))
        self.k_w   = nn.Parameter(0.1 * torch.randn(self.Q_DEPTH, self.N_QUBITS, 3))

        # Z 期望测量
        self.measure_z = None
        if _TQ_AVAILABLE:
            self.measure_z = tq.MeasureAll(tq.PauliZ)

        # q/k 投影与归一化（从 N_QUBITS -> qk_dim）
        self.q_proj = nn.Linear(self.N_QUBITS, self.qk_dim)
        self.k_proj = nn.Linear(self.N_QUBITS, self.qk_dim)
        self.qk_ln  = nn.LayerNorm(self.qk_dim) if qk_norm == 'layernorm' else nn.Identity()

        # 残差门控与注意力 dropout
        attn_gate_init = 0.5 if encoding == 'angle' else 0.2
        self.attn_gate = nn.Parameter(torch.tensor(attn_gate_init))
        self.attn_drop = nn.Dropout(p=float(attn_dropout))

        # 针对不同 head 通道数的预投影（C_head -> N_QUBITS），延迟注册
        self.preproj_by_chead = nn.ModuleDict()
        # 严格幅度编码的预投影（C_head -> 2**N_QUBITS），延迟注册
        self.amp_preproj_by_chead = nn.ModuleDict()

    # --- debug helpers ---
    def _dbg_print(self, msg: str):
        if not self._dbg_enabled:
            return
        if self._dbg_rank0_only and self._rank != 0:
            return
        try:
            print(f"[QSANNAdapter][rank={self._rank}] {msg}")
        except Exception:
            # Avoid any failure due to printing
            pass

    # --- public callable ---
    def forward(self, *args, **kwargs):
        # 首选 x-only 接口以替换 UNetBlock 的经典注意力路径
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            x_norm = args[0]
            num_heads = kwargs.get('num_heads', None)
            if num_heads is None:
                raise TypeError('QSANNAdapter(x_norm) requires num_heads=H in kwargs')
            return self._forward_x(x_norm, num_heads)
        # 禁止 qkv 接口：仅支持 x-only 模式
        if len(args) == 3 and all(isinstance(t, torch.Tensor) for t in args):
            raise TypeError('QSANNAdapter 仅支持 x-only 接口 (x_norm, num_heads=H)，不支持 (q, k, v) 接口')
        raise TypeError('QSANNAdapter expects either (x_norm, num_heads=H) or (q, k, v)')

    # --- TorchQuantum QSANN: x-only path ---
    def _forward_x(self, x_norm: torch.Tensor, num_heads: int) -> torch.Tensor:
        B, C, H, W = x_norm.shape
        assert C % num_heads == 0, f"Channels {C} must be divisible by num_heads={num_heads}"
        C_head = C // num_heads

        # 以 head 重排为序列：xh [B*H, C_head, HW]
        xh = x_norm.reshape(B * num_heads, C_head, H * W)
        B_H, C_h, S = xh.shape

        # Debug: entry info
        if self._dbg_enabled and self._dbg_calls_x < self._dbg_max_calls:
            self._dbg_print(
                f"x-only path: B={B} C={C} H={H} W={W} | heads={num_heads} C_head={C_head} | x.dtype={x_norm.dtype} x.device={x_norm.device}"
            )
            self._dbg_print(
                f"config: N_QUBITS={self.N_QUBITS} Q_DEPTH={self.Q_DEPTH} qk_dim={self.qk_dim} encoding={self.encoding} force_fp32_attention={self.force_fp32_attention} TQ_available={_TQ_AVAILABLE}"
            )

        # 确保 TorchQuantum 可用（在 __init__ 中已强制检查；此处仅作防御性断言）
        if not _TQ_AVAILABLE:
            raise RuntimeError('TorchQuantum 不可用：无法执行量子注意力，请安装并启用 torchquantum')

        # 针对 head 的线性预投影（C_head -> N_QUBITS），按需注册，并确保设备一致
        key = f"{C_head}->${self.N_QUBITS}"
        if key not in self.preproj_by_chead:
            self.preproj_by_chead[key] = nn.Linear(C_head, self.N_QUBITS)
        preproj = self.preproj_by_chead[key]
        dev = xh.device
        # 在混合精度训练下，确保预投影在与输入一致的设备，并在需要时使用 FP32 以避免 dtype 冲突
        preproj_dtype = torch.float32 if (self.force_fp32_attention and torch.cuda.is_available()) else xh.dtype
        if preproj.weight.device != dev or preproj.weight.dtype != preproj_dtype:
            preproj = preproj.to(device=dev, dtype=preproj_dtype)
            self.preproj_by_chead[key] = preproj

        # 量子设备 batch 尺寸：B*H*S（每个 token 一个量子态）
        bsz = B_H * S
        dev = xh.device
        # 同步 q/k 投影与归一化模块设备到输入设备，避免 CPU/CUDA 混用
        try:
            self.q_proj.to(dev)
            self.k_proj.to(dev)
            self.qk_ln.to(dev)
        except Exception:
            pass
        device_name = self.device_name or ('cuda' if dev.type == 'cuda' else 'cpu')

        # 准备编码输入（angle 或 amplitude）
        x_tokens = xh.permute(0, 2, 1).reshape(bsz, C_head)   # [B*H*S, C_head]
        if self.encoding == 'amplitude':
            # 严格幅度编码：将输入向量直接映射到态幅并归一化，维度为 2**N_QUBITS
            amp_dim = 1 << self.N_QUBITS
            key_amp = f"amp:{C_head}->{amp_dim}"
            if key_amp not in self.amp_preproj_by_chead:
                self.amp_preproj_by_chead[key_amp] = nn.Linear(C_head, amp_dim)
            amp_preproj = self.amp_preproj_by_chead[key_amp]
            # 设备与 dtype 同步；在混合精度下强制使用 FP32 计算幅度，再转复数态
            amp_dtype = torch.float32 if (self.force_fp32_attention and torch.cuda.is_available()) else xh.dtype
            if amp_preproj.weight.device != dev or amp_preproj.weight.dtype != amp_dtype:
                amp_preproj = amp_preproj.to(device=dev, dtype=amp_dtype)
                self.amp_preproj_by_chead[key_amp] = amp_preproj
            xt = x_tokens.to(amp_preproj.weight.dtype)
            amp = amp_preproj(xt)                                 # [bsz, 2**N_QUBITS]
            # 直接幅度归一化（L2），避免数值不稳定
            norm = amp.norm(p=2, dim=1, keepdim=True) + self.eps
            amp = amp / norm
            # 转为复数态向量（实部为幅度，虚部为 0）
            amp_complex = torch.complex(amp, torch.zeros_like(amp))
            # 在 FP32/复数中执行量子编码与测量，避免混合精度冲突
            if self.force_fp32_attention and torch.cuda.is_available():
                ctx = _autocast(device_type='cuda', enabled=False) if _AUTOCAST_SUPPORTS_DEVICE_TYPE else _autocast(enabled=False)
                with ctx:
                    attn_out = self._qsann_attention_amplitude(amp_complex, bsz, B_H, S, dev, device_name, xh.float())
            else:
                attn_out = self._qsann_attention_amplitude(amp_complex, bsz, B_H, S, dev, device_name, xh)
        else:
            # 角度编码：归一化 -> tanh -> 映射到 [-pi, pi]
            x_tokens = x_tokens.to(preproj.weight.dtype)
            x_qubits = preproj(x_tokens)                          # [bsz, N_QUBITS]
            x_qubits = x_qubits - x_qubits.mean(dim=1, keepdim=True)
            std = x_qubits.std(dim=1, keepdim=True) + self.eps
            x_qubits = x_qubits / std
            theta = math.pi * torch.tanh(x_qubits)
            # 在 FP32 中进行量子编码与测量，避免混合精度冲突
            if self.force_fp32_attention and torch.cuda.is_available():
                ctx = _autocast(device_type='cuda', enabled=False) if _AUTOCAST_SUPPORTS_DEVICE_TYPE else _autocast(enabled=False)
                with ctx:
                    attn_out = self._qsann_attention(theta.float(), bsz, B_H, S, dev, device_name, xh.float())
            else:
                attn_out = self._qsann_attention(theta, bsz, B_H, S, dev, device_name, xh)

        # 门控与 dropout，仅返回注意力分支输出（不做残差相加；UNetBlock 将执行 proj(attn_out) + x）
        attn_out = attn_out.to(xh.dtype)
        attn_out = self.attn_drop(attn_out)
        attn_out = self.attn_gate * attn_out

        # 回写到 [B, C, H, W]
        out = attn_out.reshape(B, num_heads * C_head, H, W)
        if self._dbg_enabled and self._dbg_calls_x < self._dbg_max_calls:
            self._dbg_print(
                f"x-only path end: out.shape={out.shape} out.dtype={out.dtype} attn_gate={float(self.attn_gate.data)} device={out.device}"
            )
            self._dbg_calls_x += 1
        return out

    def _qsann_attention_amplitude(self, amp_complex: torch.Tensor, bsz: int, B_H: int, S: int, dev: torch.device, device_name: str, xh_value: torch.Tensor) -> torch.Tensor:
        """严格幅度编码路径：
        - 初态由 amp_complex 直接设定（set_states），维度为 [bsz, 2**N_QUBITS]。
        - 对 q/k 分支分别应用 enc_w 与 q_w/k_w 的 PQC，然后测量 Z 并线性投影到 qk_dim。
        - 注意：xh_value 用作 v（保留原通道特征）。
        """
        # q 分支
        qdev_q = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        # 设定初态（严格幅度编码）
        qdev_q.set_states(amp_complex.to(torch.complex64))
        self._apply_pqc(qdev_q, self.enc_w)
        self._apply_pqc(qdev_q, self.q_w)
        z_q = self._measure_z(qdev_q)           # [bsz, N_QUBITS]
        q_vec = self.q_proj(z_q)
        q_vec = self.qk_ln(q_vec)
        q = q_vec.reshape(B_H, S, self.qk_dim)  # [B*H, S, qk_dim]

        # k 分支
        qdev_k = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        qdev_k.set_states(amp_complex.to(torch.complex64))
        self._apply_pqc(qdev_k, self.enc_w)
        self._apply_pqc(qdev_k, self.k_w)
        z_k = self._measure_z(qdev_k)
        k_vec = self.k_proj(z_k)
        k_vec = self.qk_ln(k_vec)
        k = k_vec.reshape(B_H, S, self.qk_dim)

        # v 使用原通道（保持维度一致）：[B*H, C_head, S] -> [B*H, S, C_head]
        v = xh_value.transpose(1, 2)

        # RBF 注意力与归一化
        if self.tau_trainable:
            tau_eff = F.softplus(self.raw_tau) + 1e-9
        else:
            tau_eff = self.tau_value + 1e-9
        dist_sq = torch.cdist(q, k, p=2) ** 2   # [B*H, S, S]
        alpha = torch.exp(-dist_sq / tau_eff)
        alpha_sum = alpha.sum(dim=-1, keepdim=True) + 1e-9
        alpha_norm = alpha / alpha_sum

        # 注意力聚合
        attn_out = torch.einsum('bsj,bjd->bsd', alpha_norm, v)
        attn_out = attn_out.transpose(1, 2)
        return attn_out

    def _qsann_attention(self, theta: torch.Tensor, bsz: int, B_H: int, S: int, dev: torch.device, device_name: str, xh_value: torch.Tensor) -> torch.Tensor:
        """执行 TorchQuantum QSANN 注意力：构造 q/k（通过测量 Z 并线性投影），使用 RBF 核得到权重，对 v 使用原 xh_value。"""
        # 构造量子设备
        qdev_q = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        # 角度编码：对每个量子比特施加 RY(theta)
        for i in range(self.N_QUBITS):
            tqf.ry(qdev_q, wires=i, params=theta[:, i])
        # enc + q_w 的 PQC
        self._apply_pqc(qdev_q, self.enc_w)
        self._apply_pqc(qdev_q, self.q_w)
        z_q = self._measure_z(qdev_q)           # [bsz, N_QUBITS]
        q_vec = self.q_proj(z_q)
        q_vec = self.qk_ln(q_vec)
        q = q_vec.reshape(B_H, S, self.qk_dim)  # [B*H, S, qk_dim]

        # k 分支：重新构造设备并应用 enc + k_w
        qdev_k = tq.QuantumDevice(n_wires=self.N_QUBITS, bsz=bsz, device=device_name)
        for i in range(self.N_QUBITS):
            tqf.ry(qdev_k, wires=i, params=theta[:, i])
        self._apply_pqc(qdev_k, self.enc_w)
        self._apply_pqc(qdev_k, self.k_w)
        z_k = self._measure_z(qdev_k)
        k_vec = self.k_proj(z_k)
        k_vec = self.qk_ln(k_vec)
        k = k_vec.reshape(B_H, S, self.qk_dim)  # [B*H, S, qk_dim]

        # v 使用原通道（保持维度一致）：[B*H, C_head, S] -> [B*H, S, C_head]
        v = xh_value.transpose(1, 2)            # [B*H, S, C_head]

        # RBF 注意力：alpha_{s,j} = exp(-||q_s - k_j||^2 / tau)
        if self.tau_trainable:
            tau_eff = F.softplus(self.raw_tau) + 1e-9
        else:
            tau_eff = self.tau_value + 1e-9
        dist_sq = torch.cdist(q, k, p=2) ** 2   # [B*H, S, S]
        alpha = torch.exp(-dist_sq / tau_eff)
        alpha_sum = alpha.sum(dim=-1, keepdim=True) + 1e-9
        alpha_norm = alpha / alpha_sum

        # 注意力聚合：attn_out [B*H, S, C_head]
        attn_out = torch.einsum('bsj,bjd->bsd', alpha_norm, v)
        # 回到 [B*H, C_head, S]
        attn_out = attn_out.transpose(1, 2)
        return attn_out

    def _apply_pqc(self, qdev: 'tq.QuantumDevice', weights: torch.Tensor):
        """RX+RY -> CNOT 链 -> RY（与 tq_qsann_min_train 一致）。weights: [depth, N_QUBITS, 3]."""
        depth = weights.shape[0]
        for l in range(depth):
            rx_params = weights[l, :, 0]
            ry_params = weights[l, :, 1]
            ent_ry_params = weights[l, :, 2]
            # 局部旋转
            for i in range(self.N_QUBITS):
                tqf.rx(qdev, wires=i, params=rx_params[i])
                tqf.ry(qdev, wires=i, params=ry_params[i])
            # 线性 CNOT 链
            for i in range(self.N_QUBITS - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            # 纠缠后旋转
            for i in range(self.N_QUBITS):
                tqf.ry(qdev, wires=i, params=ent_ry_params[i])

    def _measure_z(self, qdev: 'tq.QuantumDevice') -> torch.Tensor:
        if self.measure_z is None:
            raise RuntimeError('TorchQuantum not available: cannot perform Z measurement')
        z = self.measure_z(qdev)  # [bsz, N_QUBITS]
        return z

    # 保留设备/类型一致性检查（用于防御性编程场景）
    def _check_same_device_dtype(self, *tensors):
        dtypes = {t.dtype for t in tensors}
        devices = {t.device for t in tensors}
        if len(dtypes) != 1:
            raise TypeError(f'All tensors must have the same dtype, got {dtypes}')
        if len(devices) != 1:
            raise TypeError(f'All tensors must be on the same device, got {devices}')