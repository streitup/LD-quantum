import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicMHSAAdapter(nn.Module):
    """
    经典多头自注意力适配器（x-only 接口）。

    主接口：adapter(x_norm, num_heads=H) -> attn_out
      - x_norm: [B, C, H, W]
      - attn_out: [B, C, H, W]（仅注意力输出，不含残差；残差与 1x1 proj 在 UNetBlock 中完成）

    设计要点：
    - 与 QSANNAdapter 相同的 x-only 调用方式，便于在 UNetBlock 中用同一入口并行选择量子或经典注意力。
    - 标准 MHSA：线性 q/k/v、缩放点积注意力 softmax、输出投影（在本适配器内部不做 1x1 空间投影，保持与量子适配器一致的返回形状）。
    - AMP 兼容：在 FP16/AMP 下正常工作，无需强制 FP32；可选提供 force_fp32_attention 以在数值不稳定时强制注意力核为 FP32。
    """

    def __init__(self,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 force_fp32_attention: bool = False,
                 **kwargs):
        super().__init__()
        # 兼容量子适配器的 kwargs，忽略无关项
        attn_dropout = kwargs.get('attn_dropout', attn_dropout)
        proj_dropout = kwargs.get('proj_dropout', proj_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.force_fp32_attention = bool(kwargs.get('force_fp32_attention', force_fp32_attention))

        # 运行时构造 qkv/proj 层：因 C 依赖输入，初始化时延后到首次调用
        self._inited = False
        self.C = None
        self.qkv = None
        self.proj = None

        # Debug 控制
        self._dbg_enabled = bool(os.environ.get('QTRANSFORMER_DEBUG') in ('1', 'true', 'True', 'yes', 'on'))
        self._dbg_calls_x = 0
        self._dbg_max_calls = 5

    def _dbg_print(self, *args, **kwargs):
        if self._dbg_enabled:
            print('[ClassicMHSAAdapter]', *args, **kwargs)

    def _maybe_init(self, C: int, device: torch.device, dtype: torch.dtype):
        if self._inited and self.C == C:
            return
        self.C = C
        # 使用 1x1 卷积实现通道内 qkv 和输出投影，以保持空间维度 HxW 不变
        self.qkv = nn.Conv2d(in_channels=C, out_channels=3 * C, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1, bias=True)
        self.qkv.to(device=device, dtype=dtype)
        self.proj.to(device=device, dtype=dtype)
        self._inited = True
        self._dbg_print(f'init qkv/proj with C={C}, device={device}, dtype={dtype}')

    def forward(self, *args, **kwargs):
        # 仅支持 x-only 接口
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            x_norm = args[0]
            num_heads = kwargs.get('num_heads', None)
            if num_heads is None:
                raise TypeError('ClassicMHSAAdapter(x_norm) requires num_heads=H in kwargs')
            return self._forward_x(x_norm, num_heads)
        if len(args) == 3 and all(isinstance(t, torch.Tensor) for t in args):
            raise TypeError('ClassicMHSAAdapter 仅支持 x-only 接口 (x_norm, num_heads=H)，不支持 (q, k, v) 接口')
        raise TypeError('ClassicMHSAAdapter expects (x_norm, num_heads=H)')

    def _forward_x(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        标准 MHSA：
        - 输入 x: [B, C, H, W]
        - 分头：C 必须能被 num_heads 整除，head_dim=C//H
        - qkv: [B, 3C, H, W] -> 重排为 [B*H, head_dim, HW]
        - 注意力：softmax((q·k^T)/sqrt(head_dim))，与 v 聚合
        - 合并 heads 并回写到 [B, C, H, W]
        - 返回注意力分支输出（不含残差）；UNetBlock 将执行 proj(attn_out) + x
        """
        assert isinstance(x, torch.Tensor)
        B, C, H, W = x.shape
        if C % num_heads != 0:
            raise RuntimeError(f'C={C} must be divisible by num_heads={num_heads}')
        head_dim = C // num_heads
        device = x.device
        dtype = x.dtype

        # 延迟初始化 qkv/proj
        self._maybe_init(C, device, dtype)

        # 生成 qkv 并重排
        qkv = self.qkv(x)  # [B, 3C, H, W]
        qkv = qkv.reshape(B, 3, num_heads, head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, H, head_dim, S]
        # 变形为 [B*H, head_dim, S]
        q = q.reshape(B * num_heads, head_dim, H * W)
        k = k.reshape(B * num_heads, head_dim, H * W)
        v = v.reshape(B * num_heads, head_dim, H * W)

        # 缩放点积注意力
        scale = 1.0 / math.sqrt(head_dim)

        # 可选在 FP32 中执行注意力核，提高数值稳定性（AMP 下通常不需要）
        if self.force_fp32_attention and (dtype in (torch.float16, torch.bfloat16)):
            q_ = q.float()
            k_ = k.float()
            v_ = v.float()
            attn_logits = torch.einsum('bci,bcj->bij', q_, k_) * scale
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.einsum('bij,bcj->bci', attn_weights, v_)
            out = out.to(dtype)
        else:
            attn_logits = torch.einsum('bci,bcj->bij', q, k) * scale
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.einsum('bij,bcj->bci', attn_weights, v)

        # 回写并合并 heads -> [B, C, H, W]
        out = out.reshape(B, num_heads, head_dim, H * W)
        out = out.reshape(B, C, H, W)

        # 可选输出投影与 dropout（保留与量子适配器一致的接口，返回注意力分支本身）
        out = self.proj_dropout(out)

        if self._dbg_enabled and self._dbg_calls_x < self._dbg_max_calls:
            self._dbg_print(f'x-only end: out.shape={out.shape} dtype={out.dtype} device={out.device}')
            self._dbg_calls_x += 1

        return out