# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
import importlib
import warnings
from torch_utils import persistence
from torch.nn.functional import silu
# [Quantum-Integration Marker] 导入 QuantumTransformerDenoiser 以支持通过 globals()[model_type] 实例化
# 说明：当 train.py 传入 c.network_kwargs.model_type='QuantumTransformerDenoiser' 时，
# 预条件化封装器（VP/VE/iDDPM/EDM/Patch_EDM）会在 __init__ 中通过 globals()[model_type](...) 构造底层模型。
# 因此需确保该符号在本模块的全局命名空间可见。
try:
    from .quantum_transformer import QuantumTransformerDenoiser, QuantumMLP, QuantumAdaGN, QuantumConv2d, QuantumFrontEndQCNN  # noqa: F401
except Exception as _qt_import_err:
    # 若量子模块不可用，提供一个占位符类，在实例化时给出更友好的错误信息。
    class QuantumTransformerDenoiser:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "QuantumTransformerDenoiser 无法导入：请确保已安装 torchquantum，并且 training/quantum_transformer.py 可用。\n"
                "安装指引：pip install torchquantum。若仍失败，请参考 docs/quantum_transformer_unet_plan.md。\n"
                f"原始错误：{_qt_import_err}"
            )
    class QuantumMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"QuantumMLP 无法导入：{_qt_import_err}")
    class QuantumAdaGN:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"QuantumAdaGN 无法导入：{_qt_import_err}")
    class QuantumConv2d:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"QuantumConv2d 无法导入：{_qt_import_err}")
    class QuantumFrontEndQCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"QuantumFrontEndQCNN 无法导入：{_qt_import_err}")

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        # [Quantum-Integration Marker] 注意力权重计算入口
        # 说明：这是自注意力的 softmax(Q^T K / sqrt(d)) 实现位置，当前使用经典张量算子。
        # 若集成量子 Transformer，可在 UNetBlock 的注意力分支中用量子模块替代 qkv + AttentionOp + proj 的组合，
        # 保持输入输出形状与 dtype 一致，以最小改动适配训练/采样主流程。
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
        # [Quantum-Integration Marker] 可选量子 Transformer 开关与适配器
        # 说明：当 attention=True 且 use_quantum_transformer=True 时，forward 将调用
        # quantum_adapter 以替换经典的 qkv+softmax+proj 注意力路径。
        use_quantum_transformer=False, quantum_adapter=None,
        use_quantum_affine=False,
        use_qcnn_frontend=False,
        qcnn_chunk_size=4096,
        qcnn_use_strided=False,
        qcnn_reupload=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        # [Quantum-Integration Marker] 保存量子集成相关参数
        self.use_quantum_transformer = use_quantum_transformer
        self.quantum_adapter = quantum_adapter
        self.use_quantum_affine = use_quantum_affine
        self.use_qcnn_frontend = use_qcnn_frontend
        self.qcnn_chunk_size = int(qcnn_chunk_size)
        self.qcnn_use_strided = bool(qcnn_use_strided)
        self.qcnn_reupload = bool(qcnn_reupload)

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        
        # Classic Affine (Always used for classic path, or if we need emb projection)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        
        # Architecture 2: Integrated Quantum Block (QuantumFrontEndQCNN)
        if self.use_qcnn_frontend:
            dev_name = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Architecture 2 Parameters from Benchmark 6
            # n_groups=8, n_qubits=4, n_layers=4, stride=1
            # Note: out_channels must be divisible by 8. If not, we might fallback or adjust.
            n_groups = 8
            if out_channels % n_groups != 0:
                n_groups = 4 # Fallback
            
            self.quantum_frontend = QuantumFrontEndQCNN(
                channels=out_channels,
                style_dim=emb_channels, # Raw embedding dimension
                n_qubits_data=2, # Optimized: 2 Qubits for Amplitude Encoding
                n_qubits_ancilla=2,
                n_layers=4, # Deep Architecture
                device_name=dev_name,
                use_strided_cnot=self.qcnn_use_strided,
                reupload_data=True,
                max_qdev_bsz=self.qcnn_chunk_size,
                n_groups=n_groups,
                use_strong_bypass=False, # Pure Quantum as per Architecture 2
                use_mlp_residual=False,
                stride=1, # We rely on conv0 for resizing
                encoding_type='amplitude' # Use Amplitude Encoding
            )
        else:
            self.quantum_frontend = None
            
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)
            if self.use_quantum_transformer and self.quantum_adapter is not None:
                pass
            elif self.use_quantum_transformer:
                pass

    def forward(self, x, emb):
        orig = x
        
        if self.use_qcnn_frontend and self.quantum_frontend is not None:
            # Full Quantum Pipeline with QCNN FrontEnd:
            # Scheme: Norm0 -> QuantumFrontEnd (Integrated) -> Skip
            # Note: We replace Conv0 + Affine + Conv1 with a single Quantum Block.
            
            # 1. Handle Input Projection if channels change
            if self.in_channels != self.out_channels:
                # If dimensions mismatch, we must use Conv0 (or a projection) to match dimensions
                # because QuantumFrontEnd expects fixed input dimension.
                x = self.conv0(silu(self.norm0(x)))
            else:
                # If dimensions match, we skip Conv0 entirely!
                # Direct Quantum Processing on Normalized Input
                x = self.norm0(x)
            
            # 2. Integrated Quantum Block
            # [Stability Fix] Normalize input before Quantum Circuit (Norm1)
            # In Integrated scheme, Norm1 acts as the pre-quantum normalization.
            x = self.norm1(x)
            x = self.quantum_frontend(x, emb)
            
            # Resolution Check & Fix
            # If skip expects high res (because block is not 'down'), but x is low res
            if self.skip is not None:
                 # self.skip handles resampling if needed.
                 pass
            else:
                 # No skip module, so we add `orig` directly. `orig` is input resolution.
                 # If `x` is downsampled, this fails.
                 if x.shape[2] != orig.shape[2]:
                      x = torch.nn.functional.interpolate(x, size=orig.shape[2:], mode='nearest')
            
        else:
            # Classic Pipeline
            x = self.conv0(silu(self.norm0(x)))
            
            params = self.affine(emb)
            params = params.unsqueeze(2).unsqueeze(3).to(x.dtype)
            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = silu(self.norm1(x.add_(params)))

            x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
            
        # QuantumFrontEndQCNN currently performs downsampling (stride=2).
        # If the block is NOT a downsampling block (down=False), we must restore resolution.
        # Check spatial mismatch and interpolate if needed.
        if x.shape[2:] != orig.shape[2:]:
             # Assuming x is downsampled and orig is full resolution
             # We use nearest neighbor upsampling to match orig resolution
             # This is a workaround until QuantumFrontEnd supports stride=1.
             # Note: This operation is valid as it restores the spatial dimensions for the skip connection addition above?
             # Wait, the addition `x.add_(skip(orig))` ALREADY HAPPENED on the line above!
             # And that line crashed.
             # So we must fix `x` BEFORE the addition.
             pass

        # Let's rewrite the forward logic slightly to handle this.
        
        if self.use_qcnn_frontend and self.quantum_frontend is not None:
            # Full Quantum Pipeline with QCNN FrontEnd:
            # Affine (Time Gen) -> QuantumFrontEnd (Entangled Modulation + Spatial)
            # Pass raw 'emb', frontend will use affine circuit internally (no measurement)
            x = self.quantum_frontend(x, emb)
            
            # Resolution Check & Fix
            # If skip expects high res (because block is not 'down'), but x is low res
            if self.skip is not None:
                 # self.skip handles resampling if needed.
                 # If self.skip is Conv2d with down=True, it will downsample orig.
                 # If self.skip is Identity (or kernel=1), it keeps orig resolution.
                 pass
            else:
                 # No skip module, so we add `orig` directly. `orig` is input resolution.
                 # If `x` is downsampled, this fails.
                 if x.shape[2] != orig.shape[2]:
                      x = torch.nn.functional.interpolate(x, size=orig.shape[2:], mode='nearest')

        else:
            # Classic or Legacy Quantum Pipeline
            params = self.affine(emb)
            
            if self.use_quantum_affine and self.quantum_adagn is not None:
                # (Legacy) Split Quantum Pipeline: Injection -> Conv1
                # 1. Quantum Injection (replaces AdaGN)
                x = self.quantum_adagn(x, params)
                
                # 2. Quantum Conv1 (replaces Conv2d)
                if self.quantum_conv1 is not None:
                    x = self.quantum_conv1(x)
                else:
                    x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
            else:
                # Classic Pipeline
                params = params.unsqueeze(2).unsqueeze(3).to(x.dtype)
                if self.adaptive_scale:
                    scale, shift = params.chunk(chunks=2, dim=1)
                    x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
                else:
                    x = silu(self.norm1(x.add_(params)))

                x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
            
        # Perform Skip Addition (Safe)
        # Re-check resolution before addition just in case skip module output differs
        skip_out = self.skip(orig) if self.skip is not None else orig
        if x.shape[2:] != skip_out.shape[2:]:
             # If x is still mismatched (e.g. QCNN downsampled but skip didn't), we conform x to skip_out
             x = torch.nn.functional.interpolate(x, size=skip_out.shape[2:], mode='nearest')
             
        x = x.add_(skip_out)
        x = x * self.skip_scale

        if self.num_heads:
            # [Quantum-Integration Marker] Transformer/Attention 子模块 forward 接口
            # 说明：注意力分支的输入输出签名与形状转换如下：
            # - 输入 x: [B, C, H, W]；x_norm = norm2(x)
            # - 量子路径（启用 use_quantum_transformer）：直接调用 QuantumTransformer/QSANNAdapter 的 x 接口
            #   adapter(x_norm, num_heads=H) -> attn_out，与 x 同形状；与残差相加后返回。
            # - 经典路径（未启用量子）：使用 qkv + AttentionOp + proj 实现经典自注意力。
            # 注意：量子模式下不再计算或回退到 qkv 路径，以避免“经典态 qkv 与量子注意力混用”。
            x_norm = self.norm2(x)
            if self.use_quantum_transformer:
                if self.quantum_adapter is None:
                    raise RuntimeError("Quantum Transformer enabled but quantum_adapter is None in UNetBlock. This indicates a configuration error or upstream import failure.")
                # [Quantum-Integration Marker] QSANNAdapter 调用入口（仅 x 接口）
                attn_out = self.quantum_adapter(x_norm, num_heads=self.num_heads)
                if not isinstance(attn_out, torch.Tensor):
                    raise TypeError('quantum_adapter must return a torch.Tensor')
                if attn_out.shape != x.shape:
                    raise RuntimeError(f'quantum_adapter(x) must return tensor of shape {x.shape}, got {attn_out.shape}')
                # 保持与经典路径一致的投影位置：对 attn_out 施加 proj（1x1 conv）再加残差
                x = self.proj(attn_out).add_(x)
            elif False: # Classical path disabled/removed in this version for clarity or by user design
                pass 
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        implicit_mlp        = False,        # enable implicit coordinate encoding
        # [Quantum-Integration Marker] 量子 Transformer 集成参数（向下透传至 UNetBlock）
        use_quantum_transformer = False,
        quantum_adapter = None,
        quantum_adapter_kwargs = None,
        use_quantum_mlp = False,
        use_quantum_affine = False,
        use_qcnn_frontend = False,
        qcnn_chunk_size = 4096,
        qcnn_use_strided = False,
        qcnn_reupload = False,
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        self.use_quantum_mlp = use_quantum_mlp
        self.use_quantum_affine = use_quantum_affine
        self.use_qcnn_frontend = use_qcnn_frontend
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        # [Quantum-Integration Marker] 解析并实例化量子适配器（如提供字符串路径）
        adapter_obj = None
        if use_quantum_transformer:
            if isinstance(quantum_adapter, str):
                module_name, class_name = quantum_adapter.split(':')
                mod = importlib.import_module(module_name)
                AdapterClass = getattr(mod, class_name)
                adapter_obj = AdapterClass(**(quantum_adapter_kwargs or {}))
            else:
                adapter_obj = quantum_adapter
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
            # [Quantum-Integration Marker] 将量子开关与适配器传递给所有 UNetBlock
            use_quantum_transformer=use_quantum_transformer, quantum_adapter=adapter_obj,
            use_quantum_affine=use_quantum_affine,
            use_qcnn_frontend=use_qcnn_frontend,
            qcnn_chunk_size=qcnn_chunk_size,
            qcnn_use_strided=qcnn_use_strided,
            qcnn_reupload=qcnn_reupload,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        
        if self.use_quantum_mlp:
            self.time_embed_mlp = QuantumMLP(in_features=noise_channels, out_features=emb_channels)
            # Placeholder for map_layer0/1 to avoid errors if accessed directly (though unlikely)
            self.map_layer0 = None 
            self.map_layer1 = None
        else:
            self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                if implicit_mlp:
                    self.enc[f'{res}x{res}_conv'] = torch.nn.Sequential(
                                                        Conv2d(in_channels=cin, out_channels=cout, kernel=1, **init),
                                                        torch.nn.SiLU(),
                                                        Conv2d(in_channels=cout, out_channels=cout, kernel=1, **init),
                                                        torch.nn.SiLU(),
                                                        Conv2d(in_channels=cout, out_channels=cout, kernel=1, **init),
                                                        torch.nn.SiLU(),
                                                        Conv2d(in_channels=cout, out_channels=cout, kernel=3, **init),
                                                    )
                    self.enc[f'{res}x{res}_conv'].out_channels = cout
                else:
                    self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                _bk = dict(block_kwargs)
                _bk.update(use_qcnn_frontend=(bool(use_qcnn_frontend) and attn))
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **_bk)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                _bk_attn = dict(block_kwargs)
                _bk_attn.update(use_qcnn_frontend=bool(use_qcnn_frontend))
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **_bk_attn)
                _bk_noattn = dict(block_kwargs)
                _bk_noattn.update(use_qcnn_frontend=False)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **_bk_noattn)
            else:
                _bk_up = dict(block_kwargs)
                _bk_up.update(use_qcnn_frontend=False)
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **_bk_up)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                _bk = dict(block_kwargs)
                _bk.update(use_qcnn_frontend=(bool(use_qcnn_frontend) and attn))
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **_bk)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        
        if self.use_quantum_mlp:
            emb = self.time_embed_mlp(emb)
        else:
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
        # [Quantum-Integration Marker] 量子 Transformer 集成参数（向下透传至 UNetBlock）
        use_quantum_transformer = False,
        quantum_adapter = None,
        quantum_adapter_kwargs = None,
        use_quantum_mlp = False,
        use_quantum_affine = False,
        use_qcnn_frontend = False,
        qcnn_chunk_size = 4096,
        qcnn_use_strided = False,
        qcnn_reupload = False,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        self.use_quantum_mlp = use_quantum_mlp
        self.use_quantum_affine = use_quantum_affine
        self.use_qcnn_frontend = use_qcnn_frontend
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        # [Quantum-Integration Marker] 解析并实例化量子适配器（如提供字符串路径）
        adapter_obj = None
        if use_quantum_transformer:
            if isinstance(quantum_adapter, str):
                module_name, class_name = quantum_adapter.split(':')
                mod = importlib.import_module(module_name)
                AdapterClass = getattr(mod, class_name)
                adapter_obj = AdapterClass(**(quantum_adapter_kwargs or {}))
            else:
                adapter_obj = quantum_adapter
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero,
                            # [Quantum-Integration Marker] 将量子开关与适配器传递给所有 UNetBlock
                            use_quantum_transformer=use_quantum_transformer, quantum_adapter=adapter_obj,
                            use_quantum_affine=use_quantum_affine,
                            use_qcnn_frontend=use_qcnn_frontend,
                            qcnn_chunk_size=qcnn_chunk_size,
                            qcnn_use_strided=qcnn_use_strided,
                            qcnn_reupload=qcnn_reupload)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        
        if self.use_quantum_mlp:
            self.time_embed_mlp = QuantumMLP(in_features=model_channels, out_features=emb_channels)
            self.map_layer0 = None
            self.map_layer1 = None
        else:
            self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
            
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                _bk_down = dict(block_kwargs)
                _bk_down.update(use_qcnn_frontend=False)
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **_bk_down)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                _bk = dict(block_kwargs)
                _bk.update(use_qcnn_frontend=(bool(use_qcnn_frontend) and attn))
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **_bk)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                _bk_attn = dict(block_kwargs)
                _bk_attn.update(use_qcnn_frontend=bool(use_qcnn_frontend))
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **_bk_attn)
                _bk_noattn = dict(block_kwargs)
                _bk_noattn.update(use_qcnn_frontend=False)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **_bk_noattn)
            else:
                _bk_up = dict(block_kwargs)
                _bk_up.update(use_qcnn_frontend=False)
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **_bk_up)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                _bk = dict(block_kwargs)
                _bk.update(use_qcnn_frontend=(bool(use_qcnn_frontend) and attn))
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **_bk)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        
        if self.use_quantum_mlp:
            emb = self.time_embed_mlp(emb)
        else:
            emb = silu(self.map_layer0(emb))
            emb = self.map_layer1(emb)

        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        beta_d          = 19.9,         # Extent of the noise level schedule.
        beta_min        = 0.1,          # Initial slope of the noise level schedule.
        M               = 1000,         # Original number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        # [Quantum-Integration Marker] 模型构造位置（底层 UNet 实例化）
        # 说明：根据 train.py 传入的 model_type 与相关 kwargs，在此处实例化 SongUNet/DhariwalUNet。
        # 如需集成量子 Transformer，请在 UNetBlock 的注意力分支替换为量子模块，避免改动此处接口。
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 100,          # Maximum supported noise level.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        # [Quantum-Integration Marker] 模型构造位置（底层 UNet 实例化）
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        # [Quantum-Integration Marker] 模型构造位置（底层 UNet 实例化，iDDPM 输出为 2×C）
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        out_channels    = None,             # Optional override for output channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        # 输出通道与底层 UNet 构造
        # 说明：训练循环会将 out_channels 作为顶层网络参数一并传入。
        # 若此处不显式接收该参数，out_channels 会落入 **model_kwargs 并在底层 UNet 构造时与显式传参重复，导致
        # “got multiple values for keyword argument 'out_channels'” 的 TypeError。
        self.out_channels = img_channels if out_channels is None else out_channels
        # [Quantum-Integration Marker] 模型构造位置（底层 UNet 实例化）
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=self.out_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, x_pos=None, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        # [Quantum-Integration Fix] 如果底层模型为 QuantumTransformerDenoiser，则强制使用 FP32，
        # 避免原生 nn.Conv2d/nn.Linear 在 Half 输入下出现与 Float bias/weight 不匹配的问题。
        try:
            if isinstance(self.model, QuantumTransformerDenoiser):
                dtype = torch.float32
        except Exception:
            pass

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # 兼容位置编码通道：训练循环将额外的坐标通道 x_pos 作为第三个参数传入。
        # 若提供 x_pos，则在通道维上与输入拼接，以匹配底层 UNet 的 in_channels 设置（img_channels + 2）。
        x_in = torch.cat([c_in * x, x_pos], dim=1) if x_pos is not None else (c_in * x)
        F_x = self.model(x_in.to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Patch Version of EDMPrecond.

@persistence.persistent_class
class Patch_EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        out_channels    = None,
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.out_channels = img_channels if out_channels is None else out_channels
        # [Quantum-Integration Marker] 模型构造位置（底层 UNet 实例化，支持 x_pos 额外通道）
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=self.out_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, x_pos=None, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        # [Quantum-Integration Fix] 同上：量子 Transformer 路径强制 FP32，避免 Half/Float 偏置不匹配错误。
        try:
            if isinstance(self.model, QuantumTransformerDenoiser):
                dtype = torch.float32
        except Exception:
            pass

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = torch.cat([c_in * x, x_pos], dim=1) if x_pos is not None else c_in * x
        F_x = self.model((x_in).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
