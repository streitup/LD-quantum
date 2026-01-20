# QSANN 量子注意力适配集成说明（quantum_adapt.md）

本文档汇总了在项目中为集成量子 Transformer/QSANNAdapter 所做的具体代码改动、接口位置与使用方法，并给出适配器的维度映射与编码方案（N_QUBITS、Q_DEPTH、qk_dim、amplitude/angle 编码，以及必要的归一化与线性映射）。

## 一、改动总览与检索标记

为便于后续检索与维护，相关代码处均加入统一前缀的非功能性注释标识：[Quantum-Integration Marker]。

1) training/networks.py
- UNetBlock：
  - 在构造函数新增可选开关与适配器参数：`use_quantum_transformer=False, quantum_adapter=None`。
  - 在注意力分支 `forward` 中，当 `use_quantum_transformer=True` 且提供 `quantum_adapter` 时，优先调用量子适配器路径；否则走经典注意力（qkv + AttentionOp + proj）。
  - 检索标记：
    - 注意力权重计算入口（AttentionOp）：`[Quantum-Integration Marker] 注意力权重计算入口`
    - Transformer/Attention 子模块 forward 接口与替换建议：`[Quantum-Integration Marker] Transformer/Attention 子模块 forward 接口`
    - QSANNAdapter 调用入口：`[Quantum-Integration Marker] QSANNAdapter 调用入口`

- SongUNet 与 DhariwalUNet：
  - 在构造器中解析 CLI 传入的 `use_quantum_transformer` 与 `quantum_adapter`（支持字符串 `module:ClassName` 动态导入），并通过 `block_kwargs` 透传至所有 UNetBlock。
  - 检索标记：`[Quantum-Integration Marker] 量子 Transformer 集成参数（向下透传至 UNetBlock）` 与 `解析并实例化量子适配器（如提供字符串路径）`。

- 预条件化封装器 VPPrecond/VEPrecond/iDDPMPrecond/EDMPrecond/Patch_EDMPrecond：
  - 保持 `self.model = globals()[model_type](..., **model_kwargs)`，确保从 train.py 传入的量子参数透传到底层 UNet（SongUNet/DhariwalUNet）。
  - 检索标记：`[Quantum-Integration Marker] 模型构造位置（底层 UNet 实例化）`。

2) Training Codes of LD-Diffusion/train.py
- 新增 CLI 选项：
  - `--use-quantum-transformer`：启用量子 Transformer 注意力。
  - `--quantum-adapter`：量子适配器的导入路径，格式 `module:ClassName`（例如 `QuantumTransformer.QSANNAdapter:QSANNAdapter`）。
- 通过 `c.network_kwargs.update(...)` 将上述量子参数透传至 `training.networks.*Precond`，进而传递到底层 UNet 与 UNetBlock。
- 检索标记：
  - 模型构造位置（选择底层 UNet 架构）：`[Quantum-Integration Marker] 模型构造位置（选择底层 UNet 架构）`
  - 模型构造位置（选择预条件化封装器）：`[Quantum-Integration Marker] 模型构造位置（选择预条件化封装器）`
  - 实例化触发点：`[Quantum-Integration Marker] 实例化触发点`

## 二、UNetBlock 注意力替换接口（x-only 模式）

经典路径（未启用量子）保持不变：
- 输入 x 形状 `[B, C, H, W]`，归一化后经 1×1 qkv 投影得到 `[B, 3C, H, W]`；
- 重排为 `[B*num_heads, C_per_head, 3, HW]` 并拆分 q/k/v；
- 调用 `AttentionOp.apply(q, k)` 得到权重 w，随后与 v 做加权求和得到 a；
- 通过 1×1 `proj` 回到 `[B, C, H, W]` 并与残差相加。

量子路径（启用 `use_quantum_transformer=True`）改为严格的 x-only：
- 直接调用 `QuantumTransformer/QSANNAdapter` 的 x 接口：`adapter(x_norm, num_heads=H) -> attn_out`；
- 返回的 `attn_out` 与 `x` 同形状 `[B, C, H, W]`，随后与残差相加；
- 量子模式下不再计算或回退到 qkv 路径，避免“经典态 qkv 与量子注意力混用”。

## 三、QSANNAdapter 的维度映射与编码方案（x-only）

适配器的配置项（可从构造器传入或使用默认值）：
- `N_QUBITS`：用于限制编码维度的上限（最大通道映射维度不超过 `2**N_QUBITS`）。
- `Q_DEPTH`：编码/混合的层数（使用蝶形/旋转等无参混合算子，模拟量子层级叠加）。
- `qk_dim`：q/k 的计算通道维度（若小于输入通道，则按通道分组做无参线性降维：组均值）；
- `encoding`：`'amplitude'` 或 `'angle'`；
  - amplitude：去中心化 + L2 归一化，模拟振幅编码；
  - angle：去中心化 + 标准化后映射到角度域，再用 `[cos, sin]` 展开，模拟角度编码；
- 归一化与线性映射：
  - 去中心化（按通道减均值）与 L2 归一化（防止尺度不一致）；
  - 线性降维采用“通道分组均值”（无参线性投影），保证形状稳定与优化器兼容。如需可学习映射，可将其替换为 `nn.Linear(C_head, qk_dim)` 并在模型构建阶段传入固定的 `channels_per_head`。

计算流程（x 接口 adapter(x_norm, num_heads))：
1. 重排：`xh = x_norm.reshape(B*H, C_head, HW)`；
2. 维度裁剪：`target_dim = min(qk_dim或C_head, 2**N_QUBITS)`，对 `xh` 做通道分组均值降维得到 `q_proj`，并令 `k_proj = q_proj`；
3. 编码与混合：对 `q_proj/k_proj` 按 `encoding` 执行 amplitude 或 angle 编码，并进行 `Q_DEPTH` 次无参混合；
4. 权重计算：`w = softmax( (q_embed^T k_embed) / sqrt(dim_embed), dim=2 )`，全程 FP32 计算、输出转回原始 dtype；
5. 加权求和：以 `xh` 为 `v` 做汇聚，得到 `a = bmm(v, w^T)`，再 reshape 回 `[B, C, H, W]`。

## 四、使用方法

命令行示例（以 NCSN++ + Patch_EDM 为例）：

```bash
torchrun --standalone --nproc_per_node=1 "Training Codes of LD-Diffusion/train.py" \
  --outdir=training-runs \
  --data=datasets/cifar10-32x32.zip \
  --arch=ncsnpp --precond=pedm \
  --use-quantum-transformer=True \
  --quantum-adapter=QuantumTransformer.QSANNAdapter:QSANNAdapter \
  --dropout=0.1 --fp16=False
```

若以对象方式注入适配器（不使用字符串路径），可在构造底层 UNet 时通过 `quantum_adapter=QSANNAdapter(...)` 传入，接口保持一致。

## 五、兼容性、回退与注意事项

- 量子模式下为严格 x-only，不提供 qkv 回退；若适配器返回值形状或 dtype 不符合预期，将直接抛出明确错误提示。
- 采用无参线性降维与混合算子，保证训练流程无需改动优化器配置。如需可学习的线性映射，请在模型构建阶段传入稳定的 `channels_per_head` 并将降维替换为 `nn.Linear`。
- 注意力层是否启用由 `SongUNet/DhariwalUNet` 的 `attn_resolutions` 控制；仅在这些分辨率处会调用适配器。

## 六、最小测试建议

- 形状一致性：构造一个 batch，在启/停量子路径下分别前向一次，确认各层输出形状与 dtype 一致；
- 数值稳定性：在 FP32 下前向/反向各一次，检查梯度能正常传播；
- 兼容性：在 DDPM++/NCSN++/ADM 三种架构、VP/VE/EDM/Patch_EDM 四种封装下执行 100 步训练，观察显存与速度；

## 七、已知限制与后续计划

- 当前的线性降维为无参版本，后续可增加可学习投影并通过适配器参数从 CLI 传入；
- 将在模型设计报告中同步“扩散模型适配”章节，并补充基于 TorchQuantum/自研电路的门级实现选项；
- 增加更细致的单元测试与基准评测（通道维度变动、头数变化、不同分辨率下的稳定性）。