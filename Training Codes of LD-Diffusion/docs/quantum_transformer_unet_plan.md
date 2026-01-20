# Quantum Transformer 替换 UNet 方案（EDM + 潜在空间）

重要迁移说明（arch 默认与启用方式）
- 训练脚本 train.py 的网络架构参数 arch 现在默认值为 ncsnpp（原生 UNet）。
- 仅当显式传入 --arch=quantum_transformer 时，才会启用 Quantum Transformer 路径（QSANN 注意力）。
- 其他架构（ddpmpp、adm、ncsnpp）下会强制关闭量子注意力相关开关，避免误用。
- 若需从旧版行为迁移：此前可能通过 --use-quantum-transformer 直接在 UNet 内开启量子注意力；新版本中请改为设置 --arch=quantum_transformer，并可选传入量子参数（n_qubits、q_depth、encoding、attn_dropout 等）。
- 旧版 CLI（如 --use-quantum-transformer、--quantum-adapter 等）已弃用但兼容：仅在 --arch=quantum_transformer 时生效；在其他 arch 下会被忽略并给出提示，建议完成迁移。

QSANN 输入维度与编码规范（当前默认采用幅度编码 amplitude）
- Patch 设定：patch_size=4。对于 EDM latent z_t ∈ [B, C_lat=4, 16, 16]，用 nn.Unfold(kernel=4, stride=4) 在通道维上按 4×4×4 展平得到每个 token 的 64 维向量（L=16 个 token）。
- 幅度编码（encoding=amplitude）：将每个 token 的 64 维向量做 L2 归一化（加 eps）后直接作为 2^N_QUBITS 的量子态幅度，N_QUBITS=6（2^6=64）精确匹配输入维度，无需额外线性扩/降维。
- Q/K/V 三分支均为量子线路，且三者的输入向量为同一个 64 维向量（严格与 /home/zzn/qfl_tq/ffhq_workspace/Qencoding_classify_test/tq_qsann_min_train.py 一致）：
  - 公共前置 PQC 参数 enc_w 与分支专属 PQC 参数 q_w/k_w/v_w，结构为按层的 RX+RY → 线性 CNOT 链 → RY。
  - Q/K：对每个量子比特进行 Z 测量得到 (B,S,6) 后，线性投影到 qk_dim（默认 4），再做 LayerNorm（可选）。
  - V：从量子态获取概率分布 (B,S,64)，对 64 维作 LayerNorm 以缓解尺度问题。
- 注意力权重为 RBF：alpha_{s,j} = exp(-||q_s - k_j||^2 / tau)，其中 tau 为可训练温度（softplus 保证正）；归一化后与 V 聚合得到 attn_out，并通过门控 attn_gate 与 dropout 做残差加回。
- Heads 说明：量子注意力内部为单头 RBF 注意力（不分头）；train.py 中的 heads 仅用于经典路径（如 MLP/投影的形状约束），量子注意力不使用 heads 进行切分。

默认超参数选择（Quantum Transformer / DiT 风格，更新版）
- depth=4，heads=8（主干保持多层与多头的经典配置，量子注意力内部单头）。
- model_dim=384（经典 token 嵌入维度，以及量子注意力输出 64→384 的线性映射维度）。
- mlp_ratio=2.0（FFN 隐藏层宽度为 384×2=768）。
- dropout=0.1，attn_dropout=0.1（注意：量子注意力的输出残差前应用 0.1 的 dropout）。
- patch_size=4（将 16×16 潜在特征映射为 L=16 个 token，每个 token 原始维度为 64）。
- N_QUBITS=6（2^6=64，与每个 token 的原始维度严格匹配）。
- 建议在量子注意力计算中使用 FP32（force_fp32_attention=True），以获得更稳定的数值与梯度。

本文件固化“使用 Quantum Transformer 替换 UNet”的详细方案，确保与 EDM 的“预测干净目标（clean target x0）”语义保持一致，并优先在 VAE 潜在空间（16×16×C_lat）上训练与采样，以控制注意力复杂度与显存占用。

决议与默认参数（确认）
- 选择 ldm 的 AutoencoderKL，统一像素 128×128 → 潜在 16×16×4。
- 默认超参：c0=64、num_heads=8、n_qubits=6、q_depth=2、qk_dim=4、layers=4、patch_size=4、attn_dropout=0.1、pos-embed=sincos（可选）、force_fp32_attention=True。
- 量子自注意力实现与 /home/zzn/qfl_tq/ffhq_workspace/Qencoding_classify_test/tq_qsann_min_train.py 严格对齐（Q/K/V 全量子、输入向量为同一 64 维、RBF 注意力、可训练温度与残差门控）。FFN 等模块沿用 DiT/Transformer 规范。
- EDMLoss 遵循原有代码逻辑，采用“预测干净目标 x0”的语义；若与噪声预测存在冲突，以现有 EDMLoss 逻辑为准。

一、整体架构与数据流

- 输入图像 x ∈ [B, C_img, 128, 128]（RGB 或灰度）。
- VAE 编码：z = VAE.encode(x)，得到 z ∈ [B, C_lat, 16, 16]。
  - 使用 latent-diffusion/ldm 的 AutoencoderKL，冻结权重。
  - AutoencoderKL 通常 latent 通道为 C_lat=4，分辨率缩放 8×（128→16）。
  - 注意：AutoencoderKL 需要图像归一化到 [-1,1]（Stable Diffusion 约定），训练前处理要一致。
- 加噪与条件：
  - 采样 σ（EDM），生成噪声 ε ∼ N(0,I)。
  - z_t = z + σ · ε。
- QSANN Transformer 去噪器：
  - 预测 x̂0_z = Denoiser(z_t, σ)，输出形状 [B, C_lat, 16, 16]（预测干净目标 x0）。
- 损失与优化：
  - 在 latent 空间使用 EDMLoss（不传 patch_size），以 x̂0_z 对齐 z（clean target）。
- 采样：
  - 在 latent 空间跑 EDM 采样得到 z_sample。
  - 图像恢复：x_rec = VAE.decode(z_sample)。

二、QuantumTransformerDenoiser 规范（替代 U-Net 主干）

- 输入输出契约：
  - 输入：z_t ∈ [B, C_lat, 16, 16]，σ ∈ [B] 或 [B,1]，可选类标签 `class_labels`。
  - 输出：x̂0_z ∈ [B, C_lat, 16, 16]（预测干净目标）。
- 结构设计（DiT 风格 Transformer block 堆叠）：
  - PatchEmbed2D：
    。
    - 量子分支：nn.Unfold(kernel=4, stride=4) 得到 tokens_64 ∈ [B, L=16, 64]，对每个 token 做 L2 归一化以适配幅度编码。
  - σ 条件嵌入：
    - 使用 MLP：σ → embed ∈ R^{model_dim}，在每个 block 入口加到 tokens_384（或做 FiLM scale/shift）。
    - 与 EDM 保持一致，统一用 σ 而非 t。
  - 位置嵌入（可选）：在 16×16 网格上构建 2D sin-cos 位置编码，线性映射到 model_dim 后叠加到 tokens_384；PEDM 模式下可使用 `x_pos` 作为位置偏置。
  - TransformerBlockQuantum × L（建议 L=4~6，以显存/速度调节）：
    - PreNorm：LayerNorm 作用于 tokens_384。
    - 量子注意力（单头 RBF）：
      - 输入 tokens_64（同一个 64 维向量同时作为 Q/K/V 的输入）。
      - Q/K：Z 测量 → 线性映射到 qk_dim=4 → LayerNorm（可选）。
      - V：量子态概率 → 64 维 → LayerNorm。
      - 注意力：alpha=exp(-||q−k||^2/τ)，softplus(tau) 保证正，归一化后与 V 聚合；attn_dropout=0.1；残差门控：tokens_384 ← tokens_384 + proj(64→384) 后的 gate*attn_out。
    - FFN：MLP（GELU、Dropout）与残差，保持 DiT Block 的接口与张量形状。
  - Unpatchify2D：将 tokens_384 逆映射为 [B, C_lat, 16, 16] 的图像空间输出。
  - 复杂度控制：
    - 在 16×16 latent 上运行（L=16，注意力复杂度 O(L^2) 可控）。
    - 初始 model_dim=384、L=4、attn_dropout=0.1，如显存紧张可降低 model_dim 或减少层数。

三、与 AutoencoderKL 的集成细节

- 模型加载：
  - 从 latent-diffusion/ldm 导入 AutoencoderKL（优先使用项目中已有的 PyTorch 版本）。
  - 冻结权重（vae.requires_grad=False）。
- 输入归一化：
  - AutoencoderKL 期望输入在 [-1,1]，确保数据预处理与训练/采样一致。
- 潜空间形状：
  - 对 128×128 输入，默认得到 [B, 4, 16, 16]。
- 推理与采样：
  - 训练：仅用 encode → latent；采样：用 decode → 图像。

四、训练管线的对接点（不改动太多，保持 EDM 框架）

- 构建模型：
  - 新增 arch=quantum_transformer。
  - model = LatentEDMWrapper(vae=AutoencoderKL_frozen, denoiser=QuantumTransformerDenoiser)。
- 训练步骤：
  - 从数据集获取 x → 归一化为 [-1,1]。
  - z = vae.encode(x)。
  - 采样 σ，生成 z_t = z + σ · ε。
  - x̂0_z = denoiser(z_t, σ)。
  - loss = EDMLoss(x̂0_z, z, σ, …)（保持现有 EDM 配置，不传 patch_size；遵循原代码逻辑）。
- 采样步骤：
  - 在 latent 空间执行 EDM 采样过程（扩散调度与步长与像素空间相似，参数可沿用）。
  - 最终 z_sample 用 vae.decode 生成图像。

五、配置与 CLI 选项（更新）

- arch: quantum_transformer。
- 主要参数：
  - --model-dim 384
  - --heads 8（量子注意力内部单头；heads 仅用于经典形状对齐）
  - --patch-size 4
  - --quantum-encoding amplitude
  - --quantum-n-qubits 6
  - --q-depth 2
  - --qk-dim 4
  - --quantum-attn-dropout 0.1
  - --force-fp32-attn true
  - --pos-embed sincos（可选）
  - --layers 4（或更多）

六、验证与风险控制

- Dummy 前向：
  - 随机 z_t ∈ [B, C_lat, 16, 16] 与 σ，跑 denoiser 前向，检查形状与 dtype；确认显存使用与速度。
- 1 tick dry-run：
  - 真实数据 → encode → z_t → denoiser → loss，记录 sec/tick、gpumem、loss 初值与稳定性。
- 性能与稳定性：
  - QSANN 注意力是 O(S^2)（S=256），初期只开 16×16 层，num_heads=4，C0=64。
  - 如显存过高：降低 C0 或减少层数 L。
  - 混合精度冲突：force_fp32_attention=True 保守稳妥，后续再优化。

七、交付步骤与时间线（建议）

- 第 1 步：接口草案与代码骨架。
  - 新增 QuantumTransformer/TransformerDenoiser.py（定义 Denoiser 与 Block）。
  - 在 networks.py 注册 arch=quantum_transformer（构造 denoiser 与载入 AutoencoderKL）。
  - 在 train.py 对接 latent 流程（x→encode→z_t→denoiser→EDM Loss）。
- 第 2 步：Dummy 前向与 dry-run。
  - Dummy 检查形状一致性、类型与设备一致性。
  - 1 tick dry-run，确认训练能跑通、日志正常（特别是 Dataset resolution=128、latent 16×16）。
- 第 3 步：采样集成。
  - 完成 latent 空间 EDM 采样 → decode → 图片输出。
- 第 4 步：性能优化。
  - 视显存与速度调整 C0、num_heads、层数 L；必要时引入 attn_dropout。

八、兼容性与现状说明

- 本方案明确选择 ldm 的 AutoencoderKL，统一到潜在 16×16×4。
- 需要同步更新训练循环与 generate.py 的潜在维度设定与解码参数，以匹配 ldm 的编码/解码规范与 [-1,1] 归一化约定。

实现参考与路径
- 量子注意力实现（严格对齐）：/home/zzn/qfl_tq/ffhq_workspace/Qencoding_classify_test/tq_qsann_min_train.py（TQ_QSANN：Q/K/V 全量子，同一 64 维输入，RBF 注意力、可训练 τ 与门控）。
- DiT/Block 与 FFN 参考：沿用当前工程中的 Transformer 规范与实现习惯。