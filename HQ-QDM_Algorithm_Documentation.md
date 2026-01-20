# 混合量子-经典扩散模型（HQ-QDM）算法架构文档

## 1. 架构概览 (Architecture Overview)

本模型基于经典扩散模型（如 EDM/DDPM）的 U-Net 骨干网络构建，但在关键的特征提取、时间嵌入与注意力机制环节引入了变分量子线路（Variational Quantum Circuits, VQC）。模型采用**“经典流控，量子核心”**的设计原则：经典模块负责整体数据流、上下采样与维度变换，量子模块负责在特征空间内进行高维纠缠与特征提取。

### 1.1 核心量子组件
1.  **Quantum Time Embedding (Q-MLP)**: 利用量子线路将标量噪声水平 $\sigma_t$ 映射为高维时间嵌入向量，并通过量子态纠缠直接调制特征提取过程。
2.  **Quantum Feature Extraction (Q-FrontEnd)**: 一种基于量子卷积神经网络（QCNN）的前端模块，利用辅助量子比特（Ancilla Qubits）实现时间信息与空间特征的**相干量子耦合（Coherent Quantum Coupling）**。
3.  **Quantum Attention (Q-Attention)**: 基于 64 维幅度编码（Amplitude Encoding）与 RBF 核的量子注意力机制，替代经典 Self-Attention。

---

## 2. 详细算法流程 (Detailed Algorithm Flow)

### 2.1 整体数据流 (Global Data Flow)

假设输入噪声图像为 $x_t \in \mathbb{R}^{B \times C \times H \times W}$，时间步/噪声水平为 $\sigma_t$。

1.  **时间嵌入生成**:
    *   $\sigma_t \xrightarrow{\text{Classical Proj}} \theta_{time} \in \mathbb{R}^{N_{qubits}}$
    *   $\theta_{time} \xrightarrow{\text{Q-MLP Circuit}} |\psi_{time}\rangle \xrightarrow{\text{Measure}} e_{time} \in \mathbb{R}^{D_{emb}}$
    *   *注：在 Q-FrontEnd 中，不进行测量，直接利用 $|\psi_{time}\rangle$ 对应的线路作用于辅助量子比特。*

2.  **U-Net 编码器/解码器 (Encoder/Decoder)**:
    *   图像 $x$ 经过层级处理。在每个分辨率层级（Level $l$），数据流经 `UNetBlock`。
    *   **经典路径**: 卷积/下采样/上采样/跳跃连接保持不变。
    *   **量子增强路径**:
        *   **特征提取**: $x_{in} \xrightarrow{\text{Q-FrontEnd}} x_{feat}$（替代经典 Conv+AdaGN）。
        *   **特征精炼**: $x_{feat} \xrightarrow{\text{Q-Attention}} x_{out}$（替代经典 Self-Attention）。

---

## 3. 模块级算法与线路设计 (Module Specifications)

### 3.1 量子时间嵌入 (Quantum MLP)

该模块将低维时间信号映射为高维控制信号，并支持作为子线路嵌入其他量子模块。

*   **输入**: 噪声水平 $\sigma_t$ 或时间步 $t$。
*   **线路结构**:
    1.  **编码 (Encoding)**: $N$ 个量子比特。输入经全连接层映射为角度 $\phi \in [-\pi, \pi]^N$，施加 $RY(\phi)$ 门。
    2.  **变分层 (Variational Layers)**: 重复 $L$ 层。
        *   **纠缠**: 环形 CNOT 连接（$q_i \to q_{(i+1)\%N}$）。
        *   **旋转**: 单比特 $RX(\theta_1), RY(\theta_2), RZ(\theta_3)$，参数可训练。
    3.  **数据重上传 (Re-uploading)**: 在每层重新施加缩放后的输入编码 $RY(w \cdot \phi)$，增强非线性能力。
*   **输出**: 对所有量子比特进行 Pauli-Z 测量，得到期望值向量，经线性投影得到 $e_{time}$。

### 3.2 量子前端特征提取器 (Quantum FrontEnd QCNN)

该模块实现了**时间-空间特征的量子纠缠**，是本架构的核心创新点。

*   **输入**: 图像特征 $x \in \mathbb{R}^{B \times C \times H \times W}$，时间嵌入源（Q-MLP）。
*   **预处理 (Unfold & Encoding)**:
    *   利用 `Unfold` 提取 $p \times p$ 的图像块（Patch）。
    *   每个 Patch 被映射为 $N_{data}$ 个角度参数 $\theta_{data}$。
*   **量子线路架构**:
    *   **量子比特分配**: $N_{total} = N_{data} + N_{ancilla}$。
        *   $Q_{data}$: 用于承载图像 Patch 信息。
        *   $Q_{ancilla}$: 用于承载时间嵌入信息。
    *   **步骤 1: 数据编码**: 对 $Q_{data}$ 施加 $RY(\theta_{data})$。
    *   **步骤 2: 相干时间注入与态缩减 (Coherent Time Injection & State Reduction)**:
        *   **并行制备 (Parallel Preparation)**: 
            *   **基态重置**: 每一个 Patch 对应的辅助量子比特组 $Q_{ancilla}$ 在计算开始前均**独立初始化为基态 $|0\rangle$**。
            *   **线路复制**: Q-MLP 的线路结构（即制备“配方”）被复制应用到每一组辅助比特上。这意味着我们不是在“分发”一个已制备好的量子态，而是在成百上千个局部位置（针对每个 Patch）**同时执行相同的制备程序**。
        *   **参数广播 (Parameter Broadcasting)**: 经典的时间嵌入参数 $\theta_{time}$ 被广播至所有 Patch 的量子线路中，确保每个 Patch 都能获得相同的时间上下文信息。这避免了量子态不可克隆原理的限制。
        *   **全量演化**: Q-MLP 线路在全维空间（$N_{qmlp}$ Qubits）上执行，直接作用于辅助量子比特及其扩展位。
        *   **测量诱导适配**: 若 Q-MLP 的量子比特数 $N_{qmlp}$ 大于 Q-FrontEnd 的辅助接口数 $N_{ancilla}$，系统将对多余的 $N_{qmlp} - N_{ancilla}$ 个量子比特执行**测量 (Measurement)**。
        *   **物理意义**: 这种**“测量诱导的态缩减” (Measurement-Induced State Reduction)** 机制利用了量子纠缠特性——对多余比特的测量会使保留在 $Q_{ancilla}$ 上的量子态发生条件坍缩（Conditional Collapse）。这不仅解决了维度适配问题，更实现了一种非线性的、基于量子概率的特征投影。
    *   **步骤 3: 纠缠调制 (Entanglement Modulation)**:
        *   使用 **CRX (Controlled-RX)** 门。
        *   **控制位**: $Q_{ancilla}$ (时间信息)。
        *   **目标位**: $Q_{data}$ (空间信息)。
        *   *物理意义*: 时间信息直接控制空间特征的演化路径，实现比经典 AdaGN 更深层的特征融合。
    *   **步骤 4: 空间演化 (Spatial Evolution)**:
        *   在 $Q_{data}$ 上施加 CNOT 环及跨步 CNOT（Strided CNOT），提取局部空间关联。
    *   **步骤 5: 测量 (Measurement)**:
        *   仅对 $Q_{data}$ 进行 Pauli-Z 测量，获取期望值。
*   **后处理**:
    *   测量结果 $\rightarrow$ 线性投影 $\rightarrow$ 残差连接 $\rightarrow$ Reshape 回 $[B, C, H, W]$。

### 3.3 量子注意力机制 (Quantum Attention / QSANN)

基于核方法的量子注意力，利用量子态的高维希尔伯特空间特性计算特征相似度。

*   **输入**: 特征图 $x$，重塑为 Token 序列 $X \in \mathbb{R}^{B \times S \times 64}$。
*   **编码方式**: **幅度编码 (Amplitude Encoding)**。
    *   将 64 维向量 $v$ L2 归一化，直接映射为 6 个量子比特的波函数 $|\psi_v\rangle = \sum_{i=0}^{63} v_i |i\rangle$。
*   **三路量子分支 (Q/K/V Branches)**:
    *   **公共编码**: 所有分支共享基础编码线路 $U_{enc}$。
    *   **Q/K 分支**:
        *   施加独立变分线路 $U_Q, U_K$。
        *   测量: Pauli-Z 期望值 $\rightarrow$ 投影至低维空间 $\rightarrow$ $q, k$ 向量。
    *   **V 分支**:
        *   施加独立变分线路 $U_V$。
        *   **测量**: 计算计算基底概率 $P(|i\rangle) = |\langle i | \psi_V \rangle|^2$。由于 $i \in [0, 63]$，直接得到 64 维概率向量作为 $v$ 向量。
*   **注意力计算 (RBF Kernel)**:
    *   不同于经典的 Dot-Product，这里使用径向基函数（RBF）核：
        $$ \alpha_{i,j} = \exp\left( -\frac{\|q_i - k_j\|^2}{\tau} \right) $$
    *   其中 $\tau$ 为可训练的温度参数。
*   **输出**: $Attention(X) = \text{softmax}(\alpha) \cdot v$。

---

## 4. 模块间数据传递规范 (Data Transmission Protocol)

为了确保量子与经典模块的无缝衔接，定义以下数据接口标准：

| 接口位置 | 发送方 | 接收方 | 数据形式 | 维度/形状 | 处理逻辑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **时间注入** | Q-MLP | Q-FrontEnd | **量子线路操作** (Circuit Ops) | N/A (Gate Sequence) | Q-MLP 线路直接接入。若 $N_{qmlp} > N_{ancilla}$，通过测量多余比特实现**态缩减适配**。 |
| **特征输入** | Classic UNet | Q-FrontEnd | Real Tensor | $[B, C, H, W]$ | Unfold 为 Patch，转换为角度参数。 |
| **特征输出** | Q-FrontEnd | Classic UNet | Real Tensor | $[B, C, H, W]$ | 测量期望值 $\rightarrow$ 投影 $\rightarrow$ Fold 拼回图像。 |
| **注意力输入** | Classic UNet | Q-Attention | Real Tensor | $[B, C, H, W]$ | Flatten 为 $[B, S, 64]$，需满足 $D=64$ 以适配 6-Qubit 幅度编码。 |
| **注意力输出** | Q-Attention | Classic UNet | Real Tensor | $[B, C, H, W]$ | 计算结果 Reshape 回图像尺寸，直接相加于残差。 |

---

## 5. 算法优势总结 (Key Algorithmic Advantages)

1.  **参数效率**: 利用 $N$ 个量子比特即可表示 $2^N$ 维特征空间（如 Q-Attention 中 6 Qubits 处理 64 维特征），参数量显著少于同等维度的经典 MLP。
2.  **相干融合**: Q-FrontEnd 实现了时间条件与空间特征在波函数层面的纠缠，理论上比经典的加法/乘法注入（AdaGN）具有更强的非线性特征捕捉能力。
3.  **全局关联**: Q-Attention 利用量子态的全局干涉特性计算注意力权重，提供了不同于经典内积注意力的归纳偏置。

---

## 6. 特定场景流程: 128x128 图像隐空间量子扩散 (Latent Quantum Diffusion)

针对 **3×128×128 输入图像** 并在 **隐空间 (Latent Space)** 进行训练的特定场景，本节详细描述量子 UNet 的逐层级算法流程。

### 6.1 阶段一：VAE 压缩与预处理 (Compression Phase)
*   **输入**: 原始图像 $x \in \mathbb{R}^{B \times 3 \times 128 \times 128}$，归一化至 $[-1, 1]$。
*   **编码器 (Encoder)**: 使用 `stabilityai/sd-vae-ft-mse` (KL-VAE)。
    *   下采样率 $f=8$。
    *   输出分布参数 $\mu, \log\sigma^2$。
*   **采样 (Sampling)**: $z = \mu + \epsilon \cdot \sigma$，得到潜变量 $z \in \mathbb{R}^{B \times 4 \times 16 \times 16}$。
*   **缩放**: $z \leftarrow z \times 0.18215$ (标准差归一化)。

### 6.2 阶段二：量子 UNet 扩散过程 (Quantum UNet Process)
模型骨干为 **NCSN++ / SongUNet**，输入为 $16 \times 16$ 的 Latent Feature。

#### 6.2.1 全局时间嵌入 (Global Time Embedding)
*   **输入**: 噪声水平 $\sigma$。
*   **处理**: 
    1.  经典投影: $\sigma \to \text{Fourier Features} \to \text{MLP} \to \text{Emb}_{class} \in \mathbb{R}^{512}$ (用于经典模块)。
    2.  **Q-MLP**: $\sigma \to \text{Quantum Circuit} \to |\psi_{time}\rangle$ (用于 Q-FrontEnd 注入)。

#### 6.2.2 UNet 逐层级详解 (Level-wise Micro-Architecture)

**A. 输入层 (Input Layer)**
*   **操作**: $3 \times 3$ 卷积。
*   **变换**: $4 \times 16 \times 16 \to 128 \times 16 \times 16$ (假设 Base Channels = 128)。

**B. 编码器层级 1 (Encoder Level 1: 16x16 Res)**
*   **输入**: $128 \times 16 \times 16$。
*   **ResBlock x2**:
    *   **Q-FrontEnd**: 替代部分经典 ResBlock 中的 Conv2d。
        *   接收 $16 \times 16$ 特征与 $|\psi_{time}\rangle$。
        *   通过量子纠缠注入时间信息。
    *   **Q-Attention (可选)**: 
        *   若启用，需处理 $S = 16 \times 16 = 256$ Tokens。
        *   **维度投影**: $128 \xrightarrow{Proj} 64 \xrightarrow{Quantum} 64 \xrightarrow{Proj} 128$。
        *   **计算**: 量子 RBF 核计算 $256 \times 256$ 注意力图（在模拟器上可能较慢，建议硬件支持或进一步 Patchify）。
*   **下采样 (Downsample)**: $Conv(stride=2)$。
    *   输出: $128 \times 8 \times 8$ (假设 Channel Mult 保持或翻倍，此处设为翻倍至 256)。

**C. 编码器层级 2 (Encoder Level 2: 8x8 Res)**
*   **输入**: $256 \times 8 \times 8$。
*   **ResBlock x2**: 包含 Q-FrontEnd。
*   **Q-Attention (核心区域)**:
    *   **序列长度**: $S = 8 \times 8 = 64$ Tokens。
    *   **完美适配**: 64 Tokens 正好对应 QSANN 的设计甜点区间，无需额外 Patchify。
    *   **维度投影**: $256 \xrightarrow{Proj} 64 \xrightarrow{Q-Attn} 64 \xrightarrow{Proj} 256$。
    *   利用 6-Qubit 幅度编码进行高效率全局特征聚合。
*   **下采样 (若有)**: 若网络更深，可继续降至 $4 \times 4$。在此配置下，通常 8x8 已是瓶颈层。

**D. 中间层 (Middle Block: 8x8)**
*   **输入**: $256 \times 8 \times 8$。
*   **结构**: `ResBlock` $\to$ `Q-Attention` $\to$ `ResBlock`。
*   **作用**: 在最低分辨率进行最密集的量子特征交互，捕捉全局语义。

**E. 解码器 (Decoder Levels)**
*   与编码器对称，执行上采样 (Upsample) 与跳跃连接 (Skip Connection)。
*   在 8x8 层级继续应用 Q-Attention。
*   在 16x16 层级应用 Q-FrontEnd 恢复细节。

#### 6.2.3 输出层 (Output Layer)
*   **操作**: GroupNorm $\to$ SiLU $\to$ $3 \times 3$ Conv。
*   **变换**: $128 \times 16 \times 16 \to 4 \times 16 \times 16$。
*   **目标**: 预测去噪后的 Latent $z_0$ (EDM Formulation)。

### 6.3 阶段三：解码与恢复 (Decompression Phase)
*   **输入**: 预测的 Latent $z_{pred} \in \mathbb{R}^{4 \times 16 \times 16}$。
*   **反缩放**: $z_{pred} \leftarrow z_{pred} / 0.18215$。
*   **解码器 (Decoder)**: VAE Decoder。
*   **输出**: 重建图像 $x_{recon} \in \mathbb{R}^{3 \times 128 \times 128}$。

---

**附：模型架构图示 (伪代码/TikZ 逻辑)**

```mermaid
graph TD
    subgraph "Time Embedding"
        sigma[Noise Level sigma] -->|Angle Enc| QMLP[Quantum MLP Circuit]
        QMLP -->|Output State| Ancilla[Ancilla Qubits]
    end

    subgraph "UNet Block"
        Img[Image Input] -->|Unfold| Patch[Patches]
        Patch -->|Angle Enc| DataQ[Data Qubits]
        
        subgraph "Quantum FrontEnd"
            Ancilla -.->|Entanglement (CRX)| DataQ
            DataQ -->|Evolution (RY/CNOT)| DataQ
            DataQ -->|Measure Z| Feat[Classic Feature Map]
        end
        
        Feat -->|Proj & Reshape| Token[Tokens (64-dim)]
        
        subgraph "Quantum Attention"
            Token -->|Amp Enc| QState[State |psi>]
            QState -->|Branch Q| QVec
            QState -->|Branch K| KVec
            QState -->|Branch V| VVec
            QVec & KVec -->|RBF Kernel| AttnMap
            AttnMap & VVec -->|Weighted Sum| OutToken
        end
        
        OutToken -->|Fold| Output[Output Image]
    end
```
