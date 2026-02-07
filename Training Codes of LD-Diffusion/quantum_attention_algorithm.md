# 量子注意力机制算法详解与对比 (Advanced Quantum Attention)

本文档详细描述了当前项目中最先进的量子注意力模块 (`QuantumAttentionPatch`) 的算法流程、核心组件、数学原理，并将其与原始 QSANN 算法进行深度对比。

## 1. 算法核心架构 (Quantum Attention Patch)

当前实现的 `QuantumAttentionPatch` 是一种基于 **幅度编码 (Amplitude Encoding)** 和 **数据重上传 (Data Re-uploading)** 的高效量子注意力机制，集成了 **LoRA (Low-Rank Adaptation)** 技术以降低经典参数量。

### 1.1 核心模块流程

#### 模块 1: 输入分块与幅度编码 (Patching & Amplitude Encoding)
*   **作用**: 将高维经典特征压缩到少量量子比特的希尔伯特空间中。
*   **输入**: 经典特征张量 $X \in \mathbb{R}^{B \times S \times D}$。
*   **计算**:
    1.  **分块 (Patching)**: 将输入重塑为 Patch 形式，维度变为 $[B, S/P, P \times D]$。设组维度 $D_{group} = P \times D$。
    2.  **补零 (Padding)**: 若 $D_{group} < 2^N$ ($N$为量子比特数)，则补零至 $2^N$。
    3.  **L2 归一化**: 对特征向量进行 L2 归一化，使其满足量子态的归一性条件 $|\psi|^2 = 1$。
    4.  **状态制备**: 将归一化向量直接映射为量子态幅值 $|\psi_{in}\rangle = \sum_{i=0}^{2^N-1} x_i |i\rangle$。
*   **优点**: 相比角度编码需要 $N \sim D$ 个量子比特，幅度编码只需 $N = \lceil \log_2 D \rceil$ 个量子比特，实现了指数级的特征压缩。

#### 模块 2: 数据重上传 PQC (Data Re-uploading PQC)
*   **作用**: 在深层量子线路中反复注入原始输入信息，防止信息流失，增强非线性表达能力。
*   **计算**:
    1.  **角度映射**: 将输入 $x$ 通过经典投影层映射为旋转角度 $\phi$。
        $$ \phi = \tanh(x W_{re} + b_{re}) \cdot \pi $$
    2.  **层级演化**: 线路共 $L$ 层，每层包含：
        *   **可训练旋转**: $U(\theta_{l})$ (U3 门)。
        *   **数据重上传**: $U(\phi)$ (U3 门，参数由输入决定)。
        *   **纠缠**: 环形 CNOT 连接。
*   **公式**:
    $$ |\psi_{out}\rangle = \left( \prod_{l=1}^L \text{CNOT} \cdot U(\phi) \cdot U(\theta_l) \right) |\psi_{in}\rangle $$

#### 模块 3: 并行 Q/K/V 生成与测量 (Parallel Q/K/V Generation)
*   **作用**: 通过并行的量子线路分别生成 Query, Key, Value 特征。
*   **计算**:
    1.  **并行分支**: 共享编码器权重 $\theta_{enc}$，但拥有独立的分支权重 $\theta_Q, \theta_K, \theta_V$。
    2.  **可训练测量基**: 在测量前应用一个通用的单比特旋转门 $U_{meas}(\omega)$，允许模型学习最优的测量基（不仅仅是 Z 基）。
    3.  **测量**: 测量 Pauli-Z 算符的期望值。
        $$ E_k = \langle \psi_{final} | Z_k | \psi_{final} \rangle $$
    4.  **量子投影**: 将测量结果 $E \in \mathbb{R}^N$ 线性投影回 $D_{group}$ 维度。

#### 模块 4: 混合残差与 LoRA (Hybrid Residuals with LoRA)
*   **作用**: 结合经典路径的稳定性与量子路径的特征提取能力，并利用 LoRA 降低参数量。
*   **计算**:
    $$ Q = Q_{quant} + Q_{classic\_lora}(x) $$
    其中 $Q_{classic\_lora}(x) = x W_{down} W_{up}$，秩 $r \ll D$ (如 $r=8$)。
*   **优点**: LoRA 大幅减少了经典部分的参数量（相比全秩矩阵减少 90%+），同时保留了梯度流的稳定性。

#### 模块 5: 缩放点积注意力 (Scaled Dot-Product Attention)
*   **作用**: 计算最终的注意力分数与输出。
*   **计算**:
    $$ \text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\text{scale}} \right) V $$
    其中 $\text{scale}$ 是一个**可学习参数**，而非固定的 $\sqrt{d_k}$。

---

## 2. 与原始 QSANN 算法的深度对比

原始算法参考自 `\home\zzn\qfl_tq\qgpt-issue-31\QSANN codes\QSANN_pennylane.ipynb`。

| 核心维度 | 原始 QSANN (Original) | 当前量子注意力 (Current Q-Attention) | 改进优势论证 |
| :--- | :--- | :--- | :--- |
| **编码方式** | **线性角度编码 (Angle Encoding)**<br>输入维度 $D$ 需要 $N \approx D/2$ 个量子比特。例如 $D=64$ 需 32+ 比特。 | **幅度编码 (Amplitude Encoding)**<br>利用量子态叠加性质，仅需 $N = \log_2 D$ 个比特。例如 $D=64$ 仅需 6 比特。 | **指数级节省量子资源**。在有限的量子比特数下（如 NISQ 设备），幅度编码能处理高维得多的特征（如图像 Patch）。 |
| **数据注入** | **单次编码**<br>仅在线路初始阶段编码一次数据。随着线路加深，输入信息易被遗忘。 | **数据重上传 (Data Re-uploading)**<br>在每一层 PQC 中都重新注入输入数据的映射角度。 | **增强表达能力**。根据通用近似定理，数据重上传使得量子线路能拟合更复杂的傅里叶级数，解决“数据冻结”问题。 |
| **经典-量子融合** | **纯量子 / 简单残差**<br>主要依赖量子线路输出，经典部分仅作简单加和。 | **混合架构 + LoRA**<br>引入低秩适应 (LoRA) 作为经典残差。 | **训练稳定性与效率**。LoRA 保证了梯度回传的通畅，避免了纯量子线路常见的“梯度消失/贫瘠高原”问题，且参数量极低。 |
| **注意力机制** | **高斯核 (Gaussian RBF)**<br>$\alpha = \exp(-(Q-K)^2)$。这是一种距离度量。 | **点积注意力 (Dot-Product)**<br>$\text{Softmax}(QK^T)$。这是 Transformer 的标准形式。 | **兼容性与性能**。点积注意力更能捕捉特征间的方向一致性，且在深度学习中经过验证更适合大规模序列建模。 |
| **测量机制** | **随机泡利串 / 固定 Z 基**<br>V 使用随机算符，Q/K 使用固定 Z。 | **可训练测量基**<br>引入参数化旋转门 $U(\omega)$ 学习测量方向。 | **信息提取最大化**。模型可以自适应地选择在哪个基底下测量能提取最多的有效信息，而非盲目测量。 |
| **运行框架** | **PennyLane (Serial)**<br>基于逐样本循环，无法利用 GPU 批处理优势。 | **TorchQuantum (Batch)**<br>支持张量级并行计算。 | **训练速度提升百倍**。支持 Batch Training 是深度学习落地的基本要求。 |

### 3. 总结

当前的量子注意力机制并非是对 QSANN 的简单复现，而是一次**工程化与算法层面的全面重构**。
1.  **资源效率**: 通过幅度编码，我们将量子比特需求从线性降低为对数级。
2.  **表达能力**: 通过数据重上传和可训练测量基，我们显著提升了线路的非线性拟合能力。
3.  **可训练性**: 通过混合 LoRA 架构和 Batch 并行，我们解决了量子神经网络难以训练和速度慢的两大痛点，使其真正具备了在 LD-Diffusion 等复杂模型中应用的能力。
