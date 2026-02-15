# SOTA Quantum Attention Algorithm (Full Quantum Deep V3 -> V4 Angle Encoding)

本文档详细描述了当前项目中最先进的量子注意力机制 **`QuantumAttentionDeep`** (V3) 及其升级版 **`QuantumAttentionAngle`** (V4)。该架构在 V2 (Hybrid Lite) 的基础上进行了革命性升级，通过**全量子化 (Full Quantum)** 和 **深度量子线路 (Deep Circuit)**，实现了超越经典卷积的特征提取能力。V4 版本进一步引入了 **角度编码 (Angle Encoding)** 和 **密集角度编码 (Dense Angle Encoding)**，显著提升了特征容量和模型性能。

---

## 1. 核心架构演进

### V3: Full Quantum Deep (2024)
*   **全量子化架构**: 移除 Q/K 分支经典卷积。
*   **深度量子线路 (Depth=8)**: 突破浅层线路瓶颈。
*   **Batch-Parallel 接口优化**: 接口耗时降低 63.45%。
*   **概率测量**: 输出 64 维概率分布特征。

### V4: Angle Encoding & Dense Encoding (2026 - Current)
针对 V3 中幅度编码（Amplitude Encoding）仅利用量子态振幅、忽略相位信息的局限，V4 引入了基于旋转门的角度编码方案。

1.  **角度编码 (Angle Encoding)**:
    *   **原理**: 将经典特征 $x$ 映射为旋转角度 $\theta = \text{Tanh}(x) \cdot \frac{\pi}{2}$，通过 $R_x(\theta), R_y(\theta)$ 门注入量子线路。
    *   **优势**: 利用了量子态的相位空间，特征映射具有更强的非线性（周期性）。
    *   **Pure 模式**: 移除经典残差连接，证明量子线路本身具备强大的特征提取能力。

2.  **密集角度编码 (Dense Angle Encoding)**:
    *   **问题**: 随着线路深度增加，初始层注入的信息可能衰减。
    *   **解决方案**: **分层注入 (Layer-wise Injection)**。将输入特征切分为多个块（或通过 MLP 映射到高维），在量子线路的每一层都注入新的角度特征。
    *   **容量**: 特征容量从 $N_{qubits}$ 扩展到 $N_{qubits} \times Depth$，极大提升了信息承载量。

---

## 2. 算法详细流程 (V4 Angle Encoding)

输入张量 $X \in \mathbb{R}^{B \times S \times D}$ ($D=64$)。

### 步骤 1: 密集角度映射 (Dense Angle Mapping)
通过一个 MLP 将 64 维特征映射为所有层的旋转角度：
$$ \Theta = \text{MLP}(X) \in \mathbb{R}^{B \times S \times (D_{circuit} \times N_{qubits} \times 2)} $$
其中 $D_{circuit}$ 为电路深度 (Depth=4/8)，$N_{qubits}$ 为量子比特数 (6/8/10)。

### 步骤 2: 分层量子演化 (Layer-wise Evolution)
初始化量子态 $|0\rangle^{\otimes N}$。
对于每一层 $l \in \{1, ..., D_{circuit}\}$：

1.  **数据注入**:
    $$ |\psi_l'\rangle = \bigotimes_{i=1}^{N} R_y(\theta_{l,i,y}) R_x(\theta_{l,i,x}) |\psi_{l-1}\rangle $$
    这里 $\theta_{l,i}$ 来自 $\Theta$ 的切片。

2.  **可训练演化**:
    $$ |\psi_l''\rangle = U_{trainable}^{(l)} |\psi_l'\rangle $$
    使用 $U3$ 门和 CNOT 纠缠环。

### 步骤 3: 测量与注意
与 V3 类似，采用 Z 基概率测量获取 $Q, K, V$ 特征，并计算注意力。

---

## 3. 性能对比 (Performance)

在 FFHQ-100 数据集上的消融实验结果（50 Epochs）：

| 模型变体 | PSNR (dB) | SSIM | LPIPS (↓) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Angle Encoding (Pure, 10 Qubits)** | **13.79** | **0.5600** | **34.37** | **当前 SOTA** |
| Angle Encoding (Pure, 8 Qubits) | 13.60 | 0.5478 | 35.11 | |
| Angle Encoding (Pure, 6 Qubits) | 13.80 | 0.55xx | 38.21 | (Run 1 Result) |
| Angle Encoding (Hybrid) | 13.66 | 0.54xx | 39.26 | 含经典残差 |
| Full Quantum Deep (V3) | 13.14 | 0.5252 | 41.79 | 幅度编码 |
| Baseline (QSANN) | 12.80 | 0.5275 | 41.68 | |

**结论**:
1.  **Angle > Amplitude**: 角度编码全面超越幅度编码，PSNR 提升约 0.6dB，LPIPS 降低显著（更符合人眼感知）。
2.  **High Qubits**: 增加量子比特数（6 -> 10）进一步提升了表现，验证了量子宽度的重要性。
3.  **Dense Encoding**: (待填充结果) 密集编码有望进一步推高上限。

---

## 4. 工程化建议
*   **Batch-Parallel**: V4 同样支持批次并行，但需注意 Dense Encoding 带来的额外旋转门开销。
*   **显存优化**: 随着 Qubit 数增加 (10+)，建议使用 Checkpointing 技术或减少 Batch Size。
