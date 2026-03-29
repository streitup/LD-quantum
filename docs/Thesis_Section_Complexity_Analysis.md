# 4.X 算法复杂度分析

本节将对本文提出的 SOTA_QCNN（State-of-the-Art Quantum Convolutional Neural Network）与 SOTA_QAttention（State-of-the-Art Quantum Attention）算法进行严谨的复杂度分析。分析涵盖时间复杂度、空间复杂度及参数量三个维度，并将这两个算法与其经典对应物（经典卷积神经网络和经典 Transformer 自注意力机制）以及早期量子算法（如 QSANN）进行对比。

为了全面评估算法的性能与可行性，我们将分析分为两个层面：
1.  **真实量子复杂度 (Quantum Circuit Complexity)**：基于理想量子硬件模型，衡量算法在未来量子计算时代的理论优势。
2.  **经典模拟复杂度 (Classical Simulation Complexity)**：基于当前的经典 GPU/CPU 模拟环境，评估算法在现有实验条件下的计算代价。

需要特别说明的是，为了在近期量子设备（NISQ）的可训练性与高维特征的表达能力之间取得平衡，本文在两个模块中采用了不同的量子编码策略：
*   **SOTA_QCNN**：采用 **角度编码 (Angle Encoding)**，旨在利用其低深度特性（$O(1)$）在前端特征提取阶段实现高效的非线性映射。
*   **SOTA_QAttention**：采用 **幅度编码 (Amplitude Encoding)**，旨在利用其指数级数据压缩能力（$N$ 比特编码 $2^N$ 维数据）处理高维语义特征。

#### 4.X.1 SOTA_QCNN 复杂度分析

SOTA_QCNN 旨在利用量子电路的高维希尔伯特空间（Hilbert Space）映射能力替代经典卷积核的局部特征提取过程。PQC 部分采用了 **硬件高效拟设 (Hardware-Efficient Ansatz, HEA)**。

**1. 符号定义**
设输入特征图尺寸为 $H \times W$，通道数为 $C_{in}$。卷积核大小为 $K \times K$，输出通道数为 $C_{out}$。
对于量子部分，设量子比特数为 $N$，变分量子电路（PQC）深度为 $L_{depth}$。我们将特征图展开（Unfold）后的 Patch 总数记为 $M = H_{out} \times W_{out}$。
**$S_{shots}$ (采样次数)**：在真实量子设备上，为了从量子态的坍缩结果中统计出概率分布（或期望值），需要对同一个电路重复运行并测量多次。$S_{shots}$ 代表这一重复测量的次数，它决定了结果的统计精度（误差 $\propto 1/\sqrt{S_{shots}}$）。通常 $S_{shots}$ 取值在 $10^3 \sim 10^4$ 量级。

**2. 时间复杂度分析**

*   **经典 CNN (Conv2d)**：
    $$ T_{CNN} \approx O(M \cdot C_{out} \cdot C_{in} \cdot K^2) $$

*   **SOTA_QCNN (基于 HEA + 角度编码)**：

    *   **层面一：真实量子复杂度**
        由于采用角度编码，数据加载仅需单层旋转门，深度为 $O(1)$。
        *   **电路深度**：$D = D_{enc} + D_{PQC} \approx O(1) + O(L_{depth})$。
        *   **执行时间**：$T_{real} \approx O(M \cdot S_{shots} \cdot L_{depth})$。
        *   **优势**：利用量子并行性，时间复杂度与输入通道 $C_{in}$ 完全解耦，实现了对高维通道数据的 $O(1)$ 处理能力。

    *   **层面二：经典模拟复杂度**
        *   **态演化**：$O(M \cdot N \cdot L_{depth} \cdot 2^N)$。
        *   **测量与投影**：$O(M \cdot 2^N \cdot C_{out})$。
        *   **总复杂度**：$T_{sim} \approx O(M \cdot 2^N \cdot (N \cdot L_{depth} + C_{out}))$。

**3. 空间复杂度与参数量分析 (Spatial Complexity & Parameter Efficiency)**

空间复杂度分析重点关注模型的参数量（Parameter Count），这直接决定了模型的存储需求与训练难度。

*   **经典 CNN (Conv2d)**：
    *   **参数量**：$P_{CNN} = C_{out} \cdot C_{in} \cdot K^2 + C_{out}$。参数量随输入通道 $C_{in}$ 和输出通道 $C_{out}$ 呈二次增长。对于高维特征（如 $C_{in}=256, C_{out}=512$），参数量巨大。

*   **SOTA_QCNN (基于 HEA + 角度编码)**：
    *   **参数效率 (Parameter Efficiency)**：
        $$ P_{QCNN} = P_{HEA} + P_{Proj} = 3 \cdot N \cdot L_{depth} + 2^N \cdot C_{out} $$
        *   **HEA 部分**：参数量仅与量子比特数 $N$ 和电路深度 $L_{depth}$ 线性相关，与输入通道 $C_{in}$ **完全无关**。这是量子卷积最显著的优势——即使输入通道数增加到数千维，HEA 的参数量依然保持恒定（仅由电路结构决定）。
        *   **投影部分**：虽然投影层引入了 $2^N \cdot C_{out}$ 的参数，但通常 $2^N \ll C_{in} \cdot K^2$（例如 $N=6, 2^N=64$ vs $256 \times 9 = 2304$）。
        *   **结论**：SOTA_QCNN 在参数量上实现了极大的压缩（Compression Ratio > 10x），这使得在资源受限设备（如边缘计算）上部署成为可能。

#### 4.X.2 SOTA_QAttention 复杂度分析

SOTA_QAttention 通过量子电路生成 Query (Q)、Key (K) 和 Value (V) 向量。为了处理 Transformer 中典型的高维 Token（如 $D=64$ 或更高），此处采用 **幅度编码**。

**1. 符号定义**
设输入序列长度为 $S$，嵌入维度为 $D$（在此对应 $2^N$）。注意力头数为 $H$，每头维度为 $d_k$。

**2. 时间复杂度对比**

*   **SOTA_QAttention (基于 HEA + 幅度编码)**：
    *   **真实量子复杂度**：
        幅度编码的制备代价较高，深度通常为 $O(2^N)$ 或通过近似变分加载优化到 $O(poly(N))$。假设采用标准幅度编码：
        $$ T_{real} \approx O(S \cdot S_{shots} \cdot (2^N + L_{depth})) + O(S^2 \cdot d_k) $$
        尽管数据加载项 $2^N$ 较高，但后续的特征融合与纠缠演化 $L_{depth}$ 仍保持了量子优势，且整体量子调用次数为 $O(S)$（线性）。
    *   **经典模拟复杂度**：
        $$ T_{sim} \approx O(S \cdot 2^N \cdot (2^N + N \cdot L_{depth})) + O(S^2 \cdot d_k) $$
        其中第一个 $2^N$ 来自幅度编码的状态初始化开销。

**3. 空间复杂度分析**

*   **经典 Self-Attention**：
    *   需存储 $Q, K, V$ 矩阵及注意力分数矩阵 $A \in \mathbb{R}^{S \times S}$。
    *   空间复杂度为 $O(S^2 + S \cdot D)$。当序列长度 $S$ 增加时，$S^2$ 项成为瓶颈。

*   **SOTA_QAttention**：
    *   **量子态存储**：利用幅度编码，我们将 $2^N$ 维的经典数据压缩到 $N$ 个量子比特中。在真实量子设备上，这实现了**指数级的空间压缩**（$N = \log_2 D$）。
    *   **参数量优势**：SOTA_QAttention 使用 PQC 替代了经典 Transformer 中庞大的线性投影层（$W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$）。HEA 的参数量为 $O(N \cdot L)$，而经典线性层为 $O(D^2) = O((2^N)^2)$。这意味着量子注意力机制在参数效率上具有**双指数级优势**（相对于比特数 $N$）。

#### 4.X.3 复杂度总结

表 4-1 总结了 SOTA 算法与经典及相关量子算法的复杂度对比。

**表 4-1 算法复杂度对比汇总**

| 算法模型 | 编码方式 | 真实量子时间复杂度 | 经典模拟时间复杂度 | 参数量规模 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **经典 CNN** | N/A | N/A | $O(M \cdot C_{in} C_{out} K^2)$ | $O(C_{in} C_{out})$ | 计算量随通道数二次增长 |
| **SOTA_QCNN** | **角度编码** | $\mathbf{O(M \cdot S_{shots} \cdot L)}$ | $O(M \cdot 2^N \cdot C_{out})$ | $O(N L + 2^N C_{out})$ | **通道解耦**，硬件友好 |
| **经典 Self-Attention** | N/A | N/A | $O(S^2 D + S D^2)$ | $O(D^2)$ | 线性投影参数量大 |
| **QSANN (基线)** | 角度/幅度 | $O(S^2 \cdot S_{shots} \cdot L)$ | $O(S^2 \cdot 2^N)$ | $O(N L)$ | 量子调用次数 $O(S^2)$ |
| **SOTA_QAttention** | **幅度编码** | $O(S \cdot S_{shots} \cdot 2^N)$ | $O(S \cdot 2^N \cdot 2^N)$ | $O(N L + 2^N d_k)$ | **高维压缩**，线性复杂度 |

综上所述，SOTA_QCNN 利用角度编码在前端实现高效特征提取，SOTA_QAttention 利用幅度编码在深层实现高维语义压缩，两者结合实现了在 NISQ 约束下的性能最大化。
