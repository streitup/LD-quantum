# SOTA 量子注意力算法文档 (SOTA Quantum Attention Algorithm Documentation)

## 1. 算法概述

本节介绍的 SOTA 量子注意力机制（QuantumAttentionAngleDense）是量子扩散模型的核心组件，旨在利用量子计算的高维特征映射能力来捕捉数据中的复杂依赖关系。该算法将经典 Transformer 中的线性变换（Linear Projections）替换为参数化量子电路（Parameterized Quantum Circuits, PQC），通过量子态的演化和测量来提取 Query (Q)、Key (K) 和 Value (V) 特征。

### 1.1 核心特性
*   **密集角度编码 (Dense Angle Encoding)**：采用密集角度编码策略，将输入数据注入到量子电路的每一层中，最大化了量子电路的表达能力和数据重上传（Data Re-uploading）的效率。
*   **深层变分电路 (Deep Variational Circuit)**：使用 12 层深度的 PQC 结构，确保了足够的量子纠缠和非线性映射能力。
*   **混合架构 (Hybrid Architecture)**：保留了经典的点积注意力和 Softmax 归一化计算，确保了数值稳定性和与现有深度学习框架的兼容性。
*   **量子-经典接口**：实现了从经典张量到量子旋转角度的映射，以及从量子测量结果到经典特征空间的投影。

## 2. 数学表述

### 2.1 输入变换与 QCNN 导出
量子注意力模块的输入张量记为 $z_q$。在端到端的量子扩散模型中，$z_q$ 通常由前置的量子卷积神经网络（QCNN）或经典编码器输出得到。

假设前置模块输出的特征为 $z_{in} \in \mathbb{R}^{B \times S \times D_{in}}$，其中 $B$ 为批次大小，$S$ 为序列长度（Token数），$D_{in}$ 为输入维度。
为了适配量子电路的输入要求，首先通过一个经典线性层将 $z_{in}$ 映射到潜在的量子维度空间，得到 $z_q$：

$$
z_q = \text{Linear}(z_{in}) \in \mathbb{R}^{B \times S \times D_{q}}
$$

### 2.2 角度映射 (Angle Mapping)
在进入量子电路之前，$z_q$ 需要被转换为量子门的旋转角度参数 $\Theta$。采用了 **Tanh 激活函数** 将数值约束在 $(-\pi, \pi)$ 区间内，以符合量子相位旋转的物理意义：

$$
\Theta = \text{tanh}(\text{Linear}(z_q)) \times \pi
$$

这里得到的 $\Theta \in \mathbb{R}^{B \times S \times (L \times N_q \times 3)}$，其中 $L$ 是电路深度，$N_q$ 是量子比特数，$3$ 代表通用单量子比特旋转门 $U3(\theta, \phi, \lambda)$ 的三个参数。

### 2.3 参数化量子电路 (PQC)
PQC 定义了一个酉变换 $U(\Theta, \Phi)$，其中 $\Theta$ 是由数据导出的编码参数，$\Phi$ 是可训练的变分参数。对于每一层 $l \in \{1, \dots, L\}$ 和每一个量子比特 $i \in \{1, \dots, N_q\}$，电路演化过程如下：

1.  **数据编码 (Data Re-uploading)**：
    使用由输入数据决定的旋转门 $R$ 对量子态进行调制：
    $$
    |\psi^{(l)}_0\rangle = R(\Theta^{(l)}_{i}) |\psi^{(l-1)}\rangle
    $$

2.  **变分处理 (Variational Processing)**：
    应用带有可训练参数 $\Phi$ 的旋转门：
    $$
    |\psi^{(l)}_1\rangle = U(\Phi^{(l)}_{i}) |\psi^{(l)}_0\rangle
    $$

3.  **量子纠缠 (Entanglement)**：
    应用环形 CNOT 门（Circular CNOT）引入量子比特间的关联：
    $$
    \text{CNOT}(i, (i+1) \mod N_q)
    $$

最终的量子态 $|\psi_{final}\rangle$ 在 Pauli-Z 基底下进行测量，得到期望值或概率分布。

### 2.4 量子 Q/K/V 投影
为了生成注意力机制所需的 Query ($Q$)、Key ($K$) 和 Value ($V$) 向量，我们并行运行三个 PQC（或共享部分参数）：

$$
Q_{raw} = \text{Measure}(U_Q(z_q)), \quad K_{raw} = \text{Measure}(U_K(z_q)), \quad V_{raw} = \text{Measure}(U_V(z_q))
$$

测量结果 $M \in \mathbb{R}^{B \times S \times N_q}$ 随后被线性投影回注意力头的维度 $d_h$：

$$
Q = \text{Linear}(Q_{raw}), \quad K = \text{Linear}(K_{raw}), \quad V = \text{Linear}(V_{raw})
$$

### 2.5 注意力计算
获得量子生成的 $Q, K, V$ 后，采用标准的缩放点积注意力计算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_h}}\right) V
$$

## 3. 算法流程

1.  **输入准备**：接收张量 $z_q$（来自 QCNN 或上一层）。
2.  **角度编码**：将 $z_q$ 映射为旋转角度张量 $\Theta$。
3.  **量子电路执行 (并行 Q/K/V)**：
    *   初始化量子态为 $|0\rangle^{\otimes N}$。
    *   **循环** $l=1$ 到 $L$ (层数)：
        *   应用数据旋转门 $\Theta^{(l)}$ (Data Encoding)。
        *   应用变分旋转门 $\Phi^{(l)}$ (Trainable Weights)。
        *   应用纠缠层 (CNOT Ring)。
    *   执行 Pauli-Z 测量。
4.  **特征投影**：将测量结果映射为 $Q, K, V$ 向量。
5.  **注意力聚合**：计算注意力分数并加权求和。
6.  **输出**：输出包含上下文信息的特征张量。

## 4. 基础 QSANN 及其局限性

量子自注意力神经网络（Quantum Self-Attention Neural Network, QSANN）是早期尝试将量子计算引入注意力机制的代表性工作。它为后续的量子注意力研究奠定了基础，但在处理复杂任务时存在一定局限性。

### 4.1 QSANN 核心机制
QSANN 的核心思想是利用量子态的内积来模拟注意力分数（Attention Score）。对于输入向量 $x_i$ 和 $x_j$，QSANN 将其分别编码为量子态 $|\psi(x_i)\rangle$ 和 $|\psi(x_j)\rangle$。

注意力系数 $\alpha_{ij}$ 定义为两个量子态的保真度（Fidelity）或重叠积分：
$$
\alpha_{ij} = |\langle \psi(x_i) | \psi(x_j) \rangle|^2
$$
这种方法直接利用了量子力学的几何特性来计算相似度，无需显式的 $Q \cdot K^T$ 矩阵乘法。

### 4.2 局限性分析
1.  **编码能力受限**：基础 QSANN 通常采用简单的幅度编码或单层角度编码，难以将高维经典数据映射到具有丰富特征的希尔伯特空间。
2.  **特征变换单一**：仅依赖量子态重叠计算注意力分数，缺乏可训练的 Query/Key 投影变换，导致模型难以学习特定的语义关联。
3.  **计算复杂度**：为了计算所有 $i, j$ 对的重叠，需要 $O(S^2)$ 次量子电路执行或复杂的 SWAP 测试，硬件实现难度大。

## 5. SOTA 量子注意力与 QSANN 的对比与改进

本节详细对比本文提出的 SOTA 量子注意力机制与基础 QSANN，阐述关键改进点及其优势。

### 5.1 关键改进点

| 特性 | 基础 QSANN | SOTA Quantum Attention (本文) | 改进意义 |
| :--- | :--- | :--- | :--- |
| **注意力计算** | 量子态重叠 $|\langle \psi_i | \psi_j \rangle|^2$ | 经典缩放点积 $\text{softmax}(QK^T/\sqrt{d})$ | **数值稳定性**：保留了 Transformer 的经典注意力结构，避免了量子噪声对注意力权重的直接干扰，且梯度反传更稳定。 |
| **Q/K/V 生成** | 无显式投影（直接使用编码态） | **参数化量子电路 (PQC)** | **特征提取能力**：引入深层 PQC 作为可训练的非线性特征提取器，使模型能够学习特定的语义空间映射。 |
| **数据编码** | 简单幅度/角度编码 | **密集角度编码 (Data Re-uploading)** | **表达能力**：通过在电路每一层重复注入数据，突破了单层编码的表达瓶颈，显著增强了对复杂数据的拟合能力。 |
| **纠缠结构** | 较弱或无纠缠 | **深层环形 CNOT (Circular Entanglement)** | **量子关联**：深层纠缠结构使得量子比特间能够充分交互，捕捉特征维度间的深层关联。 |
| **可扩展性** | $O(S^2)$ 量子操作 | $O(S)$ 量子操作 (并行生成 Q/K/V) | **计算效率**：仅需 $3S$ 次量子电路执行即可生成 Q/K/V，随后进行经典矩阵乘法，大幅降低了量子硬件的调用开销。 |

### 5.2 改进总结
SOTA 量子注意力机制并未完全摒弃经典 Transformer 的优势，而是采取了 **“量子特征提取 + 经典注意力聚合”** 的混合策略。
*   相比 QSANN 试图用量子内积完全替代经典点积，SOTA 方法更务实地将量子计算算力集中在 **特征变换（Q/K/V Projection）** 这一最需要高维映射能力的环节。
*   **密集角度编码** 和 **深层 PQC** 的引入，解决了 QSANN 在处理高维复杂数据（如图像、长文本）时特征表达不足的问题。
*   这种混合架构不仅保留了量子计算的潜在优势（高维特征空间），还具备了经典深度学习的可训练性和鲁棒性，是当前量子机器学习领域更为成熟和高效的解决方案。

## 6. SOTA 变种：窗口化量子注意力 (Windowed Quantum Attention)

为了在处理长序列时降低计算复杂度并聚焦局部特征，我们引入了 SOTA 量子注意力的窗口化变种。

### 6.1 窗口划分 (Window Partitioning)
将长度为 $S$ 的输入序列划分为互不重叠的固定大小窗口。窗口大小 $W$ 设定为总 Token 数的四分之一：
$$
W = \frac{S}{4}
$$
这通过将输入张量 $z_q$ 重塑（Reshape）来实现：
$$
z_q \in \mathbb{R}^{B \times S \times D} \rightarrow z_{win} \in \mathbb{R}^{B \times 4 \times W \times D}
$$
此时，批次维度实际上被扩大了 4 倍，即 $B' = B \times 4$。

### 6.2 独立窗口处理 (Independent Window Processing)
每个窗口 $w \in \{1, 2, 3, 4\}$ 被视为一个独立的子序列。上述的 SOTA 量子注意力机制（第 2 节所述）在每个窗口内部独立运行：

$$
\text{Output}_w = \text{SOTA\_QAttention}(z_{win}^{(w)})
$$

这意味着 Token 只能与同一窗口内的其他 Token 进行注意力交互，计算复杂度从 $O(S^2)$ 降低为 $O(S \cdot W) = O(S^2/4)$。

### 6.3 窗口合并 (Window Merging)
在所有 4 个窗口计算完成后，将输出张量沿序列维度拼接，恢复原始形状：
$$
\text{Output} = \text{Concat}(\text{Output}_1, \text{Output}_2, \text{Output}_3, \text{Output}_4) \in \mathbb{R}^{B \times S \times D}
$$

### 6.4 差异总结
| 特性 | SOTA 量子注意力 (全局) | 窗口化变种 (局部) |
| :--- | :--- | :--- |
| **作用范围** | 全局序列 (Full Sequence) | 局部窗口 (Fixed Window $W=S/4$) |
| **复杂度** | $O(S^2)$ | $O(S^2/4)$ |
| **上下文** | 长程依赖 (Long-range) | 局部特征 (Local Features) |
| **应用场景** | 需要全局感受野的任务 | 长序列、注重局部细节的任务 |
