# 4.Z.3 修正后的量子注意力计算流程 (Corrected Quantum Attention Workflow)

本节基于 SOTA 算法的实际实现（采用全概率 V 分支与 RBF 核注意力），对量子注意力机制的计算流程进行严谨的数学描述。

#### a) 量子态演化与测量 (Quantum Evolution & Measurement)

对于输入序列中的第 $i$ 个 Token，我们分别通过三个并行的参数化量子电路（PQC）生成 Query、Key 和 Value 分支的量子态：$|\psi_{Q,i}\rangle, |\psi_{K,i}\rangle, |\psi_{V,i}\rangle$。

**测量过程**：
*   **Q/K 分支（降维特征）**：对 $|\psi_{Q,i}\rangle$ 和 $|\psi_{K,i}\rangle$ 执行 Pauli-Z 算符期望值测量，得到 $N$ 维特征向量 $\mathbf{m}_{Q,i}, \mathbf{m}_{K,i} \in \mathbb{R}^N$。
    $$ \mathbf{m}_{Q,i} = \left[ \langle \sigma_z^{(0)} \rangle_Q, \dots, \langle \sigma_z^{(N-1)} \rangle_Q \right]^T $$
    $$ \mathbf{m}_{K,i} = \left[ \langle \sigma_z^{(0)} \rangle_K, \dots, \langle \sigma_z^{(N-1)} \rangle_K \right]^T $$

*   **V 分支（全概率特征）**：对 $|\psi_{V,i}\rangle$ 执行计算基全概率测量，得到 $2^N$ 维特征向量 $\mathbf{m}_{V,i} \in \mathbb{R}^{2^N}$。
    $$ \mathbf{m}_{V,i} = \left[ P(0), P(1), \dots, P(2^N-1) \right]^T, \quad P(x) = |\langle x | \psi_{V,i} \rangle|^2 $$

#### b) 线性投影 (Linear Projection)

为了计算注意力权重，我们将 Q/K 的测量结果投影到注意力头维度 $d_h$；而 V 分支由于使用了全概率测量，其维度 $2^N$ 天然对应于输出维度，因此直接使用（或仅做归一化）。

$$ \mathbf{q}_i = \text{Linear}_Q(\mathbf{m}_{Q,i}) \in \mathbb{R}^{d_h} $$
$$ \mathbf{k}_i = \text{Linear}_K(\mathbf{m}_{K,i}) \in \mathbb{R}^{d_h} $$
$$ \mathbf{v}_i = \mathbf{m}_{V,i} \in \mathbb{R}^{2^N} \quad (\text{无需降维投影}) $$

#### c) 注意力分数计算 (Attention Score with RBF Kernel)

为了更好地衡量量子特征空间中的相似度，我们采用高斯径向基函数（RBF Kernel）代替传统的点积。注意力权重矩阵 $A \in \mathbb{R}^{S \times S}$ 的元素 $\alpha_{i,j}$ 计算如下：

$$ \alpha_{i,j} = \frac{\exp\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_j\|^2}{\tau}\right)}{\sum_{l=1}^S \exp\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_l\|^2}{\tau}\right)} $$

其中 $\tau$ 为可训练的温度参数。

#### d) 上下文聚合与输出 (Context Aggregation & Output)

最后，利用计算出的注意力权重对 V 分支的高维特征进行加权求和，得到包含上下文信息的输出向量 $\mathbf{z}_{score, i}$：

$$ \mathbf{z}_{score, i} = \sum_{j=1}^S \alpha_{i,j} \mathbf{v}_j, \quad \mathbf{z}_{score, i} \in \mathbb{R}^{2^N} $$

最终输出经过一个线性层恢复到模型的原始通道维度 $C_{model}$（若 $2^N = C_{model}$，则此步可为恒等映射）：

$$ \mathbf{z}_{out} = \text{Linear}_{out}(\mathbf{z}_{score}) \in \mathbb{R}^{B \times S \times C_{model}} $$

该流程清晰地展示了如何利用非对称测量策略（Q/K 降维 + V 全概率）在保证计算效率的同时，最大化量子注意力的语义表达能力。
