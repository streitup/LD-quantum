# 4.Z.2 量子注意力中的测量机制 (Quantum Attention Measurement Mechanism)

在 SOTA QAttention（基于 QuantumAttention64 实现）中，为了兼顾计算效率与语义表达能力，我们对 Query (Q)、Key (K) 与 Value (V) 分支采用了**非对称测量策略**。该策略在计算注意力权重时使用低维的 Pauli-Z 期望值特征，而在聚合内容信息时使用高维的全概率分布特征。

#### 1. Q/K 分支：Pauli-Z 期望值测量

Query 和 Key 分支的主要任务是计算注意力分数（Attention Scores），即衡量不同 Token 之间的相关性。为了降低 $O(S^2)$ 复杂度下的计算开销，我们采用基于单量子比特可观测量（Observables）的降维测量方式。

设经过 HEA 演化后的 Q 分支量子态为 $|\psi_Q\rangle$，K 分支量子态为 $|\psi_K\rangle$。对于 $N$ 个量子比特，我们分别测量其 Pauli-Z 算符 $\sigma_z$ 的期望值，得到 $N$ 维特征向量：

$$
\mathbf{m}_Q = \left[ \langle \psi_Q | \sigma_z^{(0)} | \psi_Q \rangle, \dots, \langle \psi_Q | \sigma_z^{(N-1)} | \psi_Q \rangle \right]^T
$$
$$
\mathbf{m}_K = \left[ \langle \psi_K | \sigma_z^{(0)} | \psi_K \rangle, \dots, \langle \psi_K | \sigma_z^{(N-1)} | \psi_K \rangle \right]^T
$$

随后，通过线性投影层将这些特征映射到低维空间 $d_k$（例如 $d_k=4$ 或 $16$），用于计算 RBF 核注意力权重：

$$
\mathbf{q} = \mathbf{W}_Q \mathbf{m}_Q, \quad \mathbf{k} = \mathbf{W}_K \mathbf{m}_K
$$
$$
\alpha_{i,j} = \text{Softmax}\left( \exp\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_j\|^2}{\tau}\right) \right)
$$

这种低维投影策略显著减少了注意力矩阵计算过程中的参数量和计算复杂度。

#### 2. V 分支：全概率分布测量 (Full Probability Measurement)

Value 分支承载了需要被聚合和传递的语义内容信息。为了最大化保留量子态中蕴含的高维特征，我们放弃了降维测量，转而直接提取量子态在计算基下的**全概率分布**。

设 V 分支演化后的量子态为 $|\psi_V\rangle = \sum_{x=0}^{2^N-1} c_x |x\rangle$。我们测量其在计算基 $\{|0\rangle, \dots, |2^N-1\rangle\}$ 上的概率分布向量 $\mathbf{v}$：

$$
\mathbf{v} = \left[ P(0), P(1), \dots, P(2^N-1) \right]^T, \quad \text{其中 } P(x) = |c_x|^2
$$

由于输入使用了幅度编码，该概率向量的维度 $2^N$（例如 $2^6=64$）天然对应于输入 Token 的嵌入维度 $D$。因此，$\mathbf{v}$ 可以直接作为 Value 向量参与加权求和，无需额外的投影层，从而无损地保留了量子电路演化产生的高维非线性特征。

$$
\text{Output}_i = \sum_{j=1}^S \alpha_{i,j} \mathbf{v}_j
$$

#### 3. 总结

| 分支 | 测量方式 | 输出特征维度 | 设计意图 |
| :--- | :--- | :--- | :--- |
| **Q / K** | **Pauli-Z 期望值** + 线性投影 | 低维 ($d_k \ll 2^N$) | 降低注意力矩阵计算复杂度，提取关键关联特征 |
| **V** | **全概率分布** (Computational Basis) | 高维 ($2^N = D$) | 最大化保留语义信息容量，实现无损特征传递 |

这种非对称设计充分利用了量子测量坍缩的不同特性，在保证模型表达能力的同时优化了计算效率。
