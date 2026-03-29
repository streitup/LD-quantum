# 4.Z 测量输出与特征重构过程 (Measurement and Feature Reconstruction)

本节详细阐述 SOTA QCNN 算法中，如何将经过变分量子电路（PQC）演化后的量子态信息提取为经典数值，并将其重构为与经典卷积神经网络兼容的特征图格式。该过程包含三个关键步骤：量子测量、组内投影与特征拼接重组。

#### 4.Z.1 量子测量 (Quantum Measurement)

在硬件高效拟设（HEA）演化结束后，量子系统处于一个包含丰富特征信息的纠缠态。为了提取这些信息，我们对每个量子比特执行 Pauli-Z 算符测量，计算其期望值。这一步将量子希尔伯特空间中的态矢量转化为经典的实数值向量。

设经过 HEA 演化后的第 $m$ 个 Patch、第 $g$ 个通道分组的量子态为 $|\psi_{m,g}\rangle$。对于该组内的 $N$ 个量子比特，我们分别测量 Pauli-Z 算符 $\sigma_z$，得到测量向量 $\mathbf{v}_{m,g} \in \mathbb{R}^N$：

$$
\mathbf{v}_{m,g} = \left[ \langle \psi_{m,g} | \sigma_z^{(0)} | \psi_{m,g} \rangle, \dots, \langle \psi_{m,g} | \sigma_z^{(N-1)} | \psi_{m,g} \rangle \right]^T
$$

其中 $\sigma_z^{(i)}$ 表示作用在第 $i$ 个量子比特上的 Pauli-Z 算符。这一过程利用了量子坍缩原理，将高维复数振幅信息压缩为 $N$ 个实数特征值，既保留了关键特征，又实现了数据的降维。

#### 4.Z.2 组内投影 (Intra-Group Projection)

为了将 $N$ 维量子特征映射回该分组对应的经典通道维度 $C_{group} = C_{out}/G$（其中 $G$ 为分组数，$C_{out}$ 为总输出通道数），我们对测量结果应用一个可训练的线性变换（全连接层）。这一步实现了量子特征到经典特征空间的维度适配与语义对齐。

$$
\mathbf{z}_{m,g} = \mathbf{W}_{proj} \mathbf{v}_{m,g} + \mathbf{b}_{proj}, \quad \mathbf{z}_{m,g} \in \mathbb{R}^{C_{out}/G}
$$

其中 $\mathbf{W}_{proj} \in \mathbb{R}^{(C_{out}/G) \times N}$ 和 $\mathbf{b}_{proj} \in \mathbb{R}^{C_{out}/G}$ 分别为投影层的权重矩阵与偏置向量。

#### 4.Z.3 特征拼接与重组 (Concatenation & Spatial Folding)

最后，我们将所有 $G$ 个分组的投影特征在通道维度上进行**拼接（Concatenate）**，恢复出完整的通道数 $C_{out}$，从而得到第 $m$ 个 Patch 的完整特征向量 $\mathbf{z}_m$：

$$
\mathbf{z}_m = \text{Concat}(\mathbf{z}_{m,1}, \mathbf{z}_{m,2}, \dots, \mathbf{z}_{m,G}) \in \mathbb{R}^{C_{out}}
$$

随后，利用逆滑动窗口操作（Fold），将 $M$ 个一维特征向量按照其原始的空间位置索引重新排列回二维空间网格 $H \times W$，最终得到输出特征图张量 $\mathbf{O}$：

$$
\mathbf{O} = \text{Fold}(\{\mathbf{z}_m\}_{m=1}^M) \in \mathbb{R}^{B \times C_{out} \times H \times W}
$$

通过上述三个步骤，SOTA QCNN 完成了从“经典 Patch”到“量子态”再回归“经典特征图”的完整闭环，确保了量子模块能够无缝嵌入现有的深度学习架构中。
