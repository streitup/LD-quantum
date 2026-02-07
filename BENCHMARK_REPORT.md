# 量子扩散模型架构对比实验报告 (Benchmark Analysis)

## 1. 实验背景与目标
本实验旨在探究在扩散模型（Diffusion Models）的核心组件——U-Net残差块中，如何高效地引入量子计算模块（VQC/QCNN）以提升模型的表达能力或参数效率。我们设计了 "Benchmark 6" 对比实验，针对三种不同的架构方案进行“时空特征调制”任务的性能评估。

**任务定义**:
学习一个高度非线性的时空调制函数：
$$Target = Roll(h \cdot style + \cos(h \cdot style))$$
其中 $h = \sin(x)$，$x$ 为图像特征，$style$ 为时间步/条件嵌入。该任务要求模型同时具备强大的特征提取能力（处理 $x$）和灵活的信号调制能力（处理 $style$ 对 $x$ 的影响）。

---

## 2. 对比架构详解

### 2.1 Classic Ref (经典参考架构)
标准的经典 U-Net 残差块，作为性能基准（Baseline）。
*   **核心机制**: FiLM (Feature-wise Linear Modulation)。
*   **参数量**: ~427k
*   **算法流程**:
    1.  **Input**: $x \in \mathbb{R}^{B \times C \times H \times W}$, $emb \in \mathbb{R}^{B \times D}$。
    2.  **Conv Block 1**: `Norm -> SiLU -> Conv2d` 提取特征得到 $x_1$。
    3.  **Affine Modulation**:
        *   通过线性层将 $emb$ 映射为 $scale$ 和 $shift$。
        *   公式: $x_{mod} = \text{Norm}(x_1) \cdot (1 + scale) + shift$。
    4.  **Conv Block 2**: `SiLU -> Conv2d` 处理调制后的特征。
    5.  **Output**: $x_{input} + x_{out}$。

### 2.2 Separated Quantum (分离式量子架构 - 推荐)
采用 **"量子特征提取 + 经典信号调制"** 的混合设计（Q-C-Q Sandwich Structure）。
*   **核心机制**: 利用量子层进行高维特征映射，利用经典层进行幅度调制。
*   **参数量**: ~433k (与经典持平)
*   **算法流程**:
    1.  **QConv0 (量子前端)**:
        *   **分组 (Grouping)**: 将 $C$ 通道分为 $G$ 组（如128通道分8组）。
        *   **量子编码**: 对每组 $3 \times 3$ patch 进行振幅编码（Amplitude Encoding）映射到希尔伯特空间。
        *   **酉变换**: 执行参数化量子线路（PQC）进行特征提取。
        *   **测量**: 得到量子特征 $x_{q0}$。
    2.  **Classic Modulation**:
        *   与经典架构完全相同，使用 `Linear` 层生成 $scale, shift$。
        *   公式: $x_{mid} = \text{Norm}(x_{q0}) \cdot (1 + scale) + shift$。
    3.  **QConv1 (量子后端)**:
        *   再次经过量子卷积层处理 $x_{mid}$ 得到 $x_{q1}$。
    4.  **Output**: $x_{input} + x_{q1}$。
*   **优势**: 完美结合了量子计算的强非线性（特征空间）和经典计算的数值缩放能力（幅度空间）。

### 2.3 Integrated Quantum (集成式量子架构)
尝试完全在量子线路内部完成特征融合，去除显式的经典乘法调制。
*   **核心机制**: 数据重上传 (Data Re-uploading) 与 参数调制。
*   **参数量**: ~155k (仅为经典的 36%)
*   **算法流程**:
    1.  **Deep QCNN**: 使用4层深度的量子线路。
    2.  **Style Injection**:
        *   将 $style$ 向量映射为旋转角度 $\theta_{style}$。
        *   将 $\theta_{style}$ 作为旋转门参数直接注入量子线路。
    3.  **量子演化**:
        *   $|\psi_{out}\rangle = U(\theta_{trainable}) \cdot U(\theta_{style}) \cdot U(x_{data}) \cdot |0\rangle$。
    4.  **Output**: 测量期望值并映射回通道空间。
*   **局限**: 量子酉变换本质上是保范数的（Norm-preserving），难以模拟大幅度的数值缩放（Amplitude Scaling），导致拟合复杂调制任务较难。

---

## 3. 实验结果与分析

| 模型架构 | 初始 Loss | 最终 Loss (200 steps) | 参数量 | 评价 |
| :--- | :--- | :--- | :--- | :--- |
| **Classic Ref** | 2.18957 | 0.02238 | 427k | 基准水平，收敛平稳 |
| **Separated (Q-C-Q)** | 2.19203 | **0.00075** | 433k | **SOTA**，误差比经典低一个数量级 |
| **Integrated (Pure Q)**| 2.29448 | 0.01417 | **155k** | 参数效率极高，性能优于经典但弱于分离式 |

### 3.1 核心发现
1.  **分离式架构的优越性**: 实验数据表明，`Separated` 架构的最终 Loss (0.00075) 远低于经典架构 (0.02238)。这证明了在保留经典调制层（处理幅度信息）的基础上，引入量子卷积层（处理相位/特征空间信息）能显著提升模型的表达能力。
2.  **纯量子模型的瓶颈**: 
    *   在 Benchmark 1 (Pure Quantum Analysis) 中，我们发现完全去除经典线性层（Linear Projection）的纯量子模型无法拟合目标函数 (Loss ~0.48)。
    *   原因在于 **振幅编码 (Amplitude Encoding)** 强制归一化输入向量，导致原始数据的**模长（Magnitude）信息丢失**。
    *   **结论**: 混合架构（Hybrid）是必须的。必须保留经典的 Input Scaling 和 Output Projection 层来处理数据的数值范围。

### 3.2 模块输入输出尺寸 (以 $C=128, H=16, W=16$ 为例)

1.  **分组处理 (Grouped Processing)**:
    *   输入: $[B, 128, 16, 16]$
    *   分组: $[B, 8 \text{ groups}, 16 \text{ ch}, 16, 16]$
    *   Unfold (3x3): 每个 Patch 大小为 $16 \times 3 \times 3 = 144$。
2.  **量子映射**:
    *   Data Projection: $144 \to 16$ (映射到 4 qubits, $2^4=16$)。
    *   Quantum State: $16$ 维复数向量。
    *   Measurement: $16$ 维实数概率分布。
    *   Output Projection: $16 \to 16$ (映射回通道组)。
3.  **重组**:
    *   $[B, 8, 16, 16, 16] \to [B, 128, 16, 16]$。

## 4. 结论与建议
针对本项目中的量子扩散模型改进，建议采用 **Separated Quantum Architecture**：
1.  保留 U-Net 中的经典 GroupNorm 和 FiLM (Affine) 层。
2.  将标准的 Conv2d 替换为 **Grouped Quantum Conv (QConv)**。
3.  保持输入输出的经典线性投影层，以规避量子态归一化带来的信息丢失问题。
