# SOTA Quantum CNN (QCNN) 算法详解

本文档详细描述了当前项目中最先进的量子卷积神经网络（SOTA QCNN, `QuantumFrontEndQCNN`）的算法流程、数学原理及输入输出示例。该架构专为扩散模型的潜空间（Latent Space）特征提取而设计，结合了量子纠缠、时间嵌入调制（Time Embedding Modulation）与混合编码策略。

## 1. 算法核心架构

QCNN 模块作为经典卷积层（Conv2d）的替代品，通过将图像局部特征映射到量子态，利用量子电路的并行性与纠缠特性提取特征。

**核心组件：**
1.  **预处理 (Pre-processing)**: `Unfold` 提取图像 Patch，并进行分组（Grouping）。
2.  **混合编码 (Hybrid Encoding)**: 支持幅度编码（Amplitude Encoding）与角度编码（Angle Encoding），并结合数据重上传（Data Re-uploading）。
3.  **量子-经典融合 (Fusion/Modulation)**: 利用辅助量子比特（Ancilla Qubits）或参数调制注入时间步（Time Step）信息。
4.  **变分量子电路 (PQC Backbone)**: 多层结构，包含参数化旋转门（RY, RZ）与纠缠门（CNOT Ring）。
5.  **测量与读出 (Measurement & Readout)**: 可训练测量基下的概率测量，并投影回经典通道维度。
6.  **残差连接 (Residuals)**: 包含经典线性/MLP 残差与强卷积旁路（Strong Bypass）。

---

## 2. 详细算法流程与公式

### 步骤 1: 输入预处理与分块 (Patch Extraction)

输入张量 $X \in \mathbb{R}^{B \times C \times H \times W}$。
设定卷积核大小 $K \times K$（通常 $3 \times 3$），步长 $S$，填充 $P$。

利用 `Unfold` 操作提取 $L$ 个局部 Patch：
$$ X_{patch} = \text{Unfold}(X) \in \mathbb{R}^{B \times (C \cdot K^2) \times L} $$
其中 $L = H_{out} \times W_{out}$。

为了适应多量子处理器或逻辑分组，将通道 $C$ 分为 $G$ 组：
$$ X_{grouped} \in \mathbb{R}^{B \cdot L \times G \times D_{sub}} $$
其中 $D_{sub} = \frac{C}{G} \cdot K^2$ 是每组的特征维度。

### 步骤 2: 量子态编码 (Quantum Encoding)

针对每个分组特征向量 $v \in \mathbb{R}^{D_{sub}}$，映射到 $N$ 个数据量子比特。

#### 方案 A: 幅度编码 (Amplitude Encoding) - *高容量*
1.  **投影**: $v' = W_{enc} v$, 其中 $v' \in \mathbb{R}^{2^N}$。
2.  **归一化**: $\hat{v} = \frac{v'}{\|v'\|_2}$。
3.  **态制备**:
    $$ |\psi_{data}\rangle = \sum_{i=0}^{2^N-1} \hat{v}_i |i\rangle $$

#### 方案 B: 角度编码 (Angle/Tanh Encoding) - *高非线性*
1.  **投影**: $\theta = \pi \cdot \tanh(W_{enc} v)$, 其中 $\theta \in \mathbb{R}^{N}$。
2.  **态制备**:
    $$ |\psi_{data}\rangle = \bigotimes_{j=0}^{N-1} R_y(\theta_j) |0\rangle $$

### 步骤 3: 时间嵌入注入 (Time Embedding Injection)

扩散模型依赖时间步 $t$。时间向量 $style \in \mathbb{R}^{D_{style}}$ 通过以下方式注入：

1.  **富参数注入 (Rich Injection)**:
    将 $style$ 映射为 PQC 中的旋转参数增量 $\Delta \theta$。
    $$ \theta_{PQC}(t) = \theta_{base} + W_{style} \cdot style $$
2.  **辅助比特调制 (Ancilla Modulation)**:
    引入 $N_{anc}$ 个辅助比特，其状态由时间嵌入控制（或来自 QMLP）。
    通过受控旋转门（CRX/CRZ）与数据比特纠缠：
    $$ U_{mod} = \prod_{j=0}^{N-1} CR(\text{control}=j\%N_{anc}, \text{target}=j, \theta=f(style)_j) $$

### 步骤 4: 变分量子电路演化 (PQC Evolution)

电路包含 $L_{depth}$ 层。第 $l$ 层的演化 $U_l$ 包含：

1.  **数据重上传 (Re-uploading)** (可选):
    再次注入输入特征或时间特征，增强非线性。
    $$ |\psi\rangle \leftarrow \left( \bigotimes_{j} R_z(\theta_{input, j} + \theta_{style, j}) \right) |\psi\rangle $$

2.  **参数化旋转 (Local Rotations)**:
    $$ U_{rot}^{(l)} = \bigotimes_{j=0}^{N-1} R_z(\phi_{l,j}) R_y(\theta_{l,j}) $$

3.  **纠缠层 (Entanglement)**:
    使用环形 CNOT 连接（Ring Topology）：
    $$ U_{ent}^{(l)} = \prod_{j=0}^{N-1} \text{CNOT}(j, (j+1) \pmod N) $$
    *可选：跨步 CNOT (Strided CNOT) 连接 $j$ 与 $(j+2)$。*

总演化：
$$ |\psi_{final}\rangle = \left( \prod_{l=1}^{L_{depth}} U_{ent}^{(l)} U_{rot}^{(l)} U_{mod}^{(l)} \right) |\psi_{data}\rangle $$

### 步骤 5: 测量与读出 (Measurement & Readout)

1.  **可训练测量基**:
    在测量前施加通用单比特旋转 $U3$：
    $$ |\psi_{meas}\rangle = \left( \bigotimes_{j=0}^{N-1} U3(\omega_j) \right) |\psi_{final}\rangle $$

2.  **概率测量**:
    测量计算基下的概率分布 $p \in \mathbb{R}^{2^N}$：
    $$ p_k = |\langle k | \psi_{meas} \rangle|^2, \quad k \in \{0, \dots, 2^N-1\} $$

3.  **经典投影**:
    将高维概率向量映射回通道维度：
    $$ y_{quant} = W_{out} p + b_{out} $$

### 步骤 6: 后处理与融合

1.  **残差连接**:
    $$ y = y_{quant} + \text{Residual}(v) $$
    其中 Residual 可以是线性层或 MLP。

2.  **重组 (Fold/Reshape)**:
    将处理后的 Patch 向量 $y \in \mathbb{R}^{B \cdot L \times C}$ 重组为图像张量：
    $$ Y \in \mathbb{R}^{B \times C \times H_{out} \times W_{out}} $$

3.  **强旁路 (Strong Bypass)** (可选):
    $$ Y_{final} = Y + \text{Conv2d}_{classic}(X) $$

---

## 3. 输入输出示例

假设配置：
- 输入: `[B=2, C=4, H=32, W=32]` (Latent Feature)
- Kernel: 3x3, Stride: 2, Padding: 1
- Group: 1
- N_qubits: 6 (Data) + 2 (Ancilla)
- Encoding: Amplitude

**流程数据流：**

1.  **Input**:
    Tensor `[2, 4, 32, 32]`

2.  **Unfold**:
    Patch Size = $4 \times 3 \times 3 = 36$。
    $H_{out} = W_{out} = 16$。
    $L = 256$。
    Output: `[2, 36, 256]` -> Flatten -> `[512, 36]` (Batch of Patches).

3.  **Encoding (Amplitude)**:
    Project `36` -> `2^6 = 64`。
    Normalize -> State Vector `[512, 64]` (Complex).
    Pad with Ancilla -> `[512, 2^8 = 256]` (Complex).

4.  **Quantum Processing**:
    Input State: `[512, 256]`
    Operations: $U_{PQC}$ applied to 8 qubits.
    Output State: `[512, 256]`

5.  **Measurement**:
    Probabilities of Data Qubits (tracing out ancilla or measuring all? Code measures all or data).
    假设测量 Data Qubits (6个): Output Probabilities `[512, 64]`。

6.  **Projection**:
    Linear `64 -> 4` (Output Channels).
    Output: `[512, 4]`.

7.  **Reshape**:
    `[512, 4]` -> `[2, 256, 4]` -> Transpose -> `[2, 4, 256]`.
    View as Image -> `[2, 4, 16, 16]`.

8.  **Output**:
    Tensor `[2, 4, 16, 16]`.

## 4. 关键代码接口

```python
# 初始化
qcnn = QuantumFrontEndQCNN(
    channels=4, 
    style_dim=128, 
    n_qubits_data=6, 
    n_qubits_ancilla=2,
    n_layers=4,             # 增强深度
    encoding_type='amplitude'
)

# 前向传播
# x: [B, 4, 32, 32]
# t_emb: [B, 128]
output = qcnn(x, t_emb) 
# output: [B, 4, 16, 16] (若 stride=2)
```
