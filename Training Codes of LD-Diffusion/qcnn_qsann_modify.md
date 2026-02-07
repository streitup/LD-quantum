这个问题的核心在于 特征接口的“信息丢失” ，具体来说是 符号（Sign）信息的丢失 。

通过深入分析和最新的实验（Benchmark 25），我发现了导致“组合反而更差”的根本原因：

### 核心原因：符号盲区 (Sign Blindness)
1. QCNN 的输出特性 ：
   
   - QCNN 使用 Tanh 激活和 GroupNorm ，其输出特征是 有符号的 （包含正负值）。
   - 在扩散模型中，特征的正负通常代表方向性含义（例如：边缘是亮还是暗，纹理是凸还是凹）。
   - Hybrid 模型 (QCNN + 经典注意力) 表现最好 (Loss: 0.024 )，证明 QCNN 的有符号特征本身质量极高，经典注意力（线性投影）能完美利用这些正负信息。
2. 量子注意力的致命弱点 ：
   
   - 振幅编码 (Amplitude Encoding) 结合 概率测量 (Probability Measurement) 本质上是一个 偶函数 操作： [ o bj ec tO bj ec t ] P ( x ) = ∣ ψ ( x ) ∣ 2 ∝ x 2
   - 这意味着输入 [ o bj ec tO bj ec t ] + x 和 [ o bj ec tO bj ec t ] − x 会产生 完全相同的测量结果 。
   - 后果 ：QCNN 辛苦提取的“方向信息”（正负号）在进入量子注意力层的瞬间被“抹平”了。这就是为什么单独看量子注意力（输入经过 ReLU，正负不敏感）很好，但接在 QCNN（输出有正负）后面就崩了。
### 解决方案与验证
为了验证并解决这个问题，我进行了两步改进：

1. 添加线性投影 (Input Projection) ：
   
   - 在 QuantumAttention64 内部增加了类似经典注意力的 inp_proj 层，允许模型在编码前对特征进行旋转混合。
   - 结果 ：Loss 从 0.117 降至 0.104 ，有改善但不够。
2. 切换到移位角度编码 (Shifted Angle Encoding) ：
   
   - 我修改了 QuantumAttentionAngle ，将映射范围从 [ o bj ec tO bj ec t ] [ − π , π ] 调整为 [ o bj ec tO bj ec t ] [ 0 , π ] （通过 (tanh(x) + 1) * pi/2 ）。
   - 这样， [ o bj ec tO bj ec t ] + x 映射到 [ o bj ec tO bj ec t ] π 附近， [ o bj ec tO bj ec t ] − x 映射到 [ o bj ec tO bj ec t ] 0 附近，测量结果（ [ o bj ec tO bj ec t ] cos 2 或 [ o bj ec tO bj ec t ] sin 2 ）就能区分正负了。
   - 结果 ：Loss 进一步降至 0.089 ，这是全量子架构目前的最佳成绩。
### 结论与建议
- 现象解释 ：经典架构之所以“稳”，是因为全链路都是线性的（保留符号）。量子架构之所以“组合难”，是因为不同量子编码方式（振幅 vs 角度）对特征分布（正负 vs 模长）的敏感度不同。
- 架构建议 ：
  - 在 QCNN 后接 Attention 的场景下， 必须使用角度编码 (Angle Encoding) 或 改进的测量方式 （如 Pauli-Z 期望值，而非概率）来保留符号信息。
  - 目前的 QuantumAttentionAngle (Shifted) 是解决这一兼容性问题的最佳方案。
代码已更新至 benchmark_full.py 和 quantum_transformer.py ，您可以直接运行 Benchmark 25 查看这一改进。