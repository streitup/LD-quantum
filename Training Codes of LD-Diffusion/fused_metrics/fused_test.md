实验三：能在不测量的情况下把量子卷积和量子注意力的算法连接起来吗？
根据当前的qcnn算法，每个量子线路所编码的数据是$$X \in \mathbb{R}^{B \times C \times H \times W}$$当中$$x_{patch}\in \mathbb{R}^{G\times p \times p}$$通过线性层映射的数据，那么如果要让qcnn算法和q-attn算法全量子化，那么直接将q-attn计算的token也确定为$$x_{patch}\in \mathbb{R}^{G\times p \times p}$$即可。因此，可以得到All-qu算法的架构为：创造三个用于qkv计算的量子线路，在conv部分共享旋转门参数$$\theta_{conv}$$，进行(L=8)层HEA的计算，得到三个相同的量子态$$|\psi_{conv}\rangle$$,随后的l=8层的HEA的参数$$\theta_{attn} = ((\theta_{q}^{0},\theta_{k}^{0},\theta_{v}^{0}),(\theta_{q}^{1},\theta_{k}^{1},\theta_{v}^{1}),...,(\theta_{q}^{\frac{C}{G}},\theta_{k}^{\frac{C}{G}},\theta_{v}^{\frac{C}{G}}))$$则独立进行参数的训练，最后分别进行测量，计算q,k向量的点积距离和v向量进行注意力的矩阵运算。这实质上是在原本的注意力计算的token单位$$x_{token}\in \mathbb{R}^{C\times p \times p}$$又做了一次通道维度的分组。
实验对比算法如下：
- 算法一：经典Unet架构，卷积和注意力均使用经典算法，称为Classical baseline
- 算法二：量子卷积+测量+MLP线性层+Silu激活函数+量子注意力，称为Q-Hybrid-baseline
- 算法三：量子卷积+测量+linear线性层+量子注意力，称为Q-Hybrid-Lite
- 算法四：量子卷积+量子注意力（无中途测量），称为Q-Pure-Fused
实验通过加载100-shot-obama数据集，首先通过随机噪声打乱特征图，输入模型中，通过100个epoch的训练让模型恢复出原始图像，并评估恢复图像$$\hat x$$与原始图像$$x$$之间的mse-loss，来快速评估其在扩散模型去噪网络中的特征提取和恢复的能力。
| 算法 | Loss (MSE) | 参数量 |
| :--- | :--- | :--- |
| Classical baseline | 0.7736 | 378,240 |
| Q-Hybrid-baseline | 0.8543 | 392,429 |
| Q-Hybrid-Lite | 0.7835 | 375,917 |
| Q-Pure-Fused | 0.9300 | 268,971 |可以看出，效果最好的是Q-Hybrid-baseline，其将时间嵌入+卷积特征提取之后的特征，使用线性层+激活函数进行经典的特征重塑，随后编码进量子注意力的线路当中进行注意力的计算，有效地利用了两个量子模块，并使用经典线性层+激活函数，使得梯度反传更加流畅，取得了相较经典网络更好的参数量和loss。然而，我们可以发现，使用纯量子线路却起到了反作用：量子线路的深度过深，从而导致梯度反传和训练极为困难，因此我们可以得到一个初步的结论：在不同的量子模块之间，进行经典的激活函数进行特征重塑，有利于训练过程中梯度的反传，能够有效利用量子模块的特征提取能力，又避免了量子算法的贫瘠高原问题。
- 结论：使用Q-Hybrid-baseline算法，替换经典baseline架构，能够取得最好的效果。