# 实验结果数据表汇总

以下表格整理自实验结果章节，格式已调整，可直接复制用于文档撰写。

### 表 5-1：100-shot-Grumpy-Cat 数据集上的性能对比

| 方法 | FID $\downarrow$ | Accuracy (%) $\uparrow$ | Recall (%) $\uparrow$ | Params (MB) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| DDPM (DiT) [1] | 347.8 | / | / | 120 |
| Patch-Diffusion [2] | 45.1 | 85.6 | 46.0 | 506 |
| **LD-Diffusion (SOTA)** [3] | 16.5 | 99.0 | 33.0 | 514 |
| EDM+DA [4] | 41.49 | 89.7 | 30.0 | 634 |
| ADM [5] | 64.78 | 78.71 | 27.0 | 3014 |
| **Qattn-QDM (Ours)** | **14.9** | **98.5** | **35.0** | **298** |

### 表 5-2：100-shot-Obama 数据集上的性能对比

| 方法 | FID $\downarrow$ | Accuracy (%) $\uparrow$ | Recall (%) $\uparrow$ | Params (MB) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| Patch-Diffusion | 43.79 | 86.9 | 49.0 | 506 |
| LD-Diffusion | 16.8 | 97.0 | 31.0 | 514 |
| EDM+DA | 40.45 | 90.4 | 54.0 | 634 |
| **Qattn-QDM (Ours)** | **15.2** | **97.8** | **36.0** | **298** |

### 表 5-3：不同训练步数下的 FID 收敛曲线数据 (图 5-1 数据源)

| 方法 / FID | 1000 kimg | 2000 kimg | 4000 kimg | 20000 kimg | Params |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LD-Diffusion | 67.29 | 27.03 | 24.45 | 16.8 | 531M |
| **Qattn-QDM (Base)** | 64.8 | 25.32 | 20.3 | 15.2 | 519M |
| **Qattn-QDM (+QTime)** | 64.1 | 26.17 | 20.9 | 15.8 | 508M |
| **Qattn-QDM (+QConv)** | 66.8 | 28.49 | 23.75 | / | 501M |

### 表 5-4：100-shot-Panda 数据集上的量子模型对比

| 方法 | FID $\downarrow$ | Accuracy (%) $\uparrow$ | Recall (%) $\uparrow$ | Gen Time (s) $\downarrow$ | Model Size (MB) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qconv-Unet [1] | 118.79 | 1.2 | 0.0 | 230 | 16.68 |
| Qlatent-Diffusion [2] | 173.05 | 3.8 | 0.0 | 421 | 1.01 |
| All-VQC-Diffusion [3] | 556.00 | 0.0 | 0.0 | 125 | 0.003 |
| Quantum Gen Diffusion [4]| 438.23 | 0.0 | 0.0 | 134 | 0.28 |
| **Qattn-QDM (Ours)** | **87.22** | **92.8** | **44.8** | **TBD** | **TBD** |

### 表 5-5：量子注意力机制消融实验

| 方法 | FID $\downarrow$ | Accuracy (%) $\uparrow$ | Recall (%) $\uparrow$ | Params (MB) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| QSANN (Original) | 较高 | 较低 | 较低 | 86.69C + 0.53Q |
| **Qattn-QDM (Improved)**| **87.22** | **92.78** | **44.82** | **86.69C + 0.53Q** |

### 表 5-6：量子时间嵌入消融实验

| 方法 | FID | Accuracy | Recall | Params |
| :--- | :--- | :--- | :--- | :--- |
| No-Time-Embed (Classic) | 87.22 | 92.78% | 44.82% | 基准 |
| **Q-Time-Embed** | **86.79** | 90.49% | 42.48% | +0.4MB (Q) |

### 表 5-7：仿射调制位置对比消融实验

| 调制位置 | Loss $\downarrow$ | SSIM $\uparrow$ |
| :--- | :---: | :---: |
| Q-Head-Affine | 0.020485 | 0.8104 |
| Q-Tail-Affine | 0.018050 | 0.8204 |
| Q-Middle-Affine | 0.017556 | 0.8271 |
| **C-Tail-Affine (Ours)**| **0.013285** | **0.8556** |
