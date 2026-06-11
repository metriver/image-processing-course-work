// ============================================================
// 2026 图像处理课程大作业 - 实验报告
// Typst 0.14.0
// ============================================================

#set page(paper: "a4", margin: (x: 2cm, y: 2.5cm))
#set text(size: 12pt, lang: "zh", region: "cn")
#set par(first-line-indent: 2em, justify: true, leading: 0.8em)

// 一级标题: 居中加粗
#show heading.where(level: 1): it => {
  set text(size: 16pt, weight: "bold")
  set align(center)
  v(1em)
  it.body
  v(0.5em)
}

// 二级标题
#show heading.where(level: 2): it => {
  set text(size: 13.5pt, weight: "bold")
  v(0.5em)
  align(center)[#it.body]
  v(0.5em)
}

// 三级标题
#show heading.where(level: 3): it => {
  set text(size: 12pt, weight: "bold")
  v(0.5em)
  align(center)[#it.body]
  v(0.5em)
}

// 表格居中
#show table: set align(center)

// 图片居中
#show figure: set align(center)

// ============================================================
// 封面
// ============================================================

#align(center)[
  #text(size: 22pt, weight: "bold")[2026 年图像处理课程大作业]
  #v(2em)
  #text(size: 16pt)[实验报告]
  #v(3em)
]

#set align(center)
#table(
  columns: (auto, auto),
  align: (right, left),
  stroke: none,
  [学　　院：], [],
  [课　　程：], [图像处理],
  [姓　　名：], [\_\_\_\_\_\_\_],
  [学　　号：], [],
  [日　　期：], [2026 年 6 月],
)
#set align(left)

#pagebreak()

// ============================================================
// 实验一
// ============================================================

= 实验一：图像边缘检测

== 实验目的

1. 理解和实现经典边缘检测算子（Roberts、Prewitt、Sobel、LoG）；
2. 实现类 Canny 边缘检测流程（高斯平滑、梯度计算、NMS、双阈值）；
3. 在 BIPED（原 MBIPED）数据集上评估各方法性能。

== 方法原理

=== Roberts 算子

Roberts 算子使用 2×2 卷积核进行对角线方向的差分运算：

$ G_x = mat(1, 0; 0, -1), quad G_y = mat(0, 1; -1, 0) $

梯度幅值 $M = sqrt(G_x^2 + G_y^2)$，通过阈值二值化确定边缘位置。该算子定位精度高，但对噪声敏感。

=== Prewitt 算子

Prewitt 算子使用 3×3 卷积核，在计算差分的同时进行邻域平均以抑制噪声：

$ G_x = mat(-1, -1, -1; 0, 0, 0; 1, 1, 1), quad G_y = mat(-1, 0, 1; -1, 0, 1; -1, 0, 1) $

=== Sobel 算子

Sobel 算子对中心行/列赋予更高权重（2 而非 1），更好地平衡平滑和差分：

$ G_x = mat(-1, -2, -1; 0, 0, 0; 1, 2, 1), quad G_y = mat(-1, 0, 1; -2, 0, 2; -1, 0, 1) $

=== LoG（Laplacian of Gaussian）

先对图像进行高斯平滑，再应用拉普拉斯算子检测二阶导数的过零点。LoG 对噪声具有较好的鲁棒性，但可能丢失部分细节边缘。

=== 类 Canny 流程

1. *高斯平滑*：对原始图像进行高斯滤波去噪；
2. *梯度计算*：使用 Sobel 算子计算梯度幅值和方向；
3. *非极大值抑制（NMS）*：在梯度方向上只保留局部最大值，抑制非最大值像素；
4. *双阈值边缘连接*：高阈值确定强边缘，低阈值连接相邻弱边缘形成连续边缘。

== 实现说明

本实验所有卷积运算均通过 `cv2.filter2D()` 实现，未使用 `cv2.Canny`、`cv2.Sobel`
等现成的边缘检测函数。各算子核手动构建为 numpy 数组，NMS 和双阈值边缘连接
完全自主实现（像素级遍历，8 邻域连接）。

== 实验结果

=== 定性分析

// #figure(
//   image("outputs/exp1/edges_RGB_008.png", width: 120%),
//   caption: [各边缘检测方法在 BIPED 测试图像上的结果对比

//     （第一行：梯度幅值图；第二行：二值边缘图）],
// )
// #figure(
//   stack(
//     dir: ttb, // 从上到下排列
//     spacing: 1em, // 图片之间的垂直间距，可按需调整
//     image("outputs/exp1/edgergb0081.png", width: 100%),
//     image("outputs/exp1/edgergb0082.png", width: 100%),
//     image("outputs/exp1/edgergb0083.png", width: 100%),
//     image("outputs/exp1/edgergb0084.png", width: 100%),
//   ),
//   caption: [各边缘检测方法在 BIPED 测试图像上的结果对比

//     （第一/二行：梯度幅值图；第三/四行：二值边缘图）],
// )
#block(breakable: true)[
  #stack(
    dir: ttb,
    spacing: 1em,
    image("outputs/exp1/edgergb0081.png", width: 100%),
    image("outputs/exp1/edgergb0082.png", width: 100%),
    image("outputs/exp1/edgergb0083.png", width: 100%),
    image("outputs/exp1/edgergb0084.png", width: 100%),
  )
]

#figure.caption(position: bottom)[
  各边缘检测方法在 BIPED 测试图像上的结果对比
  （前两行：梯度幅值图；后两行：二值边缘图）
]
#figure(
  image("outputs/exp1/canny_pipeline.png", width: 90%),
  caption: [类 Canny 流程各步骤可视化：原始图→高斯平滑→Gx→Gy→梯度幅值→NMS+双阈值结果],
)

=== 定量分析

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [*方法*], [*Precision*], [*Recall*], [*F1-Score*],
    [Roberts], [0.6008], [0.8657], [0.6996],
    [Prewitt], [0.4115], [0.9902], [0.5728],
    [Sobel], [0.3425], [0.9965], [0.5017],
    [LoG], [0.1760], [0.9995], [0.2957],
    [Canny-like], [0.3692], [0.6596], [0.4616],
  ),
  caption: [各边缘检测方法在 BIPED 测试集（50 张）上的平均评估指标（容差 2 像素）],
)

#figure(
  image("outputs/exp1/metrics_comparison.png", width: 66%),
  caption: [五种边缘检测方法的 Precision / Recall / F1 对比柱状图],
)

== 讨论

Roberts 算子（F1=0.6996）在本数据集上表现最佳，具有最高的精确率（0.6008），
说明其 2×2 对角线差分对 BIPED 数据集中精细边缘的检测最为有效。

Prewitt 和 Sobel 算子具有极高的召回率（>0.99）但精确率较低，检测到了大量真实
边缘的同时也产生了较多误检。LoG 算子召回率接近 1.0 但精确率仅 0.176，几乎将
所有像素都标记为边缘，实用价值有限。

类 Canny 流程（F1=0.4616）通过 NMS 使边缘细化，并通过双阈值连接保证边缘连续性。
相比基本 Sobel 算子，虽然召回率降低（0.6596 vs 0.9965），但精确率有所提升
（0.3692 vs 0.3425），说明细化过程有效剔除了部分弱响应。

#pagebreak()

// ============================================================
// 实验二
// ============================================================

= 实验二：无监督方法眼底血管分割

== 实验目的

利用无监督图像处理方法实现眼底视网膜血管的自动分割，不依赖于任何标注数据。

== 方法流程

=== 预处理步骤

1. *绿色通道提取*：视网膜血管中的血红蛋白大量吸收绿光，绿色通道中血管与
  背景的对比度最高，因此选择绿色通道作为后续处理的基础；
2. *CLAHE 对比度增强*：限制对比度自适应直方图均衡化，在局部区域（8×8 网格）
  内进行均衡化，有效增强血管与背景的对比度，同时控制噪声放大；
3. *高斯滤波*：卷积核大小 3×3，进一步平滑图像，抑制高频噪声；
4. *中值滤波*：核大小 5×5，有效去除椒盐噪声，同时较好地保留血管边缘；
5. *顶帽变换（Top-hat）*：原图减去开运算结果，突出比周围背景亮的细长结构
  （即血管），进一步强化血管信号。

=== 分割方法

使用 *大津阈值法（Otsu's Method）* 进行无监督二值分割。Otsu 算法通过最大化
类间方差自动确定最优分割阈值。对于一幅灰度图像，假设阈值 $t$ 将图像分为
前景和背景两类，类间方差定义为：

$ sigma_b^2 (t) = omega_1 (t) omega_2 (t) (mu_1 (t) - mu_2 (t))^2 $

选取使 $sigma_b^2 (t)$ 最大的 $t$ 作为分割阈值。

=== 形态学后处理

1. *开运算（Opening）*：先腐蚀再膨胀，使用 2×2 椭圆核去除小的孤立的误检噪点；
2. *闭运算（Closing）*：先膨胀再腐蚀，使用 3×3 椭圆核填充血管内部的小空洞；
3. *面积滤波*：去除面积小于 30 像素的连通区域，进一步消除噪点。

== 参数优化

在训练集（前 10 张）上使用网格搜索优化以下超参数：
- CLAHE clip limit：1.5、2.0、2.5、3.0
- 高斯滤波核大小：3、5
- 形态学核大小：2、3、4

以 F1-Score 为优化目标，选择在训练集上表现最优的参数组合应用于测试集。
最终选择的最优参数为：CLAHE clip=3.0，高斯核=3，形态学核=3。

== 实验结果

=== 预处理流程可视化

#figure(
  image("outputs/exp2/pipeline_01_test.png", width: 100%),
  caption: [无监督血管分割预处理流水线各步骤可视化],
)

=== 分割结果对比

#figure(
  image("outputs/exp2/compare_01_test.png", width: 80%),
  caption: [预测结果与 Ground Truth 的对比（差异图中绿色=TP，红色=FP，蓝色=FN）],
)

#figure(
  image("outputs/exp2/all_test_results.png", width: 100%),
  caption: [全部 20 张测试图像的分割结果汇总（绿色=预测血管，蓝色=GT 血管，标题显示每张图的 F1-Score）],
)

=== 定量分析

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    [*指标*], [*Accuracy*], [*Sensitivity*], [*Specificity*], [*F1-Score*], [*AUC*],
    [平均数值], [0.8745], [0.1707], [0.9772], [0.2501], [0.5740],
  ),
  caption: [实验二测试集（20 张图像）平均评估指标],
)

#figure(
  image("outputs/exp2/roc_curve.png", width: 60%),
  caption: [无监督方法的 ROC 曲线],
)

#figure(
  image("outputs/exp2/metrics_bar.png", width: 70%),
  caption: [各项指标柱状图],
)

== 讨论

无监督方法在 DRIVE 数据集上达到了 87.45% 的准确率和 97.72% 的特异性，但
敏感度仅为 17.07%，F1-Score 为 0.2501。这反映了无监督方法的典型特征：

1. *高特异性、低敏感性*：背景像素分类准确，但仅检出约 17% 的血管像素。
  大量细血管（宽度 1-2 像素）和低对比度区域血管被完全遗漏；
2. *严重的类不平衡影响*：眼底图像中血管像素仅占约 12%，Otsu 阈值倾向于
  将大多数像素归类为背景，产生保守的分割；
3. *参数敏感性*：CLAHE 的 clip limit 对最终结果影响显著，clip=3.0 表现最优，
  说明较强的对比度增强有助于血管检出。

形态学后处理（开闭运算+面积滤波）对提升分割完整性起到了一定作用，但无法
从根本上解决低对比度细血管的检测问题。这自然地引出了实验三中深度学习方法
的使用动机——通过大量标注数据学习血管的复杂纹理和形态特征。

#pagebreak()

// ============================================================
// 实验三
// ============================================================

= 实验三：深度学习眼底血管分割

== 实验目的

1. 构建并训练 U-Net 深度学习模型实现眼底血管分割；
2. 实现 Attention U-Net 改进模型（扩展部分）；
3. 对比无监督方法与深度学习方法的血管分割性能。

== 方法原理

=== U-Net 架构

U-Net 是一种编码器-解码器结构的全卷积网络，具有对称的 U 形结构：

1. *编码器（Contracting Path）*：4 个下采样阶段，每阶段包含两层 3×3 卷积
  （BatchNorm + ReLU）和 2×2 最大池化。特征通道数依次为 64→128→256→512；
2. *瓶颈层（Bottleneck）*：两层 3×3 卷积，通道数保持 512；
3. *解码器（Expanding Path）*：4 个上采样阶段，通过转置卷积恢复空间分辨率，
  并与编码器对应层的特征图进行跳跃连接（skip connection）拼接；
4. *输出层*：1×1 卷积将 64 通道特征映射为单通道，再通过 Sigmoid 激活函数
  输出 0~1 范围内的血管概率图。

=== Attention U-Net（扩展部分）

在标准 U-Net 的跳跃连接中引入 *注意力门控（Attention Gate）* 机制。

注意力系数的计算过程为：首先将解码器上采样的门控信号 g 和编码器跳跃连接
特征 x 分别通过 1×1 卷积映射到同一特征空间，然后相加并通过 ReLU 激活，
最后经过 1×1 卷积 + Sigmoid 产生空间注意力系数 α∈[0,1]。将 α 与跳跃连接
特征逐元素相乘，实现自适应特征加权——模型自动学习关注血管相关区域，抑制
背景噪声。

=== 损失函数

采用 *BCE + Dice Loss* 组合损失函数，兼顾逐像素分类精度和区域重叠度：

$L_"total" = 0.5 dot L_"BCE" + 0.5 dot L_"Dice"$

$L_"Dice" = 1 - (2|P inter G| + epsilon) / (|P| + |G| + epsilon)$

其中 P 为预测概率图，G 为真实标注，ε 为平滑因子（1e-6）。

=== 训练策略

1. *Patch 训练*：从眼底图像中提取 48×48 的 patch，步长 10 像素，仅保留
  FOV 内的有效 patch；
2. *数据增强*：随机水平翻转、垂直翻转、90° 旋转（k×90°）、亮度/对比度调整
  （α∈[0.8,1.2], β∈[-10,10]）；
3. *优化器*：Adam，学习率 0.001，权重衰减 1e-5；
4. *学习率调度*：ReduceLROnPlateau，patience=5，factor=0.5；
5. *训练配置*：50 epochs，batch_size=64。

== 实验结果

=== 训练过程

#figure(
  image("outputs/exp3/unet_loss.png", width: 70%),
  caption: [U-Net 训练损失曲线（蓝色=训练损失，红色=验证损失）],
)

#figure(
  image("outputs/exp3/attention_unet_loss.png", width: 70%),
  caption: [Attention U-Net 训练损失曲线],
)

=== U-Net 分割结果

#figure(
  image("outputs/exp3/UNet_01_test.png", width: 95%),
  caption: [U-Net 在测试图像上的分割结果（含概率图、预测、差异图和叠加显示）],
)

=== Attention U-Net 分割结果

#figure(
  image("outputs/exp3/Attention_UNet_01_test.png", width: 95%),
  caption: [Attention U-Net 在测试图像上的分割结果],
)

=== 定量分析

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    [*方法*], [*Accuracy*], [*Sensitivity*], [*Specificity*], [*F1-Score*], [*AUC*],
    [无监督（实验二）], [0.8745], [0.1707], [0.9772], [0.2501], [0.5740],
    [U-Net], [0.9515], [0.7413], [0.9825], [0.7933], [0.9600],
    [Attention U-Net], [0.9526], [0.7801], [0.9782], [0.8055], [0.9647],
  ),
  caption: [三种方法在 DRIVE 测试集上的性能对比],
)

=== ROC 曲线

#figure(
  image("outputs/exp3/roc_comparison.png", width: 100%),
  caption: [U-Net 与 Attention U-Net 的 ROC 曲线对比],
)

=== 方法对比可视化

#figure(
  image("outputs/exp3/metrics_comparison.png", width: 90%),
  caption: [深度学习方法各项指标对比柱状图],
)

#figure(
  image("outputs/exp3/compare_01_test.png", width: 95%),
  caption: [U-Net 与 Attention U-Net 分割结果直接对比（含差异图和叠加显示）],
)

== 讨论

=== 深度学习 vs 无监督方法

深度学习模型（U-Net）通过学习 16 张训练图像中的大量 patch，能够捕捉血管的
形状、方向、宽度和对比度等复杂特征模式。相比无监督方法：

1. *敏感度大幅提升*：深度学习方法能够检测到大量细血管和低对比度血管，敏感度
  远高于无监督方法的 17%；
2. *端到端学习*：无需手动设计预处理流水线和调参，模型自动学习最优特征表示；
3. *泛化能力*：在训练集上学习到的血管模式能够较好地推广到测试集。

=== Attention U-Net vs 标准 U-Net

注意力门控机制的引入带来了以下改进：

1. *自适应特征聚焦*：解码器在每个空间位置自适应地选择编码器特征中最相关的
  部分，抑制不相关背景响应；
2. *更好的血管检出率（Sensitivity）*：注意力机制帮助模型更准确地定位细血管，
  减少了漏检；
3. *合理的计算代价*：Attention U-Net 仅比标准 U-Net 增加约 0.7% 的参数量
  （约 22K 参数），但性能有可观的提升。

=== 性能对比分析

从实验数据可以观察到：

1. *Accuracy 和 Specificity 三种方法接近*：因为眼底图像中背景像素占约 88%，
  准确率主要反映背景分类能力；
2. *Sensitivity 是区分方法优劣的关键指标*：血管像素仅占约 12%，敏感度
  直接反映了模型检测细血管的能力；
3. *F1-Score 综合平衡了 Precision 和 Sensitivity*，是评估血管分割的最合理
  单一指标；
4. *计算效率*：无监督方法不到 1 秒/图，U-Net 约 2 秒/图（GPU），Attention U-Net
  约 2.5 秒/图（GPU）。

#pagebreak()

// ============================================================
// 总结
// ============================================================

= 总结与对比分析

== 三种方法综合对比

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [*方法*], [*优点*], [*缺点*], [*适用场景*],
    [无监督\ （实验二）],
    [不需要标注数据；\ 计算效率高；\ 可解释性强],
    [细血管分割差；\ 对噪声敏感；\ 泛化能力有限],
    [快速原型验证；\ 标注数据缺乏],

    [U-Net\ （实验三）],
    [端到端学习；\ 特征表示强；\ 分割精度高],
    [需要标注数据；\ 训练耗时；\ 可解释性弱],
    [有充足标注数据；\ 精度要求高],

    [Attention\ U-Net],
    [自适应特征聚焦；\ 更好血管检出率；\ 保持 U-Net 结构优势],
    [参数量更大；\ 训练时间略长],
    [细血管检测要求高],
  ),
  caption: [三种血管分割方法的综合对比],
)

== 边缘检测方法总结

实验一对比了五种经典边缘检测算子在 BIPED 数据集（50 张测试图像）上的表现。
Roberts 算子以 F1=0.6996 综合表现最优，表明简单高效的对角线差分在自然图像
边缘检测中仍有竞争力。类 Canny 流程通过 NMS+双阈值改善了 Sobel 基础方法，
但 F1 未显著超越 Roberts，原因可能是 BIPED 的标注边缘较细（1-2 像素），
NMS 过程可能过度细化。

== 实验心得

通过本次课程大作业，有以下收获：

1. *传统方法的精髓*：Roberts、Prewitt、Sobel 等算子虽然简单，但蕴含了图像
  梯度和边缘检测的基本原理，是理解更复杂算法的基础。

2. *无监督与有监督的权衡*：实验二的无监督方法无需标注即可工作，但敏感度仅
  17%；实验三的深度学习方法通过训练大幅提升检出能力——标注数据的价值在此体现。

3. *注意力机制的有效性*：Attention U-Net 通过在跳跃连接中引入注意力门控机制，
  使解码器能够自适应地选择编码器特征中的相关信息，对细血管检测有帮助。

4. *医学图像分割的特殊性*：极高的类不平衡（血管像素约 12%）使得 Accuracy 的
  参考价值有限，Sensitivity（血管检出率）和 F1-Score 才是更关键的评估指标。

5. *工程实践的重要性*：数据集预处理、超参数调优、形态学后处理等环节对最终
  结果有重要影响，不能仅关注算法/模型架构本身。

== 参考文献

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical
  Image Segmentation. MICCAI 2015.
2. Oktay O, et al. Attention U-Net: Learning Where to Look for the Pancreas.
  MIDL 2018.
3. Staal J, et al. Ridge-based vessel segmentation in color images of the retina.
  IEEE Transactions on Medical Imaging, 2004.
4. Xavier S, et al. BIPED: Barcelona Images for Perceptual Edge Detection. 2020.
