# 论文来源与工程转译

本项目围绕商品身份保真、阴影几何、透明材质和环境融合开展方法调研。这里列出与当前设计直接相关的研究，以及从论文转译到项目时保留和舍弃的部分。

“参考”表示吸收问题分解、中间表示或约束思路；它不表示已复现论文网络、训练流程或达到论文报告指标。当前核心实现是现有图像算法、确定性合成、材质／身份路由与受限候选恢复的组合。算法说明见 [ALGORITHMS.md](ALGORITHMS.md)。

## 1. Pixel Height Maps

**Controllable Shadow Generation Using Pixel Height Maps**，ECCV 2022。

- [作者项目页](https://shengcn.github.io/SSG/)
- [项目中的光影研究摘要](REALISM_RESEARCH.md)

论文将物体、地面与相机之间的关系表达为像素高度，再利用投影几何控制硬阴影方向和形状，并通过学习方法产生软阴影。

项目吸收的是“接触像素不应产生脱离地面的位移，高处像素对应更远投影”的几何思路。工程实现依赖二维 Alpha、接触轮廓、光向和支持面，使用伪高度近似；脚点、统一基线和多脚接触需要结合商品姿态选择。

**转译边界：**没有证据表明本仓库训练了论文的高度预测或软阴影网络。二维轮廓高度也不等于真实三维几何。投影公式是否合理与脚点是否正确，是两个需要分别检查的问题。

## 2. PixHt-Lab

**PixHt-Lab: Pixel Height Based Light Effect Generation for Image Compositing**，CVPR 2023。

- [作者项目页](https://shengcn.github.io/PixHtLab/)

论文将像素高度显式映射到三维空间，用重建几何和材质处理软阴影及不同光泽度反射，并使用具有三维信息的缓冲通道指导神经渲染。

对项目的启发是把接触、支持面、阴影、反射与材质作为相关因素处理，而不把所有效果压进一个统一模糊图层。项目中的倒影采用屏幕空间逆采样、粗糙度相关模糊、距离衰减和支持面裁剪等近似。

**转译边界：**本项目没有完整复现 PixHt-Lab 的几何重建和神经渲染系统。屏幕空间倒影是可控视觉线索，不能等同于具有真实几何和材质的物理反射。

## 3. TOM-Net

**TOM-Net: Learning Transparent Object Matting From a Single Image**，CVPR 2018。

- [CVF 论文页面](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_TOM-Net_Learning_Transparent_CVPR_2018_paper.html)
- [作者代码仓库](https://github.com/guanyingc/TOM-Net)

TOM-Net 将透明物体抠图表述为折射流估计问题，输出物体 mask、衰减 mask 和折射流。这个表示指出：透明物体的观测颜色已经混有源背景，仅用普通 RGB 与 Alpha 不能充分描述它在新环境中的外观。

项目使用显式 `OpticalMatte` 表示覆盖率、透射颜色、二维折射位移和反射／高光代理，并在合成时读取目标背景。这与“表示物体如何作用于环境”的研究方向一致。

**转译边界：**当前 OpticalMatte 主要使用图像证据和启发式近似，没有直接部署或重新训练 TOM-Net。不能将启发式折射位移称为已经准确恢复的真实折射流。

## 4. Environment Matting

**Environment Matting and Compositing**，SIGGRAPH 1999。

- [华盛顿大学项目页](https://grail.cs.washington.edu/projects/digital-matting/envmatte/)
- [作者提供的论文 PDF](https://grail.cs.washington.edu/projects/digital-matting/papers/sig1999.pdf)

这项研究将传统 opacity matte 扩展为物体对环境光的反射、折射、衰减和散射描述，使物体放入新环境时能够采用新场景的光线信息。

它帮助解释项目中的根本矛盾：商品身份需要保留，但透明物体外观不能把旧环境逐像素原样搬入。项目因此将身份、结构与环境相关光学分开，再分别选择硬恢复和受限外观修改。

**转译边界：**论文研究包含特定采集与环境测量过程；本项目使用普通单张商品照片，缺少同等信息。当前表示只能提供受约束的外观近似，不能恢复任意未知环境下的全部成像规律。

## 5. SSN

**SSN: Soft Shadow Network for Image Compositing**，CVPR 2021。

- [CVF 论文页面](https://openaccess.thecvf.com/content/CVPR2021/html/Sheng_SSN_Soft_Shadow_Network_for_Image_Compositing_CVPR_2021_paper.html)
- [作者预印本](https://arxiv.org/abs/2007.08211)

SSN 以二维物体 mask 和环境光信息生成可控软阴影，并使用中间 AO 信息提供几何线索。

项目从这一类研究中关注接触遮蔽与方向投影的区分：接触 AO 负责底部贴合，方向投影表达共同光向，半影描述软化程度。将它们拆开后，可以分别定位悬浮、方向错误、过黑与过度模糊。

**转译边界：**当前确定性 AO、轮廓投影和 Gaussian 软化不是 SSN 网络的完整复现。引用论文用于说明方法背景，不能用其用户研究或质量指标替代本项目评估。

## 6. 线性颜色空间中的图像协调

**Learning Image Harmonization in the Linear Color Space**，ICCV 2023。

- [CVF 论文页面](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Learning_Image_Harmonization_in_the_Linear_Color_Space_ICCV_2023_paper.html)

论文从相机图像形成过程出发，学习将颜色反处理到更接近线性场景辐射量的中间表示，再进行协调并映射回输出颜色空间。

对项目的启发是：编码 RGB 的加减乘和物理光能运算不是同一件事。透明光学与受限重光照分支需要明确颜色空间，原始细节与低频外观也应分开处理。

**转译边界：**将 sRGB 解码到线性光不能等同于复现该论文学习到的相机 unprocessing 和 HDR 中间表示。当前代码不同分支仍采用不同的颜色运算，不能概括为所有路径均完成了物理一致的颜色协调。

## 7. SPRIH

**Semi-supervised Parametric Real-world Image Harmonization**，CVPR 2023，简称 SPRIH。

- [作者项目页](https://kewang0622.github.io/sprih/)

论文采用参数化表达，以 RGB 曲线校正全局颜色和色调，以 shading map 表达局部外观变化，并研究从真实合成图中学习协调。

项目借鉴的是“预测受控参数或局部变化，保留可解释修改方式”的思路。低频协调和 Learned Effects 合同允许有限增益及残差，而不是直接接收新的整件商品 RGB。

**转译边界：**这里的 Alpha 归一化低通、有限增益和高频保留属于工程设计，没有复现 SPRIH 的半监督双流训练、RGB 曲线网络或论文基准结果。

## 8. 从阅读到实现的对应关系

| 实际问题 | 研究提供的视角 | 当前项目中的转译 |
|---|---|---|
| 商品像贴纸，阴影不接地 | 像素高度、接触和软阴影 | 接触轮廓、二维高度代理、分离 AO 与投影 |
| 透明物体带着旧背景 | 环境 matting、衰减和折射流 | OpticalMatte 与目标背景采样 |
| 图像协调改坏文字 | 参数化外观修改、光色分解 | 原始细节保留、身份 mask、有限增益 |
| 生成结果自然但改动过大 | 生成结果作为候选而非权威底图 | 材质路由、`allow − guard`、硬恢复 |
| 单张效果改善却不稳定 | 方法假设与独立实验 | 固定变量对比、跨场景测试、分项拒绝 |

当前设计中的身份／结构保护、授权候选恢复和原子回退，是围绕项目要求形成的算法组合与工程约束。文献为问题表示和方法选择提供背景；这些文献并不直接证明当前实现具有新的基础模型、完整物理正确性或更高泛化指标。

## 9. 复现与引用说明

- 本仓库没有附带上述论文的第三方模型权重或研究数据集。
- 使用第三方代码、权重和数据时，需要分别查阅其发布条款；论文公开与代码可任意再发布不是同一回事。
- 当前测试可检查蒙版边界、数值合同、恢复与回退，视觉真实感仍需要独立实验。
- 发布说明中的测试结果应对应本次公开代码；不能以论文结果或历史实验记录代替。
