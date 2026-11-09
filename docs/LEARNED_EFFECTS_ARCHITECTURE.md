# 受限学习光影层架构（Learned Effects V1）

## 1. 这次重构解决什么问题

当前确定性合成已经能锁住商品 RGBA、文字、图案、轮廓、背景和摆位，
但接触/投影代码逐渐堆积了大量针对底座、细脚、斜底、连续底边和透明材质
的特判。继续给这些函数增加 `if + GaussianBlur + morphology` 会导致规则互相
影响：修好一种商品，另一种商品退化。

V1 不删除旧算法，也不把最终图片交给生成模型。它只在核心渲染器中建立一条
窄而可替换的接口，让后续本地网络、ComfyUI 或 RunningHub 最多交付三类张量：

1. 接收面上的方向投影 Alpha；
2. 商品内部的单通道低频光照增益；
3. 商品内部的微小线性光高光残差。

任何模型都不能交付最终商品 RGB、商品 Alpha、背景 RGB 或最终合成图。

## 2. 像素所有权

| 像素/信息 | 最终所有者 | 学习模型权限 |
|---|---|---|
| 商品 Alpha 与轮廓 | 原始 RGBA + 确定性 Homography | 无 |
| 商品文字、Logo、图案 | 原始 RGBA + Identity Guard | 无，应用后逐像素恢复 |
| 商品高频纹理 | 原始 RGBA 的高频残差 | 无替换权 |
| 商品低频照明 | 原始低频 × 有界 Gain | 可预测 Gain |
| 小范围高光 | 原始 RGB + 有界 residual | 仅非身份内部 |
| 接触 AO | 确定性渲染器 | V1 无 |
| 方向投影 | 确定性 fallback 或 learned Alpha | 只可替换 Alpha |
| 反射/玻璃 | 显式材质与光学分支 | V1 无 |
| 背景 | 原背景 + 授权蒙版外硬恢复 | 无 |
| 摆位/尺度/层级 | 几何与语义布局 | 无 |

## 3. 运行时分层

```text
原始 RGBA
  -> 确定性颜色预处理（可选 RGB gain）
  -> 预乘 Alpha + Homography
  -> 材质/透明光学
  -> 确定性 contact / cast / reflection
  -> Identity 与 Structure Guard
  -> [可选] 整场 Learned Effects Provider
       输入：背景、支持面、保护区、三商品图层、光照、可选 Depth/Normal
       输出：每商品 shadow_alpha / illumination_gain / specular_residual
       验证：尺寸、数值范围、越界、身份区、批次完整性
       失败：整场原子回退确定性图层
  -> cast darken
  -> reflection
  -> contact darken
  -> 原商品图层合成
  -> [可选] 已有局部 Hybrid Candidate
  -> 授权蒙版外恢复原背景
  -> 审计与 PNG 输出
```

Learned Effects 位于确定性商品图层已经完成、接收面尚未真正写入最终图之间。
它与后置的 Hybrid Candidate 完全独立：前者是受限张量，后者仍是授权区域内的
候选 RGB 恢复。

## 4. 核心数据合同

代码位置：`strict_compositor/learned_effects.py`。

### 4.1 `LearnedEffectRequest`

一次请求包含整个场景和全部商品，而不是逐商品发三个互不知情的请求。这样三件
商品共享同一主光、接收面和保护区，避免每件商品的阴影风格不一致。

主要字段：

- `background_rgb`: `H×W×3 uint8`
- `support_mask`: `H×W bool`
- `protected_mask`: `H×W bool`
- `lighting`: 已解析的共同光照
- `products`: 全部 `EffectContext`
- `depth`: 可选 `H×W float32`
- `normal`: 可选 `H×W×3 float32`

传给第三方 Provider 的不是核心数组引用，而是隔离副本。同一个 7K 背景只复制
一次并在商品上下文之间共享，避免三件商品重复复制数百 MB。

场景分析的 Deep 路线会把相对深度保存为 `H×W float32 [0,1]`，把相机坐标
法线保存为 `H×W×3 float32` 单位向量，两者都注册到原始背景全分辨率并写入
`SceneAnalysis`。UI 只传 `.npy` 路径；`Job.depth_map` / `Job.normal_map` 将路径
固定在本次任务及后续 refinement 中。只有实际配置 Learned Provider 时合成器才
读取文件，因此默认确定性路线没有额外大文件 I/O。

进入 Provider 前会再次检查后缀、精确 `float32`、形状、有限值、Depth 范围与
Normal 单位长度；必要的画布缩放由核心完成，法线插值后重新单位化。文件哈希、
源/目标尺寸、插值方式、数值范围与画布数组哈希进入审计。证据损坏不会中止整个
合成，而是在 Provider 调用前触发整场 `strict_fallback`。

### 4.2 `LearnedEffectPrediction`

每件商品只能返回：

- `shadow_alpha`: `H×W uint8 [0,255]` 或 `float [0,1]`
- `illumination_gain`: `H×W float32 [0.82,1.18]`
- `specular_residual`: `H×W` 或 `H×W×3 float`，线性光范围 `[-0.08,0.08]`

不提供 `rgb`、`rgba`、`alpha`、`final`、`reflection_rgb` 等字段，因此普通 Provider
无法通过正常接口请求这些权限。把 `H×W×3` 图塞入 Gain、返回错误尺寸、NaN/Inf
或超范围 residual 都会被拒绝。

### 4.3 整场原子性

Provider 必须返回和请求完全一致的商品 ID 集合。所有商品先验证、后应用；任意
商品失败时三件商品全部使用旧确定性结果，禁止出现一件走模型、两件走旧规则的
混合光照。

## 5. 重光照公式

不能直接模糊或重绘商品 RGB，也不直接在编码 sRGB 上乘梯度。V1 在商品 ROI 内
转到线性光空间并做 Alpha 归一化低通：

```text
low  = alpha_normalized_gaussian(linear_original)
high = linear_original - low

relit = low * lowpass(illumination_gain)
        + high
        + bounded_specular_residual
```

随后：

1. 只把结果写入商品 Alpha 内；
2. Identity Guard 和 Structure Guard 从原商品 RGB 逐像素恢复；
3. Alpha 外保持原图层的哨兵 RGB 不变；
4. 中性 Gain=1、Residual=0 时走字节稳定旁路，不进行浮点往返。

这样模型只能改变大尺度受光关系和少量高光，文字/花纹的源高频残差仍来自原图。

## 6. 阴影合同

Learned Shadow 只替换 `cast_shadow`，不和旧 cast 做 `max()` 或叠加。确定性
`contact_shadow` 暂时保留，负责商品底座的零缝隙接地线。这样第一阶段不需要
模型同时分清 AO、接缝和长投影，也不会因为 learned cast 轻微断开就出现浮空。

输入 Alpha 会被硬裁到：

```text
support_mask AND NOT protected_mask
```

环境漫射场景（`ambient_only`）不允许 Provider 凭空生成方向长投影；非零 cast 会
触发严格回退。

## 7. ComfyUI / RunningHub 接法

### 7.1 当前在线契约：RunningHub Shadow V1

在线 V1 已冻结为一条最小权限路径：一个场景只创建一个 RunningHub 任务，只下载
一张单通道灰度 PNG，并把它解释为**相对阴影形状/密度**。工作流必须同时接收：

```text
无影预合成图
商品 ID 蒙版
Depth / Support / Protected 三通道证据
Normal
包含光向、长度、柔化、材质的结构化提示
```

缺少任一必要节点、密钥、深度或法线时，请求在上传前失败。V1 明确禁止配置
`illumination_gain` 和 `specular_residual` 输出节点，避免云端改写文字、Logo 或高频
商品纹理。云端输出还要经过以下本地边界：

```text
确定性投影生成不可变几何包络
-> 模型阴影裁入包络
-> 必须与确定性接触投影相连
-> 本地光照参数决定最大黑度
-> 三件商品与审计报告全部成功后原子提交
```

因此模型返回全白桌面也不能扩大背景可改区；报告生成阶段发生异常，也不会出现
“审计写严格回退、实际像素却已使用模型”的半提交状态。

### 7.2 本地张量实验与未来 V2

V1 已提供 `RegisteredNpzEffectProvider`，开发阶段可把工作流输出注册为每商品一个
NPZ 文件。NPZ 只允许下列键：

```text
shadow_alpha
illumination_gain
specular_residual
```

ComfyUI 常规图片输出可先通过打包工具转换：

```powershell
python tools\pack_learned_effect_maps.py `
  --shadow product_1_shadow.png `
  --gain product_1_gain.png `
  --specular product_1_specular.png `
  --output product_1_effects.npz
```

图片编码约定：Shadow 黑到白对应 `0..1`；Gain 中灰是精确 `1.0`、黑白分别是
`0.82/1.18`；Specular 中灰是 `0`、黑白分别是 `-0.08/+0.08`。需要更高精度时
可直接输入 float32 `.npy`。

命令行实验时必须为作业内每件商品各注册一次，不能只让其中一件走模型：

```powershell
python -m strict_compositor.cli compose `
  --job examples\job.json `
  --out output\learned_ab `
  --learned-effect product_1=product_1_effects.npz `
  --learned-effect product_2=product_2_effects.npz `
  --learned-effect product_3=product_3_effects.npz
```

推荐工作流：

```text
背景 RGB + 商品 Alpha + 商品位置 + Support + Depth + Normal + Lighting
  -> Shadow 网络
  -> shadow_alpha

背景 RGB + 已放置商品 RGB/Alpha + Lighting + Identity Guard
  -> Relighting 网络
  -> illumination_gain + specular_residual

三个商品张量打包
  -> RegisteredNpzEffectProvider
  -> 核心 Validator
  -> 严格合成
```

RunningHub/ComfyUI 负责推理和导出张量；核心程序仍负责坐标、裁剪、身份恢复、
线性光合成和背景硬恢复。尚未配置真实 Shadow/Relight Workflow 时，系统必须明确
显示 `not_requested`，不能把旧 OpenCV 阴影冒充为 learned 结果。

## 8. 审计

`audit.json -> learned_effects` 记录：

- provider 名称；
- `not_requested / applied / strict_fallback`；
- 每件商品输入 RGB、Alpha、身份区、旧阴影的 SHA-256；
- 每个预测张量的 SHA-256 与数值范围；
- 阴影越界被裁像素数；
- 商品外变化像素数；
- 身份区变化像素数；
- 最终商品 RGB 与 Shadow Alpha 的 SHA-256；
- Provider 自带 provenance。

输出图层目录还会保存可视化的 learned shadow、gain 和 residual 强度图，便于前端
直接检查模型到底返回了什么。

## 9. 当前状态与明确未完成项

已完成：

- 核心 Protocol、Request、Prediction、Batch；
- 数值与像素权限验证；
- 低/高频分解重光照；
- 整场原子应用和失败回退；
- NPZ 开发 Provider；
- 审计哈希、范围和诊断图层；
- 默认无 Provider 时保持旧路线；
- 新旧回归测试。
- Deep 场景分析的全分辨率 Depth/Normal NPY 持久化；
- `SceneAnalysis -> 前端状态 -> Job -> Provider` 的几何证据链；
- 证据严格加载、只读隔离、哈希审计与原子回退。

尚未完成，不能对外宣称已启用：

- 配对训练的 Shadow Network；
- 配对训练的 Relighting Network；
- RunningHub learned-shadow workflow ID；
- RunningHub learned-relight workflow ID；
- 冻结测试集的一次性盲测。

现有 336 张分类/参考图片适合肉眼诊断与问题覆盖，但不是带 shadow-alpha、gain-map
真值的配对训练集，不能直接当作监督训练数据。

## 10. 后续迁移顺序

1. 冻结旧 Shadow 规则，不再增加产品类别特判。
2. 在公开开发集上离线生成合法的 NPZ 效果张量，验证图层合同和肉眼效果。
3. 接一个真实 Learned Shadow Provider，只 A/B 替换 cast，contact 保留。
4. 跨材质、跨视角肉眼通过后，再逐步减少旧 cast 分支。
5. 接 Learned Relight Gain/Residual，先对无文字普通商品开放。
6. 透明玻璃继续使用显式光学；只有无文字、授权内部区域允许后置局部修复。
7. 路由与参数冻结后，只对 `未命名文件夹 3` 做一次盲测，禁止边看边调。

## 11. 不做的事情

- 不继续扩写旧阴影特判；
- 不让全图 Diffusion 控制商品或背景；
- 不把 IMPRINT、ObjectMate、HopNet 当成可直接插入的现成 API；
- 不在本轮重写布局、玻璃、反射和阴影四条链；
- 不因为模型调用成功就宣称视觉通过；
- 不用分数替代最终肉眼商品美感判断。
