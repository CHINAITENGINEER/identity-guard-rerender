# MiniMax-M3 场景顾问迁移说明

> 本文说明历史适配设计。模型名、端点与 API 能力随服务更新，使用者需按实际账号配置；
> 公开测试使用 mock，不代表真实云端连接或当前模型能力已验证。

## 结论

项目已经迁移为“MiniMax-M3 提供结构化候选，本地系统控制最终像素”的双层架构。
MiniMax 不参与抠图输出、不生成最终商品、不重绘背景，也不能绕过本地几何门禁。

```text
背景 + 商品源图
  ├─ MiniMax-M3：语义支持面、保护区、光向、反射、布局与路线候选
  └─ 本地模型：语义分割、深度、支持面和光照证据
            ↓
      JSON/坐标/面积/一致性校验
            ↓
      本地支持面、底座、碰撞、保护区门禁
            ↓
      确定性 RGBA 合成与像素审计
```

## 为什么不是让 MiniMax 直接“给最终答案”

MiniMax 官方的 Chat Completions 接口支持图片理解，但当前官方接口文档没有声明
`response_format=json_schema`。模型可能在 JSON 外输出思考标签、代码围栏或说明文字。
因此本项目采用以下防线：

1. 提示词要求唯一 JSON 对象和固定字段。
2. 清除 `<think>` 与 Markdown 围栏后，仅提取第一个合法 JSON 对象。
3. 对所有字段进行类型、范围、有限数值、坐标、面积和产品 ID 校验。
4. 支持面必须与本地深度/语义候选达到最低重合度，才可成为最终支持面。
5. MiniMax 的商品位置只是布局种子，最终必须通过本地底座、碰撞和保护区门禁。
6. 网络错误、限流、无效 JSON 或低置信度都会自动回退到本地链路。

官方依据：

- [MiniMax Chat Completions API](https://platform.minimaxi.com/docs/api-reference/text-chat-openai)
- [MiniMax API 接口概览](https://platform.minimaxi.com/docs/api-reference/api-overview)

## MiniMax 输出契约

```json
{
  "support_polygon": [[0.08, 0.20], [0.92, 0.20], [0.92, 0.75], [0.08, 0.75]],
  "protected_regions": [
    {
      "label": "telephone",
      "polygon": [[0.10, 0.30], [0.25, 0.30], [0.25, 0.50], [0.10, 0.50]]
    }
  ],
  "lighting": {
    "light_direction": [-0.7, 0.7],
    "shadow_direction": [0.7, -0.7]
  },
  "surface": {
    "reflectivity": 0.18,
    "reflection_recommended": false
  },
  "products": [
    {
      "product_id": "product_1",
      "center_x": 0.30,
      "bottom_y": 0.62,
      "height_ratio": 0.15,
      "z_order": 1,
      "footprint_width": 0.08,
      "source_light_compatibility": 0.81
    }
  ],
  "route": "strict_composite",
  "confidence": 0.88,
  "warnings": []
}
```

坐标均为背景图归一化坐标。`bottom_y` 是商品接触基线，而非包围框底部的随意位置。
商品主体允许越出支持面边界，但接触脚印必须位于支持面内。

## 本地仲裁规则

### 支持面

- MiniMax 置信度至少为 `0.68`。
- MiniMax 支持面与本地支持面的 IoU 至少为 `0.55`。
- 坐标必须全部位于 `[0,1]`，多边形面积至少为画面的 `1.5%`。
- 不通过时保留本地支持面，并在 `advisor.warnings` 中记录分歧。

### 保护区

- 本地模型保护区始终保留。
- MiniMax 保护区只有与已验证支持面存在足够关联时才并入。
- 合成前仍由像素级保护掩版拒绝碰撞。

### 光照和反射

- 光向必须为非零有限向量，进入系统前自动归一化。
- MiniMax 置信度至少为 `0.62` 才能参与光向和反射判断。
- 反射率使用本地证据与 MiniMax 判断加权融合，不直接照抄模型数值。

### 商品布局

- MiniMax 输出 `center_x / bottom_y / height_ratio / z_order`。
- 本地程序按商品真实 Alpha 计算接触脚印。
- 任何越界、保护区碰撞或商品主体碰撞都会触发最小修复。
- 修复失败时回退到本地确定性美学布局。
- 高置信度 `route=reject` 会阻止自动构图。

## 密钥配置

已在聊天中发送过的密钥必须先在 MiniMax 控制台撤销并重新创建。
项目不会读取前端输入、工作流 JSON 或仓库文件中的密钥，只读取进程环境变量
`MINIMAX_API_KEY`。

推荐启动方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_ui_with_minimax.ps1
```

脚本会隐藏密钥输入，只把新密钥放入当前服务进程的内存环境；服务退出后变量会被清除。

如果不配置密钥，系统仍可正常启动，并自动使用原有本地场景分析与确定性构图。

## UI 中如何确认

- `MiniMax 已校验 xx%`：M3 结果已通过本地仲裁。
- `MiniMax 异常，已本地回退`：接口或 JSON 失败，但本地流程仍可工作。
- `MiniMax 建议拒绝`：场景兼容性不足，需要人工确认。
- `MiniMax 未配置`：当前服务进程没有读取到环境变量。

`/api/health` 只返回是否配置、模型名和服务地址，绝不返回密钥。

## 缓存与审计

已验证的结构化建议缓存在 `ui_workspace/minimax_advice/`，缓存中只有：

- 模型名和请求 ID；
- 经过解析的结构化 JSON；
- 提示词版本。

缓存不保存 API Key，也不保存 MiniMax 的原始思考文本。该目录已加入 `.gitignore`。

## 当前边界

- 本次迁移没有接入 MiniMax 图像生成接口，因为它不应拥有最终像素。
- RunningHub/ComfyUI 仍可作为 RGBA、深度、法线或局部残差提供方。
- BrushNet、Differential Diffusion 和重光照代理仍是可选阶段，默认关闭。
- 最终输出仍必须通过背景像素门禁和人工视觉评分，模型成功不等于交付成功。
