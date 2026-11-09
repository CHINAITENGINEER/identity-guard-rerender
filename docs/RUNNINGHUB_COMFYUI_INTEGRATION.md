# RunningHub / ComfyUI 协同接入

> 这是保留的集成设计说明。公开发布未验证实际云端工作流，也不提供账号工作流或私人素材。

## 已实现的接口边界

RunningHub 在本项目中是“云端蒙版与代理能力”，本地严格合成器是最终像素所有者。
二者不是替代关系：

```text
商品原图
  → RunningHub AI App / ComfyUI：只预测 Alpha
  → 本地检查尺寸、比例、Alpha 完整性
  → 丢弃云端 RGB
  → 用原始商品 RGB + 云端 Alpha 重建 RGBA
  → 本地摆位、接触 AO、方向投影、倒影、暖色回弹
  → 覆盖原始商品层
  → 恢复授权蒙版外原背景
  → 像素审计与人工视觉门禁
```

只要云结果改变长宽比、缺失 Alpha、输出异常或未通过蒙版质量门禁，本次云结果就会
被拒绝并自动转入本地抠图。RunningHub 成功返回文件不等于合成成功，更不等于
`visual_score >= 90`。

## 客户如何配置

1. 启动 `tools/start_workbench.ps1`。
2. 打开页面左侧的“云端能力配置（只需设置一次）”。
3. 粘贴轮换后的 MiniMax 与 RunningHub 密钥。
4. RunningHub 中国账号选择 `.cn`，国际账号选择 `.ai`。
5. 默认使用“AI 应用（推荐）”。填写自己审核并有权调用的抠图应用 ID。
6. 点击“测试 RunningHub”。系统从官方 API 读取 `nodeInfoList`，自动选择第一个
   `fieldType=IMAGE` 的节点，并保存 `nodeId + fieldName`。
7. 勾选“优先使用 RunningHub 云抠图”，保存配置。

密钥保存在 `ui_workspace/private/integrations.json`。文件中的密钥是 Windows DPAPI
密文，只能由同一 Windows 用户解密；此目录已被 `.gitignore` 排除。API 返回给前端
的设置对象只包含 `configured=true/false`，不会回显密钥或密钥片段。

## AI App 与 ComfyUI 工作流

AI App 模式适合非技术客户。它使用：

- `GET /api/webapp/apiCallDemo` 获取可编辑输入节点；
- `POST /task/openapi/upload` 上传输入图；
- `POST /task/openapi/ai-app/run` 创建任务；
- `POST /task/openapi/outputs` 轮询结果。

ComfyUI 模式适合团队自有工作流。它使用：

- `POST /api/openapi/getJsonApiFormat` 检查工作流；
- `POST /task/openapi/upload` 上传输入图；
- `POST /task/openapi/create` 覆盖指定节点并创建任务；
- `POST /task/openapi/outputs` 轮询结果。

自有抠图工作流必须保持输入尺寸和长宽比，输出为同尺寸 RGBA PNG 或灰度 Alpha。
推荐工作流结构为：

```text
LoadImage
  → RMBG-2.0 / BiRefNet / Inspyrenet（选择一种经过验证的主模型）
  → 轻量边缘细化
  → 保持原尺寸
  → SaveImage (RGBA)
```

不要在此工作流内换背景、扩图、重绘纹理或裁掉商品底座。框选功能已经在本地先裁出
目标区域，再把同一裁图交给云端或本地模型，因此“框选后抠图”和“直接抠图”的算法
边界一致。

## 与 MiniMax 的关系

MiniMax 只输出支持面、保护区、光向、反射属性和布局建议 JSON。RunningHub 只输出
Alpha/深度/法线/局部代理。本地系统做坐标统一、几何仲裁与最终合成。任何一层失效
都不能放宽本地门禁：

- MiniMax 无效 JSON：本地场景分析回退；
- RunningHub 超时或失败：本地抠图回退；
- 商品底座不在支持面：拒绝或最小吸附；
- 保护区碰撞：拒绝或最小吸附；
- 授权区外背景变化：立即拒绝；
- 人工视觉评分低于 90：`reject`，禁止批量放行。

## 当前验收

自动化测试覆盖 Windows DPAPI 密钥往返、明文不落盘、RunningHub 请求端点、排队态、
失败态、Alpha 质量、支持面、接触线、投影长度随商品高度增长、保护区、授权区外
逐像素不变等规则。历史真实页面实验使用私人素材；这些素材不随公开仓库分发。
公开验证范围见 [VALIDATION.md](VALIDATION.md)，不据此推断云端工作流已完成端到端验证。

官方接口依据：

- [RunningHub 完整接入示例](https://www.runninghub.cn/runninghub-api-doc-en/doc-8287469)
- [RunningHub 启动 AI App 任务](https://www.runninghub.cn/runninghub-api-doc-en/api-425761096)
- [RunningHub 高级 ComfyUI 任务](https://www.runninghub.cn/runninghub-api-doc-en/api-425761093)
- [RunningHub 上传资源](https://www.runninghub.cn/runninghub-api-doc-en/api-425761099)
