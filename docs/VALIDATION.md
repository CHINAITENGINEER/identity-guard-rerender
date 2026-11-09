# 公开发布验证

验证日期：2026 年 11 月 9 日。

环境：Windows x64、CPython 3.13.9；测试包含 Node.js 24.14.0 的内嵌 JavaScript 语法检查。
在独立虚拟环境中只安装 `requirements-dev.txt`，不安装 rembg、PyTorch、
Transformers 或浏览器模型包。解析后的依赖见 `constraints-tested.txt`。

## 公开测试

```powershell
python -m pip install -r requirements-dev.txt -c constraints-tested.txt
python -m pytest -q --durations=5 --tb=short
```

本地结果：

```text
396 passed, 1 skipped, 33 subtests passed in 29.23s
```

跳过项为 `tests/test_heldout_manifest.py` 中需要私人 holdout 图片的测试。
仓库只包含对应元数据，不包含原始图片；跳过原因由测试明确报告。
耗时是本机这次执行的观测值，不是性能保证。

测试涵盖 Alpha、几何约束、布局、透明光学、身份蒙版、候选恢复、
学习效果校验、场景级回退、内容哈希与前端合同等。
云端调用使用 mock；没有在本轮调用真实付费图像或模型服务。

## 离线示例

```powershell
python tools/run_synthetic_demo.py --out output/clean_demo
```

输入全部由脚本生成，候选故意修改整个画面。脚本在快速预览合成后，
独立重新计算背景像素审计和严格底图到混合结果的差异：

```json
{
  "image_size": [640, 420],
  "eligible_pixels": 7050,
  "changed_pixels_inside": 7050,
  "identity_changed_pixels": 0,
  "outside_changed_pixels": 0,
  "max_outside_delta": 0,
  "background_pixel_audit": {
    "changed_pixels": 0,
    "max_delta": 0,
    "verdict": "pass"
  },
  "visual_acceptance": "not_evaluated",
  "batch_unlock": false
}
```

## CLI 与工作台入口

```powershell
python -m strict_compositor.cli compose --job output/clean_demo/hybrid_job.json --out output/clean_cli
python -m pip check
```

CLI 返回背景像素 `pass`，交付状态为 `pending_visual_review`，批量放行关闭。
`pip check` 未发现依赖冲突。

用 FastAPI TestClient 请求工作台 `/` 和 `/api/health`，两项均为 HTTP 200。
这证明本地页面和健康接口可加载，不等于执行了真实浏览器、自动模型或云端端到端测试。

## 本次未证明的内容

- 真实商品照片达到 90 分视觉验收。
- 多场景、不同材质与视角下的稳定通过率。
- 全自动 OCR、Logo 定位或真实三维材质恢复。
- RunningHub / MiniMax 的当前账号配置、真实调用或吞吐能力。
- 新 Shadow / Relighting 网络的训练效果。
- 生产服务的鉴权、隔离、并发与资源控制。

像素审计证明的是编辑边界与恢复行为；视觉和泛化需要独立实验。
GitHub Actions 执行公开测试、离线示例和 CLI 重放，实际运行状态以仓库 Actions 为准。
