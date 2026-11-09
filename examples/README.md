# 离线合成示例

在仓库根目录运行：

```powershell
python tools/run_synthetic_demo.py --out output/synthetic
```

脚本生成桌面、两个带字商品、一个透明器皿示意图，以及故意修改整张图的候选。
随后运行严格合成与局部候选恢复，并直接检查身份保护区和允许编辑区外的像素差。
输入全部由 NumPy/Pillow 生成，不需要原始照片、权重、API 密钥或网络请求。

主要输出：

- `background.png`、`*_rgba.png`：示例输入。
- `strict_job.json`、`hybrid_job.json`：以所在目录为基准的可重放任务。
- `strict/`、`hybrid/`：完整分层、蒙版与审计。
- `untrusted_candidate.png`：故意越界的候选。
- `comparison.png`：四图对照。
- `pixel_contract.json`：独立逐像素检查结果。

重放 CLI 路线：

```powershell
python -m strict_compositor.cli compose --job output/synthetic/strict_job.json --out output/cli_strict
python -m strict_compositor.cli compose --job output/synthetic/hybrid_job.json --out output/cli_hybrid
```

该示例验证像素权限与恢复行为，不用于证明摄影真实感或 90 分视觉验收。
生成器使用快速调试审计，正式 CLI 仍保存待视觉评审状态。
