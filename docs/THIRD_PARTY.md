# 第三方依赖与素材

项目运行依赖 NumPy、OpenCV、Pillow、Requests；工作台另依赖 FastAPI 和 Uvicorn。
它们的许可证与版权归各自项目所有，以安装的具体发行版本为准。

可选自动视觉能力通过 rembg、ONNX Runtime、PyTorch、Transformers 等接入。
模型代码的许可不自动覆盖权重、训练数据或云平台工作流。
U²-Net、IS-Net、SAM2、语义分割和深度模型等权重不随仓库分发，
应在使用前根据实际下载来源核对条款与适用范围。

`docs/RESEARCH.md` 中的论文链接用于说明方法来源，不表示完整复现、
作者背书或对论文实现的再授权。

`tools/run_synthetic_demo.py` 生成几何商品、桌面与候选。
`docs/assets/synthetic_demo.png` 来自该脚本，不使用第三方商品照片、
字体文件、生成服务或私人实验图像。

`tests/fixtures/heldout_set_b_manifest.json` 仅保留历史实验的相对文件名、
哈希和大小等元数据。对应图片不随仓库提供，也不应据此推断素材可再分发。

本仓库尚未选定单独的代码开源许可证。
