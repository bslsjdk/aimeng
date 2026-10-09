# 固定基础模型：Ornith-1.5-9B-GGUF Q4_K_M

## 来源与身份

- Hugging Face 仓库：`ornith-ai/Ornith-1.5-9B-GGUF`
- 文件：`Ornith-1.5-9B-Q4_K_M.gguf`
- 量化：Q4_K_M
- 公开文件页面显示文件约 5.78 GB；实际下载后必须记录文件大小和 SHA-256。
- 上游仓库同时提供 MLX 版本，但 Android 训练/运行基线选择 GGUF，而不是 MLX/Safetensors 的 MLX 专用执行路径。

## 下载

在有足够磁盘空间、网络和 Python 环境的训练机上运行：

```bash
python -m pip install -r requirements.txt
python scripts/fetch_model.py
python scripts/inspect_gguf.py --output runs/gguf_manifest.json
```

默认文件保存在 `models/ornith-1.5-9b-q4km/`，该目录被 Git 忽略。下载器先检查至少 8 GiB 可用磁盘空间，使用 Hugging Face Hub 缓存，并且不会把模型二进制提交进仓库。

## “用 GGUF 训练”在本项目中的准确含义

第一阶段将 GGUF 作为**冻结的教师模型/推理基线**，用于生成输出、测量延迟和内存，并评估候选策略。GGUF 量化权重不是适合直接用常规 AdamW 训练的标准训练检查点；不应假装可以直接对 Q4_K_M 文件做常规反向传播。

要训练能选择 FFN 通道组的路由器，必须让执行后端暴露所需的层/通道信息，并支持真正跳过对应的结构化计算。仅从输入和最终输出训练一个路由分类器，不能证明它知道哪些内部神经元有用；仅在逻辑上屏蔽通道也不能证明省了内存或算力。

后续训练需要分清三类实验：
1. **冻结 GGUF 基线**：可先完成，记录准确模型 SHA、任务质量、速度和峰值内存。
2. **路由策略原型**：先用可观测、可复现的特征训练轻量控制器；把无法观测的通道贡献标记为未知，不能编造标签。
3. **真实结构化稀疏执行**：需要修改/扩展 llama.cpp 或另一执行后端，验证质量、速度和内存均有实测改善。

## Android 4 GiB 目标

5.78 GB 是磁盘文件大小，不代表运行时必然占用同样大小，也不代表 mmap 后只占很少内存。实际峰值受权重页驻留、KV cache、计算缓冲区、后端与驱动影响。4 GiB 限制必须通过目标 Android 设备的运行时测量验证，训练机器上的结果不能替代手机实测。
