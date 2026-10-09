# 自有模型工件格式 v0：设计草案

状态：设计草案，尚未实现，不是稳定 ABI。

## 目标

为 AIMENG 自己的模块化/任务相关 AI 定义一种版本化、可校验、支持按模块索引的 tensor container。它首先是工件容器，不负责定义神经网络本身，也不直接带来低内存或高速度。

## 必需元数据

- magic bytes 与 major/minor format version；
- header 长度、目录偏移和目录长度；
- 架构 ID 与架构版本；
- tensor name、dtype、shape、offset、stored length、logical length、alignment；
- 模块 ID 与 tensor ID 的映射；
- 存储编码（raw、压缩或具体量化 scheme + 参数）；
- 每个块的 checksum，建议 SHA-256 用于整个工件清单、快速校验可使用 CRC32C；
- tokenizer/词表版本（若模型使用 token 化）；
- 训练来源、代码提交、数据 manifest、参数版本和创建工具版本。

## 安全与健壮性

1. 所有整数使用固定宽度与明确字节序。
2. 读取器先校验文件大小、目录范围、整数乘法溢出、tensor shape、重叠区间和对齐，再分配内存。
3. 未知 major version 必须拒绝；未知可选字段可跳过，但不能忽略未知的关键张量编码。
4. 不允许在读取模型时执行 pickle 或其他任意代码。
5. 目录与权重分离可支持按模块读取，但只有 loader 真实按需读取/释放时，才可能减少驻留内存。
6. 格式转换要提供可复现工具和 round-trip / numerical tolerance 测试。
7. 训练 checkpoint 与部署工件分离：checkpoint 可能还包含 optimizer、scheduler、RNG 和训练步数；部署工件不需要这些状态。

## 初始二进制布局（建议）

- 固定头：magic、版本、flags、metadata_len、index_offset、index_len。
- metadata：UTF-8 JSON 或规范化 CBOR，限制最大长度。
- tensor index：按名称排序的记录；记录 shape/dtype、模块归属、偏移、长度、alignment、编码和校验。
- tensor payload：按 alignment 放置，不允许与 header/index 越界或互相重叠。
- 可选模块目录：把一组 tensor 映射到模块 ID，便于调度器决定加载哪些工件片段。

第一版不要追求最小文件大小，也不要设计复杂的通用压缩。先追求安全、可验证、可读写和可迁移。

## v0 验收测试

- 随机 tensor 写入后读回，shape/dtype/字节完全一致；
- 空 tensor、零维 tensor、极大 shape、整型溢出和未知 dtype；
- 截断文件、伪造偏移、重叠区间、损坏 checksum 必须拒绝；
- 未知 major version 必须拒绝；
- 单模块索引只读取指定 tensor 的 payload；
- 基准报告分别记录全文件读取、单模块读取时间与实际峰值 RSS/PSS；
- 测试输出不得把“文件可分块读”误称为“系统已释放内存”。

## 决策门槛

只有 Python 原型和安全测试通过后，才固定字节布局；在格式稳定前不把它作为 Android runtime ABI，也不要求模型训练代码依赖最终二进制细节。
