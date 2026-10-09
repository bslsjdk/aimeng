# AIMENG 神经元管理、持久化与运行日志

此工具管理用户实际导入的工件，不会在缺少数据时创建伪造神经元。当前批量评测适配器只支持 JSON 线性神经元工件；其他模型格式会明确报“不支持”，不会返回伪造分数。

## 当前实现范围

- 导入真实工件并登记 ID、版本、SHA-256；重复 ID、无效工件会失败。
- 显式选择 ID 后批量保存、导出 ZIP、启用/停用、独立数据集评测。
- runs/neuron-manager.jsonl 追加记录成功和失败事件，包含 run ID、事件 ID、时间、结果、错误类型/消息。
- 保存快照含 manifest 与哈希；导出包包含工件和 manifest。
- 评测后按 held-out MSE 阈值标记 excellent、candidate 或 rejected。excellent 与 candidate 都会自动保存带哈希的快照；达到 --reject-above-mse 的神经元会被停用并归档到 rejected 目录，但不会物理删除，保留复查与恢复能力。默认 excellent 阈值 0.05、rejected 阈值 1.0；阈值应按具体任务尺度校准，不能跨任务直接比较。
- 缺注册表时 list 会报错并写日志；只有显式 import 真实有效工件才会创建注册表。不会拿测试用例伪装成实际神经元。

## 工件格式（当前评测适配器）

一个真实、可评测的 JSON 工件需要包含格式标记、非空有限权重和有限偏置：

    {"format":"aimeng.linear_neuron.v1","weights":[0.25,-0.5],"bias":0.1}

上面只是格式说明，不是已训练模型，也不应被登记为真实神经元。其他架构需要实现对应的评测适配器后才能进行公平评测。

评测 JSONL 每行包含 input 数组和 target 数值；请使用独立于训练过程的 held-out 数据，不要用训练集自评后就宣布“优秀”。

## 命令示例

在仓库根目录执行：

    python scripts/neuron_manager.py import --id N-001 --artifact /path/to/real-neuron.json
    python scripts/neuron_manager.py import --id N-003 --artifact /path/to/another-real-neuron.json
    python scripts/neuron_manager.py list
    python scripts/neuron_manager.py save --ids N-001 N-003 --output runs/saved-neurons
    python scripts/neuron_manager.py export --ids N-001 N-003 --output runs/selected-neurons.zip
    python scripts/neuron_manager.py disable --ids N-001 N-003
    python scripts/neuron_manager.py enable --ids N-001 N-003
    python scripts/neuron_manager.py evaluate --ids N-001 N-003 --dataset /path/to/heldout.jsonl --good-mse-threshold 0.05

全局参数 --registry、--root、--log 写在子命令之前。例如：

    python scripts/neuron_manager.py --registry data/neuron_registry.json --root . --log runs/neuron-manager.jsonl list

每条命令都输出机器可读 JSON；失败时返回非零退出码并尝试写入错误日志。若工件在登记后被改动，SHA-256 校验会阻止启用/停用、保存、导出和评测。批量启用/停用先校验全部选中工件，避免只改一半。

## 日志回传

向我提供 runs/neuron-manager.jsonl、相关评测 JSONL、导出的 ZIP manifest，以及出错时的终端 JSON 输出即可。日志可能含本地文件路径或任务输入，请先检查并删除不希望分享的隐私内容。

## 自动化测试

    python -m unittest discover -s tests -p 'test_neuron_manager.py'

测试会在临时目录里造明确仅供测试的合成工件，验证导入、哈希、批量操作、保存、导出、评测和失败日志；测试数据不会写入默认注册表，也不会冒充真实训练成果。

## 尚未完成的边界

这次增加的是可执行的 Python 管理与日志基础层，不是 Android 图形管理页。它尚未集成真实神经元训练器、任意模型格式推理后端或自动晋级/永久淘汰策略。手机导航栏遮挡和顶部 UI 重叠，需要对实际承载神经元管理界面的 Android Activity/layout 单独修改并在设备上验收。


## 可下载的导入冒烟测试文件

目录 tests/fixtures/neuron_manager_smoke/ 内有一个显式标记 TEST-ONLY 的合成工件和独立测试数据。它们只验证文件导入与评测链路，不是训练结果。可按目录内 README 操作。GitHub Actions 会自动运行这条冒烟测试；它不会把 TEST-ONLY ID 注册到正式默认数据目录。
