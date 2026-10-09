# 人类式想法、可塑计算与外部神经元库：实现规格 v0.1

> 状态：设计规格，尚不代表代码已经实现或实验已经证明有效。目标是先把概念拆成可编码、可测试、可回滚的机制，再逐阶段实现。本文遵守 AIMENG 的统一核心模型方向：**一个核心模型，不采用 MoE，不另建一个“想法 AI”或独立专家模型。**

## 1. 目标与非目标

### 目标

让同一个核心模型逐步具备以下可训练行为：

1. **自由联想**：根据问题生成多个不同方向的候选想法，允许暂时大胆、反常规、跨领域。
2. **展开假设**：明确每个想法依赖的前提、可能机制、可推导后果和适用边界。
3. **自我质疑**：主动找反例、隐藏假设、矛盾、资源限制与可能的替代解释。
4. **证据分级**：区分已知事实、推导结果、待验证假设、纯探索性设想，不能把流畅表达当成证据。
5. **实验验证**：尽量把关键不确定性转成低成本、可重复的测试；代码和数学任务优先使用可执行检查。
6. **经验更新**：把任务结果转成可复用的策略经验；经过验证的内部计算变化才有资格固化。
7. **可塑计算**：允许同一个模型内部一个受限的可训练区域在任务期间尝试变化；任务结束后对候选变化进行验证、保留或回滚。
8. **外部神经元库**：存放经过版本管理、具有输入输出接口和测试证据的可复用计算模块，供同一个核心模型按需检索和加载。
9. **资源受控**：任何探索、缓存、适配器加载和训练都受预算限制；Android 整应用峰值 RAM 硬上限为 4096 MiB，目标门槛低于 3800 MiB。

### 非目标与禁止误解

- 不把系统拆成多个互相独立的 AI，不引入 MoE 作为默认方案。
- 不把“生成很多答案”称为学习成功。
- 不把模型自己的自评、置信度或教师声明当作独立验证证据。
- 不在每次推理时无条件更新核心权重。
- 不允许候选参数直接覆盖稳定核心权重。
- 不把单个孤立神经元当作天然可迁移的能力。实际可复用单元必须带有连接/参数、接口、兼容版本与测试记录。
- 不把外部文本记忆等同于神经计算能力；文本知识和可执行参数模块是两类工件。
- 不声称人类式创造力、自主学习或神经元增长已经得到证明，必须以实验结果为准。

## 2. 统一核心架构

逻辑上只有一个核心模型。外围是普通软件基础设施，不是第二个智能模型。

```text
输入 / 环境事件
      |
      v
统一核心模型
  ├─ 理解当前目标与约束
  ├─ 选择：直接回答 / 联想探索 / 分解任务 / 请求验证
  ├─ 生成并比较候选想法
  ├─ 调用工具、测试器或检索接口
  ├─ 反思失败并提出修订
  └─ 决定是否申请可塑性实验或调用已验证模块
      |
      +------> 任务状态与原始轨迹（持久化）
      +------> 证据 / 工具 / 确定性验证器（非 AI 或外部独立检查）
      +------> 可塑工作区（临时参数、候选结构、预算）
      +------> 外部神经元库（已版本化、已验证的计算模块）
      +------> 评测与回滚门（独立决定是否晋级）
```

“想法”“可塑区域”“神经元库”是同一模型生命周期内的不同机制，而不是三个模型：
- 想法机制改变当前探索的候选方案与推理路线。
- 可塑区域尝试改变模型内部一小部分参数或结构。
- 神经元库保存验证通过的计算工件及其兼容性信息。
- 任务状态记录当前工作进度，不把整个历史对话无差别塞进上下文。
- 验证器提供外部可检查的结果，不负责代替核心模型进行通用思考。

## 3. 一次任务的状态机

每个任务都必须有持久化状态和明确终态，以便进程中断后恢复。

状态集合：
- `CREATED`：收到任务，生成 task_id。
- `CONTEXT_READY`：已读取目标、约束、相关记忆和预算。
- `EXPLORE`：按任务难度选择直接处理或产生多样候选。
- `HYPOTHESIS_READY`：候选方案已写入工作区，带有前提与可检验预测。
- `CHECKING`：执行逻辑检查、检索、代码/数学测试或外部验证。
- `REVISING`：根据反例或失败结果修改假设；受最大轮数限制。
- `SOLVED_PENDING_REVIEW`：任务暂时解决，但结果仍需验收。
- `PLASTIC_TRIAL`：可选阶段；只对任务有明确必要性且资源允许的任务启用。
- `CANDIDATE_EVALUATION`：将临时参数/模块与稳定基线做成对照评测。
- `CONSOLIDATE`：记录结果、更新策略经验、提交合格模块或回滚。
- `DONE`：结果、状态、日志和工件引用已持久化。
- `FAILED_RECOVERABLE`：错误已记录，可以从检查点恢复。
- `ABORTED_BUDGET`：预算耗尽，保留现场并安全退出。

每次状态转移都记录 `task_id`、前状态、后状态、时间、原因、输入/输出工件哈希。程序重启后从最近一次原子提交的检查点恢复，不依赖进程内存。

## 4. 人类式“想法”循环

### 4.1 什么时候探索

核心模型先判断任务风险、难度、不确定性与预算。简单、明确的问题可直接回答；高不确定、开放式或失败代价高的问题才进入更深入探索。选择器属于同一个模型的输出策略，不是另一个模型。

建议初始策略：
- 低复杂度且可验证：1 个主方案，快速验证。
- 中等复杂度：2–4 个方向不同的候选。
- 高不确定开放问题：3–6 个方向不同的候选，之后按预算裁剪。
- 最大探索轮数、候选数和 token 数都必须可配置，并有硬上限。
这些是初始工程默认值，之后应由消融实验调整，不是认知科学定律。

### 4.2 每个想法的数据结构

每个 idea 至少包含：
- `idea_id`、`task_id`、`parent_idea_id`；
- `statement`：简短假设；
- `novelty_axis`：与其他候选的差异方向；
- `assumptions[]`：依赖前提；
- `mechanism`：为什么它可能有效；
- `predictions[]`：若假设成立，能观察到什么；
- `counterexamples_to_seek[]`：优先寻找的反例；
- `evidence_refs[]`：证据工件引用，不能只存无法追溯的“我觉得”；
- `risk_and_cost`：失败风险、资源需求、可逆性；
- `epistemic_status`：`speculative` / `plausible` / `supported` / `refuted` / `unknown`；
- `verification_plan`：怎样区分它和替代解释；
- `result_refs[]`：实际测试与结果；
- `created_by_model_version`、`prompt_or_policy_version`。

### 4.3 多样性和反思

不能仅靠重复改写同一个答案伪造“多想法”。候选要尽量覆盖不同的机制，例如：直接解法、类比迁移、分解重组、边界条件变化、反例驱动、资源受限替代方案。通过相似度/规则检查标记重复候选，但不能只靠语义相似度判断正确性。

每轮反思至少回答：
1. 哪个假设最脆弱？
2. 哪条证据会使当前首选方案被否定？
3. 是否存在成本更低或机制不同的解释？
4. 当前失败是目标错误、知识缺失、步骤错误、工具错误，还是验证不足？
5. 是否值得继续探索，还是应该停止、提问或承认未知？

这是一种训练目标和可观察的任务协议，不要求把模型私密的内部思维链全部保存。只记录可审计的简要理由、假设、测试和结论。

### 4.4 证据等级

- `speculative`：新设想，没有充分证据。
- `plausible`：机制在逻辑上可解释，但未被关键测试支持。
- `supported`：有相关证据或测试支持，仍须声明范围。
- `verified_for_scope`：指定版本、输入范围和测试集上通过预定义独立验收。
- `refuted`：指定假设在某个测试/条件下失败。
- `unknown`：现有证据不足，不能判定。

`verified_for_scope` 不是“普遍真理”；记录必须注明适用范围。一个反例可以否定普遍化主张，但不一定否定更窄的条件性主张。

## 5. 验证与停止规则

验证器必须尽可能独立于产生候选的同一段生成过程：
- 数学：精确计算、符号检查、多个独立测试输入。
- 代码：单元测试、静态检查、沙箱执行、边界与错误输入测试。
- 事实：可信来源、来源日期、交叉验证；无法查证则保留 unknown。
- 设计/架构：显式不变量、可执行原型、对照实验和失败案例。
- 主观创意任务：使用预先定义的多维量表和盲评/多评者；不能把主观评分伪装成客观事实。
- 自我评价：只能作为辅助信号，不能单独把状态提升到 verified。

停止条件任一成立即停止当前探索：已达到任务验收标准；达到轮数/token/时间/内存预算；新增候选不再带来可测信息；缺少关键输入且继续推断风险过高；验证器/工具不可用。预算停止也必须保存检查点和未解决项。

## 6. 可塑计算：先安全试验，再考虑结构增长

### 6.1 两种学习时间尺度

**任务内短期可塑性**
- 只改临时工作区内的候选参数/适配器。
- 稳定核心权重只读。
- 保存优化器状态、随机种子、训练配置、数据哈希和基线 checkpoint 引用。
- 任务结束后，默认丢弃或冻结候选；只有进入独立评测并通过门槛才有资格进入库。

**任务后长期固化**
- 将候选与稳定基线在固定的验证集、旧能力回归集、新任务迁移集上比较。
- 通过质量、泛化、回归、兼容性和资源门后，作为版本化外部模块发布。
- 是否把模块合并进核心权重是后续独立实验，不能作为第一阶段默认操作。
- 每个已发布模块可撤销；核心 checkpoint 永不被候选直接覆盖。

### 6.2 实现顺序

**阶段 A：不修改主模型权重**
先把“想法生成 → 验证 → 经验记录 → 下次检索使用”闭环跑通。它能验证想法质量和经验利用，不等于神经元增长。

**阶段 B：同一模型的小型可训练适配器**
在训练机上尝试 LoRA/小型 adapter 作为可塑区域。adapter 是一个可控的参数模块，不是字面上的单个神经元，也不等于已实现动态神经元拓扑。训练结束做对照评测，通过后保存为外部可复用模块。

**阶段 C：可插拔计算子图**
只有当后端能定义明确的输入/输出张量、参数格式、连接语义、加载/卸载和执行接口时，才研究真正的子图模块。先做静态子图，不允许任意改写主干拓扑。

**阶段 D：真正的动态神经元/连接增长**
最后再研究新增单元、连接初始化、梯度/优化器状态、结构剪枝、模块合并与后端映射。必须先定义拓扑不变量、内存上限、可序列化格式和回滚策略。未实现这些机制前，不得宣称系统会实时创造可工作的神经元。

### 6.3 可塑性候选生命周期

`CREATED -> TRAINING -> FROZEN_CANDIDATE -> EVALUATING -> ACCEPTED`

任意训练失败、超预算、指标不足或兼容性错误转到 `REJECTED` / `ROLLED_BACK`。只有 `ACCEPTED` 才能出现在正常检索结果中。候选模块不能自我批准。

对照实验至少比较：
- 基线模型；
- 基线 + 候选模块；
- 如适用，基线 + 随机/无关模块控制组。
评估固定测试、不同表面形式的新任务、组合任务、旧能力回归、延迟、峰值 RAM、模块加载失败率。只在原题上变好不算迁移成功。

## 7. 外部神经元库的模块契约

库中保存的是可加载的计算工件，而不只是文本说明。v0 首先支持 adapter 参数；后续可扩展到计算子图。

建议每个模块目录包含：
- `module.json`：元数据和接口；
- `weights.safetensors` 或后端明确支持的无损参数文件；
- `compatibility.json`：基模型、层/维度、tokenizer、运行时 ABI 等兼容条件；
- `eval_report.json`：基线与候选指标、测试集哈希、失败案例；
- `manifest.sha256`：文件完整性校验；
- 可选的短说明文本和适用任务标签。

元数据必填字段：
- `schema_version`、`module_id`、`module_version`、`status`；
- `module_type`（adapter / subgraph / future_neuron_graph）；
- `base_model_id`、`base_model_revision`、`interface_signature`；
- `input_contract`、`output_contract`、`dependencies`；
- `task_families`、`known_limits`、`resource_cost`；
- `training_data_fingerprint`、`eval_suite_fingerprint`；
- `validation_status`、`validated_scope`、`created_at`；
- `artifact_hashes`、`rollback_target`。

模块库状态：`draft`、`training`、`candidate`、`accepted`、`deprecated`、`rejected`。默认只检索 `accepted` 且兼容当前核心版本的模块。检索可以根据任务标签、接口和历史收益排序，但模块选择最终仍由同一个核心模型与确定性策略共同决定。不得加载哈希不匹配或接口不兼容的工件。

### 模块复用的安全规则

1. 先在元数据层检查兼容性、状态、大小和依赖。
2. 在预算内加载，超预算则拒绝或卸载低优先级的非 pinned 模块。
3. 通过接口校验后才允许执行。
4. 对关键任务保留无模块基线回退路径。
5. 记录模块版本、加载成本、任务结果和失败。
6. 新结果可更新模块的统计证据，但不能仅凭一次成功自动修改模块参数或升级其状态。

## 8. 经验记忆和计算模块必须分开存储

至少分为四类，不要求四个模型：

1. **情节/轨迹记录**：发生了什么、工具输出、状态转移、失败与恢复点；尽量保留原始记录并按需检索。
2. **策略经验**：哪种任务策略在什么条件下有效/无效，附独立验证、样本量和版本。
3. **知识记录**：事实、来源、时间、可信等级、适用条件；易变事实优先外部检索。
4. **计算模块**：adapter 或子图参数、接口、依赖、评测报告、hash 和版本。

不要把未经验证的候选直接混入已验证记忆。写入流程为：
`candidate -> validate -> deduplicate -> scope -> commit`
每一步都可记录拒绝原因。知识与经验记录也应允许过期、撤销和版本更替。

## 9. 推荐代码目录与接口

以下为目标目录设计；实施前要先检查仓库现有模块，复用现有的任务轨迹 schema、评测工具和数据门，避免重复造轮子。

```text
aimeng/
  schemas/
    idea_record.schema.json
    plastic_module.schema.json
    task_state.schema.json
  aimeng/
    thinking/
      policy.py              # 选择直接回答、探索、分解或验证
      idea_pool.py           # 候选管理、去重、上限
      hypothesis.py          # 前提、预测、反例与状态
      critique.py            # 生成可检查的反例/质疑点
      verification.py        # 验证器协议与结果归一化
      stopping.py            # 预算、收益和停止规则
    plasticity/
      interface.py           # PlasticBackend 协议，不绑定特定推理后端
      trial.py               # 临时候选训练/冻结/回滚
      evaluate.py            # 基线、候选、回归和迁移评测
    neuron_library/
      manifest.py            # 模块元数据与兼容性
      store.py               # 原子写入、索引、hash、状态迁移
      loader.py              # 预算/兼容检查后加载和卸载
    memory/
      episodic.py            # 持久任务状态和轨迹
      strategy.py            # 带证据的策略经验
      knowledge.py           # 有来源的事实知识
  scripts/
    run_idea_cycle.py
    validate_idea_records.py
    evaluate_idea_cycle.py
    evaluate_plastic_module.py
  tests/
    test_idea_state_machine.py
    test_idea_verification.py
    test_stopping_and_resume.py
    test_module_manifest.py
    test_module_compatibility.py
    test_module_rollback.py
    test_no_unverified_promotion.py
```

建议先沿用仓库已存在的 `schemas/task_trajectory.schema.json`、`scripts/validate_training_data.py`、固定 held-out/regression 评测集与训练数据门；不要再建一套与现有 task_id、split、provenance 不兼容的体系。上述目录是建议接口，不表示文件已经存在。

### 核心接口草案

```python
@dataclass
class Idea:
    idea_id: str
    task_id: str
    statement: str
    assumptions: list[str]
    predictions: list[str]
    counterexamples_to_seek: list[str]
    epistemic_status: str
    evidence_refs: list[str]
    verification_plan: dict

@dataclass
class VerificationResult:
    status: str  # pass, fail, inconclusive, unavailable
    verifier_id: str
    scope: dict
    evidence_refs: list[str]
    metrics: dict
    limitations: list[str]

class Verifier(Protocol):
    def verify(self, idea: Idea, context: dict) -> VerificationResult: ...

class PlasticBackend(Protocol):
    def create_trial(self, base_revision: str, budget: dict) -> str: ...
    def train_trial(self, trial_id: str, approved_data_ref: str, config: dict) -> dict: ...
    def freeze_trial(self, trial_id: str) -> dict: ...
    def evaluate_trial(self, trial_id: str, suites: dict) -> dict: ...
    def rollback_trial(self, trial_id: str) -> None: ...

class ModuleStore(Protocol):
    def register_candidate(self, manifest: dict, artifact_paths: list[str]) -> str: ...
    def validate_compatibility(self, module_id: str, base_revision: str) -> dict: ...
    def promote(self, module_id: str, evaluation_ref: str) -> None: ...
    def reject(self, module_id: str, reason: str) -> None: ...
```

接口只规定契约；真实实现需要在现有仓库语言、依赖和模型后端上确认后落地。任何后端不支持的功能必须明确返回 `unavailable`，不能静默伪装成功。

## 10. 主循环伪代码

```python
def run_task(task, core_model, stores, verifiers, budget):
    state = stores.state.create(task, budget)
    baseline = core_model.revision
    try:
        context = stores.memory.retrieve_relevant(task, budget)
        state.transition("CONTEXT_READY")

        if core_model.choose_mode(task, context) == "direct":
            ideas = [core_model.solve_direct(task, context)]
        else:
            ideas = core_model.generate_diverse_ideas(
                task, context, max_ideas=budget.max_ideas
            )
        stores.ideas.save_candidates(ideas)
        state.transition("EXPLORE")

        for idea in ideas:
            if budget.exhausted():
                state.transition("ABORTED_BUDGET")
                break
            critique = core_model.propose_counterexamples(idea, context)
            result = verifiers.select(idea).verify(idea, {
                "task": task, "context": context, "critique": critique
            })
            stores.results.append(idea.idea_id, result)
            idea.update_status_from_evidence(result)

        best = core_model.compare_ideas(ideas, stores.results.for_task(task.id))
        answer = core_model.prepare_scoped_answer(best)

        if core_model.should_try_plasticity(task, best, budget):
            trial = stores.plasticity.create_isolated_trial(baseline, budget)
            stores.plasticity.train_only_on_approved_data(trial)
            report = stores.plasticity.compare_against_baseline(
                trial, frozen_eval_suites=True
            )
            if report.passes_all_hard_gates:
                stores.module_library.register_candidate(trial, report)
            stores.plasticity.rollback_or_discard(trial)

        stores.memory.consolidate_verified_outcomes(task, ideas, answer)
        state.commit_result(answer, baseline_revision=baseline)
        state.transition("DONE")
        return answer
    except RecoverableError as exc:
        state.checkpoint_error(exc)
        state.transition("FAILED_RECOVERABLE")
        raise
```

伪代码是流程约束，不是可以直接执行的现有实现。生产实现还要补全原子写入、异常分类、锁、重复任务幂等性、超时取消、随机种子、进程重启恢复与资源遥测。塑性试验必须在独立候选工件中进行；即使通过评测，v0 也先注册外部模块，不直接覆盖核心权重。

## 11. 训练数据怎么教会模型这样思考

不要只训练“最终答案”。训练样本应覆盖可观察、可验证的行为：

- 开放问题：多种机制不同的候选，而非同义改写。
- 假设标记：把事实、推导、假设、未知分开。
- 反例任务：给出貌似合理但存在漏洞的方案，让模型定位漏洞并修正。
- 预测任务：要求提出可区分竞争假设的测试结果。
- 失败恢复：从工具错误/测试失败中找出原因，更新方案。
- 停止任务：证据不足或预算耗尽时正确停止，而不是编造确定性。
- 迁移任务：改变表面形式、输入范围和组合条件，测试能否泛化。
- 自我改进任务：比较改进前后策略，只有固定评测证实收益才标记成功。

每条训练候选记录来源、任务族、难度、预期验证器、证据和数据拆分。新生成数据默认 `pending/unassigned/ineligible`；只有独立验证、去重、拆分和审查全部完成后才可训练。测试集不能参与候选生成、提示调参或模块筛选。

## 12. 实验与验收：先证明机制有用

### 实验 0：纯软件闭环，不训练权重
用小型合成任务测试状态机、idea schema、预算、恢复、验证状态与记忆写入。验收：失败注入后可恢复；未经验证的想法绝不被标记 verified；重复运行不重复提交；预算耗尽能保存现场。

### 实验 1：想法质量基线
比较直接回答、单候选、多个多样候选 + 反例检查三种策略。固定相同任务集、模型版本、token/时间预算。测量正确率、任务完成率、独立验证通过率、候选多样性、错误自信率、成本。测试集与生成/开发数据隔离。

### 实验 2：经验是否真的改变后续行为
对照组不启用经验检索，实验组启用已验证策略经验。用未见过的同类任务测试迁移，并使用经验撤销/过期测试避免陈旧经验持续误导。若只提高重复题而不提高新任务，不算充分成功。

### 实验 3：adapter 可塑性
先在训练机的小规模可训练模型上做。固定数据、训练步数、随机种子和评测套件；比较 baseline 与 adapter 候选。必须报告旧能力回归、不同表面形式、组合任务、延迟、训练成本和峰值内存。未通过门槛的 adapter 不入库。

### 实验 4：模块复用与兼容
跨多个任务重用同一已验证模块，检查成功率、加载成本、错误模块拒绝、版本不兼容、哈希损坏、卸载和基线回退。模块只在原题有效则标注窄范围，不宣称通用能力。

### 实验 5：真正的动态神经元增长
只有前面阶段稳定后再做。需要能观察新增单元/连接的结构差异、初始化、训练信号、剪枝、序列化、跨任务复用和后端真实执行证据。只在 Python 元数据中新增“神经元”记录不算神经网络拓扑真的增长。

### 预注册的硬门槛

- 正确率/独立验收率达到预定目标，且相对基线有可信提升。
- 旧能力回归不超过预定容忍值。
- 测试任务与训练/模块选择数据无泄漏。
- 模块可重复加载、校验和回滚。
- 记录运行时峰值 RAM 与延迟；Android 整应用峰值必须低于 4096 MiB，晋级目标低于 3800 MiB。
- 任一硬门槛失败即拒绝晋级，不允许用其他指标抵消。
- 每个结论都注明模型版本、数据哈希、评测版本、设备/后端与样本数量。

## 13. 实施顺序

1. **仓库盘点**：复用现有轨迹 schema、在线学习闭环、评测与数据门；确定实际包结构和依赖。
2. **schema + 验证器**：实现 idea record、模块 manifest、状态机和严格状态校验；先加单元测试。
3. **纯软件想法循环**：接入当前核心模型调用接口，完成候选、反例、验证、停止与可恢复状态；不改权重。
4. **评测脚本**：固定数据快照，比较 direct / diverse / critique 策略并生成 JSON 报告。
5. **经验记忆**：仅保存有证据、带适用范围的策略经验，验证检索是否改善未见任务。
6. **adapter 插槽**：确认真实训练 checkpoint 与后端支持后，实现隔离训练、候选冻结、评测、拒绝和回滚。
7. **模块库**：实现原子保存、hash、兼容性、版本、预算加载、卸载和回退。
8. **目标设备资源门**：测量整应用实际峰值内存；不以模型文件大小或逻辑预算代替实测。
9. **动态拓扑研究**：在静态 adapter/子图已证实有收益后再实现真实神经元与连接增长。

## 14. 当前未解决的工程问题

实施前必须从仓库/实际后端核实，不能凭文档假定：
- 核心模型的真实训练架构、权重格式与训练 checkpoint 来源。
- 运行时是否支持 adapter 加载、卸载、层级插入和并发安全。
- 后端能否导出或执行子图，以及是否能真实改变拓扑。
- 如何在目标 Android 后端测量整应用峰值 RAM、临时训练内存和模块驻留。
- 独立验证器有哪些真实可用实现及其失败模式。
- 模块版本与 tokenizer、层索引、张量形状、量化方式和运行时 ABI 的兼容契约。

这些问题未确认前，软件闭环可以先做，权重适配与动态神经元增长必须保持为明确的未实现接口。

## 15. 最终设计原则

**先允许想法自由，再要求证据严格；先让经验改变下一次行为，再尝试改变参数；先验证可复用 adapter，再研究真正动态神经元；任何变化都可追溯、可测量、可拒绝、可回滚。**

这套规格的成功标准不是“模型说自己像人”，而是它能在固定实验中提出更多有差异的可检验方案、发现自己的错误、把有效经验迁移到新任务，并在不破坏旧能力与资源边界的前提下复用经过验证的计算模块。


## 16. Initial repository implementation status

The first software-contract layer has now been added on the research branch:
- `schemas/idea_record.schema.json`: idea records with explicit assumptions, predictions, counterexamples, epistemic status, provenance, and scoped verification result.
- `schemas/plastic_module.schema.json` + `scripts/validate_plastic_module.py`: module manifest contract and executable acceptance gate; an `accepted` module requires a passing validation status, non-empty evaluation scope/report reference, and rollback target.
- `schemas/task_state.schema.json`: persistent task-state document contract.
- `scripts/validate_idea_records.py`: JSONL structural validator; `verified_for_scope` requires a passing verifier result, verifier identity/version, scope, and evidence references.
- `scripts/idea_cycle_state.py`: atomic JSON checkpoint writes, legal state transitions, recoverable failure/resume, artifact hashes, and bounded counters. It does not call a model or modify weights.
- `tests/test_idea_records.py`, `tests/test_idea_cycle_state.py`, and `tests/test_plastic_module_manifest.py`: contract, state-machine, and module-promotion tests.
- `.github/workflows/idea-cycle-tests.yml`: CI job for schema JSON parsing and those unit tests.

These are the initial contracts and persistence layer, not the complete thinking runtime. They have not yet been confirmed by a successful CI run in this document. Next, connect a real core-model adapter to idea generation/critique, add independent verifier adapters, create a fixed evaluation suite, and only later implement isolated adapter training and module loading. No dynamic neuron growth, model-weight update, real backend integration, or Android memory result is claimed at this stage.
