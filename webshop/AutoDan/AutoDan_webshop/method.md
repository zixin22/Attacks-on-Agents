# AutoDan系统化模板优化方法

## 方法概述

本方法通过模板级别的元优化，实现对trigger instruction模板的系统化搜索和优化。与传统的个体级别进化不同，这种方法关注的是找到在各种场景下都表现优秀的通用模板模式。

## 核心理念

- **模板为中心**: 优化对象是trigger instruction模板，而不是填充了具体内容的个体
- **采样评估**: 每个模板在随机采样的一组训练场景下进行评估，计算平均表现
- **精英选择**: 选择平均得分最高的模板进行后续进化
- **泛化优先**: 确保选出的模板具有良好的泛化能力

## 详细流程

### 阶段1: 初始化 (Generation 0)

#### 1.1 数据准备
```
读取文件:
├── trigger_instruction.txt → 获得N个基础模板 (N≈10)
├── dataset.txt → 划分数据集:
│   ├── 训练集: 80个pair (用于进化评估)
│   └── 测试集: 19个pair (用于最终验证)
```

#### 1.2 采样评估
```
为每个模板随机采样K个训练pair进行评估:
for template in all_templates:          # N个模板
    sampled_pairs = random.sample(all_training_pairs, K)  # K=5
    for pair in sampled_pairs:
        # 组合: template + pair → 完整prompt
        filled_prompt = template.format(host_instruction=pair.host, ...)

        # 评估: 在该pair上测试prompt表现
        score = evaluate_prompt(filled_prompt, pair)

        # 记录: template在该pair上的得分
        scores[template].append(score)
```

#### 1.3 平均分计算
```
计算每个模板的平均表现:
for template in all_templates:
    # 该模板在采样训练pair上的平均得分
    avg_score = sum(scores[template]) / len(scores[template])  # 基于5个采样点

    # 记录统计信息
    template_stats[template] = {
        'avg_score': avg_score,
        'std_dev': calculate_std_dev(scores[template]),  # 稳定性
        'min_score': min(scores[template]),             # 最差表现
        'max_score': max(scores[template])              # 最好表现
    }
```

#### 1.4 精英选择
```
选择平均得分最高的K个模板作为Generation 0精英:
elite_templates = sort_by_avg_score(all_templates)[:K]  # K=5

返回: generation_0_elites = [template_A, template_B, template_C]
```

### 阶段2: 模板进化 (Generation 1, 2, ...)

#### 2.1 模板变异
```
基于当前精英模板生成新的模板变体:
new_templates = []
for elite_template in current_elites:
    # 多种变异策略
    variants = generate_variants(elite_template)
    new_templates.extend(variants)

# 变异策略包括:
# - LLM改写 (paraphrase, expand, synonym)
# - 结构变异 (改变关键词顺序, 添加修饰词)
# - 组合变异 (从多个精英模板组合元素)
```

#### 2.2 候选评估
```
对所有新模板进行采样评估:
candidate_scores = {}
for new_template in new_templates:
    # 随机采样K个训练pair进行评估
    sampled_pairs = random.sample(all_training_pairs, K)  # K=5
    scores = []
    for pair in sampled_pairs:
        filled_prompt = new_template.format(host_instruction=pair.host, ...)
        score = evaluate_prompt(filled_prompt, pair)
        scores.append(score)

    # 计算平均分
    avg_score = sum(scores) / len(scores)
    candidate_scores[new_template] = {
        'avg_score': avg_score,
        'detailed_scores': scores,
        'stability': calculate_stability(scores)
    }
```

#### 2.3 下一代精英选择
```
从候选模板中选择最好的K个:
next_elites = sort_by_avg_score(candidate_scores.keys())[:K]

# 加入原始精英 (可选 elitism)
if use_elitism:
    next_elites.extend(current_elites)
    next_elites = sort_by_avg_score(next_elites)[:K]

返回: generation_next_elites
```

### 阶段3: 最终验证

#### 3.1 测试集评估
```
对最终精英模板进行泛化能力测试:
test_results = {}
for elite_template in final_elites:
    # 从19个测试pair中随机采样5个进行评估
    sampled_test_pairs = random.sample(test_pairs, 5)
    test_scores = []
    for test_pair in sampled_test_pairs:
        filled_prompt = elite_template.format(host_instruction=test_pair.host, ...)
        score = evaluate_prompt(filled_prompt, test_pair)
        test_scores.append(score)

    test_results[elite_template] = {
        'avg_test_score': sum(test_scores) / len(test_scores),
        'test_stability': calculate_stability(test_scores),
        'train_vs_test_gap': abs(train_avg - test_avg)  # 过拟合检测
    }
```

#### 3.2 结果选择
```
选择在测试集上表现最好的模板:
best_template = max(test_results, key=lambda x: test_results[x]['avg_test_score'])

返回: best_template, performance_metrics
```

## 方法特点

### 优势
- **公平评估**: 每个模板在相同采样条件下测试，确保公平比较
- **平均分选择**: 选择整体性能稳定的模板，避免场景特定过拟合
- **模板进化**: 在模板空间进行系统化搜索
- **计算高效**: 采样评估大幅降低计算成本
- **统计可靠性**: 5个采样点提供可靠的性能估计

### 挑战
- **采样噪声**: 随机采样可能引入评估方差
- **代表性**: 5个样本可能无法完全代表80个训练pair的分布
- **随机性**: 不同运行可能因采样不同而有结果差异

### 适用场景
- **离线优化**: 有充足计算资源和时间的场景
- **关键应用**: 需要最高可靠性和泛化能力的应用
- **研究目的**: 深入理解prompt优化模式的学术研究

## 与当前方法的区别

### 当前方法 (个体级别进化)
- 优点: 计算效率高，多样性好，实用性强
- 缺点: 评估不够全面，可能错过最佳模板

### 本方法 (模板级别元优化)
- 优点: 评估全面，选择可靠，理论最优
- 缺点: 计算成本高，实现复杂度大

## 实现建议

### 分阶段实现
1. **原型阶段**: 先实现单代版本，验证基本逻辑
2. **优化阶段**: 添加缓存机制，减少重复评估
3. **扩展阶段**: 实现并行评估，加速计算过程

### 成本控制
1. **采样优化**: 不是所有pair都评估，可以分层采样
2. **增量评估**: 重用之前代的评估结果
3. **早期停止**: 如果收敛则提前终止

### 监控指标
- **收敛曲线**: 各代精英的平均分变化
- **稳定性指标**: 模板在不同场景下的表现方差
- **过拟合检测**: 训练集vs测试集性能差距

## 总结

这种改进的模板优化方法平衡了科学性和实用性，通过采样评估确保公平比较模板质量，同时大幅降低计算成本。每个模板在5个随机采样场景下进行评估，既保持了统计可靠性，又避免了过高的计算开销，能够找到在各种场景下都表现优秀的通用trigger instruction模式。

## 实验设置（Experiment settings）

以下为建议记录的实验设置项 —— 在每次跑实验前把这些项填入实验日志以保证可复现性与可比性。

- **代码与路径**
  - 项目根目录: `D:/rap-main/webshop`
  - 主要脚本: `main.py`、`AutoDan/AutoDan_webshop/coherence_evaluator.py`、`rule_and_profile/rule_checker.py`
  - 数据集示例: `dataset_baseline_test.json`、`dataset_test_10.json`、`retrieve_datasets.jsonl`

- **运行环境**
  - 操作系统: Windows (示例: 10.0.26100)
  - Conda 环境: `Perplexity`（用于 GPT‑2/coherence 评估）或 `rap-py310`（实验主环境）
  - 建议导出环境：`conda env export -n Perplexity > env_perplexity.yml`

- **关键软件版本（示例）**
  - Python: 3.10
  - transformers: 5.0.0
  - torch: 2.10.0 (cpu 或 gpu 版需记录完整标识)
  - numpy: 1.25.x
  - httpx: 0.28.x
  - matplotlib / seaborn / scipy: 版本同 env 导出

- **模型与代理配置**
  - 模型命名（CLI 用法示例）:
    - Claude: `--model claude-sonnet-4-5-20250929`
    - Gemini: `--model gemini-xxxx`
    - OpenAI‑style: `--model gpt-4o` / `--model gpt2`（本地 GPT‑2 用于 coherence）
  - Claude 初始化（在代码中）：
    - API key 文件优先路径：`AutoDan/AutoDan_webshop/Claude_api_key.txt` 或 `D:\rap-main\webshop\Claude_api_key.txt`
    - base_url（relay/proxy）：`http://148.113.224.153:3000/v1`
    - 在代码中以 OpenAI 客户端方式初始化：OpenAI(api_key=..., base_url=..., http_client=...)
  - Gemini 初始化：
    - API key 文件类似查找 `Gemini_api_key.txt`
    - base_url（relay/proxy）：`http://148.113.224.153:3000`
  - 通用代理（非 Claude/Gemini）的 base_url：`http://152.53.53.64:3000/v1`（见 fallback）

- **CLI flags（常用）**
  - `--model <model_name>`  : 模型名称（必须）
  - `--attack`             : 启用攻击流程（布尔 flag）
  - `--attack_dataset <path>` : 攻击数据集路径（默认示例：`dataset_baseline_test.json`）
  - `--cont_number <int>`  : 控制样本/对抗次数（示例：100）
  - `--enable_rule_checker` : 使用 RuleChecker（布尔 flag）
  - `--defense_mode <mode>` : 防御模式（示例：`rule_checker`）
  - `--skip_trigger`       : 跳过 trigger 攻击（布尔 flag）
  - `--output <path>`      : 输出目录/文件前缀（示例：`ablation/target_with_prompt_injection_claude4_5`）
  - `--cont_seed <int>`    : 随机种子（建议传入以保证复现）

- **Coherence evaluator（coherence_evaluator.py）配置**
  - 运行模式:
    - 简化模式（`use_simplified=True`）：不依赖 transformers，返回近似/代理分数（用于无法安装 transformers 的环境）
    - GPT‑2 模式（`use_simplified=False`）：加载 HuggingFace GPT‑2 模型计算真实 NLL（平均负对数似然）
  - 重要参数:
    - `model_name`（默认 `"gpt2"`） — 指定用于计算 NLL 的 causal LM
    - `device`（`cpu` 或 `cuda`）
    - `batch_size`（tokenization 与推理批次大小）
  - 输出指标:
    - `coherence_loss`（NLL，以 nats 计，越小越连贯）
    - `perplexity = exp(coherence_loss)`（直观不确定度）
  - 注意：NLL/Perplexity 受 tokenizer granularity 影响（不同 tokenizer/Piece 会导致不可直接比较）

- **实验随机性与复现**
  - 全局随机种子（建议设置）：`PYTHONHASHSEED`, `random.seed()`, `numpy.random.seed()`, `torch.manual_seed()`
  - 建议在每次实验日志记录：
    - seed 值
    - 数据集文件名与 sha256/hash
    - 模型名与 checkpoint（若使用自定义模型）

- **采样与进化超参（与方法正文一致，但需记录实际运行值）**
  - 训练集 pairs 数量（示例）：80
  - 测试集 pairs 数量（示例）：19
  - 每模板评估采样 K（示例）：5
  - 每代精英数量 K_elite（示例）：3
  - 变异策略说明（记录采用哪些策略、LLM paraphrase 是否使用外部 API）

- **批量/攻击实验设置（batch attack）**
  - 批次输出日志路径：`batch_attack/<batch_name>/`
  - 单次攻击最大上下文数 `--cont_number`（示例：100）
  - 并行化/多线程设置（如有）：记录 worker 数量与内存要求

- **硬件与资源**
  - CPU/GPU 型号与内存（如有 GPU：CUDA 版本）
  - 可用磁盘空间（记录用于大规模模型缓存）

- **实验结果与保存**
  - 日志（建议 JSON 结构）应包含：
    - `experiment_id`, `timestamp`, `seed`, `env_file`（conda 导出）, `cli_args`, `model_versions`, `dataset_hashes`
  - 输出文件（示例）:
    - `ablation/target_with_prompt_injection_claude4_5`（攻击输出）
    - `batch_attack/batch_attack_<n>/analysis.json`（批次分析汇总）
    - `promptarmor/promptarmor_osagent.json`（PromptArmor 检测结果）

按照上面模板把运行时实际参数补齐到实验日志（或 README）中，就能保证每次对比实验都是可复现且可审计的。
</xai:function_call">Write contents to c:\Users\22749\Desktop\rap-main\webshop\AutoDan\method.md
