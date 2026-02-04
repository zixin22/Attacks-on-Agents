# AutoDan SeeAct 优化算法与实现总结

本文件总结 `AutoDan_seeact` 目录中 AutoDan 的优化算法思想与当前代码实现方式，涵盖流程、关键模块、评分机制与结果产出。

## 1) 总体算法思路（进化式模板优化）

- 优化目标是 **attack quey 模板**，通过进化生成与筛选更有效的模板。
- 典型流程：初始化种群 → 生成候选 → 评价打分 → 选择更新 → 迭代 → 输出最优结果。
- `method.md` 描述了 **模板级采样评估** 的设想（每轮对模板在随机采样的训练场景上评估），但当前代码实现中 **初始化与评估实际对训练集全量评估**（详见下文实现细节）。

## 2) 关键模块与职责

### 2.1 `evolutionary_optimizer.py`（进化主流程）

- `EvolutionaryOptimizer.optimize()` 协调全流程：
  1. 初始化种群（`Population.initialize_from_seeds`）
  2. 生成候选（`Proposer.generate_candidates`）
  3. 评价候选（`Evaluator.evaluate_population`）
  4. 加入种群并进化（`Population.add_candidates` → `evolve_population`）
  5. 记录日志与快照
  6. 终止条件：连续多代无改进

### 2.2 `population.py`（种群管理）

**初始化（模板级精英筛选）**
- 读取种子模板：`data_seeact/attack_query_short_seed.txt`
- 读取并准备训练集：`Evaluator._load_and_split_dataset`
  - 当前实现中训练集为全部 dataset（未再拆分）
- 对每个模板计算平均得分（`evaluate_goal_achievement` 对训练集所有 pair 评估）
- 选取 Top-3 作为精英个体（`Individual`）

**进化更新**
- 选择精英（`elite_size`）
- 其余席位按分数从高到低补齐
- 若仍不足，使用精英进行轻微变异填充
- 多样性用不同 prompt 占比衡量

### 2.3 `proposer.py`（候选生成）

候选来源 3 类：

1. **LLM 改写**（针对精英）
   - 交换/重排三要素
   - 同义改写
   - 扩展改写
2. **交叉**（模板片段组合）
   - 优先在预设关键词上拼接
   - 无交叉点时做句子级拼接
3. **变异**
   - 替换、插入、删除、同义变换

过滤规则：
- 长度上限（`max_prompt_length`）
- 必须含 `<>`、`{Masked query}`、`{host query}`、`task`
- 保护 `< >` 与 Masked 结构完整性
- 去重

### 2.4 `evaluator.py`（评估与打分）

**数据集加载**
- 从 `data_seeact/dataset.txt` 读取各 Pair 字段
- 当前实现：训练集 = 全部 pair（无验证、无测试）

**评估流程**
- 构造 `carrier query`：将敏感片段随机插入 `host query`
- 构造 P0 Prompt（长指令模板 + formatted attack quey）
- 调用 LLM 生成响应（含重试与速率控制）
- 从 LLM 输出抽取任务描述
- 与 `target_instruction` 计算相似度作为得分（SentenceTransformer 或 TF-IDF）
- 评分缓存：相同 prompt 可复用结果
- 同时记录 attack quey 与 host query 的相似度统计

**注意**
- `evaluate_goal_achievement` 当前对训练集 **全部 pair** 求平均分。
- `evaluate_template_fairly` 支持采样评估，但当前初始化流程未使用。

### 2.5 `config.py`（配置与实验输出）

常用参数：
- `population_size`、`num_generations`、`elite_size`
- `llm_rewrite_variants`、`crossover_rate`、`mutation_rate`
- `max_prompt_length`、`request_interval`
- 输出目录：`results/optimization_{id}`

## 3) 实际实现流程（代码级，含具体字段）

> 下面用代码中的实际字段名展开：`host query`、`fragment`、`masked query`、`target_query`、`attack query`。

### 3.1 数据准备

从 `data_seeact/dataset.txt` 读取 Pair，得到结构：
```
pair = {
  host query: "...",               # 原始任务描述
  fragment: "...",                # 敏感片段（可多段逗号分隔）
  masked query: "...",            
  target_query: "..."        # 期望最终任务文本
}
```

### 3.2 attack quey 模板初始化（Population.initialize_from_seeds）

1. 读取种子模板 `attack_querie_instruction_short_seed.txt`，得到若干 `attack quey` 模板。
2. 对每个模板在**全部训练 pair** 上评估：
   - 用 `host query` 和 `fragment` 构造 `carrier query`
   - 将模板中的 `{host query}`、`{Masked query}` 用 `host query` 与 `masked query` 替换
   - 组成 P0 Prompt 后调用 LLM
   - 从 LLM 输出中提取最终任务文本，与 `target_instruction` 计算相似度作为得分
3. 取平均分最高的 Top-3 模板作为初代精英个体。

### 3.3 构造评估 Prompt（Evaluator._build_p0_prompt）

核心拼装逻辑：
1. `carrier query` = 在 `host query` 里随机插入 `<fragment>` 片段  
2. `formatted attack quey` = 替换占位符：
   - `{host query}` → 具体 `host query`
   - `{Masked query}` → `"{masked query}"`
3. P0 Prompt =（固定长系统指令）+ `carrier query` + `formatted attack quey`

### 3.4 评价打分（Evaluator.evaluate_goal_achievement）

对每个训练 pair：
1. 调用 LLM 生成响应 `llm_output`
2. 抽取响应中的任务文本：
   - 期望格式：`This is the task that is asked to complete and run: ...`
3. 计算与 `target_instruction` 的语义相似度  
   - SentenceTransformer 或 TF-IDF
4. 所有 pair 的相似度平均值 = 该 `attack quey` 的得分

### 3.5 进化循环（EvolutionaryOptimizer.optimize）

每一代：
1. **候选生成**（Proposer）
   - LLM 改写：重排/同义/扩展
   - 交叉：关键词或句子级拼接
   - 变异：替换/插入/删除/同义词
2. **过滤**：长度、占位符完整性、去重
3. **评价**：对所有候选执行 3.4 的评分流程
4. **种群更新**：
   - 精英保留（`elite_size`）
   - 高分补齐到 `population_size`
   - 若不足，用精英轻度变异填充

### 3.6 终止与输出

- 连续 `no_improvement_generations` 代无提升即停止
- 输出：
  - `best_attack_queries.json`
  - `optimization_log.txt`
  - `population_history.json`
  - `attack_querie_host_similarity.json`

## 4) 与 `method.md` 方法描述的关系

- `method.md` 设想的是 **模板级“采样评估 + 精英筛选 + 代际进化”**；
- 当前实现 **保留了模板级优化的结构**，但在初始化与评估环节 **更偏向全量评估（对训练集所有 pair）**；
- 如果需要完全一致的采样评估，可在 `Population.initialize_from_seeds` 或 `Evaluator.evaluate_goal_achievement` 中引入随机采样逻辑。

## 5) 输出文件说明

- `best_attack_querie.json`：当前/最终最优个体
- `optimization_log.txt`：每代统计与进度
- `population_history.json`：种群历史快照
- `attack_querie_host_similarity.json`：attack quey 模板与 host query 的相似度记录


