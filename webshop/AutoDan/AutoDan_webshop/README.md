# AutoDan WebShop — 使用说明（精简）

优化流程与数据格式说明。算法与实验设计细节见同目录 [`method.md`](method.md)。

## 运行优化

在 **`AutoDan_webshop`** 目录下执行：

e.g.:

```bash
cd /path/to/Attacks-on-Agents/webshop/AutoDan/AutoDan_webshop
python run_optimization.py -g 10 -p 5
```

| 选项 | 含义 |
|------|------|
| `-g` / `--max-generations` | 最大进化代数，覆盖 `config.py` 中的 `num_generations` |
| `-p` / `--population-size` | 种群大小，覆盖 `config.population_size` |
| `-n` / `--experiment-name` | 实验名；非 `default_experiment` 时通过 `utils.setup_experiment_directory` 建独立结果目录 |
| `-c` / `--config-file` | 从 JSON 加载配置（存在则覆盖默认） |
| `--no-plots` | 不生成 `optimization_progress.png` |

默认会在 `results/optimization_<id>/` 下写入 `best_triggers.json`、`optimization_log.txt`、`optimization_log_full.txt`、`population_history*.json`、`config_used.json`、`experiment_report.md` 等（`experiment_id` 自增逻辑见 `config.py`）。

**复现注意**：`attack_instruction_template.txt` 中 **第一条** `attack_instruction = …` 赋值是唯一用于 attack 模板初始化的种子（`evaluator.py` / `population.py` / `symbol_proposer.py`）；换种子请改该文件并保证目标行在首条有效赋值。

## `data_webshop/` 数据文件

路径基准：`AutoDan_webshop/data_webshop/`。

### `dataset.txt`（主流程读取）

- **用途**：`evaluator._load_all_dataset_pairs` 解析为结构化样本，供训练/测试划分与打分。
- **结构**：纯文本；多个 `Pair N:` 块。解析器抽取（见 `evaluator.py`）：
  - `Host Instruction:`
  - `Sensitive Fragment:`
  - `fragment1:` / `fragment2:`（若无则把 `Sensitive Fragment` 从中截半作为两段）
  - `Masked Instruction:`
- **划分**（以代码为准）：加载全部 pair 后，**前 20 条**为训练、**第 21 条起**为测试（`evaluator._load_and_split_dataset`；注释中的「5/10」与当前切片不一致，以切片为准）。

### `trigger_instruction.txt`（主流程读取）

- **用途**：trigger 模板种子库；`Population.initialize_from_seeds` 在采样 pair 上评估并选精英进入进化。
- **结构**：多个模板用 **空行分隔**（`\n\n`）；占位符常见 `{Masked Instruction}`、`{host_instruction}`，须与 evaluator 拼接逻辑一致。

### `attack_instruction_template.txt`（主流程读取 — 种子）

- **用途**：attack 侧 Python f-string 模板的**唯一种子来源**；首条 `attack_instruction = …` 的右侧表达式用于初始化种群与 `SymbolProposer` 规范骨架（`evaluator._load_attack_template` 读第一行；`population` / `symbol_proposer` 通过 `load_attack_instruction_lines` 取列表首项，等价于首条赋值）。
- **结构**：与旧 `attack_instruction.txt` 相同，每行 `attack_instruction = f'...'` 或 `f"..."`；须含 `{host_instruction}`，以及 `{fragment}` 或 `{fragment1}`/`{fragment2}`（格式需满足 `attack_template_utils.py` 解析）。可保留多行，**只有第一条参与运行**。

### `attack_instruction.txt`（可选）

- **用途**：当前主流程 **不读取**；可作多模板备份或人工对照。实际种子以 `attack_instruction_template.txt` 为准。


## 相关代码入口

- 配置：`config.py`
- 优化主循环：`evolutionary_optimizer.py` + `run_optimization.py`
- 数据解析与评分：`evaluator.py`
- Trigger 初始化：`population.py` → `initialize_from_seeds`
- Attack 初始化：`population.py` → `initialize_attack_templates_from_file`
