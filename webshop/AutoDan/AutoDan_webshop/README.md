# AutoDan WebShop — （）

。 [`method.md`](method.md)。

## 

 **`AutoDan_webshop`** ：

e.g.:

```bash
cd /path/to/Attacks-on-Agents/webshop/AutoDan/AutoDan_webshop
python run_optimization.py -g 10 -p 5
```

|  |  |
|------|------|
| `-g` / `--max-generations` | ， `config.py`  `num_generations` |
| `-p` / `--population-size` | ， `config.population_size` |
| `-n` / `--experiment-name` | ； `default_experiment`  `utils.setup_experiment_directory`  |
| `-c` / `--config-file` |  JSON （） |
| `--no-plots` |  `optimization_progress.png` |

 `results/optimization_<id>/`  `best_triggers.json`、`optimization_log.txt`、`optimization_log_full.txt`、`population_history*.json`、`config_used.json`、`experiment_report.md` （`experiment_id`  `config.py`）。

****：`attack_instruction_template.txt`  **** `attack_instruction = …`  attack （`evaluator.py` / `population.py` / `symbol_proposer.py`）；。

## `data_webshop/` 

：`AutoDan_webshop/data_webshop/`。

### `dataset.txt`（）

- ****：`evaluator._load_all_dataset_pairs` ，/。
- ****：； `Pair N:` 。（ `evaluator.py`）：
  - `Host Instruction:`
  - `Sensitive Fragment:`
  - `fragment1:` / `fragment2:`（ `Sensitive Fragment` ）
  - `Masked Instruction:`
- ****（）： pair ，** 20 **、** 21 **（`evaluator._load_and_split_dataset`；「5/10」，）。

### `trigger_instruction.txt`（）

- ****：trigger ；`Population.initialize_from_seeds`  pair 。
- ****： ****（`\n\n`）； `{Masked Instruction}`、`{host_instruction}`， evaluator 。

### `attack_instruction_template.txt`（ — ）

- ****：attack  Python f-string ****； `attack_instruction = …`  `SymbolProposer` （`evaluator._load_attack_template` ；`population` / `symbol_proposer`  `load_attack_instruction_lines` ，）。
- ****： `attack_instruction.txt` ， `attack_instruction = f'...'`  `f"..."`； `{host_instruction}`， `{fragment}`  `{fragment1}`/`{fragment2}`（ `attack_template_utils.py` ）。，****。

### `attack_instruction.txt`（）

- ****： ****；。 `attack_instruction_template.txt` 。


## 

- ：`config.py`
- ：`evolutionary_optimizer.py` + `run_optimization.py`
- ：`evaluator.py`
- Trigger ：`population.py` → `initialize_from_seeds`
- Attack ：`population.py` → `initialize_attack_templates_from_file`
