# AutoDan SeeAct (standalone bundle)

Evolutionary search over **SeeAct-style trigger instructions**: propose variants (LLM + crossover + mutation), score them against `data_seeact/dataset.txt` using an OpenAI-compatible chat model, keep elites, repeat.

**Research-only.** Do not use against systems you do not own or lack authorization to test.

---

## What you need

1. **Python 3.10+** (recommended).
2. **Dependencies** (minimal run):

   ```bash
   pip install -r requirements.txt
   ```

   For better similarity scoring / plots, optionally install:

   ```bash
   pip install scikit-learn sentence-transformers matplotlib
   ```

3. **OpenAI API key** — either:
   - Environment variable: `OPENAI_API_KEY` (or `API_KEY`), **or**
   - File: **`openai_key.txt`** in **this same directory** as `run_optimization.py` (repository ships an empty placeholder; paste your key as a single line and save).

   Resolution order: env vars → `openai_key.txt` → `OpenAI_api_key.txt` (same folder).

4. **Data** (included under `data_seeact/`):
   - `dataset.txt` — host / fragment / masked / target pairs  
   - `trigger_instruction_short_seed.txt` — blank-line separated seed triggers  
   - `attack_instruction.txt` — first line template for building the “similar task” block in the evaluator prompt  

All paths are **relative to this folder** (`config.base_dir`). You can zip **`AutoDan_seeact/`** alone and run from inside it; no parent `webshop` tree is required.

---

## Layout

```
AutoDan_seeact/
  run_optimization.py      # CLI entry
  config.py
  population.py
  proposer.py
  evaluator.py
  evolutionary_optimizer.py
  utils.py
  openai_key.txt           # put your key here (or use env)
  requirements.txt
  data_seeact/
    dataset.txt
    attack_instruction.txt
    trigger_instruction_short_seed.txt
    ...
  results/                   # created on run
    optimization_<id>/
      best_triggers.json
      optimization_log.json
      population_history.json
      config_used.json
      trigger_host_similarity.json   # when recorded
  experiments/               # only if you pass --experiment-name
    <name>/results/...
```

---

## Commands

From **`AutoDan_seeact/`**:

```bash
# Default run (new results/optimization_<n>/)
python run_optimization.py
```

```bash
# Overrides
python run_optimization.py -g 5 -p 12
```

```bash
# Named run → experiments/<name>/results/
python run_optimization.py -n seeact_smoke -g 2 -p 8
```

```bash
# Resume after Ctrl+C (checkpoint written to results dir)
python run_optimization.py -r experiments/seeact_smoke/results/checkpoint_interrupt.json
```

```bash
# Load JSON overrides (keys matching Config attributes)
python run_optimization.py -c path/to/config.json
```

```bash
# Smoke test (API + data; may incur small cost)
python run_optimization.py --test
```

---

## Pipeline (short)

1. **Load / split** `data_seeact/dataset.txt` → fixed **80% train** / **20% test** (order preserved).
2. **Seed population** — score each block in `trigger_instruction_short_seed.txt` on **all training pairs**; keep top **3** as initial individuals.
3. Each **generation**: Proposer builds candidates from elites → Evaluator scores each candidate on **all training pairs** (mean similarity of model output to each pair’s **target Instruction** field in `dataset.txt`) → population **selection / refill**.
4. **Stop** if `no_improvement_generations` reached (default **10**) or max generations completed.
5. **Final** best trigger is evaluated on the **test** split; scores written to `best_triggers.json` / log.

Model and API base live in `config.py` → `llm_config` (`model`, `temperature`, `max_tokens`, `api_base`). Point `api_base` at any OpenAI-compatible server if needed.

---

## Outputs

| File | Role |
|------|------|
| `best_triggers.json` | Best individuals, `final_best`, optional `test_set_score` |
| `optimization_log.json` | Per-generation stats + optional test-evaluation entry |
| `population_history.json` | Per-generation member dicts (+ optional test block) |
| `config_used.json` | Snapshot of config used for that run |
| `experiment_report.md` | Short auto-generated summary |
| `optimization_progress.png` | If matplotlib is installed and `--no-plots` not set |

---

## Optional scripts

- `evaluate_specific_trigger.py` — evaluate one trigger string.  
- `scripts/compute_trigger_host_similarity.py` — offline similarity helper.  
- `semantic_evaluator.py`, `coherence_evaluator.py`, `ner_processor.py` — extra utilities (not required for `run_optimization.py`).

---

## License / ethics

Provided for **authorized security research** only. Misuse is prohibited.
