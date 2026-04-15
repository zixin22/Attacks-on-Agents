# AutoDan SeeAct — run report

Generated: 2026-04-14 20:21:23

## Artifacts
- results dir: `/Users/zixinrao/Desktop/rap-fragment/Attacks-on-Agents/webshop/AutoDan/AutoDan_seeact/experiments/seeact_smoke/results`
- `best_triggers.json` — best individuals + final test score
- `optimization_log.json` — per-generation stats
- `population_history.json` — population snapshots

## Summary
- generations: 2
- best rows: 2
- score mean: 0.000
- score max: 0.000
- final best score: 0.000

## Commands

```bash
cd AutoDan_seeact   # this folder
python run_optimization.py
```

```bash
python -c "from utils import print_optimization_summary; print_optimization_summary('results/optimization_1/best_triggers.json')"
```

API key: `openai_key.txt` in this folder or env `OPENAI_API_KEY`.