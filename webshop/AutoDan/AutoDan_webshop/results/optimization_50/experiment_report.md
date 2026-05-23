# AutoDan Evolutionary Optimization Experiment Report

Generated at: 2026-01-14 05:03:30

## File Structure
- Result directory: results/optimization_50
- Best triggers: results/best_triggers.json
- Optimization log: results/optimization_log.txt
- Population history: results/population_history.json

## Optimization Summary
- Total generations: 5
- Number of best individuals: 5
.3f.3f.3f

## Usage

### 1. Run Optimization
```bash
cd AutoDan
python run_optimization.py
```

### 2. View Results
```bash
python -c "from utils import print_optimization_summary; print_optimization_summary('results/best_triggers.json')"
```

### 3. Plot Progress
```bash
python -c "from utils import plot_optimization_progress; plot_optimization_progress('results/optimization_log.txt')"
```