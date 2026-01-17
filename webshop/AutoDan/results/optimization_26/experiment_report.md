# AutoDan 进化优化实验报告

生成时间: 2026-01-13 21:56:11

## 文件结构
- 结果目录: C:\Users\22749\Desktop\rap-main\webshop\AutoDan\results\optimization_26
- 最佳triggers: results/best_triggers.json
- 优化日志: results/optimization_log.txt
- 种群历史: results/population_history.json

## 优化结果摘要
- 总进化代数: 1
- 最佳个体数量: 1
.3f.3f.3f

## 使用说明

### 1. 运行优化
```bash
cd AutoDan
python run_optimization.py
```

### 2. 查看结果
```bash
python -c "from utils import print_optimization_summary; print_optimization_summary('results/best_triggers.json')"
```

### 3. 绘制进度图
```bash
python -c "from utils import plot_optimization_progress; plot_optimization_progress('results/optimization_log.txt')"
```