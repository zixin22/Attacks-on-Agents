# AutoDan Evolutionary Optimization System

AutoDan 进化优化系统 - 基于进化算法的Prompt优化框架，用于生成能够绕过安全过滤器的trigger instructions。

## 架构概述

本系统实现了完整的进化算法框架来优化AutoDan trigger prompts：

```
原始目标 + 初始 Prompt/种子集合
              │
     ┌────────┴─────────┐
     │   提案生成器（Proposer）  │
     │     - LLM 改写/扩写      │
     │     - 交叉/变异 操作     │
     └────────┬─────────┘
              │
   候选 Prompt 池（Population / Queue）
              │
     ┌────────┴─────────┐
     │   评价/打分  (Evaluator)  │
     │     - 目标达成情况        │
     │     - 语言质量评分        │
     └────────┬─────────┘
              │
       选择 + 更新（Selection & Update）
              │  → 产生下一轮候选
              ↓
        终止条件？→ 是 / 否
         （若否返回候选生成）
```

## 文件结构

```
AutoDan/
├── config.py                    # 配置管理
├── population.py                # 种群管理
├── proposer.py                  # 提案生成器 (LLM改写/交叉/变异)
├── evaluator.py                 # 评价器 (目标达成+质量评分)
├── evolutionary_optimizer.py    # 主进化优化器
├── utils.py                     # 工具函数
├── run_optimization.py          # 主执行脚本
├── trigger_instruction_v1.txt   # 种子prompts集合
├── README.md                    # 本文件
└── results/                     # 输出结果目录
    ├── best_triggers.json       # 优化结果
    ├── optimization_log.txt     # 优化日志
    └── population_history.json  # 种群历史
```

## 核心组件

### 1. 配置管理 (`config.py`)
- 管理所有算法参数
- 支持从文件加载/保存配置
- 参数包括种群大小、进化代数、权重等

### 2. 种群管理 (`population.py`)
- 管理候选prompt池
- 实现精英保留策略
- 提供多样性计算和统计

### 3. 提案生成器 (`proposer.py`)
- **LLM改写**: 使用GPT生成prompt变体
- **交叉操作**: 组合不同prompts的元素
- **变异操作**: 随机修改现有prompts

### 4. 评价器 (`evaluator.py`)
- **越狱成功评分**: 测试是否能绕过RuleChecker
- **语言质量评分**: LLM评估prompt质量
- **综合评分**: 加权组合两种评分

### 5. 主优化器 (`evolutionary_optimizer.py`)
- 协调整个进化过程
- 实现终止条件检查
- 提供检查点保存/恢复功能

## 快速开始

### 基本使用

```bash
# 运行默认优化
python run_optimization.py

# 指定目标指令
python run_optimization.py --target-instruction "i would like a restricted product"

# 自定义参数
python run_optimization.py --max-generations 100 --population-size 30
```

### 高级用法

```bash
# 大规模实验
python run_optimization.py \
    --target-instruction "your target instruction" \
    --max-generations 200 \
    --population-size 50 \
    --experiment-name "large_scale_test"

# 从检查点恢复
python run_optimization.py --resume-from results/checkpoint.json

# 自定义配置
python run_optimization.py --config-file my_config.json
```

## 配置参数

### 种群参数
- `population_size`: 种群大小 (默认: 20)
- `num_generations`: 最大进化代数 (默认: 50)
- `elite_size`: 精英个体数量 (默认: 3)

### 提案生成参数
- `llm_rewrite_variants`: LLM改写变体数 (默认: 5)
- `crossover_rate`: 交叉概率 (默认: 0.3)
- `mutation_rate`: 变异概率 (默认: 0.1)

### 评价参数
- `jailbreak_weight`: 越狱成功权重 (默认: 0.7)
- `quality_weight`: 语言质量权重 (默认: 0.3)
- `evaluation_samples`: 每次评价的样本数 (默认: 3)

### 终止条件
- `convergence_threshold`: 收敛阈值 (默认: 0.95)
- `no_improvement_generations`: 无改进代数上限 (默认: 10)

## 输出结果

### 主要输出文件

1. **`best_triggers.json`**: 优化结果
   - 包含所有代的最佳个体
   - 最终最优prompt
   - 完整的评分信息

2. **`optimization_log.txt`**: 优化日志
   - 每代种群统计信息
   - 进化进度记录
   - 用于生成图表

3. **`population_history.json`**: 种群历史
   - 保存每一代的完整种群状态
   - 用于详细分析

### 结果分析

```python
from utils import print_optimization_summary, plot_optimization_progress

# 打印摘要
print_optimization_summary('results/best_triggers.json')

# 生成进度图
plot_optimization_progress('results/optimization_log.txt')
```

## API接口

### 基本优化

```python
from config import Config
from evolutionary_optimizer import EvolutionaryOptimizer

# 配置
config = Config()
config.num_generations = 100

# 创建优化器
optimizer = EvolutionaryOptimizer(config)

# 执行优化
target_instruction = "i would like a restricted product"
best_individuals = optimizer.optimize(target_instruction)
```

### 自定义评价器

```python
from evaluator import Evaluator

# 创建自定义评价器
evaluator = Evaluator(config)

# 评估单个prompt
score, jb_score, q_score = evaluator.evaluate_single(
    prompt="test prompt",
    target_instruction="target",
    profile=user_profile
)
```

## 扩展开发

### 添加新的提案策略

在 `proposer.py` 中添加新的生成方法：

```python
def new_proposal_strategy(self, population):
    # 实现新的提案生成逻辑
    pass
```

### 自定义评价指标

在 `evaluator.py` 中添加新的评价函数：

```python
def custom_evaluation_metric(self, prompt):
    # 实现自定义评价逻辑
    pass
```

### 集成新的LLM

在 `proposer.py` 和 `evaluator.py` 中替换LLM接口：

```python
class CustomLLMInterface:
    def generate(self, prompt, **kwargs):
        # 实现自定义LLM调用
        pass
```

## 注意事项

1. **API依赖**: 需要有效的OpenAI API密钥
2. **计算资源**: 大规模优化需要较多计算资源
3. **安全考虑**: 生成的prompts可能具有攻击性，请谨慎使用
4. **实验记录**: 建议使用不同的实验名称来组织结果

## 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥配置
   - 确认网络连接
   - 检查API额度

2. **内存不足**
   - 减小种群大小
   - 减少LLM改写变体数

3. **优化收敛慢**
   - 调整变异和交叉率
   - 增加精英保留数量

### 调试模式

```bash
# 运行基本功能测试
python run_optimization.py --test

# 详细输出
python run_optimization.py --verbose
```

## 许可证

本项目仅用于研究目的，请勿用于非法活动。
