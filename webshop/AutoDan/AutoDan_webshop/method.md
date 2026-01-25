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
elite_templates = sort_by_avg_score(all_templates)[:K]  # K=3

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

这种改进的模板优化方法平衡了科学性和实用性，通过采样评估确保公平比较模板质量，同时大幅降低计算成本。每个模板在5个随机采样场景下进行评估，既保持了统计可靠性，又避免了过高的计算开销，能够找到在各种场景下都表现优秀的通用trigger instruction模式。</content>
</xai:function_call">Write contents to c:\Users\22749\Desktop\rap-main\webshop\AutoDan\method.md
