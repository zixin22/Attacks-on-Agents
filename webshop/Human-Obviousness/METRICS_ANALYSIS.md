# Human-Obviousness 文件夹指标分析

## 概述

`Human-Obviousness` 文件夹包含用于分析片段攻击指令和触发攻击指令相对于原始指令的"人类可察觉性"的代码。该分析从多个维度评估攻击指令是否容易被人类或自动化系统检测到。

## 计算的指标

### 1. Embeddings（嵌入向量）

**定义**：使用 SentenceTransformer 将文本转换为高维向量，用于衡量文本的语义相似度。

**计算方法**：
- **模型**：`sentence-transformers/all-MiniLM-L6-v2`
- **方法**：`model.encode(texts)`，返回 384 维向量
- **位置**：`generate_embeddings()`（第 189-202 行）

**统计指标**：
- `mean_norm`：嵌入向量 L2 范数的均值
- `std_norm`：嵌入向量 L2 范数的标准差
- 计算：`np.linalg.norm(emb)` 对每个嵌入向量

**可视化**：
- **PCA**：降维到 2 维，显示主成分分布
- **t-SNE**：非线性降维（样本数 ≤ 300 时）

### 2. Perplexity（困惑度）

**定义**：语言模型对文本的"意外程度"，值越低表示文本对模型来说越常见/可预测。

**计算方法**：
- **模型**：GPT-2（`gpt2`）
- **公式**：`PPL = exp(negative_log_likelihood)`
- **步骤**：
  1. 使用 GPT-2 tokenizer 编码文本
  2. 输入 GPT-2 计算负对数似然（`outputs.loss`）
  3. `perplexity = exp(neg_log_likelihood)`
- **位置**：`PerplexityCalculator.calculate_perplexity()`（第 63-92 行）

**统计指标**：
- `mean`：平均困惑度
- `std`：标准差
- `min`：最小值
- `max`：最大值
- `median`：中位数

### 3. Length（长度）

**定义**：文本的字符数和词数。

**计算方法**：
- **字符长度**：`len(text)`
- **词数**：`len(text.split())`
- **位置**：`calculate_lengths()`（第 226-230 行）

**统计指标**：
- `mean_words`：平均词数
- `std_words`：词数标准差
- `min_words`：最小词数
- `max_words`：最大词数

### 4. Embedding Distance（嵌入距离）

**定义**：各组的嵌入向量到 Group A（原始指令）质心的欧氏距离。

**计算方法**：
- **步骤**：
  1. 计算 Group A 的质心：`centroid_A = np.mean(embeddings['A'], axis=0)`
  2. 对每个组的每个嵌入向量：`distance = np.linalg.norm(emb - centroid_A)`
- **位置**：`visualize_distributions()`（第 372-393 行）

**用途**：衡量攻击指令与原始指令在语义空间中的偏离程度。

## 三个组别

- **Group A**：原始指令（Original Instructions）
- **Group B**：Fragment Attack Instructions（片段攻击指令）
- **Group C**：Trigger Attack Instructions（触发攻击指令）

## 可视化输出

### 1. `distribution_analysis.png`（2x2 子图）

#### 子图 1：Embedding Distribution (PCA)
- **数据**：每个样本的 PCA 降维后的 2D 坐标
- **计算**：
  - 输入：所有组的原始嵌入向量（384 维）
  - PCA 降维到 2 维：`pca_embeddings = PCA.fit_transform(combined_embeddings)`
  - 绘制：每个样本的 `(PC1, PC2)` 坐标点

#### 子图 2：Embedding Distribution (t-SNE)
- **数据**：每个样本的 t-SNE 降维后的 2D 坐标
- **计算**：
  - t-SNE 降维到 2 维：`tsne_embeddings = TSNE.fit_transform(combined_embeddings)`
  - 绘制：每个样本的 `(t-SNE1, t-SNE2)` 坐标点

#### 子图 3：Perplexity Distribution（直方图）
- **数据**：每个样本的困惑度值（原始值列表）
- **绘制**：
  - 输入：`perplexities[group_name]`（每个样本的困惑度）
  - 方法：`ax.hist(ppl_values, bins=30)`，展示分布密度

#### 子图 4：Length Distribution (Word Count)（直方图）
- **数据**：每个样本的词数（原始值列表）
- **绘制**：
  - 输入：`lengths[group_name][1]`（词数列表）
  - 方法：`ax.hist(word_lengths, bins=20)`，展示分布密度

### 2. `comparison_boxplots.png`（1x3 子图）

#### 子图 1：Perplexity Comparison（箱线图）
- **数据**：每个样本的困惑度值（原始值列表）
- **绘制**：
  - 输入：`perplexities[group_name]`（过滤掉 inf）
  - 方法：`ax.boxplot(box_data)`，展示中位数、四分位数、异常值等

#### 子图 2：Length Comparison（箱线图）
- **数据**：每个样本的词数（原始值列表）
- **绘制**：
  - 输入：`lengths[group_name][1]`（词数列表）
  - 方法：`ax.boxplot(box_data)`

#### 子图 3：Embedding Distance Comparison（箱线图）
- **数据**：每个样本到 Group A 质心的欧氏距离（原始值列表）
- **计算**：
  - 质心：`centroid_A = np.mean(embeddings['A'], axis=0)`
  - 距离：`dist = np.linalg.norm(emb - centroid_A)`（每个样本一个距离值）
- **绘制**：`ax.boxplot(box_data)`，展示距离分布

## 统计数据 vs 可视化数据

| 统计指标 | 保存在 JSON | 绘制到 PNG | PNG 中的实际数据 |
|---------|------------|-----------|----------------|
| `mean_norm` | ✅ | ❌ | 不绘制，仅统计 |
| `std_norm` | ✅ | ❌ | 不绘制，仅统计 |
| `mean` (perplexity) | ✅ | ❌ | 不绘制，仅统计 |
| `std` (perplexity) | ✅ | ❌ | 不绘制，仅统计 |
| **原始 perplexity 值** | ❌ | ✅ | 每个样本的困惑度值（直方图+箱线图） |
| **原始 word_count 值** | ❌ | ✅ | 每个样本的词数（直方图+箱线图） |
| **原始 embedding 向量** | ❌ | ✅ | 每个样本的嵌入向量（PCA/t-SNE 散点图） |
| **embedding distance** | ❌ | ✅ | 每个样本到 Group A 质心的距离（箱线图） |

**关键点**：
- PNG 图片绘制的是**原始数据点的分布**，而不是汇总统计值
- `mean_norm`、`std_norm` 等统计值仅保存在 `statistics.json` 中，不直接可视化
- 可视化方式：
  - **散点图**：展示每个样本在降维空间中的位置
  - **直方图**：展示值的分布密度
  - **箱线图**：展示分布的中位数、四分位数、异常值等

## 统计结果示例（statistics.json）

从结果看：
- **Group C（Trigger Attack）**的困惑度最低（mean=62.26），说明对模型来说更"可预测"
- **Group A（Original）**的困惑度最高（mean=526.53），变化最大（std=985.93）
- **Group C**的词数最多（mean=48.19），因为 trigger instruction 包含更多指令文本

## 分析维度总结

该分析从三个维度评估攻击指令的"人类可察觉性"：

1. **语义相似度（Embeddings）**：攻击指令与原始指令在语义空间的距离
2. **语言模型意外度（Perplexity）**：文本对语言模型来说是否"异常"
3. **文本长度（Length）**：攻击是否显著改变文本长度

这些指标用于评估攻击指令是否容易被人类或自动化系统检测到。

