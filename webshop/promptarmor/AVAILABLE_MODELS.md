# PromptArmor 可用模型列表

PromptArmor 支持通过环境变量 `PROMPTARMOR_MODEL` 配置不同的模型。由于使用 OpenAI 兼容的 API，理论上支持所有兼容 OpenAI API 格式的模型。

## 已验证支持的模型

### OpenAI 官方模型

1. **gpt-4o** (默认)
   - 使用 ChatCompletion API
   - 推荐用于最佳检测效果
   - 命令：`export PROMPTARMOR_MODEL="gpt-4o"`

2. **gpt-4-0613**
   - 使用 ChatCompletion API
   - 命令：`export PROMPTARMOR_MODEL="gpt-4-0613"`

3. **gpt-3.5-turbo-instruct**
   - 使用 Completion API（与其他模型不同）
   - 成本更低，速度更快
   - 命令：`export PROMPTARMOR_MODEL="gpt-3.5-turbo-instruct"`

### 其他兼容模型

由于使用 OpenAI 兼容的 API (`http://152.53.53.64:3000/v1`)，理论上支持：

- **Claude 系列**（如果 API 兼容）
- **Llama 系列**（如果通过兼容 API 提供）
- **其他开源模型**（如果通过兼容 API 提供）

## 使用方法

### 方法1：环境变量（推荐）

```bash
# 使用 gpt-4o（默认）
python promptarmor/batch_test_attacks.py

# 使用 gpt-3.5-turbo-instruct
export PROMPTARMOR_MODEL="gpt-3.5-turbo-instruct"
python promptarmor/batch_test_attacks.py

# 使用 gpt-4-0613
export PROMPTARMOR_MODEL="gpt-4-0613"
python promptarmor/batch_test_attacks.py
```

### 方法2：在脚本中修改配置

```python
from promptarmor.config import PromptArmorConfig
from promptarmor.detector import PromptArmorDetector

# 创建自定义配置
config = PromptArmorConfig()
config.DETECTION_MODEL = "gpt-3.5-turbo-instruct"  # 或其他模型

# 使用自定义配置初始化检测器
detector = PromptArmorDetector(config)
```

### 方法3：在 batch_test_attacks.py 中修改

编辑 `batch_test_attacks.py`，在初始化检测器前设置：

```python
import os
os.environ["PROMPTARMOR_MODEL"] = "gpt-3.5-turbo-instruct"

from promptarmor.detector import PromptArmorDetector
detector = PromptArmorDetector()
```

## 模型选择建议

### 检测准确度优先
- **推荐**: `gpt-4o`
- 理由：最强的理解能力，能更准确识别复杂的注入模式

### 成本优先
- **推荐**: `gpt-3.5-turbo-instruct`
- 理由：成本更低，速度更快，但准确度可能略低

### 平衡选择
- **推荐**: `gpt-4-0613`
- 理由：在准确度和成本之间取得平衡

## 当前 API 配置

- **API Base**: `http://152.53.53.64:3000/v1`
- **API Key**: 从 `OpenAI_api_key.txt` 读取

## 测试不同模型

可以创建一个测试脚本来比较不同模型的效果：

```bash
# 测试 gpt-4o
export PROMPTARMOR_MODEL="gpt-4o"
python promptarmor/batch_test_attacks.py
mv promptarmor/attack_detection_results.json promptarmor/results_gpt4o.json

# 测试 gpt-3.5-turbo-instruct
export PROMPTARMOR_MODEL="gpt-3.5-turbo-instruct"
python promptarmor/batch_test_attacks.py
mv promptarmor/attack_detection_results.json promptarmor/results_gpt35.json

# 比较结果
python -c "import json; r1=json.load(open('promptarmor/results_gpt4o.json')); r2=json.load(open('promptarmor/results_gpt35.json')); print('gpt-4o:', sum(1 for v in r1['results'] for i in v['instructions'].values() if i.get('is_injected'))); print('gpt-3.5:', sum(1 for v in r2['results'] for i in v['instructions'].values() if i.get('is_injected')))"
```

## 注意事项

1. **API 兼容性**: 确保你的 API 服务支持所选模型
2. **成本**: 不同模型的 API 调用成本不同
3. **性能**: `gpt-3.5-turbo-instruct` 使用不同的 API 端点（Completion vs ChatCompletion）
4. **准确度**: 更强大的模型通常有更好的检测准确度


