# GuardAgent KeyError 分析报告

## 错误信息
```
Exception: GuardAgent initiate_chat error: KeyError: GuardAgent initiate_chat failed
```

## 可能缺失的键

根据代码分析，GuardAgent 的 `generate_init_message` 方法需要以下必需的 context 键：

### 1. 必需的 Context 键（在 `generate_init_message` 中）
- `user_request` - 用户请求描述
- `agent_specification` - Agent 规范
- `agent_input` - Agent 输入（用户配置和指令）
- `agent_output` - Agent 输出（购买请求）
- `agent_task_deco_examples` - 任务分解示例

### 2. 必需的 Config 键（在 `config_list[0]` 中）
- `api_key` - OpenAI API 密钥
- `model` - 模型名称（如 "gpt-4", "gpt-4o"）

### 3. 必需的 Memory 键（在 `self.memory[i]` 中）
- `agent input` - Agent 输入
- `agent output` - Agent 输出
- `subtasks` - 子任务
- `code` - 生成的代码

## 检查点

### 在 `webshop_guard_agent.py` 中调用 `initiate_chat` 时：
```python
self.guard_agent.initiate_chat(
    self.chatbot,
    user_request=self.user_request,              # ✓ 应该已初始化
    agent_specification=self.agent_specification, # ✓ 应该已初始化
    agent_input=agent_input,                      # ✓ 从 _format_agent_input 生成
    agent_output=agent_output,                    # ✓ 从 _format_agent_output 生成
    agent_task_deco_examples=self.decomposition_examples, # ✓ 应该已初始化
)
```

### 可能的问题：
1. **`self.user_request` 未初始化**
   - 检查 `_load_webshop_prompts()` 是否被调用
   - 检查 `request_webshop.py` 中的 `User_Request_WebShop` 是否正确导入

2. **`self.agent_specification` 未初始化**
   - 检查 `_load_webshop_prompts()` 是否被调用
   - 检查 `request_webshop.py` 中的 `Specification_WebShop` 是否正确导入

3. **`self.decomposition_examples` 未初始化**
   - 检查 `_load_webshop_prompts()` 是否被调用
   - 检查 `request_webshop.py` 中的 `Decomposition_Examples` 是否正确导入

4. **`config_list[0]` 缺少键**
   - 检查 `model_config()` 函数是否正确返回包含 `api_key` 和 `model` 的字典

## 修复建议

1. **添加调试输出**：在调用 `initiate_chat` 前打印所有 context 键的状态
2. **改进错误处理**：从 KeyError 消息中提取缺失的键列表
3. **验证初始化**：确保所有必需的属性在 `__init__` 和 `_load_webshop_prompts()` 中正确初始化

## 下一步

重新运行攻击命令，新的错误处理会显示：
- 具体缺失的键列表
- 所有 context 键的状态
- 完整的错误堆栈跟踪

