# GuardAgent Pipeline 分析

## 一、整体目标

GuardAgent 是一个**AI Agent 保护系统**，用于为其他 AI Agent（如 EHR Agent、SeeAct Agent）提供安全防护和访问控制。它通过动态生成和执行守卫代码（guardrail code）来检查 Agent 的输入输出是否符合安全规则。

## 二、Pipeline 流程概览

```
用户请求 + Agent输入输出 
    ↓
任务分解 (Task Decomposition)
    ↓
示例检索 (Example Retrieval)
    ↓
生成守卫代码 (Generate Guardrail Code)
    ↓
执行代码 (Execute Code)
    ↓
错误处理 (Error Debugging) [如果出错]
    ↓
结果验证 (Result Validation)
    ↓
记忆更新 (Memory Update) [如果正确]
```

## 三、详细流程分析

### 1. 初始化阶段 (`main.py`)

**代码位置**: `main.py:19-47`

```python
# 创建两个 Agent：
# - chatbot: AssistantAgent，负责生成守卫代码
# - user_proxy: GuardAgent，负责执行和管理整个流程
```

**关键步骤**：
- 加载配置（模型、API密钥等）
- 初始化长期记忆（从 `CodeGEN_Examples` 加载历史示例）
- 注册执行函数（`run_code_ehragent` 或 `run_code_seeact`）

### 2. 任务分解 (`task_decomposition`)

**代码位置**: `guardagent.py:47-83`

**功能**：使用 LLM 将用户请求和 Agent 输入输出分解为具体的检查子任务

**实现方式**：
- 使用 `Example_Decomposition` 提示词模板
- 调用 OpenAI API 生成子任务列表
- 返回结构化的任务分解结果

**示例**：
```
输入：用户请求 + Agent规格 + Agent输入输出
输出：任务分解，如：
- 识别用户角色（physician/nursing/general admission）
- 提取需要访问的数据库和列
- 检查访问权限
```

### 3. 示例检索 (`retrieve_examples`)

**代码位置**: `guardagent.py:85-101`

**功能**：从长期记忆中检索最相似的示例

**实现方式**：
- 使用 **Levenshtein 距离**计算相似度
- 选择距离最小的前 `num_shots` 个示例
- 格式化示例为提示词模板

**相似度计算**：
```python
levenshtein_dist[i] = Levenshtein.distance(agent_input, mem_input) + 
                      Levenshtein.distance(agent_output, mem_output)
```

### 4. 生成初始消息 (`generate_init_message`)

**代码位置**: `guardagent.py:103-127`

**功能**：整合所有信息，生成发送给 chatbot 的初始消息

**流程**：
1. 调用 `task_decomposition` 获取子任务
2. 调用 `retrieve_examples` 获取相似示例
3. 使用 `GuardAgent_Message_Prompt` 模板格式化消息
4. 返回包含示例、输入输出、子任务的完整提示

### 5. 对话启动 (`initiate_chat`)

**代码位置**: `guardagent.py:139-142`

**功能**：启动与 chatbot 的对话，chatbot 会基于提示生成守卫代码

**流程**：
- `initiate_chat` → `generate_init_message` → `send` → chatbot 接收消息
- chatbot 生成 Python 代码（守卫代码）
- 代码通过函数调用返回给 GuardAgent

### 6. 代码执行 (`execute_function`)

**代码位置**: `guardagent.py:192-247`

**功能**：执行 chatbot 生成的守卫代码

**实现细节**：
- 解析函数调用的参数（JSON格式）
- 调用注册的执行函数（`run_code_ehragent` 或 `run_code_seeact`）
- 捕获执行错误

**执行函数** (`toolset_high.py`):
- `run_code_ehragent`: 执行 EHR Agent 的守卫代码
  - 检查数据库访问权限
  - 返回：`label`, `inaccessible_db`, `guardrailed_answer`
- `run_code_seeact`: 执行 SeeAct Agent 的守卫代码
  - 检查规则违反
  - 返回：`action_denied`, `inaccessible_actions`, `guardrailed_answer`

### 7. 错误调试 (`error_debugger`)

**代码位置**: `guardagent.py:158-190`

**功能**：当代码执行出错时，使用 LLM 诊断错误原因

**实现方式**：
- 使用 `CodeDebugger` 提示词模板
- 调用 OpenAI API 分析错误
- 返回错误原因，附加到错误信息中

### 8. 结果验证 (`main.py:192-270`)

**功能**：验证 GuardAgent 的输出是否正确

**EHR Agent 验证**：
- 检查 `label` 是否正确
- 如果 `label=1`，检查 `inaccessible_db` 是否匹配
- 如果 `label=0`，检查 `guardrailed_answer` 是否匹配

**SeeAct Agent 验证**：
- 检查 `action_denied` 是否正确
- 如果 `action_denied=1`，检查违反的规则是否匹配

### 9. 记忆更新 (`main.py:274-286`)

**代码位置**: `main.py:274-286`

**功能**：如果验证正确，将当前示例添加到长期记忆

**存储内容**：
```python
{
    "agent input": agent_input,
    "agent output": agent_output,
    "subtasks": subtasks,
    "code": code
}
```

## 四、核心组件

### 1. GuardAgent 类 (`guardagent.py:14-253`)

继承自 `UserProxyAgent`，主要方法：
- `task_decomposition`: 任务分解
- `retrieve_examples`: 示例检索
- `generate_init_message`: 生成初始消息
- `execute_function`: 执行函数
- `error_debugger`: 错误调试
- `update_memory`: 更新记忆

### 2. 工具函数 (`tools.py`)

- `check_access`: 检查数据库访问权限
- `check_rule`: 检查规则违反

### 3. 提示词模板 (`prompts_guard.py`)

- `Example_Decomposition`: 任务分解提示
- `GuardAgent_Message_Prompt`: 生成守卫代码的提示
- `CodeDebugger`: 错误调试提示
- `CodeHeader`: 代码头部（导入工具函数）

### 4. 执行函数 (`toolset_high.py`)

- `run_code_ehragent`: EHR Agent 代码执行器
- `run_code_seeact`: SeeAct Agent 代码执行器

## 五、关键技术点

### 1. Few-shot Learning
- 使用长期记忆存储历史示例
- 通过 Levenshtein 距离检索相似示例
- 动态更新记忆，实现持续学习

### 2. 代码生成与执行
- LLM 生成 Python 守卫代码
- 动态执行代码并捕获结果
- 错误时自动调试和修复

### 3. 多 Agent 协作
- GuardAgent（UserProxyAgent）负责流程管理
- chatbot（AssistantAgent）负责代码生成
- 通过 AutoGen 框架实现 Agent 间通信

### 4. 领域适配
- 支持两种不同的 Agent 类型：
  - **EHR Agent**: 医疗数据库访问控制
  - **SeeAct Agent**: Web 导航安全规则检查
- 通过不同的执行函数和验证逻辑适配不同场景

## 六、数据流

```
输入数据 (dataset)
    ↓
提取 agent_input, agent_output (utils.py)
    ↓
GuardAgent 处理
    ├─ 任务分解 (LLM)
    ├─ 示例检索 (Levenshtein)
    └─ 生成守卫代码 (LLM)
    ↓
执行代码 (toolset_high.py)
    ↓
验证结果 (main.py)
    ↓
更新记忆 (如果正确)
```

## 七、关键代码路径

1. **主流程**: `main.py:148-163` → `guardagent.py:139-142`
2. **消息生成**: `guardagent.py:103-127` → `task_decomposition` + `retrieve_examples`
3. **代码执行**: `guardagent.py:192-247` → `toolset_high.py:6-65` 或 `68-161`
4. **错误处理**: `guardagent.py:239-241` → `error_debugger`
5. **记忆更新**: `main.py:274-286` → `guardagent.py:249-251`

## 八、总结

GuardAgent 实现了一个**智能的、自适应的 Agent 保护系统**：

1. **动态性**：根据输入动态生成守卫代码，而非硬编码规则
2. **学习性**：通过长期记忆实现 few-shot learning 和持续改进
3. **鲁棒性**：包含错误处理和自动调试机制
4. **通用性**：支持多种 Agent 类型和场景

整个系统通过 LLM 的强大能力，将复杂的规则检查任务转化为代码生成和执行问题，实现了灵活且可扩展的 Agent 保护方案。

