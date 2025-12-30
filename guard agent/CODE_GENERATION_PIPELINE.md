# 守卫代码生成和执行 Pipeline 详解

## 一、澄清：谁在生成守卫代码？

**答案：Chatbot（AssistantAgent）使用 LLM 生成守卫代码，但 GuardAgent 负责准备提示词和管理整个流程。**

## 二、完整的 Pipeline 流程

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: GuardAgent 准备提示词                                │
└─────────────────────────────────────────────────────────────┘

GuardAgent.generate_init_message():
  1. 调用 task_decomposition() → 使用 LLM 生成任务分解
  2. 调用 retrieve_examples() → 检索相似示例
  3. 格式化提示词模板 GuardAgent_Message_Prompt
  4. 返回完整的提示词消息

代码位置: guardagent.py:103-127

┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: GuardAgent 发送消息给 Chatbot                       │
└─────────────────────────────────────────────────────────────┘

GuardAgent.initiate_chat():
  - 调用 generate_init_message() 生成提示词
  - 调用 send() 发送消息给 chatbot
  - chatbot 接收消息

代码位置: guardagent.py:139-142

┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: Chatbot 使用 LLM 生成守卫代码                        │
└─────────────────────────────────────────────────────────────┘

Chatbot (AssistantAgent):
  - 接收 GuardAgent 的提示词
  - 使用配置的 LLM（GPT-4 或 GPT-3.5-turbo）生成代码
  - 通过函数调用（function call）返回代码
  - 调用 "python" 函数，参数是生成的守卫代码

代码位置: main.py:35-39 (chatbot 初始化)
         config.py:16-38 (函数定义)

┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: GuardAgent 接收函数调用并执行代码                    │
└─────────────────────────────────────────────────────────────┘

GuardAgent.execute_function():
  - 接收 chatbot 的函数调用（function_call）
  - 解析参数，提取守卫代码
  - 调用注册的执行函数（run_code_ehragent 或 run_code_seeact）
  - 执行守卫代码
  - 返回执行结果

代码位置: guardagent.py:192-247

┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: 执行结果返回给 Chatbot                              │
└─────────────────────────────────────────────────────────────┘

Chatbot 接收执行结果:
  - 如果成功：返回 "TERMINATE"
  - 如果出错：可能继续对话，请求修复代码

代码位置: main.py:42 (终止条件)
```

## 三、详细代码流程

### 1. GuardAgent 生成提示词

```python
# guardagent.py:103-127
def generate_init_message(self, **context):
    # 1. 任务分解（使用 LLM）
    subtasks = self.task_decomposition(
        self.config_list[0],  # 使用第一个模型配置
        self.user_request,
        self.agent_specification,
        self.agent_input,
        self.agent_output,
        self.agent_task_deco_examples
    )
    
    # 2. 检索相似示例
    examples = self.retrieve_examples(self.agent_input, self.agent_output)
    
    # 3. 格式化提示词
    init_message = GuardAgent_Message_Prompt.format(
        examples=examples,
        agent_input=self.agent_input,
        agent_output=self.agent_output,
        subtasks=subtasks
    )
    return init_message
```

### 2. GuardAgent 发送消息给 Chatbot

```python
# guardagent.py:139-142
def initiate_chat(self, recipient: "ConversableAgent", ...):
    self._prepare_chat(recipient, clear_history)
    # 发送提示词给 chatbot
    self.send(self.generate_init_message(**context), recipient, silent=silent)
```

### 3. Chatbot 使用 LLM 生成代码

```python
# main.py:35-39
chatbot = autogen.agentchat.AssistantAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,  # ← 这里配置了 LLM 模型
)

# config.py:16-38
def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",  # ← 函数名称
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,  # ← 模型配置（GPT-4 或 GPT-3.5-turbo）
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list
```

**Chatbot 的工作流程**：
1. 接收 GuardAgent 的提示词
2. LLM（GPT-4/GPT-3.5）根据提示词生成守卫代码
3. LLM 通过函数调用（function call）返回代码：
   ```json
   {
     "function_call": {
       "name": "python",
       "arguments": {
         "cell": "list_of_database = {...}\nsummary_of_database = {...}\naccess_denied, inaccessible_database = CheckAccess(...)\n..."
       }
     }
   }
   ```

### 4. GuardAgent 执行代码

```python
# guardagent.py:192-247
def execute_function(self, func_call):
    func_name = func_call.get("name", "")  # "python"
    func = self._function_map.get(func_name, None)  # run_code_ehragent 或 run_code_seeact
    
    # 解析参数
    arguments = json.loads(input_string)
    # arguments = {"cell": "守卫代码字符串"}
    
    # 执行函数
    self.code = arguments["cell"]
    content = func(**arguments)  # 调用 run_code_ehragent(cell=守卫代码)
    
    # 返回执行结果
    return is_exec_success, {
        "name": func_name,
        "role": "function",
        "content": str(content),  # 执行结果
    }
```

### 5. 执行函数（run_code_ehragent / run_code_seeact）

```python
# toolset_high.py:6-19
def run_code_ehragent(cell):
    global_var = {
        "access_denied": None,
        "inaccessible_database": None,
        "guardrailed_answer": None
    }
    # 执行守卫代码
    exec(CodeHeader + cell, global_var)
    
    # 返回结果
    return "GuardAgent results:\nlabel: {}\ninaccessible_db: {}\nguardrailed_answer: {}\n(End of results)".format(
        int(global_var['access_denied']),
        global_var['inaccessible_database'],
        global_var['guardrailed_answer']
    )
```

## 四、模型配置

### 支持的模型

**代码位置**: `config.py:1-13`

```python
def model_config(model):
    if model == 'gpt-3.5-turbo':
        config = {
            "model": "gpt-3.5-turbo",
            "api_key": "<YOUR_API_KEY>",  # 需要替换为实际的 API key
        }
    else:
        config = {
            "model": "gpt-4",
            "api_key": "<YOUR_API_KEY>",  # 需要替换为实际的 API key
        }
    return config
```

### 如何使用

**命令行参数**: `main.py:21`

```bash
python main.py --llm gpt-4 --agent ehragent --seed 42 --num_shots 3 --logs_path ./logs --dataset_path ./dataset
```

**参数说明**:
- `--llm`: 选择模型，可选 `gpt-4` 或 `gpt-3.5-turbo`（默认：`gpt-4`）
- `--agent`: 被保护的 Agent，可选 `ehragent` 或 `seeact`（默认：`ehragent`）
- `--num_shots`: Few-shot 示例数量，可选 1、2、3（默认：3）

### 模型使用位置

1. **任务分解** (`guardagent.py:47-83`)
   - 使用 `self.config_list[0]` 中的模型
   - 调用 OpenAI API 进行任务分解

2. **代码生成** (`main.py:35-39`)
   - Chatbot 使用 `llm_config` 中的模型配置
   - 通过 AutoGen 框架调用 LLM

3. **错误调试** (`guardagent.py:158-190`)
   - 使用 `self.config_list[0]` 中的模型
   - 调用 OpenAI API 进行错误诊断

## 五、关键点总结

### 1. 职责分工

| 组件 | 职责 |
|------|------|
| **GuardAgent** | - 准备提示词（任务分解、示例检索）<br>- 发送消息给 Chatbot<br>- 执行守卫代码<br>- 管理整个流程 |
| **Chatbot** | - 接收提示词<br>- 使用 LLM 生成守卫代码<br>- 通过函数调用返回代码 |
| **LLM** | - 实际生成代码的模型（GPT-4/GPT-3.5-turbo） |

### 2. 代码生成流程

```
GuardAgent 准备提示词
    ↓
发送给 Chatbot
    ↓
Chatbot 使用 LLM 生成代码
    ↓
通过函数调用返回代码
    ↓
GuardAgent 执行代码
    ↓
返回结果给 Chatbot
```

### 3. 模型选择

- **默认**: GPT-4（更准确，但更昂贵）
- **备选**: GPT-3.5-turbo（更快，更便宜）
- **配置**: 在 `config.py` 中设置 API key
- **使用**: 通过 `--llm` 参数选择

### 4. 函数调用机制

AutoGen 框架使用 OpenAI 的函数调用（function calling）功能：
- Chatbot 生成代码后，不直接返回文本
- 而是调用 "python" 函数，将代码作为参数传递
- GuardAgent 注册了 "python" 函数，对应 `run_code_ehragent` 或 `run_code_seeact`
- 这样实现了代码生成和执行的解耦

## 六、示例：完整对话流程

```
GuardAgent → Chatbot:
  "Write a guardrail code in python following the examples provided in below.
   Here are some examples:
   [示例1]
   [示例2]
   [示例3]
   (END OF EXAMPLES)
   Agent input: Identity: Nursing\nQuestion: was the amoxicillin prescribed...
   Agent output: Knowledge:\n- We can find the patient...
   Task decomposition: - Identified task "access control"...
   Guardrail code: "

Chatbot → LLM (GPT-4):
  [发送上述提示词]

LLM → Chatbot:
  [生成守卫代码]

Chatbot → GuardAgent (函数调用):
  {
    "function_call": {
      "name": "python",
      "arguments": {
        "cell": "list_of_database = {...}\nsummary_of_database = {...}\naccess_denied, inaccessible_database = CheckAccess(list_of_database, summary_of_database)\nif access_denied:\n    print('ACCESS DENIED')\n    guardrailed_answer = None\nelse:\n    print('ACCESS GRANTED')\n    guardrailed_answer = answer"
      }
    }
  }

GuardAgent → run_code_ehragent:
  [执行代码]

run_code_ehragent → GuardAgent:
  "GuardAgent results:\nlabel: 0\ninaccessible_db: {}\nguardrailed_answer: 1\n(End of results)"

GuardAgent → Chatbot:
  [返回执行结果]

Chatbot → GuardAgent:
  "TERMINATE"  # 任务完成
```

## 七、总结

1. **谁生成代码？** Chatbot（AssistantAgent）使用 LLM（GPT-4/GPT-3.5-turbo）生成守卫代码
2. **谁准备提示词？** GuardAgent 负责准备提示词（任务分解、示例检索）
3. **谁执行代码？** GuardAgent 执行生成的守卫代码
4. **如何选择模型？** 通过 `--llm` 参数选择，支持 `gpt-4` 和 `gpt-3.5-turbo`
5. **如何配置 API key？** 在 `config.py` 中设置 `api_key`

**关键理解**：GuardAgent 是"管理者"，Chatbot 是"代码生成器"，LLM 是"实际执行者"。

