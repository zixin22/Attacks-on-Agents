# GuardAgent 集成到 WebShop 后的 Pipeline

## 一、集成后的完整 Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ WebShop Session Start                                        │
└─────────────────────────────────────────────────────────────┘
    ↓
Get Instruction & Query from webshop_text
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Defense Check Point 1: Session Start                        │
│ (webshopEnv.step)                                            │
└─────────────────────────────────────────────────────────────┘
    ↓
    ├─→ defense_mode == 'rule_checker'?
    │   └─→ RuleChecker.check_all_rules(profile, instruction, query)
    │       ├─→ LLM批量检查所有10条规则
    │       └─→ 返回 (is_valid, violated_rules)
    │
    ├─→ defense_mode == 'guard_agent'?
    │   └─→ GuardAgent.check(agent_input=instruction, agent_output=query, user_profile=profile)
    │       ├─→ Step 1: Task Decomposition (任务分解)
    │       │   └─→ LLM分析输入输出，分解为子任务
    │       ├─→ Step 2: Example Retrieval (示例检索)
    │       │   └─→ 使用Levenshtein距离检索相似历史案例
    │       ├─→ Step 3: Code Generation (代码生成)
    │       │   └─→ Chatbot使用LLM生成守卫代码
    │       ├─→ Step 4: Code Execution (代码执行)
    │       │   └─→ 执行生成的Python守卫代码
    │       └─→ Step 5: Result Parsing (结果解析)
    │           └─→ 返回 (is_valid, violated_rules)
    │
    └─→ defense_mode == 'none'?
        └─→ 跳过检查，继续会话
    ↓
    ├─→ is_valid == False?
    │   └─→ BLOCK SESSION (硬阻止)
    │       ├─→ 设置 reward = 0.0
    │       ├─→ 设置 done = True
    │       └─→ 返回违规信息
    │
    └─→ is_valid == True?
        └─→ CONTINUE SESSION
            ↓
        LLM Action Generation Loop
            ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ Defense Check Point 2: Before Each LLM Call                 │
        │ (webshop_run_rap / webshop_run_react)                       │
        └─────────────────────────────────────────────────────────────┘
            ↓
            ├─→ 构建 full_prompt (包含记忆示例和当前任务)
            │
            ├─→ defense_mode == 'rule_checker'?
            │   └─→ check_prompt_with_defense()
            │       └─→ RuleChecker.check_all_rules(profile, full_prompt, query)
            │
            ├─→ defense_mode == 'guard_agent'?
            │   └─→ check_prompt_with_defense()
            │       ├─→ 提取 instruction 从 full_prompt
            │       └─→ GuardAgent.check(agent_input=instruction, agent_output=query, user_profile=profile)
            │           └─→ [重复 GuardAgent Pipeline]
            │
            └─→ defense_mode == 'none'?
                └─→ 跳过检查
            ↓
            ├─→ is_valid == False?
            │   └─→ STOP SESSION
            │       ├─→ 记录违规信息
            │       └─→ 返回 reward = 0.0
            │
            └─→ is_valid == True?
                └─→ SEND TO LLM
                    ↓
                Execute Action
                    ↓
                Continue Session...
```

## 二、关键集成点

### 2.1 webshopEnv.step() - 会话开始检查

**位置**: `webshop/main.py:417-462`

**功能**: 在会话初始化后立即检查用户请求是否违反规则

**检查时机**:
- 在获取 instruction 和 query 后
- 在第一次 action 执行前
- 只检查一次（通过 `_rules_checked` 标志避免重复检查）

**代码逻辑**:
```python
if self.defense_mode != 'none' and profile:
    instruction = self.sessions[session].get('instruction', '')
    query = self.sessions[session].get('query', '')
    
    if instruction and query and not self.sessions[session].get('_rules_checked', False):
        is_valid = True
        violated_rules = []
        
        if self.defense_mode == 'rule_checker' and self.rule_checker:
            is_valid, violated_rules = self.rule_checker.check_all_rules(
                profile, instruction, query
            )
        elif self.defense_mode == 'guard_agent' and self.guard_agent:
            is_valid, violated_rules = self.guard_agent.check(
                agent_input=instruction,
                agent_output=query,
                user_profile=profile
            )
        
        if not is_valid:
            # BLOCK SESSION
            return observation, 0.0, True, {'violated_rules': violated_rules}
```

### 2.2 webshop_run_rap() / webshop_run_react() - LLM调用前检查

**位置**: 
- `webshop/main.py:1179-1198` (webshop_run_rap)
- `webshop/main.py:977-1006` (webshop_run_react)

**功能**: 在每次发送 full_prompt 给 LLM 前检查是否包含违规内容

**检查时机**:
- 每次构建 full_prompt 后
- 在调用 `llm(full_prompt)` 前
- 每次 action 生成循环中

**代码逻辑**:
```python
full_prompt = 'Interact with a webshop application. Here are examples.\n' + init_prompt + '\nHere is the task.\n' + prompt

if (env.rule_checker or env.guard_agent) and profile:
    defense_log_file = args.output + f'/{env.defense_mode}_prompt_log_{idx}.txt'
    should_continue, violated_rules = check_prompt_with_defense(
        full_prompt, idx, profile, env.rule_checker, env.guard_agent,
        env.defense_mode, defense_log_file, i
    )
    
    if not should_continue:
        # STOP SESSION
        return 0.0, {'violated_rules': violated_rules, ...}

action = llm(full_prompt, stop=['\n']).lstrip(' ')
```

### 2.3 check_prompt_with_defense() - 统一防御检查函数

**位置**: `webshop/main.py:781-884`

**功能**: 统一的防御检查接口，支持 RuleChecker 和 GuardAgent

**参数**:
- `full_prompt`: 完整的提示词
- `session_id`: 会话ID
- `profile`: 用户档案
- `rule_checker`: RuleChecker实例（可选）
- `guard_agent`: GuardAgent实例（可选）
- `defense_mode`: 防御模式 ('rule_checker' / 'guard_agent' / 'none')
- `defense_log_file`: 日志文件路径
- `step`: 步骤编号

**返回值**:
- `(should_continue, violated_rules)`
  - `should_continue`: True表示继续，False表示停止
  - `violated_rules`: 违反的规则列表

## 三、GuardAgent vs RuleChecker 对比

### 3.1 检查方式

| 特性 | RuleChecker | GuardAgent |
|------|------------|------------|
| **检查方法** | 直接使用LLM批量检查所有规则 | 生成Python代码执行检查 |
| **输入** | full_prompt (完整提示词) | instruction + query (提取的指令和查询) |
| **规则定义** | 硬编码在RuleChecker类中 | 通过user_request描述 |
| **检查逻辑** | LLM直接判断 | 动态生成的Python代码 |
| **可解释性** | LLM输出规则名称 | 生成可执行的检查代码 |
| **性能** | 每次调用LLM API | 生成代码后本地执行 |

### 3.2 检查时机

两者都在相同的两个检查点进行检查：
1. **会话开始时** (`webshopEnv.step`)
2. **每次LLM调用前** (`webshop_run_rap` / `webshop_run_react`)

### 3.3 检查结果处理

两者返回相同格式的结果：
- `is_valid`: 是否允许继续
- `violated_rules`: 违反的规则列表

如果 `is_valid == False`，会话会被立即阻止。

## 四、使用方式

### 4.1 命令行参数

```bash
# 使用 RuleChecker
python main.py \
    --defense_mode rule_checker \
    --enable_rule_checker \
    --model gpt-4o \
    --output ./output

# 使用 GuardAgent
python main.py \
    --defense_mode guard_agent \
    --enable_guard_agent \
    --guard_agent_llm gpt-4o \
    --guard_agent_num_shots 3 \
    --model gpt-4o \
    --output ./output

# 禁用防御
python main.py \
    --defense_mode none \
    --model gpt-4o \
    --output ./output
```

### 4.2 参数说明

**RuleChecker 参数**:
- `--defense_mode rule_checker`: 选择RuleChecker防御模式
- `--enable_rule_checker`: 启用RuleChecker（必需）
- `--model`: 用于RuleChecker的LLM模型

**GuardAgent 参数**:
- `--defense_mode guard_agent`: 选择GuardAgent防御模式
- `--enable_guard_agent`: 启用GuardAgent（必需）
- `--guard_agent_llm`: GuardAgent使用的LLM模型（gpt-4, gpt-4o, gpt-3.5-turbo）
- `--guard_agent_num_shots`: Few-shot示例数量（1, 2, 3）

## 五、日志输出

### 5.1 日志文件位置

- **RuleChecker**: `{output}/{defense_mode}_prompt_log_{session_id}.txt`
- **GuardAgent**: `{output}/{defense_mode}_prompt_log_{session_id}.txt`

### 5.2 日志内容

两种防御机制的日志格式相同，包含：
- 会话信息（Session ID, Profile ID）
- 检查步骤（Step number）
- 完整提示词（Full Prompt）
- 检查结果（VALID / VIOLATED）
- 违反的规则列表（Violated Rules）

## 六、总结

### 6.1 集成优势

1. **统一接口**: 两种防御机制使用相同的检查点和接口
2. **灵活切换**: 可以通过命令行参数轻松切换防御机制
3. **并行支持**: 代码结构支持未来同时使用两种防御机制
4. **日志统一**: 两种防御机制使用相同的日志格式

### 6.2 选择建议

- **RuleChecker**: 
  - 适合需要快速检查的场景
  - 规则固定且明确
  - 需要直接使用LLM判断

- **GuardAgent**:
  - 适合需要动态规则检查的场景
  - 需要高可解释性（生成可执行代码）
  - 需要利用历史经验（记忆机制）

### 6.3 未来扩展

当前实现支持：
- ✅ RuleChecker 防御模式
- ✅ GuardAgent 防御模式
- ✅ 无防御模式

未来可以扩展：
- 🔄 同时使用两种防御机制（双重检查）
- 🔄 防御机制链（先RuleChecker后GuardAgent）
- 🔄 自定义防御策略

