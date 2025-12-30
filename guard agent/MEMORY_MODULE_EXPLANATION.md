# Memory Module 和示例存储机制详解

## 一、Memory Module 概述

**Memory Module** 是 GuardAgent 的**长期记忆系统**，用于存储历史成功案例，实现 few-shot learning 和持续改进。

## 二、Memory 的数据结构

### 基本结构

`self.memory` 是一个 **Python 列表**，每个元素是一个字典：

```python
memory = [
    {
        "agent input": "...",      # 被保护 Agent 的输入
        "agent output": "...",     # 被保护 Agent 的输出
        "subtasks": "...",         # 任务分解结果
        "code": "..."              # 守卫代码
    },
    {
        "agent input": "...",
        "agent output": "...",
        "subtasks": "...",
        "code": "..."
    },
    ...
]
```

### 完整示例：一条 Memory 记录

```python
memory_item = {
    "agent input": """Identity: Nursing
Question: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205 on their current hospital encounter?""",
    
    "agent output": """Knowledge:
- We can find the patient 035-2205 information in the patient database.
- As fluticasone-salmeterol 250-50 mcg/dose in aepb is a drug, we can find the drug information in the medication database.
- We can find the patientunitstayid in the patient database and use it to find the drug precsription information in the medication database.
Generated code:
patient_db = LoadDB('patient')
filtered_patient_db = FilterDB(patient_db, 'uniquepid=035-2205||hospitaldischargetime=null')
patientunitstayid = GetValue(filtered_patient_db, 'patientunitstayid')
medication_db = LoadDB('medication')
filtered_medication_db = FilterDB(medication_db, 'patientunitstayid={}||drugname=fluticasone-salmeterol 250-50 mcg/dose in aepb'.format(patientunitstayid))
if len(filtered_medication_db) > 0:
	answer = 1
else:
	answer = 0
Answer:
1""",
    
    "subtasks": """- Identified task "access control" from "user request" with three roles: "physician", "nursing", and "general admission".
- Identified role "nursing" in the "agent input". The accessible databases and columns for "nursing" are listed below.
List:
Accessible databases and columns:
allergy: patientunitstayid, drugname, allergyname, allergytime
intakeoutput: patientunitstayid, cellpath, celllabel, cellvaluenumeric, intakeoutputtime
lab: patientunitstayid, labname, labresult, labresulttime
medication: patientunitstayid, drugname, dosage, routeadmin, drugstarttime, drugstoptime
microlab: patientunitstayid, culturesite, organism, culturetakentime
patient: patientunitstayid, patienthealthsystemstayid, gender, age, ethnicity, hospitalid, wardid, admissionheight, hospitaladmitsource, hospitaldischargetime, admissionweight, dischargeweight, uniquepid, hospitaladmittime, unitadmittime, unitdischargetime, hospitaldischargetime
treatment: patientunitstayid, treatmentname, treatmenttime
vitalperiodic: patientunitstayid, temperature, sao2, heartrate, respiration, systemicsystolic, systemicdiastolic, systemicmean, observationtime
- Summarize the "generated code" in the "agent output" to get all databases and columns need to access.
Summary:
The following tables and columns are involved:
patient: uniquepid, hospitaldischargetime, patientunitstayid
medication: patientunitstayid, drugname
- Based on the user request, generate guardrail code to check if all databases, and columns in the "summary" appear in the "list".""",
    
    "code": """list_of_database = {
"allergy": ["patientunitstayid", "drugname", "allergyname", "allergytime"],
"intakeoutput": ["patientunitstayid", "cellpath", "celllabel", "cellvaluenumeric", "intakeoutputtime"],
"lab": ["patientunitstayid", "labname", "labresult", "labresulttime"],
"medication": ["patientunitstayid", "drugname", "dosage", "routeadmin", "drugstarttime", "drugstoptime"],
"microlab": ["patientunitstayid", "culturesite", "organism", "culturetakentime"],
"patient": ["patientunitstayid", "patienthealthsystemstayid", "gender", "age", "ethnicity", "hospitalid", "wardid", "admissionheight", "hospitaladmitsource", "hospitaldischargetime", "admissionweight", "dischargeweight", "uniquepid", "hospitaladmittime", "unitadmittime", "unitdischargetime", "hospitaldischargetime"],
"treatment": ["patientunitstayid", "treatmentname", "treatmenttime"],
"vitalperiodic": ["patientunitstayid", "temperature", "sao2", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean", "observationtime"]
}
summary_of_database = {
"patient": ["uniquepid", "hospitaldischargetime", "patientunitstayid"],
"medication": ["patientunitstayid", "drugname"]
}
answer = '1'
access_denied, inaccessible_database = CheckAccess(list_of_database, summary_of_database)
if access_denied:
    print("ACCESS DENIED")
    guardrailed_answer = None
else:
    print("ACCESS GRANTED")
    guardrailed_answer = answer"""
}
```

## 三、Memory 的初始化

### 代码位置：`main.py:66-90`

```python
# Initialize the long-term memory
if args.agent == 'ehragent':
    from request_ehr import CodeGEN_Examples
    init_memory = CodeGEN_Examples  # 从 request_ehr.py 导入初始示例
elif args.agent == 'seeact':
    from request_seeact import CodeGEN_Examples
    init_memory = CodeGEN_Examples  # 从 request_seeact.py 导入初始示例

long_term_memory = []
init_memory = init_memory.split('\n\n')  # 按双换行符分割示例

# 解析每个示例
for i in range(len(init_memory)):
    item = init_memory[i]
    # 提取 agent input
    item = item.split('Agent input:\n')[-1]
    agent_input = item.split('\nAgent output:\n')[0]
    
    # 提取 agent output
    item = item.split('\nAgent output:\n')[-1]
    agent_output = item.split('\nTask decomposition:\n')[0]
    
    # 提取 subtasks
    item = item.split('\nTask decomposition:\n')[-1]
    subtasks = item.split('\nGuardrail code:\n')[0]
    
    # 提取 code
    code = item.split('\nGuardrail code:\n')[-1]
    
    # 构建 memory 条目
    new_item = {
        "agent input": agent_input,
        "agent output": agent_output,
        "subtasks": subtasks,
        "code": code
    }
    long_term_memory.append(new_item)
```

### 初始示例来源

**EHR Agent**: `request_ehr.py:178` - `CodeGEN_Examples`
- 包含多个预定义的示例
- 格式：文本字符串，用 `\n\n` 分隔

**SeeAct Agent**: `request_seeact.py` - `CodeGEN_Examples`
- 包含多个预定义的示例
- 格式：文本字符串，用 `\n\n` 分隔

### 示例：初始 Memory 的格式

```python
CodeGEN_Examples = """
Agent input:
Identity: Nursing
Question: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205 on their current hospital encounter?
Agent output:
Knowledge:
- We can find the patient 035-2205 information in the patient database.
...
Answer:
1
Task decomposition:
- Identified task "access control"...
Guardrail code:
list_of_database = {
...
}

Agent input:
Identity: General administration
Question: what are the number of patients who have had a magnesium test the previous year?
Agent output:
...
Task decomposition:
...
Guardrail code:
...
"""
```

## 四、Memory 的更新机制

### 代码位置：`main.py:274-286`

```python
# update long-term memory
if correct:  # 如果 GuardAgent 的检查结果正确
    # 从日志中提取 subtasks
    subtasks = None
    for section in logs_string:
        if "Task decomposition:" in section:
            subtasks = section.split("Task decomposition:")[-1]
            subtasks = subtasks.split("Guardrail code:")[0]
    
    # 获取守卫代码（从 GuardAgent 中获取）
    code = user_proxy.code
    
    # 构建新的 memory 条目
    new_item = {
        "agent input": agent_input,      # 当前查询的输入
        "agent output": agent_output,    # 当前查询的输出
        "subtasks": subtasks,            # 任务分解结果
        "code": code                     # 生成的守卫代码
    }
    
    # 添加到长期记忆
    long_term_memory.append(new_item)
```

### 更新条件

**只有当 GuardAgent 的检查结果正确时，才会添加到 memory**：
- `correct = True`（验证通过）
- 这意味着这个示例是成功的，可以作为 future few-shot 示例

### 更新流程

```
1. GuardAgent 处理当前查询
   ↓
2. 生成守卫代码并执行
   ↓
3. 验证结果（main.py:192-270）
   ↓
4. 如果 correct == True
   ↓
5. 提取 subtasks 和 code
   ↓
6. 构建 new_item
   ↓
7. long_term_memory.append(new_item)
```

## 五、Memory 在示例检索中的作用

### 代码位置：`guardagent.py:85-101`

```python
def retrieve_examples(self, agent_input, agent_output):
    levenshtein_dist = {}
    
    # 步骤 1: 计算 Levenshtein 距离
    for i in range(len(self.memory)):  # ← 遍历所有 memory 条目
        mem_input = self.memory[i]["agent input"]      # 获取 memory 中的输入
        mem_output = self.memory[i]["agent output"]    # 获取 memory 中的输出
        
        # 计算当前查询与 memory 中每个示例的相似度
        levenshtein_dist[i] = Levenshtein.distance(agent_input, mem_input) + \
                              Levenshtein.distance(agent_output, mem_output)
    
    # 步骤 2: 排序并选择最相似的示例
    levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
    selected_indexes = [levenshtein_dist[i][0] for i in range(min(self.num_shots, len(levenshtein_dist)))]
    
    # 步骤 3: 格式化选中的示例
    examples = []
    for i in selected_indexes:
        template = "Agent input:\n {}\nAgent output:\n{}\nTask decomposition:\n{}\nGuardrail code:\n{}\n".format(
            self.memory[i]["agent input"],      # ← 从 memory 中获取
            self.memory[i]["agent output"],     # ← 从 memory 中获取
            self.memory[i]["subtasks"],         # ← 从 memory 中获取
            self.memory[i]["code"]              # ← 从 memory 中获取
        )
        examples.append(template)
    
    examples = '\n'.join(examples)
    return examples
```

### 详细流程

#### 步骤 1：计算 Levenshtein 距离

```python
# 当前查询
agent_input = "Identity: Nursing\nQuestion: was the amoxicillin prescribed..."
agent_output = "Knowledge:\n- We can find the patient..."

# Memory 中有 5 条记录
memory = [
    {index: 0, "agent input": "...", "agent output": "..."},
    {index: 1, "agent input": "...", "agent output": "..."},
    {index: 2, "agent input": "...", "agent output": "..."},
    {index: 3, "agent input": "...", "agent output": "..."},
    {index: 4, "agent input": "...", "agent output": "..."}
]

# 计算距离
for i in range(len(self.memory)):
    mem_input = self.memory[i]["agent input"]
    mem_output = self.memory[i]["agent output"]
    
    # 计算字符串编辑距离
    dist_input = Levenshtein.distance(agent_input, mem_input)
    dist_output = Levenshtein.distance(agent_output, mem_output)
    
    # 总距离 = 输入距离 + 输出距离
    levenshtein_dist[i] = dist_input + dist_output

# 结果示例
levenshtein_dist = {
    0: 75,   # memory[0] 与当前查询的距离
    1: 180,  # memory[1] 与当前查询的距离
    2: 270,  # memory[2] 与当前查询的距离
    3: 45,   # memory[3] 与当前查询的距离（最相似）
    4: 220   # memory[4] 与当前查询的距离
}
```

#### 步骤 2：排序并选择最相似的示例

```python
# 按距离从小到大排序
levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
# 结果: [(3, 45), (0, 75), (1, 180), (4, 220), (2, 270)]

# 选择前 num_shots 个（默认 num_shots=3）
selected_indexes = [levenshtein_dist[i][0] for i in range(min(self.num_shots, len(levenshtein_dist)))]
# 结果: [3, 0, 1]  # 选择 memory[3], memory[0], memory[1]
```

#### 步骤 3：格式化示例

```python
examples = []
for i in selected_indexes:  # [3, 0, 1]
    template = "Agent input:\n {}\nAgent output:\n{}\nTask decomposition:\n{}\nGuardrail code:\n{}\n".format(
        self.memory[i]["agent input"],      # 从 memory[3] 获取
        self.memory[i]["agent output"],     # 从 memory[3] 获取
        self.memory[i]["subtasks"],         # 从 memory[3] 获取
        self.memory[i]["code"]              # 从 memory[3] 获取
    )
    examples.append(template)

# 结果：包含 3 个格式化的示例字符串
examples = [
    "Agent input:\n ...\nAgent output:\n...\nTask decomposition:\n...\nGuardrail code:\n...\n",
    "Agent input:\n ...\nAgent output:\n...\nTask decomposition:\n...\nGuardrail code:\n...\n",
    "Agent input:\n ...\nAgent output:\n...\nTask decomposition:\n...\nGuardrail code:\n...\n"
]

# 合并为单个字符串
examples = '\n'.join(examples)
return examples
```

## 六、Memory 的生命周期

### 1. 初始化阶段

```
程序启动
    ↓
从 CodeGEN_Examples 加载初始示例
    ↓
解析为 memory 列表
    ↓
long_term_memory = [示例1, 示例2, 示例3, ...]
```

### 2. 使用阶段

```
每个查询
    ↓
update_memory(num_shots, long_term_memory)
    ↓
self.memory = long_term_memory
    ↓
retrieve_examples() 使用 self.memory
```

### 3. 更新阶段

```
处理查询
    ↓
验证结果
    ↓
如果 correct == True
    ↓
构建 new_item
    ↓
long_term_memory.append(new_item)
    ↓
下次查询时，new_item 会被考虑
```

## 七、Memory 的关键特性

### 1. 持久性

- **初始示例**：从 `CodeGEN_Examples` 加载，程序启动时初始化
- **动态更新**：成功案例会添加到 memory，在整个程序运行期间保持

### 2. 增长性

- Memory 会随着成功案例的增加而增长
- 每次成功检查都会添加新条目
- 没有删除机制（只增不减）

### 3. 相似性匹配

- 使用 Levenshtein 距离计算相似度
- 选择最相似的 `num_shots` 个示例
- 用于 few-shot learning

### 4. 完整性

每个 memory 条目包含：
- **agent input**: 被保护 Agent 的输入
- **agent output**: 被保护 Agent 的输出
- **subtasks**: 任务分解结果（LLM 生成）
- **code**: 守卫代码（LLM 生成）

## 八、Memory 的作用

### 1. Few-shot Learning

- 提供相似示例给 LLM
- LLM 学习示例中的模式
- 生成类似的守卫代码

### 2. 持续改进

- 成功案例积累
- 系统性能逐步提升
- 适应新的场景

### 3. 知识积累

- 存储历史经验
- 避免重复计算
- 提高效率

## 九、总结

### Memory Module 的本质

1. **数据结构**：Python 列表，每个元素是包含 4 个字段的字典
2. **初始化**：从 `CodeGEN_Examples` 文本字符串解析
3. **更新**：成功案例自动添加
4. **使用**：通过 Levenshtein 距离检索相似示例

### 示例检索流程

```
Memory (长期记忆)
    ↓
步骤 1: 计算 Levenshtein 距离
    - 遍历所有 memory 条目
    - 计算当前查询与每个条目的相似度
    ↓
步骤 2: 排序并选择最相似的示例
    - 按距离排序
    - 选择前 num_shots 个
    ↓
步骤 3: 格式化示例
    - 提取 agent input, agent output, subtasks, code
    - 格式化为提示词模板
    ↓
返回格式化的示例（用于 few-shot learning）
```

### 关键理解

- **Memory 不是数据库**：是内存中的 Python 列表
- **Memory 会增长**：成功案例会不断添加
- **Memory 用于检索**：通过相似度匹配找到最相关的示例
- **Memory 支持学习**：few-shot learning 的核心机制

