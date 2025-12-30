# 示例检索（retrieve_examples）详细示例

## 一、长期记忆的数据结构

长期记忆 `self.memory` 是一个列表，每个元素包含：

```python
{
    "agent input": "...",      # Agent 的输入
    "agent output": "...",     # Agent 的输出
    "subtasks": "...",         # 任务分解结果
    "code": "..."              # 守卫代码
}
```

### 示例：长期记忆中的一条记录

```python
memory_item = {
    "agent input": """Identity: Nursing
Question: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205 on their current hospital encounter?""",
    
    "agent output": """Knowledge:
- We can find the patient 035-2205 information in the patient database.
- As fluticasone-salmeterol 250-50 mcg/dose in aepb is a drug, we can find the drug information in the medication database.
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
    
    "subtasks": """- Identified task "access control" from "user request" with three roles...
- Identified role "nursing" in the "agent input"...
- Summarize the "generated code" in the "agent output" to get all databases and columns need to access.""",
    
    "code": """list_of_database = {
"allergy": ["patientunitstayid", "drugname", "allergyname", "allergytime"],
"medication": ["patientunitstayid", "drugname", "dosage", "routeadmin", "drugstarttime", "drugstoptime"],
"patient": ["patientunitstayid", "patienthealthsystemstayid", "gender", "age", ...]
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

## 二、示例检索的完整流程

### 步骤 1：计算 Levenshtein 距离

**代码位置**: `guardagent.py:85-90`

```python
def retrieve_examples(self, agent_input, agent_output):
    levenshtein_dist = {}
    for i in range(len(self.memory)):
        mem_input = self.memory[i]["agent input"]
        mem_output = self.memory[i]["agent output"]
        # 计算当前输入与记忆中的输入的编辑距离
        # 计算当前输出与记忆中的输出的编辑距离
        # 将两个距离相加作为总相似度指标
        levenshtein_dist[i] = Levenshtein.distance(agent_input, mem_input) + \
                              Levenshtein.distance(agent_output, mem_output)
```

### 步骤 2：排序并选择最相似的示例

**代码位置**: `guardagent.py:91-92`

```python
    # 按距离从小到大排序（距离越小，越相似）
    levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
    # 选择前 num_shots 个最相似的示例（默认 num_shots=3）
    selected_indexes = [levenshtein_dist[i][0] for i in range(min(self.num_shots, len(levenshtein_dist)))]
```

### 步骤 3：格式化示例

**代码位置**: `guardagent.py:93-101`

```python
    examples = []
    for i in selected_indexes:
        template = "Agent input:\n {}\nAgent output:\n{}\nTask decomposition:\n{}\nGuardrail code:\n{}\n".format(
            self.memory[i]["agent input"],
            self.memory[i]["agent output"],
            self.memory[i]["subtasks"],
            self.memory[i]["code"]
        )
        examples.append(template)
    examples = '\n'.join(examples)
    return examples
```

## 三、具体示例演示

### 场景：新的查询需要生成守卫代码

**当前查询**：
```python
agent_input = """Identity: Nursing
Question: was the amoxicillin prescribed to patient 035-2205?"""

agent_output = """Knowledge:
- We can find the patient 035-2205 information in the patient database.
- As amoxicillin is a drug, we can find the drug information in the medication database.
Generated code:
patient_db = LoadDB('patient')
filtered_patient_db = FilterDB(patient_db, 'uniquepid=035-2205||hospitaldischargetime=null')
patientunitstayid = GetValue(filtered_patient_db, 'patientunitstayid')
medication_db = LoadDB('medication')
filtered_medication_db = FilterDB(medication_db, 'patientunitstayid={}||drugname=amoxicillin'.format(patientunitstayid))
if len(filtered_medication_db) > 0:
	answer = 1
else:
	answer = 0
Answer:
1"""
```

### 假设长期记忆中有 5 条记录：

```python
memory = [
    # 记录 0: 关于 fluticasone-salmeterol 的查询（Nursing角色）
    {
        "agent input": "Identity: Nursing\nQuestion: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205...",
        "agent output": "Knowledge:\n- We can find the patient 035-2205 information...",
        ...
    },
    # 记录 1: 关于 magnesium 测试的查询（General administration角色）
    {
        "agent input": "Identity: General administration\nQuestion: what are the number of patients who have had a magnesium test...",
        "agent output": "Knowledge:\n- As magnesium is a lab test...",
        ...
    },
    # 记录 2: 关于 microbiology 测试的查询（Physician角色）
    {
        "agent input": "Identity: Physician\nQuestion: in the last hospital encounter, when was patient 031-22988's first microbiology test time?",
        "agent output": "Knowledge:\n- We can find the patient 031-22988 information...",
        ...
    },
    # 记录 3: 关于另一个 medication 查询（Nursing角色）
    {
        "agent input": "Identity: Nursing\nQuestion: was the penicillin prescribed to patient 012-3456?",
        "agent output": "Knowledge:\n- We can find the patient 012-3456 information...",
        ...
    },
    # 记录 4: 关于 cost 查询（General administration角色）
    {
        "agent input": "Identity: General administration\nQuestion: what is the total cost for patient 999-8888?",
        "agent output": "Knowledge:\n- We can find the cost information...",
        ...
    }
]
```

### 计算 Levenshtein 距离：

```python
# 对于记录 0（fluticasone-salmeterol，Nursing）
dist_input_0 = Levenshtein.distance(
    "Identity: Nursing\nQuestion: was the amoxicillin prescribed to patient 035-2205?",
    "Identity: Nursing\nQuestion: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205..."
)
# 假设距离 = 45（因为药物名称不同，但结构相似）

dist_output_0 = Levenshtein.distance(
    "Knowledge:\n- We can find the patient 035-2205 information...",
    "Knowledge:\n- We can find the patient 035-2205 information..."
)
# 假设距离 = 30（输出结构非常相似）

levenshtein_dist[0] = 45 + 30 = 75

# 对于记录 1（magnesium，General administration）
dist_input_1 = Levenshtein.distance(
    "Identity: Nursing\nQuestion: was the amoxicillin prescribed...",
    "Identity: General administration\nQuestion: what are the number of patients..."
)
# 假设距离 = 80（角色不同，问题类型不同）

dist_output_1 = Levenshtein.distance(...)
# 假设距离 = 100（输出完全不同）

levenshtein_dist[1] = 80 + 100 = 180

# 对于记录 2（microbiology，Physician）
levenshtein_dist[2] = 120 + 150 = 270

# 对于记录 3（penicillin，Nursing）
dist_input_3 = Levenshtein.distance(
    "Identity: Nursing\nQuestion: was the amoxicillin prescribed to patient 035-2205?",
    "Identity: Nursing\nQuestion: was the penicillin prescribed to patient 012-3456?"
)
# 假设距离 = 25（非常相似，只是药物名称和患者ID不同）

dist_output_3 = Levenshtein.distance(...)
# 假设距离 = 20（输出结构几乎相同）

levenshtein_dist[3] = 25 + 20 = 45

# 对于记录 4（cost，General administration）
levenshtein_dist[4] = 100 + 120 = 220
```

### 排序结果：

```python
levenshtein_dist = [
    (3, 45),   # 记录 3 最相似（penicillin，Nursing）
    (0, 75),   # 记录 0 次相似（fluticasone-salmeterol，Nursing）
    (1, 180),  # 记录 1
    (4, 220),  # 记录 4
    (2, 270)   # 记录 2 最不相似
]

# 如果 num_shots = 3，选择前 3 个：
selected_indexes = [3, 0, 1]
```

### 格式化后的示例（用于 few-shot learning）：

```python
examples = """Agent input:
 Identity: Nursing
Question: was the penicillin prescribed to patient 012-3456?
Agent output:
Knowledge:
- We can find the patient 012-3456 information in the patient database.
- As penicillin is a drug, we can find the drug information in the medication database.
Generated code:
patient_db = LoadDB('patient')
filtered_patient_db = FilterDB(patient_db, 'uniquepid=012-3456||hospitaldischargetime=null')
patientunitstayid = GetValue(filtered_patient_db, 'patientunitstayid')
medication_db = LoadDB('medication')
filtered_medication_db = FilterDB(medication_db, 'patientunitstayid={}||drugname=penicillin'.format(patientunitstayid))
if len(filtered_medication_db) > 0:
	answer = 1
else:
	answer = 0
Answer:
1
Task decomposition:
- Identified task "access control" from "user request" with three roles: "physician", "nursing", and "general admission".
- Identified role "nursing" in the "agent input". The accessible databases and columns for "nursing" are listed below.
...
Guardrail code:
list_of_database = {
"allergy": ["patientunitstayid", "drugname", "allergyname", "allergytime"],
"medication": ["patientunitstayid", "drugname", "dosage", "routeadmin", "drugstarttime", "drugstoptime"],
"patient": ["patientunitstayid", "patienthealthsystemstayid", "gender", "age", ...]
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
    guardrailed_answer = answer

Agent input:
 Identity: Nursing
Question: was the fluticasone-salmeterol 250-50 mcg/dose in aepb prescribed to patient 035-2205 on their current hospital encounter?
Agent output:
...
Guardrail code:
...

Agent input:
 Identity: General administration
Question: what are the number of patients who have had a magnesium test the previous year?
Agent output:
...
Guardrail code:
..."""
```

## 四、Few-shot Learning 的使用

### 在生成守卫代码时使用示例

**代码位置**: `guardagent.py:121-126`

```python
examples = self.retrieve_examples(self.agent_input, self.agent_output)

init_message = GuardAgent_Message_Prompt.format(
    examples=examples,              # ← 检索到的相似示例（few-shot examples）
    agent_input=self.agent_input,    # ← 当前查询的输入
    agent_output=self.agent_output,  # ← 当前查询的输出
    subtasks=subtasks                # ← 任务分解结果
)
```

### 提示词模板（`prompts_guard.py:26-41`）：

```python
GuardAgent_Message_Prompt = """Write a guardrail code in python following the examples provided in below.
Here are some examples:
{examples}                          # ← 这里插入检索到的示例（few-shot）
(END OF EXAMPLES)
Agent input:
{agent_input}                       # ← 当前查询
Agent output:
{agent_output}                      # ← 当前查询
Task decomposition:
{subtasks}                          # ← 任务分解
Here are the functions you can use:
(1) CheckAccess(DATABASE_ALLOWED, DATABASE_NEEDED)
(2) CheckRule(list_of_rules: dict, user_info: dict)
Guardrail code: """
```

### Few-shot Learning 的工作原理：

1. **提供示例**：将最相似的 3 个历史示例（包含输入、输出、任务分解、守卫代码）提供给 LLM
2. **模式学习**：LLM 从示例中学习：
   - 如何识别用户角色
   - 如何提取数据库访问需求
   - 如何生成守卫代码的结构
   - 如何使用 CheckAccess 函数
3. **生成新代码**：基于学到的模式，为当前查询生成类似的守卫代码

### 为什么使用 Levenshtein 距离？

- **字符串相似度**：Levenshtein 距离衡量两个字符串需要多少次编辑（插入、删除、替换）才能相同
- **语义相似性**：相似的输入输出通常意味着需要相似的守卫代码
- **简单有效**：不需要复杂的语义理解，计算快速

### 示例：Levenshtein 距离计算

```python
from Levenshtein import distance

# 示例 1：非常相似
str1 = "Identity: Nursing\nQuestion: was the amoxicillin prescribed to patient 035-2205?"
str2 = "Identity: Nursing\nQuestion: was the penicillin prescribed to patient 012-3456?"
distance(str1, str2)  # 约 25（只有药物名称和患者ID不同）

# 示例 2：不太相似
str1 = "Identity: Nursing\nQuestion: was the amoxicillin prescribed to patient 035-2205?"
str2 = "Identity: General administration\nQuestion: what are the number of patients who have had a magnesium test?"
distance(str1, str2)  # 约 80（角色不同，问题类型不同）
```

## 五、完整流程总结

```
1. 新查询到达
   ↓
2. 计算与所有记忆条目的 Levenshtein 距离
   ↓
3. 选择距离最小的前 num_shots 个示例
   ↓
4. 格式化示例为提示词模板
   ↓
5. 将示例 + 当前查询 + 任务分解 → 发送给 LLM
   ↓
6. LLM 基于示例学习模式，生成新的守卫代码
   ↓
7. 执行守卫代码
   ↓
8. 如果正确，将新示例添加到记忆（持续学习）
```

## 六、关键优势

1. **自适应学习**：系统从历史成功案例中学习，无需手动编写规则
2. **相似性匹配**：通过 Levenshtein 距离找到最相关的示例
3. **持续改进**：每次成功的检查都会添加到记忆，提高未来性能
4. **Few-shot 效率**：只需少量示例就能生成高质量的守卫代码

