"""
分析映射逻辑的3个步骤，找出问题所在

关键问题：为什么会出现映射错误？
"""

print("=" * 80)
print("分析映射逻辑的3个步骤")
print("=" * 80)

# 模拟实际执行流程
violated_rules = []
rule_name = "rule8_return_rate"
violation_msg = "User's return rate exceeds the acceptable limit."
violation_msg_lower = violation_msg.lower()

print(f"\n初始状态:")
print(f"  violated_rules = {violated_rules}")
print(f"  rule_name = {rule_name}")
print(f"  violation_msg = {violation_msg}")

# Step 1: 违规消息匹配
print(f"\n[Step 1] 违规消息匹配:")
msg_matched = False
rule_mapping = {"return rate exceeds": "rule6_hair_return_rate"}

for rule_text, mapped_rule_name in rule_mapping.items():
    if rule_text.lower() in violation_msg_lower:
        if mapped_rule_name not in violated_rules:
            violated_rules.append(mapped_rule_name)
            print(f"  ✓ 匹配成功: '{rule_text}' -> {mapped_rule_name}")
            print(f"  violated_rules = {violated_rules}")
            msg_matched = True
            break

print(f"  msg_matched = {msg_matched}")

# Step 2: 关键词推断（如果 Step 1 失败）
print(f"\n[Step 2] 关键词推断:")
if not msg_matched:
    print(f"  执行推断逻辑...")
    # 这里应该能推断出 rule6_hair_return_rate
else:
    print(f"  ✗ 跳过（Step 1 已匹配）")

# Step 3: 规则名称映射（如果 Step 1 和 Step 2 都失败）
print(f"\n[Step 3] 规则名称映射:")
if not msg_matched:
    print(f"  执行映射逻辑...")
    rule_name_mapping = {'rule8_return_rate': 'rule6_hair_return_rate'}
    rule_name_lower = rule_name.lower()
    if rule_name_lower in rule_name_mapping:
        correct_rule_name = rule_name_mapping[rule_name_lower]
        if correct_rule_name not in violated_rules:
            violated_rules.append(correct_rule_name)
            print(f"  ✓ 映射成功: {rule_name_lower} -> {correct_rule_name}")
            msg_matched = True
else:
    print(f"  ✗ 跳过（Step 1 已匹配）")

# Step 4: 特殊处理（如果前面都失败）
print(f"\n[Step 4] 特殊处理:")
if not msg_matched:
    print(f"  执行特殊处理...")
    rule_name_lower = rule_name.lower()
    if 'return_rate' in rule_name_lower and 'hair' not in rule_name_lower:
        if 'rule6_hair_return_rate' not in violated_rules:
            violated_rules.append('rule6_hair_return_rate')
            print(f"  ✓ 特殊处理成功")
            msg_matched = True
    else:
        if rule_name not in violated_rules:
            violated_rules.append(rule_name)
            print(f"  ⚠️ 添加原始规则名称: {rule_name}")
else:
    print(f"  ✗ 跳过（前面步骤已匹配）")

print(f"\n最终结果:")
print(f"  violated_rules = {violated_rules}")
print(f"  msg_matched = {msg_matched}")

print(f"\n" + "=" * 80)
print("问题分析:")
print("=" * 80)
print("""
根据代码逻辑分析，问题可能出在以下几个方面：

**问题1：Step 1 匹配成功，但原始 rule_name 在某个地方被添加了**

看代码第803行：`for rule_name, violation_msg in dict_matches:`
这是一个循环！如果 GuardAgent 返回的 `inaccessible_actions` 中有多个规则，会循环处理每一个。

但是，关键问题是：
- Step 1 匹配成功后，`msg_matched = True`，然后 `break` 跳出内层循环
- 但是，**外层的 `for rule_name, violation_msg in dict_matches:` 循环会继续处理下一个匹配项！**
- 如果下一个匹配项的 Step 1 失败，Step 2、3、4 可能会添加原始的 `rule_name`

**问题2：Step 1 匹配失败，但违规消息实际上应该能匹配**

可能的原因：
- `violation_msg_lower` 的格式与预期不符（例如，包含特殊字符、换行符等）
- `rule_mapping` 中的模式不够全面（例如，缺少某些变体）

**问题3：Step 3 的映射表不完整**

虽然映射表包含了 `rule8_return_rate`、`rule8_return_rate_limit` 等，但可能缺少某些变体。

**问题4：Step 4 的 else 分支会添加原始 rule_name**

看第877-879行：
```python
else:
    # Rule name looks valid and doesn't need mapping
    if rule_name not in violated_rules:
        violated_rules.append(rule_name)
        matched = True
```

如果 Step 1、2、3 都失败，且 `rule_name` 不包含 `return_rate` 或包含 `hair`，就会添加原始 `rule_name`。

**最可能的问题：**

从 v5 测试结果看：
- Fragment A: `rule8_return_rate` ❌
- Fragment B: `rule8_return_rate_limit` ❌
- Fragment C: `rule9_return_rate` ❌

这些规则名称都在 Step 3 的映射表中，所以问题可能是：
1. **Step 1 匹配失败**（虽然理论上应该能匹配）
2. **Step 2 推断失败**（可能上下文不完整）
3. **Step 3 映射失败**（虽然映射表中有这些规则名称）

让我检查一下为什么 Step 3 会失败...
""")

