"""调试映射问题：分析为什么会出现映射错误"""

print("=" * 80)
print("分析映射逻辑的3个步骤，找出问题所在")
print("=" * 80)

# 模拟实际执行流程
violated_rules = []  # 初始为空
rule_name = "rule8_return_rate"
violation_msg = "User's return rate exceeds the acceptable limit."
violation_msg_lower = violation_msg.lower()

# Step 1 的映射表
rule_mapping = {
    "return rate exceeds": "rule6_hair_return_rate",
    # ... 其他映射
}

print(f"\n初始状态:")
print(f"  violated_rules = {violated_rules}")
print(f"  rule_name = {rule_name}")
print(f"  violation_msg = {violation_msg}")

# Step 1: 违规消息匹配
print(f"\n[Step 1] 违规消息匹配:")
msg_matched = False
for rule_text, mapped_rule_name in rule_mapping.items():
    if rule_text.lower() in violation_msg_lower:
        if mapped_rule_name not in violated_rules:
            violated_rules.append(mapped_rule_name)
            print(f"  ✓ 匹配成功: '{rule_text}' -> {mapped_rule_name}")
            print(f"  violated_rules = {violated_rules}")
            msg_matched = True
            break

if not msg_matched:
    print(f"  ✗ 匹配失败")

# Step 2: 关键词推断（如果 Step 1 失败）
if not msg_matched:
    print(f"\n[Step 2] 关键词推断:")
    print(f"  执行推断逻辑...")
    # 这里应该能推断出 rule6_hair_return_rate
    inferred_rule = "rule6_hair_return_rate"  # 假设推断成功
    if inferred_rule not in violated_rules:
        violated_rules.append(inferred_rule)
        print(f"  ✓ 推断成功: {inferred_rule}")
        print(f"  violated_rules = {violated_rules}")
        msg_matched = True

# Step 3: 规则名称映射（如果 Step 1 和 Step 2 都失败）
if not msg_matched:
    print(f"\n[Step 3] 规则名称映射:")
    rule_name_mapping = {
        'rule8_return_rate': 'rule6_hair_return_rate',
        # ... 其他映射
    }
    rule_name_lower = rule_name.lower()
    if rule_name_lower in rule_name_mapping:
        correct_rule_name = rule_name_mapping[rule_name_lower]
        if correct_rule_name not in violated_rules:
            violated_rules.append(correct_rule_name)
            print(f"  ✓ 映射成功: {rule_name_lower} -> {correct_rule_name}")
            print(f"  violated_rules = {violated_rules}")
            msg_matched = True

# Step 4: 特殊处理（如果前面都失败）
if not msg_matched:
    print(f"\n[Step 4] 特殊处理:")
    rule_name_lower = rule_name.lower()
    if 'return_rate' in rule_name_lower and 'hair' not in rule_name_lower:
        if 'rule6_hair_return_rate' not in violated_rules:
            violated_rules.append('rule6_hair_return_rate')
            print(f"  ✓ 特殊处理成功: {rule_name_lower} -> rule6_hair_return_rate")
            print(f"  violated_rules = {violated_rules}")
            msg_matched = True
    else:
        # 问题可能在这里！
        if rule_name not in violated_rules:
            violated_rules.append(rule_name)
            print(f"  ⚠️ 添加原始规则名称: {rule_name}")
            print(f"  violated_rules = {violated_rules}")

print(f"\n最终结果:")
print(f"  violated_rules = {violated_rules}")
print(f"  msg_matched = {msg_matched}")

print(f"\n" + "=" * 80)
print("问题分析:")
print("=" * 80)
print("""
根据代码逻辑分析，问题可能出在以下几个方面：

1. **Step 1 匹配成功，但原始 rule_name 已经被添加**
   - 如果 violated_rules 在 Step 1 之前就已经包含了 rule_name（如 rule8_return_rate）
   - 那么 Step 1 虽然添加了 rule6_hair_return_rate，但原始 rule_name 仍然存在
   - 结果：violated_rules = ['rule8_return_rate', 'rule6_hair_return_rate']

2. **Step 1 匹配失败，Step 2 推断失败，Step 3 映射成功，但原始 rule_name 在 Step 4 被添加**
   - 如果 Step 3 的映射表不完整（缺少某些规则名称）
   - 或者 Step 3 的映射逻辑有问题
   - 那么 Step 4 可能会添加原始 rule_name
   - 结果：violated_rules = ['rule8_return_rate'] 或 ['rule8_return_rate', 'rule6_hair_return_rate']

3. **Step 1 匹配成功，但后续代码路径又添加了原始 rule_name**
   - 如果存在其他代码路径（如 else 分支）在 Step 1 之后执行
   - 可能会添加原始 rule_name
   - 结果：violated_rules = ['rule6_hair_return_rate', 'rule8_return_rate']

4. **Step 1 匹配失败，但违规消息实际上应该能匹配**
   - 如果 violation_msg_lower 的格式与预期不符
   - 或者 rule_mapping 中的模式不够全面
   - 那么 Step 1 会失败，依赖后续步骤
   - 结果：如果后续步骤也失败，就会返回原始 rule_name
""")

