"""分析映射逻辑的3个步骤，找出问题所在"""

# 模拟实际场景
test_cases = [
    {
        "rule_name": "rule8_return_rate",
        "violation_msg": "User's return rate exceeds the acceptable limit.",
        "context": "hair extensions"
    },
    {
        "rule_name": "rule8_return_rate_limit",
        "violation_msg": "User's return rate exceeds the acceptable limit.",
        "context": "hair extensions"
    },
    {
        "rule_name": "rule9_return_rate",
        "violation_msg": "Return rate exceeds the maximum allowable limit",
        "context": "hair extensions"
    }
]

# Step 1: 违规消息映射表
rule_mapping = {
    "return rate exceeds": "rule6_hair_return_rate",
    "return rate exceeds the": "rule6_hair_return_rate",
    "return rate exceeds the acceptable": "rule6_hair_return_rate",
    "return rate exceeds the allowable": "rule6_hair_return_rate",
    "return rate exceeds the maximum": "rule6_hair_return_rate",
    "return rate exceeds the maximum allowable": "rule6_hair_return_rate",
    "return rate is too high": "rule6_hair_return_rate",
    "return rate too high": "rule6_hair_return_rate",
    "return rate limit": "rule6_hair_return_rate",
    "return rate threshold": "rule6_hair_return_rate",
    "return rate maximum": "rule6_hair_return_rate",
    "return rate acceptable": "rule6_hair_return_rate",
    "return rate allowable": "rule6_hair_return_rate",
}

# Step 3: 规则名称映射表
rule_name_mapping = {
    'rule7_return_rate': 'rule6_hair_return_rate',
    'rule7_return_rate_limit': 'rule6_hair_return_rate',
    'rule7_return_rate_maximum': 'rule6_hair_return_rate',
    'rule7_return_rate_threshold': 'rule6_hair_return_rate',
    'rule8_return_rate': 'rule6_hair_return_rate',
    'rule8_return_rate_limit': 'rule6_hair_return_rate',
    'rule8_return_rate_maximum': 'rule6_hair_return_rate',
    'rule8_return_rate_threshold': 'rule6_hair_return_rate',
    'rule9_return_rate': 'rule6_hair_return_rate',
    'rule9_return_rate_limit': 'rule6_hair_return_rate',
    'rule9_return_rate_maximum': 'rule6_hair_return_rate',
    'rule9_return_rate_threshold': 'rule6_hair_return_rate',
    'rule10_return_rate': 'rule6_hair_return_rate',
    'rule6_return_rate': 'rule6_hair_return_rate',
    'return_rate_rule': 'rule6_hair_return_rate',
    'rule_return_rate': 'rule6_hair_return_rate',
    'return_rate': 'rule6_hair_return_rate',
}

print("=" * 80)
print("分析映射逻辑的3个步骤")
print("=" * 80)

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"测试用例 {i}: rule_name={case['rule_name']}, violation_msg={case['violation_msg']}")
    print(f"{'='*80}")
    
    violation_msg_lower = case['violation_msg'].lower()
    rule_name_lower = case['rule_name'].lower()
    context_lower = case['context'].lower()
    
    msg_matched = False
    
    # Step 1: 违规消息匹配
    print(f"\n[Step 1] 违规消息匹配:")
    for rule_text, mapped_rule_name in rule_mapping.items():
        if rule_text.lower() in violation_msg_lower:
            print(f"  ✓ 匹配到: '{rule_text}' -> {mapped_rule_name}")
            msg_matched = True
            break
    
    if not msg_matched:
        print(f"  ✗ 未匹配到任何模式")
        print(f"  违规消息: '{violation_msg_lower}'")
        print(f"  检查的模式:")
        for rule_text in rule_mapping.keys():
            if rule_text.lower() in violation_msg_lower:
                print(f"    - '{rule_text}' ✓ 找到")
            else:
                print(f"    - '{rule_text}' ✗ 未找到")
    
    # Step 2: 关键词推断（如果Step 1失败）
    if not msg_matched:
        print(f"\n[Step 2] 关键词推断:")
        print(f"  违规消息包含 'return rate': {'return rate' in violation_msg_lower}")
        print(f"  上下文包含 'hair': {'hair' in context_lower}")
        print(f"  上下文包含 'extension': {'extension' in context_lower}")
        if 'return rate' in violation_msg_lower and ('hair' in context_lower or 'extension' in context_lower):
            print(f"  ✓ 应该能推断出: rule6_hair_return_rate")
        else:
            print(f"  ✗ 推断失败")
    
    # Step 3: 规则名称映射（如果Step 1和Step 2都失败）
    if not msg_matched:
        print(f"\n[Step 3] 规则名称映射:")
        print(f"  规则名称: {rule_name_lower}")
        if rule_name_lower in rule_name_mapping:
            print(f"  ✓ 在映射表中找到: {rule_name_lower} -> {rule_name_mapping[rule_name_lower]}")
        else:
            print(f"  ✗ 不在映射表中")
            print(f"  映射表包含的规则:")
            for key in sorted(rule_name_mapping.keys()):
                print(f"    - {key}")
    
    # 总结
    print(f"\n[总结]")
    if msg_matched:
        print(f"  ✓ Step 1 匹配成功")
    else:
        print(f"  ✗ 所有步骤都失败，应该返回原始规则名称: {case['rule_name']}")

