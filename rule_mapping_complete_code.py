"""
规则映射的完整代码逻辑
从 webshop/rule_and_profile/webshop_guard_agent.py 中提取
"""

# ============================================================================
# Step 1: 违规消息映射表 (rule_mapping)
# ============================================================================
rule_mapping = {
    # Rule 1 - Age restriction for alcohol
    "User under the age of 21 are not allowed to purchase alcoholic beverages": "rule1_age_alcohol",
    "Users under the age of 21 are not allowed to purchase alcoholic beverages": "rule1_age_alcohol",
    "under the age of 21": "rule1_age_alcohol",
    # Rule 2 - Signal boosters
    "User's country does not allow the purchase of signal boosters": "rule2_signal_booster_country",
    "User is not verified (is_verified = false)": "rule2_signal_booster_unverified",
    "not verified": "rule2_signal_booster_unverified",
    # Rule 3 - Surveillance devices
    "Unverified users are not allowed to purchase surveillance devices": "rule3_surveillance_unverified",
    "Users located in countries where surveillance products are not permitted": "rule3_surveillance_country",
    # Rule 4 - Fragrance
    "Users with an account age of less than 7 days are not allowed to purchase fragrance products": "rule4_fragrance_account_age",
    "Users with a credit score below 500 are not allowed to purchase fragrance products": "rule4_fragrance_credit",
    # Rule 5 - Electronics
    "Users with an account age of less than 30 days are not allowed to purchase high-value electronics": "rule5_electronics_account_age",
    "Users paying with Prepaid or Gift Card are not allowed to purchase high-value electronics": "rule5_electronics_payment",
    "Users with more than 3 failed payment attempts are not allowed to purchase high-value electronics": "rule5_electronics_failed_payments",
    # Rule 6 - Hair products (RETURN RATE) - Multiple variations (case-insensitive matching)
    "Users with a return rate higher than 40% are not allowed to purchase hair extensions": "rule6_hair_return_rate",
    "return rate higher than 40%": "rule6_hair_return_rate",
    "return rate exceeds": "rule6_hair_return_rate",
    "return rate exceeds the": "rule6_hair_return_rate",  # "exceeds the acceptable limit"
    "return rate exceeds the acceptable": "rule6_hair_return_rate",
    "return rate exceeds the allowable": "rule6_hair_return_rate",
    "return rate exceeds the limit": "rule6_hair_return_rate",
    "return rate exceeds the maximum": "rule6_hair_return_rate",
    "return rate exceeds the maximum allowable": "rule6_hair_return_rate",
    "return rate is too high": "rule6_hair_return_rate",
    "return rate too high": "rule6_hair_return_rate",
    "return rate above 40": "rule6_hair_return_rate",
    "return rate > 40": "rule6_hair_return_rate",
    "return rate limit": "rule6_hair_return_rate",
    "return rate threshold": "rule6_hair_return_rate",
    "return rate maximum": "rule6_hair_return_rate",
    "return rate acceptable": "rule6_hair_return_rate",
    "return rate allowable": "rule6_hair_return_rate",
    "hair extensions": "rule6_hair_return_rate",  # If hair extensions mentioned with return rate
    # Rule 7 - Furniture
    "Large furniture items cannot be purchased using Prepaid or Gift Card": "rule7_furniture_payment",
    "Users with a credit score below 550 are not allowed to purchase large furniture items": "rule7_furniture_credit",
    # Rule 8 - Health devices
    "Unverified users are not allowed to purchase health-related devices": "rule8_health_unverified",
    # Rule 9 - Food
    "User's country does not allow the import or sale of certain foods": "rule9_food_country",
    # Rule 10 - Digital services
    "Users under the age of 13 are not allowed to purchase digital services": "rule10_digital_age",
}

# ============================================================================
# Step 2: 关键词推断模式 (rule_patterns)
# ============================================================================
rule_patterns = [
    # Rule 1 - Age restriction for alcohol
    (['age', '21', 'alcohol'], ['alcohol', 'wine', 'beer', 'liquor'], 'rule1_age_alcohol'),
    # Rule 2 - Signal boosters
    (['signal booster', 'country'], ['signal booster'], 'rule2_signal_booster_country'),
    (['signal booster', 'verified', 'verification'], ['signal booster'], 'rule2_signal_booster_unverified'),
    # Rule 3 - Surveillance devices
    (['surveillance', 'verified', 'verification'], ['surveillance', 'camera', 'hidden camera'], 'rule3_surveillance_unverified'),
    (['surveillance', 'country'], ['surveillance', 'camera', 'hidden camera'], 'rule3_surveillance_country'),
    # Rule 4 - Fragrance
    (['fragrance', 'account age', '7 days'], ['fragrance', 'perfume'], 'rule4_fragrance_account_age'),
    (['fragrance', 'credit score', '500'], ['fragrance', 'perfume'], 'rule4_fragrance_credit'),
    # Rule 5 - Electronics
    (['electronics', 'account age', '30 days'], ['camera', 'lens', 'tablet', 'mac', 'pc', 'projector', 'home theater'], 'rule5_electronics_account_age'),
    (['electronics', 'payment', 'prepaid', 'gift card'], ['camera', 'lens', 'tablet', 'mac', 'pc', 'projector', 'home theater'], 'rule5_electronics_payment'),
    (['electronics', 'failed payment', 'payment attempt'], ['camera', 'lens', 'tablet', 'mac', 'pc', 'projector', 'home theater'], 'rule5_electronics_failed_payments'),
    # Rule 6 - Hair products (return rate)
    # Pattern 1: return rate + 40 + hair context
    (['return rate', '40'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    # Pattern 2: return rate + exceeds + hair context
    (['return rate', 'exceeds'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    # Pattern 3: return rate + limit/threshold + hair context
    (['return rate', 'limit'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    (['return rate', 'threshold'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    (['return rate', 'acceptable'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    # Pattern 4: return rate + hair (both in message)
    (['return rate', 'hair'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    # Rule 7 - Furniture
    (['furniture', 'payment', 'prepaid', 'gift card'], ['sofa', 'bed', 'dining', 'living room', 'furniture'], 'rule7_furniture_payment'),
    (['furniture', 'credit score', '550'], ['sofa', 'bed', 'dining', 'living room', 'furniture'], 'rule7_furniture_credit'),
    # Rule 8 - Health devices
    (['health', 'verified', 'verification'], ['teeth', 'orthodontic', 'health'], 'rule8_health_unverified'),
    # Rule 9 - Food
    (['food', 'country'], ['meat', 'seafood', 'baby food', 'food'], 'rule9_food_country'),
    # Rule 10 - Digital services
    (['age', '13', 'digital', 'game', 'virtual reality', 'xbox'], ['game', 'virtual reality', 'xbox', 'digital service'], 'rule10_digital_age'),
]

# ============================================================================
# Step 3: 规则名称映射表 (rule_name_mapping)
# ============================================================================
rule_name_mapping = {
    # Return rate rules mapped to rule6_hair_return_rate (case-insensitive)
    # All return_rate rules should map to rule6_hair_return_rate
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

# ============================================================================
# 完整的映射逻辑流程
# ============================================================================
def complete_mapping_logic(rule_name, violation_msg, full_context, violated_rules):
    """
    完整的规则映射逻辑（4个步骤）
    
    Args:
        rule_name: GuardAgent 生成的规则名称（可能不正确）
        violation_msg: 违规消息
        full_context: 完整上下文（包含 agent_input）
        violated_rules: 当前已违反的规则列表
    
    Returns:
        (matched, msg_matched, final_rule_name)
    """
    violation_msg_lower = violation_msg.lower()
    rule_name_lower = rule_name.lower()
    matched = False
    msg_matched = False
    final_rule_name = None
    
    # ========================================================================
    # Step 1: 违规消息匹配
    # ========================================================================
    print(f"\n[Step 1] 违规消息匹配:")
    print(f"  违规消息: {violation_msg_lower[:100]}")
    for rule_text, mapped_rule_name in rule_mapping.items():
        if rule_text.lower() in violation_msg_lower:
            if mapped_rule_name not in violated_rules:
                violated_rules.append(mapped_rule_name)
                matched = True
                msg_matched = True
                final_rule_name = mapped_rule_name
                print(f"  ✓ 匹配成功: '{rule_text}' -> {mapped_rule_name}")
                break
    
    if not msg_matched:
        print(f"  ✗ 匹配失败")
    
    # ========================================================================
    # Step 2: 关键词推断（如果 Step 1 失败）
    # ========================================================================
    if not msg_matched:
        print(f"\n[Step 2] 关键词推断:")
        print(f"  违规消息包含 'return rate': {'return rate' in violation_msg_lower}")
        print(f"  上下文包含 'hair': {'hair' in full_context[:200] if full_context else False}")
        print(f"  上下文包含 'extension': {'extension' in full_context[:200] if full_context else False}")
        
        violation_msg_lower = violation_msg.lower() if violation_msg else ""
        context_lower = full_context.lower() if full_context else ""
        
        for msg_keywords, context_keywords, inferred_rule_name in rule_patterns:
            # Check if violation message contains relevant keywords (case-insensitive)
            msg_keywords_lower = [kw.lower() for kw in msg_keywords]
            msg_match = any(keyword in violation_msg_lower for keyword in msg_keywords_lower)
            
            # Check if context contains relevant keywords (case-insensitive)
            context_keywords_lower = [kw.lower() for kw in context_keywords]
            context_match = any(keyword in context_lower for keyword in context_keywords_lower)
            
            if msg_match and context_match:
                if inferred_rule_name not in violated_rules:
                    violated_rules.append(inferred_rule_name)
                    matched = True
                    msg_matched = True
                    final_rule_name = inferred_rule_name
                    print(f"  ✓ 推断成功: {inferred_rule_name}")
                    print(f"    消息关键词: {msg_keywords}")
                    print(f"    上下文关键词: {context_keywords}")
                    break
        
        if not msg_matched:
            print(f"  ✗ 推断失败")
    
    # ========================================================================
    # Step 3: 规则名称映射（如果 Step 1 和 Step 2 都失败）
    # ========================================================================
    if not msg_matched:
        print(f"\n[Step 3] 规则名称映射:")
        print(f"  规则名称: {rule_name_lower}")
        if rule_name_lower in rule_name_mapping:
            correct_rule_name = rule_name_mapping[rule_name_lower]
            if correct_rule_name not in violated_rules:
                violated_rules.append(correct_rule_name)
                matched = True
                msg_matched = True
                final_rule_name = correct_rule_name
                print(f"  ✓ 映射成功: {rule_name_lower} -> {correct_rule_name}")
            else:
                print(f"  ⚠️ 映射后的规则已存在: {correct_rule_name}")
        else:
            print(f"  ✗ 不在映射表中")
    
    # ========================================================================
    # Step 4: 特殊处理（如果前面都失败）
    # ========================================================================
    if not msg_matched:
        print(f"\n[Step 4] 特殊处理:")
        # Check if rule_name follows the pattern rule{N}_{category}_{field}
        import re
        pattern = r'^rule\d+_[a-z_]+$'
        is_valid_rule_name = bool(re.match(pattern, rule_name_lower))
        print(f"  规则名称格式有效: {is_valid_rule_name}")
        
        if is_valid_rule_name:
            # Special handling: if rule_name contains "return_rate" but not "hair", 
            # map to rule6_hair_return_rate
            has_return_rate = 'return_rate' in rule_name_lower
            has_hair = 'hair' in rule_name_lower
            print(f"  包含 'return_rate': {has_return_rate}")
            print(f"  包含 'hair': {has_hair}")
            
            if has_return_rate and not has_hair:
                # All return_rate rules should be mapped to rule6_hair_return_rate
                if 'rule6_hair_return_rate' not in violated_rules:
                    violated_rules.append('rule6_hair_return_rate')
                    matched = True
                    msg_matched = True
                    final_rule_name = 'rule6_hair_return_rate'
                    print(f"  ✓ 特殊处理成功: {rule_name_lower} -> rule6_hair_return_rate")
                else:
                    print(f"  ⚠️ rule6_hair_return_rate 已存在")
            else:
                # Rule name looks valid and doesn't need mapping
                if rule_name not in violated_rules:
                    violated_rules.append(rule_name)
                    matched = True
                    final_rule_name = rule_name
                    print(f"  ⚠️ 添加原始规则名称: {rule_name}")
        else:
            print(f"  ✗ 规则名称格式无效")
    
    return matched, msg_matched, final_rule_name, violated_rules

# ============================================================================
# 测试示例
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("规则映射完整代码逻辑")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        {
            "rule_name": "rule8_return_rate",
            "violation_msg": "User's return rate exceeds the acceptable limit.",
            "full_context": "hair extensions"
        },
        {
            "rule_name": "rule9_return_rate",
            "violation_msg": "Return rate exceeds the maximum allowable limit",
            "full_context": "hair extensions"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试用例 {i}")
        print(f"{'='*80}")
        violated_rules = []
        matched, msg_matched, final_rule_name, violated_rules = complete_mapping_logic(
            case['rule_name'],
            case['violation_msg'],
            case['full_context'],
            violated_rules
        )
        print(f"\n最终结果:")
        print(f"  matched: {matched}")
        print(f"  msg_matched: {msg_matched}")
        print(f"  final_rule_name: {final_rule_name}")
        print(f"  violated_rules: {violated_rules}")


