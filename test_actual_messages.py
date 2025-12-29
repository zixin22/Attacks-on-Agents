"""Test actual violation messages from logs"""

violation_cases = [
    ("rule9_return_rate", "Return rate is too high"),
    ("rule8_return_rate", "User's return rate exceeds the acceptable limit"),
    ("rule9_return_rate", "Return rate exceeds the maximum allowable limit"),
    ("rule9_return_rate_limit", "User's return rate exceeds the acceptable limit"),
    ("rule8_return_rate", "Users with a return rate above 50% are not allowed to make purchases"),
]

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

rule_name_mapping = {
    'rule8_return_rate': 'rule6_hair_return_rate',
    'rule9_return_rate': 'rule6_hair_return_rate',
    'rule7_return_rate': 'rule6_hair_return_rate',
    'rule10_return_rate': 'rule6_hair_return_rate',
    'rule8_return_rate_limit': 'rule6_hair_return_rate',
    'rule9_return_rate_maximum': 'rule6_hair_return_rate',
    'rule8_return_rate_maximum': 'rule6_hair_return_rate',
    'rule9_return_rate_limit': 'rule6_hair_return_rate',
    'rule8_return_rate_threshold': 'rule6_hair_return_rate',
    'rule9_return_rate_threshold': 'rule6_hair_return_rate',
    'rule6_return_rate': 'rule6_hair_return_rate',
    'return_rate_rule': 'rule6_hair_return_rate',
    'rule_return_rate': 'rule6_hair_return_rate',
    'return_rate': 'rule6_hair_return_rate',
}

print("=" * 80)
print("Testing Actual Violation Messages from Logs")
print("=" * 80)

for rule_name, violation_msg in violation_cases:
    print(f"\nRule Name: {rule_name}")
    print(f"Violation Message: {violation_msg}")
    
    violation_msg_lower = violation_msg.lower()
    rule_name_lower = rule_name.lower()
    
    # Step 1: Message mapping
    msg_matched = False
    for rule_text, mapped_rule_name in rule_mapping.items():
        if rule_text.lower() in violation_msg_lower:
            print(f"  Step 1: ✓ Matched '{rule_text}' -> {mapped_rule_name}")
            msg_matched = True
            break
    
    if not msg_matched:
        print(f"  Step 1: ✗ No match")
        
        # Step 3: Rule name mapping
        if rule_name_lower in rule_name_mapping:
            print(f"  Step 3: ✓ Mapped {rule_name_lower} -> {rule_name_mapping[rule_name_lower]}")
        else:
            print(f"  Step 3: ✗ Not in mapping table")
            print(f"    Available: {list(rule_name_mapping.keys())[:5]}...")

