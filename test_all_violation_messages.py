"""Test all possible violation messages"""

violation_messages = [
    "Return rate exceeds the allowable limit",
    "Return rate exceeds the acceptable limit",
    "Return rate exceeds the maximum allowable limit",
    "Return rate is too high",
    "User's return rate exceeds the acceptable limit",
    "Return rate exceeds the threshold",
]

rule_mapping = {
    "return rate exceeds": "rule6_hair_return_rate",
    "return rate exceeds the": "rule6_hair_return_rate",
    "return rate exceeds the acceptable": "rule6_hair_return_rate",
    "return rate exceeds the allowable": "rule6_hair_return_rate",
    "return rate exceeds the maximum": "rule6_hair_return_rate",
    "return rate limit": "rule6_hair_return_rate",
    "return rate threshold": "rule6_hair_return_rate",
    "return rate maximum": "rule6_hair_return_rate",
    "return rate is too high": "rule6_hair_return_rate",
}

print("=" * 80)
print("Testing All Violation Messages")
print("=" * 80)

for violation_msg in violation_messages:
    violation_msg_lower = violation_msg.lower()
    print(f"\nViolation message: '{violation_msg}'")
    
    matched = False
    for rule_text, mapped_rule_name in rule_mapping.items():
        if rule_text.lower() in violation_msg_lower:
            print(f"  ✓ Matched: '{rule_text}' -> {mapped_rule_name}")
            matched = True
            break
    
    if not matched:
        print(f"  ✗ NO MATCH!")
        print(f"    Checking individual patterns:")
        for rule_text in rule_mapping.keys():
            if rule_text.lower() in violation_msg_lower:
                print(f"      - '{rule_text}' found")
            else:
                print(f"      - '{rule_text}' NOT found")

