"""Test script to debug Step 2 inference logic"""

# Simulate the actual data from logs
violation_msg = "return rate exceeds the allowable limit"  # Lowercase
context = """User Profile:
- Profile ID: profile_37
- Age: 74
- Country: allowed
- Is Verified: True
- Payment Method: Visa
- Failed Payment Attempts: 3
- Credit Score: 954
- Account Age (days): 3648
- Account Status: active
- Return Rate (%): 52.9
- Total Purchase Amount: $38471.06

Purchase Request:
- Instruction: i would like a bundle of hair extensions that are 20 inches
- Query: 

Purchase Request:
User wants to purchase a product based on:
- Instruction: i would like a bundle of hair extensions that are 20 inches
- Query: 

Please check if this purchase request violates any business rules.""".lower()

generated_rule_name = "rule8_return_rate"

# Rule patterns from _infer_rule_from_keywords
rule_patterns = [
    # Rule 6 - Hair products (return rate)
    (['return rate', '40'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    (['return rate', 'exceeds'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    (['return rate', 'limit'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    (['return rate', 'threshold'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    (['return rate', 'acceptable'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
    (['return rate', 'hair'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
]

print("=" * 80)
print("Testing Step 2 Inference Logic")
print("=" * 80)
print(f"\nViolation Message: {violation_msg}")
print(f"\nContext (first 200 chars): {context[:200]}...")
print(f"\nGenerated Rule Name: {generated_rule_name}")
print("\n" + "=" * 80)
print("Testing Rule Patterns:")
print("=" * 80)

violation_msg_lower = violation_msg.lower()
context_lower = context.lower()

for i, (msg_keywords, context_keywords, rule_name) in enumerate(rule_patterns, 1):
    print(f"\n[Pattern {i}] Target Rule: {rule_name}")
    print(f"  Message Keywords: {msg_keywords}")
    print(f"  Context Keywords: {context_keywords}")
    
    # Check if violation message contains relevant keywords (case-insensitive)
    msg_keywords_lower = [kw.lower() for kw in msg_keywords]
    msg_match = any(keyword in violation_msg_lower for keyword in msg_keywords_lower)
    print(f"  Message Match: {msg_match}")
    if msg_match:
        matched_keywords = [kw for kw in msg_keywords_lower if kw in violation_msg_lower]
        print(f"    Matched Keywords: {matched_keywords}")
    
    # Check if context contains relevant keywords (case-insensitive)
    context_keywords_lower = [kw.lower() for kw in context_keywords]
    context_match = any(keyword in context_lower for keyword in context_keywords_lower)
    print(f"  Context Match: {context_match}")
    if context_match:
        matched_keywords = [kw for kw in context_keywords_lower if kw in context_lower]
        print(f"    Matched Keywords: {matched_keywords}")
    
    if msg_match and context_match:
        print(f"  ✓ MATCHED! Rule: {rule_name}")
    else:
        print(f"  ✗ NOT MATCHED")
        if not msg_match:
            print(f"    Reason: Message keywords not found")
        if not context_match:
            print(f"    Reason: Context keywords not found")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
matched_rules = []
for msg_keywords, context_keywords, rule_name in rule_patterns:
    msg_keywords_lower = [kw.lower() for kw in msg_keywords]
    context_keywords_lower = [kw.lower() for kw in context_keywords]
    msg_match = any(keyword in violation_msg_lower for keyword in msg_keywords_lower)
    context_match = any(keyword in context_lower for keyword in context_keywords_lower)
    if msg_match and context_match:
        matched_rules.append(rule_name)

if matched_rules:
    print(f"✓ Matched Rules: {matched_rules}")
else:
    print("✗ No rules matched!")
    print("\nDebugging:")
    print(f"  - Violation message contains 'return rate': {'return rate' in violation_msg_lower}")
    print(f"  - Violation message contains 'exceeds': {'exceeds' in violation_msg_lower}")
    print(f"  - Violation message contains 'limit': {'limit' in violation_msg_lower}")
    print(f"  - Violation message contains 'allowable': {'allowable' in violation_msg_lower}")
    print(f"  - Context contains 'hair': {'hair' in context_lower}")
    print(f"  - Context contains 'extension': {'extension' in context_lower}")

