"""Test the complete mapping logic"""

# Simulate actual GuardAgent results
violation_text = "{'rule9_return_rate_threshold': 'Return rate exceeds the maximum allowable limit'}"
agent_input = """User Profile:
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
- Query: """

section = f"""GuardAgent results:
action_denied: 1
inaccessible_actions: {violation_text}
guardrailed_answer: 
(End of results)"""

print("=" * 80)
print("Testing Complete Mapping Logic")
print("=" * 80)

# Step 1: Extract violation_text
violation_text_extracted = ""
if "inaccessible_actions:" in section:
    inaccessible_actions = section.split("inaccessible_actions:")[1]
    inaccessible_actions = inaccessible_actions.split("\n")[0].strip()
    violation_text_extracted = inaccessible_actions

print(f"\nExtracted violation_text: {violation_text_extracted}")

# Step 2: Parse dict format
import re
dict_pattern = r"['\"]([^'\"]+)['\"]\s*:\s*['\"]([^'\"]+)['\"]"
dict_matches = re.findall(dict_pattern, violation_text_extracted)

if dict_matches:
    print(f"\nDict matches: {dict_matches}")
    for rule_name, violation_msg in dict_matches:
        print(f"\n  Rule name: {rule_name}")
        print(f"  Violation message: {violation_msg}")
        
        violation_msg_lower = violation_msg.lower()
        
        # Build full_context
        full_context = agent_input.lower() + " " + section.lower()
        
        # Step 1: Standard mapping
        rule_mapping = {
            "return rate exceeds": "rule6_hair_return_rate",
            "return rate exceeds the": "rule6_hair_return_rate",
            "return rate exceeds the acceptable": "rule6_hair_return_rate",
            "return rate exceeds the allowable": "rule6_hair_return_rate",
            "return rate exceeds the maximum": "rule6_hair_return_rate",
            "return rate limit": "rule6_hair_return_rate",
            "return rate threshold": "rule6_hair_return_rate",
            "return rate maximum": "rule6_hair_return_rate",
        }
        
        print(f"\n  Step 1: Standard mapping")
        msg_matched = False
        for rule_text, mapped_rule_name in rule_mapping.items():
            if rule_text.lower() in violation_msg_lower:
                print(f"    ✓ Matched: '{rule_text}' -> {mapped_rule_name}")
                msg_matched = True
                break
        
        if not msg_matched:
            print(f"    ✗ No match in Step 1")
            print(f"    Violation message: '{violation_msg_lower}'")
            print(f"    Checking patterns:")
            for rule_text in rule_mapping.keys():
                if rule_text.lower() in violation_msg_lower:
                    print(f"      - '{rule_text}' found in message")
                else:
                    print(f"      - '{rule_text}' NOT found")
        
        # Step 2: Keyword inference
        if not msg_matched:
            print(f"\n  Step 2: Keyword inference")
            rule_patterns = [
                (['return rate', 'exceeds'], ['hair', 'extension'], 'rule6_hair_return_rate'),
                (['return rate', 'limit'], ['hair', 'extension'], 'rule6_hair_return_rate'),
                (['return rate', 'maximum'], ['hair', 'extension'], 'rule6_hair_return_rate'),
            ]
            
            for msg_keywords, context_keywords, inferred_rule in rule_patterns:
                msg_keywords_lower = [kw.lower() for kw in msg_keywords]
                context_keywords_lower = [kw.lower() for kw in context_keywords]
                msg_match = any(keyword in violation_msg_lower for keyword in msg_keywords_lower)
                context_match = any(keyword in full_context for keyword in context_keywords_lower)
                
                if msg_match and context_match:
                    print(f"    ✓ Inferred: {inferred_rule}")
                    msg_matched = True
                    break
            
            if not msg_matched:
                print(f"    ✗ No inference in Step 2")
        
        # Step 3: Rule name mapping
        if not msg_matched:
            print(f"\n  Step 3: Rule name mapping")
            rule_name_mapping = {
                'rule9_return_rate_threshold': 'rule6_hair_return_rate',
                'rule9_return_rate_limit': 'rule6_hair_return_rate',
                'rule8_return_rate': 'rule6_hair_return_rate',
            }
            
            rule_name_lower = rule_name.lower()
            print(f"    Rule name (lowercase): {rule_name_lower}")
            if rule_name_lower in rule_name_mapping:
                print(f"    ✓ Mapped: {rule_name_lower} -> {rule_name_mapping[rule_name_lower]}")
                msg_matched = True
            else:
                print(f"    ✗ Not in mapping table")
                print(f"    Available mappings: {list(rule_name_mapping.keys())}")

