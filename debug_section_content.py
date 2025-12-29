"""Debug script to check what section content is passed to inference"""

# Simulate the actual GuardAgent results section
section = """GuardAgent results:
action_denied: 1
inaccessible_actions: {'rule8_return_rate': 'Return rate exceeds the allowable limit'}
guardrailed_answer: 
(End of results)"""

print("=" * 80)
print("Debugging Section Content")
print("=" * 80)
print(f"\nSection content:\n{section}")
print("\n" + "=" * 80)

# Extract violation_text
violation_text = ""
if "inaccessible_actions:" in section:
    inaccessible_actions = section.split("inaccessible_actions:")[1]
    inaccessible_actions = inaccessible_actions.split("\n")[0].strip()
    violation_text = inaccessible_actions
    print(f"\nExtracted violation_text: {violation_text}")

# Parse dict format
import re
dict_pattern = r"['\"]([^'\"]+)['\"]\s*:\s*['\"]([^'\"]+)['\"]"
dict_matches = re.findall(dict_pattern, violation_text)

if dict_matches:
    print(f"\nDict matches found: {len(dict_matches)}")
    for rule_name, violation_msg in dict_matches:
        print(f"  Rule name: {rule_name}")
        print(f"  Violation message: {violation_msg}")
        
        violation_msg_lower = violation_msg.lower()
        section_lower = section.lower()
        
        print(f"\n  violation_msg_lower: {violation_msg_lower}")
        print(f"  section_lower (first 200 chars): {section_lower[:200]}...")
        
        # Check if section contains hair-related keywords
        hair_keywords = ['hair', 'extension', 'wig']
        print(f"\n  Checking for hair keywords in section:")
        for kw in hair_keywords:
            found = kw in section_lower
            print(f"    '{kw}' in section: {found}")
        
        # Test Step 1 matching
        rule_mapping = {
            "return rate exceeds": "rule6_hair_return_rate",
            "return rate exceeds the": "rule6_hair_return_rate",
            "return rate exceeds the acceptable": "rule6_hair_return_rate",
            "return rate exceeds the allowable": "rule6_hair_return_rate",
        }
        
        print(f"\n  Testing Step 1 (standard mapping):")
        msg_matched = False
        for rule_text, mapped_rule_name in rule_mapping.items():
            if rule_text.lower() in violation_msg_lower:
                print(f"    ✓ Matched: '{rule_text}' -> {mapped_rule_name}")
                msg_matched = True
                break
        
        if not msg_matched:
            print(f"    ✗ No match in Step 1")
            
            # Test Step 2 inference
            print(f"\n  Testing Step 2 (keyword inference):")
            rule_patterns = [
                (['return rate', 'exceeds'], ['hair', 'extension', 'wig'], 'rule6_hair_return_rate'),
                (['return rate', 'limit'], ['hair', 'extension', 'wig'], 'rule6_hair_return_rate'),
            ]
            
            for msg_keywords, context_keywords, rule_name in rule_patterns:
                msg_keywords_lower = [kw.lower() for kw in msg_keywords]
                context_keywords_lower = [kw.lower() for kw in context_keywords]
                msg_match = any(keyword in violation_msg_lower for keyword in msg_keywords_lower)
                context_match = any(keyword in section_lower for keyword in context_keywords_lower)
                
                print(f"    Pattern: {msg_keywords} + {context_keywords}")
                print(f"      msg_match: {msg_match}, context_match: {context_match}")
                if msg_match and context_match:
                    print(f"      ✓ INFERRED: {rule_name}")
                    break
else:
    print("\nNo dict matches found!")

