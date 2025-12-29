import re
import json
from pathlib import Path

path = Path(r"webshop/guardagent_logs/guardagent_execution_20251227_050705.log")
log = path.read_text(encoding="utf-8")

# Extract code cells
pattern = r'"cell"\s*:\s*"((?:[^"\\]|\\.)*)"'
matches = re.findall(pattern, log)

print(f"Found {len(matches)} code cells\n")

for i, cell_escaped in enumerate(matches[:1], 1):  # Only show first one
    # Decode escaped characters
    cell = cell_escaped.encode().decode('unicode_escape')
    
    print("=" * 80)
    print(f"CODE CELL {i} (FULL)")
    print("=" * 80)
    print(cell)
    print("=" * 80)
    
    # Analyze issues
    print("\n[ANALYSIS]")
    print("-" * 80)
    
    # Check 1: Function definition vs execution
    if "def " in cell:
        func_lines = [l for l in cell.split('\n') if l.strip().startswith('def ')]
        print(f"✓ Function definitions found: {len(func_lines)}")
        for fl in func_lines:
            func_name = fl.split('def ')[1].split('(')[0].strip()
            print(f"  - {func_name}")
            # Check if called
            if f"{func_name}(" in cell:
                print(f"    ✓ Function {func_name} IS called")
            else:
                print(f"    ✗ Function {func_name} is NOT called")
    else:
        print("✓ No function definitions (code executes directly)")
    
    # Check 2: ACTION DENIED/GRANTED
    if "ACTION DENIED" in cell:
        print("✓ Contains 'ACTION DENIED'")
    elif "ACTION GRANTED" in cell:
        print("✓ Contains 'ACTION GRANTED'")
    else:
        print("✗ Missing 'ACTION DENIED' or 'ACTION GRANTED'")
    
    # Check 3: CheckRule usage
    if "CheckRule(" in cell:
        print("✓ Uses CheckRule function")
        # Extract CheckRule call
        import re as re2
        checkrule_calls = re2.findall(r'CheckRule\([^)]+\)', cell)
        for call in checkrule_calls:
            print(f"  CheckRule call: {call[:100]}")
    else:
        print("✗ Does not use CheckRule function")
    
    # Check 4: list_of_rules format
    if "list_of_rules" in cell:
        print("✓ Defines list_of_rules")
        # Try to extract the structure
        lines = cell.split('\n')
        in_rules = False
        rules_lines = []
        for line in lines:
            if 'list_of_rules' in line and '=' in line:
                in_rules = True
            if in_rules:
                rules_lines.append(line)
                if line.strip().endswith('}') and '{' in ''.join(rules_lines):
                    break
        if rules_lines:
            print(f"  Rules definition (first 10 lines):")
            for rl in rules_lines[:10]:
                print(f"    {rl}")
    else:
        print("✗ Does not define list_of_rules")
    
    # Check 5: user_info format
    if "user_info" in cell:
        print("✓ Defines user_info")
    else:
        print("✗ Does not define user_info")
    
    # Check 6: Print statements
    print_lines = [l for l in cell.split('\n') if 'print(' in l]
    if print_lines:
        print(f"✓ Contains {len(print_lines)} print statements")
        for pl in print_lines[:3]:
            print(f"  {pl.strip()}")
    else:
        print("✗ No print statements found")

