import re
import json
from pathlib import Path

path = Path(r"webshop/guardagent_logs/guardagent_execution_20251227_050705.log")
log = path.read_text(encoding="utf-8")

pattern = r'"cell"\s*:\s*"((?:\\.|[^"\\])*)"'
matches = re.findall(pattern, log)

print("Found", len(matches), "cells")
print("=" * 80)

for i, cell in enumerate(matches, 1):
    # Decode escaped characters
    cell_decoded = cell.encode().decode('unicode_escape')
    print(f"\n[Cell {i}]")
    print("=" * 80)
    # Print full code (no truncation)
    print(cell_decoded)
    print("=" * 80)
    
    # Analyze the code
    if "def " in cell_decoded and "(" in cell_decoded.split("def ")[1]:
        func_def = cell_decoded.split("def ")[1].split("(")[0].strip()
        print(f"\n[Analysis] Function defined: {func_def}")
        if func_def not in cell_decoded.split("def ")[1]:
            print(f"[Analysis] Function '{func_def}' is NOT called in the code")
        else:
            # Check if it's called
            after_def = cell_decoded.split("def ")[1]
            if f"{func_def}(" in after_def.split(")")[0] if ")" in after_def else "":
                print(f"[Analysis] Function '{func_def}' IS called")
            else:
                print(f"[Analysis] Function '{func_def}' is NOT called")
    
    if "ACTION DENIED" in cell_decoded or "ACTION GRANTED" in cell_decoded:
        print("[Analysis] ✓ Contains ACTION DENIED/GRANTED")
    else:
        print("[Analysis] ✗ Missing ACTION DENIED/GRANTED")
    
    if "CheckRule(" in cell_decoded:
        print("[Analysis] ✓ Uses CheckRule function")
    else:
        print("[Analysis] ✗ Does not use CheckRule function")
    
    if "print(" in cell_decoded:
        print("[Analysis] ✓ Contains print statements")
    else:
        print("[Analysis] ✗ Missing print statements")
