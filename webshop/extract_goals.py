import json
import os

# File paths
merged_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\selected_reward_0.5\merged_reward_ge_05.json"
human_goals_file = r"C:\Users\22749\Desktop\rap-main\WebShop-master\baseline_models\data\human_goals.json"
output_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\selected_reward_0.5\extracted_goals.json"

# Read merged_reward_ge_05.json
print(f"Reading {merged_file}...")
with open(merged_file, 'r', encoding='utf-8') as f:
    merged_data = json.load(f)

fixed_numbers = merged_data.get('fixed_numbers', [])
print(f"Found {len(fixed_numbers)} fixed numbers")

# Read human_goals.json
print(f"Reading {human_goals_file}...")
with open(human_goals_file, 'r', encoding='utf-8') as f:
    human_goals = json.load(f)

print(f"Total goals in human_goals.json: {len(human_goals)}")

# Create mapping: fixed_number -> goal
# fixed_number + 2 = line number in human_goals.json
# But since human_goals.json is a JSON array, index = fixed_number
# However, user says "fixed_number + 2 = line number", so we need to read by line
# Actually, if it's a JSON array, index 0 is the first element
# But user wants line numbers, so let's read the file line by line

print("Reading human_goals.json line by line...")
with open(human_goals_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in human_goals.json: {len(lines)}")

# Extract goals based on fixed_numbers
# fixed_number + 2 = line number in human_goals.json
# Since human_goals.json is a JSON array, index = fixed_number
# But user wants to use line numbers, so we'll use both methods

result = {}
missing_indices = []

for fixed_num in fixed_numbers:
    # Method 1: Use JSON array index directly (most reliable)
    if fixed_num < len(human_goals):
        goal = human_goals[fixed_num]
        result[str(fixed_num)] = goal
    else:
        # Method 2: Try to get from line number (fixed_num + 2)
        line_num = fixed_num + 2
        if line_num <= len(lines):
            line_content = lines[line_num - 1].strip()
            # Clean up the line content
            if line_content.endswith(','):
                line_content = line_content[:-1]
            if line_content.startswith('"') and line_content.endswith('"'):
                line_content = line_content[1:-1]
            result[str(fixed_num)] = line_content
        else:
            missing_indices.append(fixed_num)
            print(f"Warning: fixed_num {fixed_num} (line {line_num}) not found")

if missing_indices:
    print(f"\nWarning: {len(missing_indices)} fixed_numbers could not be found:")
    print(missing_indices[:10])  # Print first 10

# Save result
print(f"\nSaving results to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Successfully extracted {len(result)} goals")
print(f"Output saved to: {output_file}")

# Print some examples
print("\nFirst 5 examples:")
for i, (key, value) in enumerate(list(result.items())[:5]):
    print(f"  fixed_number {key}: {value[:80]}...")

