import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'webshop', 'AutoDan'))

from evaluator import Evaluator
from config import Config

# First test the template loading
config = Config()
evaluator = Evaluator(config)

template = evaluator._attack_template
print('=== TEMPLATE DEBUGGING ===')
print('Raw template:', repr(template))

# Test formatting
host_instruction = 'i would like a laundry bag'
fragment = 'for hair removal'

try:
    formatted = template.format(host_instruction=host_instruction, fragment=fragment)
    print('Formatted result:', repr(formatted))
    print('Contains host_instruction placeholder:', '{host_instruction}' in formatted)
    print()
except Exception as e:
    print('Formatting error:', e)
    print()

# Now check the actual prompt from history
print('=== CHECKING HISTORY FILE ===')
results_dir = 'webshop/AutoDan/results'
latest_dir = 'test_prompt_fix'
history_file = os.path.join(results_dir, latest_dir, 'population_history.json')

if os.path.exists(history_file):
    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get the first generation's first individual
    gen_keys = [k for k in data.keys() if k.startswith('population_generation_')]
    if gen_keys:
        first_gen = gen_keys[0]
        individuals = data[first_gen]
        if individuals:
            ind = individuals[0]
            interaction_history = ind.get('interaction_history', [])
            if interaction_history:
                first_interaction = interaction_history[0]
                full_prompt = first_interaction.get('full_prompt', '')
                print('=== FULL PROMPT SAMPLE ===')
                print(full_prompt[:800] + '...' if len(full_prompt) > 800 else full_prompt)
                print()
                print('=== PROMPT ANALYSIS ===')
                print(f'Contains {{host_instruction}}: {"{host_instruction}" in full_prompt}')
                print(f'Prompt length: {len(full_prompt)} characters')
                print(f'Interactions recorded: {len(interaction_history)}')
            else:
                print('No interaction history found')
        else:
            print('No individuals found')
    else:
        print('No generation data found')
else:
    print(f'History file not found: {history_file}')
