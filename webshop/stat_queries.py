import json
from pathlib import Path
from collections import Counter

base_dir = Path(r'data/normal_output_0_12000')
queries = set()
query_count = Counter()

print("Scanning memory_1.json files...")
for mem_file in base_dir.rglob('memory_1.json'):
    try:
        print(f'Reading {mem_file}...')
        with open(mem_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            query = item.get('Query', '')
            if query:
                queries.add(query)
                query_count[query] += 1
    except Exception as e:
        print(f'Error reading {mem_file}: {e}')

print(f'\n{"="*60}')
print(f'Total unique queries: {len(queries)}')
print(f'{"="*60}\n')

print("All unique queries (sorted alphabetically):")
print("-" * 60)
for query in sorted(queries):
    count = query_count[query]
    print(f"{query} (appears {count} times)")

