import json

with open('promptarmor_agentgharm.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计每个group中is_injected为true的数量
group_true_counts = {}
group_total_counts = {}

for result in data['results']:
    group = result['group']
    is_injected = result['is_injected']
    
    if group not in group_total_counts:
        group_total_counts[group] = 0
        group_true_counts[group] = 0
    
    group_total_counts[group] += 1
    if is_injected:
        group_true_counts[group] += 1

print('=' * 60)
print('PromptArmor 检测结果统计')
print('=' * 60)
print('总检测数量: {}'.format(sum(group_total_counts.values())))
print()

for group in sorted(group_total_counts.keys()):
    true_count = group_true_counts[group]
    total = group_total_counts[group]
    false_count = total - true_count
    true_rate = (true_count / total * 100) if total > 0 else 0
    
    print('Group: {}'.format(group))
    print('  - is_injected = true:  {}/{} ({:.1f}%)'.format(true_count, total, true_rate))
    print('  - is_injected = false: {}/{} ({:.1f}%)'.format(false_count, total, 100-true_rate))
    print()









