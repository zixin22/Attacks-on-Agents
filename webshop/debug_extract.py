log_file = 'batch_attack/batch_attack_gpt5_1/attackplan_webshoplog.txt'

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 找到 id_18_fix_68 的完整块
import re
pattern = r'ATTACK PLAN FOR CASE id_18_fix_68.*?(?=ATTACK PLAN FOR CASE|\Z)'
match = re.search(pattern, content, re.DOTALL)

if match:
    block = match.group(0)
    lines = block.split('\n')
    
    # 找到 Session ID: id_18_fix_68 出现在 FRAGMENT ATTACK header 中的位置
    for i, line in enumerate(lines):
        if 'Session ID: id_18_fix_68' in line:
            print(f'找到 Session ID 在行 {i}: {line.strip()}')
            
            # 找到header开始：向上查找第一个 '=' 行
            header_start = None
            for j in range(i-1, max(0, i-20), -1):
                if '=' in lines[j] and '---' not in lines[j]:
                    header_start = j
                    break
            
            na = "N/A"
            print(f'Header 起始行: {header_start}')
            if header_start is not None:
                print(f'Header 起始行内容: {lines[header_start].strip()}')
            else:
                print(f'Header 起始行内容: {na}')
            
            # 找到header结束：向下查找第一个 '-' 行
            header_end = None
            for j in range(i, min(len(lines), i+20)):
                if '-' in lines[j] and '=' not in lines[j]:
                    header_end = j
                    break
            
            print(f'Header 结束行: {header_end}')
            if header_end is not None:
                print(f'Header 结束行内容: {lines[header_end].strip()}')
            else:
                print(f'Header 结束行内容: {na}')
            
            if header_start is not None and header_end is not None:
                header_text = '\n'.join(lines[header_start:header_end])
                print()
                print('提取的 header 文本:')
                print('=' * 60)
                print(header_text)
                print('=' * 60)
                
                # 检查 Reward
                reward_match = re.search(r'Reward:\s*([0-9.]+)', header_text)
                if reward_match:
                    print(f'找到 Reward: {reward_match.group(1)}')
                else:
                    print('未找到 Reward')









