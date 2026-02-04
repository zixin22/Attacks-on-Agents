log_file = 'batch_attack/batch_attack_gpt5_1/attackplan_webshoplog.txt'

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

import re

# 找到 id_18_fix_68 的完整块
pattern = r'ATTACK PLAN FOR CASE id_18_fix_68.*?(?=ATTACK PLAN FOR CASE|\Z)'
match = re.search(pattern, content, re.DOTALL)

if match:
    block = match.group(0)
    lines = block.split('\n')
    
    print("id_18_fix_68 的完整日志块:")
    print("=" * 80)
    
    # 打印所有行
    for i, line in enumerate(lines):
        if 'Session ID' in line or 'Reward' in line or '====' in line or 'FRAGMENT' in line or 'TRIGGER' in line or 'INCOMPLETE' in line or 'FAILED' in line or 'Status' in line:
            print(f"{i:3d}: {line}")
    
    print("=" * 80)
    
    # 找到 FRAGMENT A ATTACK - INCOMPLETE 中的 Session ID
    for i, line in enumerate(lines):
        if 'Session ID: id_18_fix_68' in line:
            # 检查上下文
            context_start = max(0, i - 5)
            context_end = min(len(lines), i + 20)
            
            print(f"\n找到 Session ID 在行 {i}")
            print("上下文 (行 {context_start} - {context_end}):")
            for j in range(context_start, context_end):
                marker = " >>> " if j == i else "    "
                print(f"{marker}{j:3d}: {lines[j]}")
            
            # 模拟 _extract_session_header_precise
            session_start = content.find(line, content.find(block))
            if session_start != -1:
                print(f"\n模拟 _extract_session_header_precise:")
                
                # 找到header开始
                header_start_search = content.rfind("====", content.find(block), session_start)
                if header_start_search != -1:
                    full_header_start = content.rfind("\n", content.find(block), header_start_search)
                    if full_header_start != -1:
                        header_start = full_header_start + 1
                    else:
                        header_start = header_start_search
                else:
                    header_start = session_start
                
                print(f"  header_start_search: {header_start_search}")
                print(f"  full_header_start: {full_header_start}")
                print(f"  header_start: {header_start}")
                print(f"  header_start 内容: {content[header_start:header_start+100]}...")
                
                # 找到header结束
                dash_end = content.find("----", session_start)
                eq_end = content.find("====", session_start)
                
                header_end_candidates = []
                if dash_end != -1:
                    header_end_candidates.append(dash_end)
                if eq_end != -1:
                    header_end_candidates.append(eq_end)
                
                if header_end_candidates:
                    header_end = min(header_end_candidates)
                    
                    # 检查分隔符后的行
                    extended_end = content.find("\n", header_end)
                    if extended_end != -1:
                        next_line_end = content.find("\n", extended_end + 1)
                        if next_line_end != -1:
                            next_line = content[extended_end:next_line_end].strip()
                            if next_line.startswith("Reward:") or next_line.startswith("Status:"):
                                header_end = next_line_end
                    
                    print(f"  header_end: {header_end}")
                    print(f"  header_end 内容: {content[header_end:header_end+100]}...")
                    
                    header_text = content[header_start:header_end].strip()
                    print(f"\n提取的 header_text:")
                    print("-" * 60)
                    print(header_text)
                    print("-" * 60)
                    
                    # 检查 Reward
                    reward_match = re.search(r'Reward:\s*([0-9.]+)', header_text)
                    if reward_match:
                        print(f"\n找到 Reward: {reward_match.group(1)}")
                    else:
                        print(f"\n未找到 Reward!")



