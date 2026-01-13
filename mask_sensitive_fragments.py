#!/usr/bin/env python3
"""
处理violated_instructions_with_sensitive_fragments.txt文件
为每个pair添加Masked Instruction，其中Sensitive Fragments被[MASK]替换
"""

import re
import ast

def parse_sensitive_fragments(file_path: str) -> list:
    """解析文件，提取所有pairs的信息"""
    pairs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按Pair ID分割
    pair_pattern = r'(\d+)\. Pair ID: (\d+)\s*\n\s*Instruction: ([^\n]+)\s*\n\s*Sensitive Fragments: (\[[^\]]+\])'
    matches = re.findall(pair_pattern, content, re.MULTILINE)

    for match in matches:
        pair_num, pair_id, instruction, fragments_str = match

        try:
            # 解析fragments列表
            fragments = ast.literal_eval(fragments_str)

            pairs.append({
                'pair_num': pair_num,
                'pair_id': pair_id,
                'instruction': instruction.strip(),
                'sensitive_fragments': fragments
            })

        except (ValueError, SyntaxError) as e:
            print(f"解析Pair {pair_num}失败: {e}")
            continue

    return pairs

def create_masked_instruction(instruction: str, sensitive_fragments: list) -> str:
    """创建masked instruction，将sensitive fragments替换为[MASK]"""
    masked_instruction = instruction

    # 按照fragments的长度排序（长的先替换，避免短的干扰长的）
    sorted_fragments = sorted(sensitive_fragments, key=len, reverse=True)

    for fragment in sorted_fragments:
        # 移除fragment前后的引号（如果有的话）
        clean_fragment = fragment.strip('"\'')

        # 替换为[MASK]
        masked_instruction = masked_instruction.replace(clean_fragment, '[MASK]')

    return masked_instruction

def process_file(input_file: str, output_file: str = None):
    """处理文件并生成新格式"""
    if output_file is None:
        output_file = input_file.replace('.txt', '_masked.txt')

    # 解析pairs
    pairs = parse_sensitive_fragments(input_file)
    print(f"解析到 {len(pairs)} 个pairs")

    # 处理并写入
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("100个有明确sensitive fragments的violated instruction案例 (含Masked Instruction)\n")
        f.write("=" * 90 + "\n")

        for pair in pairs:
            f.write(f"--------------------------------------------------\n\n")
            f.write(f"{pair['pair_num']}. Pair ID: {pair['pair_id']}\n")
            f.write(f"   Instruction: {pair['instruction']}\n")
            f.write(f"   Sensitive Fragments: {pair['sensitive_fragments']}\n")

            # 生成masked instruction
            masked_instruction = create_masked_instruction(
                pair['instruction'],
                pair['sensitive_fragments']
            )
            f.write(f"   Masked Instruction: {masked_instruction}\n\n")

    print(f"处理完成，结果保存到: {output_file}")

    # 显示前3个示例
    print("\n前3个处理示例:")
    print("-" * 50)
    for i, pair in enumerate(pairs[:3]):
        masked = create_masked_instruction(pair['instruction'], pair['sensitive_fragments'])
        print(f"{i+1}. 原始: {pair['instruction']}")
        print(f"   敏感片段: {pair['sensitive_fragments']}")
        print(f"   掩码后: {masked}")
        print()

if __name__ == "__main__":
    input_file = "webshop/AutoDan/data/violated_instructions_with_sensitive_fragments.txt"
    process_file(input_file)
