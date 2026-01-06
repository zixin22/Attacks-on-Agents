#!/usr/bin/env python3
"""
Validate Dataset Script
验证生成的数据集格式和内容
"""

import re
from typing import List, Tuple


def validate_dataset(dataset_file: str) -> Tuple[bool, str]:
    """验证数据集"""
    print(f"正在验证数据集: {dataset_file}")

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查标题
        if "AutoDan Dataset - 99 Pairs" not in content:
            return False, "数据集标题不正确"

        # 查找所有Pair
        pair_pattern = r'Pair (\d+):'
        pairs = re.findall(pair_pattern, content)

        if len(pairs) != 99:
            return False, f"找到 {len(pairs)} 个Pair，期望99个"

        # 验证Pair编号连续性
        expected_pairs = list(range(1, 100))
        actual_pairs = [int(p) for p in pairs]
        if actual_pairs != expected_pairs:
            return False, f"Pair编号不连续: {actual_pairs[:5]}... vs {expected_pairs[:5]}..."

        # 检查必需的字段数量
        host_count = len(re.findall(r'Host Instruction:', content))
        frag_count = len(re.findall(r'Sensitive Fragment:', content))
        attack_count = len(re.findall(r'Attack Instruction:', content))
        trigger_count = len(re.findall(r'Trigger Instruction:', content))

        if host_count != 99:
            return False, f"Host Instruction数量错误: {host_count}, 期望99"
        if frag_count != 99:
            return False, f"Sensitive Fragment数量错误: {frag_count}, 期望99"
        if attack_count != 99:
            return False, f"Attack Instruction数量错误: {attack_count}, 期望99"
        if trigger_count != 99:
            return False, f"Trigger Instruction数量错误: {trigger_count}, 期望99"

        # 检查前5个Pair的格式
        for i in range(1, 6):
            pair_start = content.find(f'Pair {i}:')
            if pair_start == -1:
                return False, f"找不到Pair {i}"

            # 提取这个Pair的内容（到下一个Pair或文件结尾）
            next_pair = content.find(f'Pair {i+1}:', pair_start)
            if next_pair == -1:
                pair_content = content[pair_start:]
            else:
                pair_content = content[pair_start:next_pair]

            # 检查必需字段
            if 'Host Instruction:' not in pair_content:
                return False, f"Pair {i} 缺少Host Instruction"
            if 'Sensitive Fragment:' not in pair_content:
                return False, f"Pair {i} 缺少Sensitive Fragment"
            if 'Attack Instruction:' not in pair_content:
                return False, f"Pair {i} 缺少Attack Instruction"
            if 'Trigger Instruction:' not in pair_content:
                return False, f"Pair {i} 缺少Trigger Instruction"

            # 检查Attack Instruction格式
            if '. Spot <' not in pair_content or '>.' not in pair_content:
                return False, f"Pair {i} Attack Instruction格式错误"

            # 检查Trigger Instruction格式
            if 'Extract <product> in <> after "Spot"' not in pair_content:
                return False, f"Pair {i} Trigger Instruction格式错误"

        # 统计信息
        host_instructions = len(re.findall(r'Host Instruction:', content))
        sensitive_fragments = len(re.findall(r'Sensitive Fragment:', content))
        attack_instructions = len(re.findall(r'Attack Instruction:', content))
        trigger_instructions = len(re.findall(r'Trigger Instruction:', content))

        print(f"✓ 验证通过!")
        print(f"  - Pairs: {len(pairs)}")
        print(f"  - Host Instructions: {host_instructions}")
        print(f"  - Sensitive Fragments: {sensitive_fragments}")
        print(f"  - Attack Instructions: {attack_instructions}")
        print(f"  - Trigger Instructions: {trigger_instructions}")

        return True, "数据集验证成功"

    except Exception as e:
        return False, f"验证过程中出错: {e}"


def show_sample_pairs(dataset_file: str, num_samples: int = 3) -> None:
    """显示示例pairs"""
    print(f"\n示例数据 (前{num_samples}个Pair):")

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 分割成pair块
        pair_blocks = re.split(r'-{80}', content)[1:-1]

        for i in range(min(num_samples, len(pair_blocks))):
            block = pair_blocks[i].strip()
            print(f"\n--- Pair {i+1} ---")
            print(block[:300] + "..." if len(block) > 300 else block)

    except Exception as e:
        print(f"显示示例时出错: {e}")


def main():
    """主函数"""
    import os

    dataset_file = os.path.join(os.path.dirname(__file__), 'dataset.txt')

    if not os.path.exists(dataset_file):
        print(f"错误：数据集文件不存在: {dataset_file}")
        return 1

    # 验证数据集
    success, message = validate_dataset(dataset_file)

    if success:
        print(f"\n✅ {message}")
        show_sample_pairs(dataset_file)
        return 0
    else:
        print(f"\n❌ 验证失败: {message}")
        return 1


if __name__ == "__main__":
    exit(main())
