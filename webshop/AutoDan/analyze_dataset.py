#!/usr/bin/env python3
"""
分析dataset.txt数据集，评估其适合度作为训练/验证/测试集
"""

import os
import re
from collections import Counter
from typing import List, Dict, Set

def analyze_dataset(file_path: str):
    """分析数据集的结构和质量"""
    print("AutoDan Dataset 分析报告")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"错误: 数据集文件不存在: {file_path}")
        return

    # 读取数据集
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 基本统计
    lines = content.split('\n')
    total_lines = len(lines)
    print(f"总行数: {total_lines}")

    # 解析数据对
    pairs = parse_pairs(content)
    print(f"数据对数量: {len(pairs)}")

    if len(pairs) == 0:
        print("错误: 未找到任何数据对")
        return

    # 分析各个字段
    analyze_fields(pairs)

    # 评估数据集质量
    evaluate_dataset_quality(pairs)

def parse_pairs(content: str) -> List[Dict]:
    """解析数据集中的所有pair"""
    pairs = []

    # 分割每个pair块
    pair_blocks = re.split(r'Pair \d+:', content)[1:]  # 跳过文件头

    for i, block in enumerate(pair_blocks, 1):
        pair = {}

        # 提取Host Instruction
        host_match = re.search(r'Host Instruction:\s*(.+?)(?=\n\s*Sensitive Fragment:|$)', block, re.DOTALL)
        if host_match:
            pair['host_instruction'] = host_match.group(1).strip()

        # 提取Sensitive Fragment
        fragment_match = re.search(r'Sensitive Fragment:\s*(.+?)(?=\n\s*Attack Instruction:|$)', block, re.DOTALL)
        if fragment_match:
            pair['sensitive_fragment'] = fragment_match.group(1).strip()

        # 提取Attack Instruction
        attack_match = re.search(r'Attack Instruction:\s*(.+?)(?=\n\s*Trigger Instruction:|$)', block, re.DOTALL)
        if attack_match:
            pair['attack_instruction'] = attack_match.group(1).strip()

        # 提取Trigger Instruction
        trigger_match = re.search(r'Trigger Instruction:\s*(.+?)(?=---|$)', block, re.DOTALL)
        if trigger_match:
            pair['trigger_instruction'] = trigger_match.group(1).strip()

        if all(key in pair for key in ['host_instruction', 'sensitive_fragment', 'attack_instruction', 'trigger_instruction']):
            pair['pair_id'] = i
            pairs.append(pair)

    return pairs

def analyze_fields(pairs: List[Dict]):
    """分析各个字段的统计信息"""
    print("\n" + "=" * 40)
    print("字段分析")
    print("=" * 40)

    # Host Instructions
    host_lengths = [len(p['host_instruction'].split()) for p in pairs]
    print(f"Host Instructions: {len(pairs)} 个")
    print(f"  平均词数: {sum(host_lengths)/len(host_lengths):.1f}")
    print(f"  最短: {min(host_lengths)} 词")
    print(f"  最长: {max(host_lengths)} 词")

    # Sensitive Fragments
    fragment_lengths = [len(p['sensitive_fragment'].split()) for p in pairs]
    fragments = [p['sensitive_fragment'] for p in pairs]
    unique_fragments = len(set(fragments))
    print(f"\nSensitive Fragments: {len(fragments)} 个")
    print(f"  唯一值: {unique_fragments} 个 ({unique_fragments/len(fragments)*100:.1f}%)")
    print(f"  平均词数: {sum(fragment_lengths)/len(fragment_lengths):.1f}")
    print(f"  最短: {min(fragment_lengths)} 词")
    print(f"  最长: {max(fragment_lengths)} 词")

    # 显示最常见的fragments
    fragment_counts = Counter(fragments)
    print("\n  最常见的fragments:")
    for fragment, count in fragment_counts.most_common(10):
        print(f"    '{fragment}': {count} 次")

    # Attack Instructions
    attack_lengths = [len(p['attack_instruction'].split()) for p in pairs]
    print(f"\nAttack Instructions: {len(pairs)} 个")
    print(f"  平均词数: {sum(attack_lengths)/len(attack_lengths):.1f}")

    # Trigger Instructions
    trigger_lengths = [len(p['trigger_instruction'].split()) for p in pairs]
    print(f"\nTrigger Instructions: {len(pairs)} 个")
    print(f"  平均词数: {sum(trigger_lengths)/len(trigger_lengths):.1f}")

def evaluate_dataset_quality(pairs: List[Dict]):
    """评估数据集质量和适用性"""
    print("\n" + "=" * 40)
    print("数据集质量评估")
    print("=" * 40)

    # 检查数据完整性
    complete_pairs = len([p for p in pairs if all(k in p for k in ['host_instruction', 'sensitive_fragment', 'attack_instruction', 'trigger_instruction'])])
    print(f"完整数据对: {complete_pairs}/{len(pairs)} ({complete_pairs/len(pairs)*100:.1f}%)")

    # 多样性分析
    fragments = [p['sensitive_fragment'] for p in pairs]
    unique_fragments = len(set(fragments))

    print("\n多样性评估:")
    if unique_fragments >= len(pairs) * 0.8:
        print("  ✅ 高度多样: 几乎每个pair都有独特的fragment")
    elif unique_fragments >= len(pairs) * 0.5:
        print("  ⚠️ 中等多样: 一半以上的fragments是独特的")
    else:
        print("  ❌ 低多样: 存在大量重复的fragments")

    # 检查fragment质量
    print("\nFragment质量检查:")
    short_fragments = [f for f in fragments if len(f.split()) < 2]
    long_fragments = [f for f in fragments if len(f.split()) > 8]

    if short_fragments:
        print(f"  ⚠️ 过短fragments ({len(short_fragments)}个): {short_fragments[:3]}...")

    if long_fragments:
        print(f"  ⚠️ 过长fragments ({len(long_fragments)}个): {long_fragments[0][:50]}...")

    # 适用性评估
    print("\n训练/测试适用性:")
    if len(pairs) >= 50:
        print("  ✅ 数据量充足: ≥50个样本")
    else:
        print("  ❌ 数据量不足: <50个样本")

    if unique_fragments >= 20:
        print("  ✅ Fragment多样性良好: ≥20个独特值")
    else:
        print("  ⚠️ Fragment多样性不足: <20个独特值")

    # 分割建议
    print("\n数据集分割建议:")
    if len(pairs) >= 60:
        train_size = int(len(pairs) * 0.7)
        val_size = int(len(pairs) * 0.15)
        test_size = len(pairs) - train_size - val_size
        print(f"  建议分割: 训练{train_size} | 验证{val_size} | 测试{test_size}")
    else:
        print("  建议: 仅用于测试或小规模验证")

    print("\n总体评估:")
    if len(pairs) >= 50 and unique_fragments >= len(pairs) * 0.6:
        print("  ✅ 适合作为训练/验证/测试数据集")
    elif len(pairs) >= 20:
        print("  ⚠️ 基本适合，但建议扩充数据")
    else:
        print("  ❌ 不适合，需要更多数据")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.txt')
    analyze_dataset(dataset_path)
