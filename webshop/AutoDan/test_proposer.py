#!/usr/bin/env python3
"""
Test Proposer Enhanced Functionality
测试增强后的Proposer功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from proposer import Proposer


def test_enhanced_proposer():
    """测试增强后的Proposer功能"""
    print("=" * 60)
    print("测试增强后的Proposer功能")
    print("=" * 60)

    # 初始化配置和Proposer
    config = Config()
    proposer = Proposer(config)

    # 测试数据
    test_population = [
        "Extract <product> in <> after 'Spot' in instruction. Buy <product> only!",
        "Find the bracketed term after 'Spot' and return it as <product>.",
        "Locate <product> within <> following 'Spot' in the instruction and output it."
    ]

    print(f"初始种群大小: {len(test_population)}")
    print("初始prompts:")
    for i, prompt in enumerate(test_population, 1):
        print(f"  {i}. {prompt}")
    print()

    # 测试生成候选
    print("正在生成候选prompts...")
    candidates = proposer.generate_candidates(test_population)

    print(f"生成了 {len(candidates)} 个候选prompts")
    print("\n前10个候选prompts:")
    for i, candidate in enumerate(candidates[:10], 1):
        print(f"  {i}. {candidate}")
    print()

    # 测试过滤
    filtered = proposer._filter_candidates(candidates)
    print(f"过滤后剩余 {len(filtered)} 个有效prompts")

    # 测试不同类型的变异
    print("\n测试不同变异类型:")
    test_prompt = test_population[0]

    print(f"原始prompt: {test_prompt}")

    # 测试各种变异
    mutations = [
        ("替换变异", proposer._mutate_replace),
        ("插入变异", proposer._mutate_insert),
        ("删除变异", proposer._mutate_delete),
        ("释义变异", proposer._mutate_paraphrase),
        ("扩展变异", proposer._mutate_expand),
        ("同义变异", proposer._mutate_synonym),
    ]

    for name, mutation_func in mutations:
        try:
            result = mutation_func(test_prompt)
            changed = "✓" if result != test_prompt else "✗"
            print(f"  {name}: {changed} {result}")
        except Exception as e:
            print(f"  {name}: ✗ 错误: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


def test_llm_features():
    """测试LLM相关功能"""
    print("\n测试LLM相关功能:")

    config = Config()
    proposer = Proposer(config)

    test_prompt = "Extract <product> in <> after 'Spot' in instruction. Buy <product> only!"

    print(f"测试prompt: {test_prompt}")

    # 测试LLM改写
    try:
        rewrites = proposer.llm_rewrite(test_prompt, 2)
        print(f"✓ LLM改写: 生成了 {len(rewrites)} 个变体")
        for i, rewrite in enumerate(rewrites[:2], 1):
            print(f"    {i}. {rewrite}")
    except Exception as e:
        print(f"✗ LLM改写失败: {e}")

    # 测试同义改写
    try:
        synonyms = proposer.llm_synonym_rewrite(test_prompt, 2)
        print(f"✓ 同义改写: 生成了 {len(synonyms)} 个变体")
        for i, synonym in enumerate(synonyms[:2], 1):
            print(f"    {i}. {synonym}")
    except Exception as e:
        print(f"✗ 同义改写失败: {e}")

    # 测试扩展改写
    try:
        expands = proposer.llm_expand_rewrite(test_prompt, 2)
        print(f"✓ 扩展改写: 生成了 {len(expands)} 个变体")
        for i, expand in enumerate(expands[:2], 1):
            print(f"    {i}. {expand}")
    except Exception as e:
        print(f"✗ 扩展改写失败: {e}")


if __name__ == "__main__":
    test_enhanced_proposer()
    test_llm_features()
