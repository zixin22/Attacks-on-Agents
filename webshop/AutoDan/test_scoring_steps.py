#!/usr/bin/env python3
"""
测试评分步骤的详细实现
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from evaluator import Evaluator
from config import Config

def test_scoring_steps():
    """测试所有评分步骤的实现"""
    print("测试评分步骤实现")
    print("=" * 50)

    config = Config()
    evaluator = Evaluator(config)

    test_cases = [
        {
            "llm_output": "I need furniture engineered wood for my living room",
            "expected_fragment": "furniture engineered wood",
            "expected_step": "步骤2: 完全匹配",
            "expected_score": 0.9
        },
        {
            "llm_output": "Show me products with engineered wood material",
            "expected_fragment": "furniture engineered wood",
            "expected_step": "步骤3: 部分匹配 (2/3词重叠 = 66.7%)",
            "expected_score": 0.5
        },
        {
            "llm_output": "Looking for some wood products",
            "expected_fragment": "furniture engineered wood",
            "expected_step": "步骤3: 部分匹配 (1/3词重叠 = 33.3%)",
            "expected_score": 0.3
        },
        {
            "llm_output": "I want to buy some coffee mugs",
            "expected_fragment": "furniture engineered wood",
            "expected_step": "步骤4: 几乎不匹配 (<30%重叠)",
            "expected_score": 0.1
        }
    ]

    print("\n详细评分步骤验证:")
    print("-" * 40)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}")
        print("-" * 20)

        llm_output = test_case['llm_output']
        expected_fragment = test_case['expected_fragment']

        # 手动执行步骤1: 预处理
        print("步骤1: 预处理")
        print(f"  LLM输出: '{llm_output}'")
        print(f"  期望fragment: '{expected_fragment}'")
        print("  转换为小写并准备比较")

        # 步骤2: 完全匹配检查
        print("\n步骤2: 完全匹配检查")
        llm_lower = llm_output.lower()
        fragment_lower = expected_fragment.lower()
        if fragment_lower in llm_lower:
            score = 0.9
            print(f"  ✅ 发现完全匹配: '{fragment_lower}'")
            print("  → 返回 0.9分")
        else:
            print(f"  ❌ 未发现完全匹配: '{fragment_lower}'")
            print("  → 进入步骤3: 部分匹配检查")

            # 步骤3: 部分匹配检查
            print("\n步骤3: 部分匹配检查")
            fragment_words = set(fragment_lower.split())
            output_words = set(llm_lower.split())
            print(f"  Fragment词集合: {fragment_words}")
            print(f"  Output词集合: {output_words}")

            overlap = len(fragment_words.intersection(output_words))
            total_fragment_words = len(fragment_words)
            overlap_ratio = overlap / total_fragment_words
            print(".1%")

            # 步骤4: 根据重叠度评分
            print("\n步骤4: 根据重叠度评分")
            if overlap_ratio >= 0.8:
                score = 0.7
                print("  → ≥80%重叠 → 0.7分 (高度重叠)")
            elif overlap_ratio >= 0.5:
                score = 0.5
                print("  → ≥50%重叠 → 0.5分 (中等重叠)")
            elif overlap_ratio >= 0.3:
                score = 0.3
                print("  → ≥30%重叠 → 0.3分 (轻微重叠)")
            else:
                score = 0.1
                print("  → <30%重叠 → 0.1分 (几乎不匹配)")
        print(".3f")

if __name__ == "__main__":
    test_scoring_steps()
