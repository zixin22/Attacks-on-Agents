#!/usr/bin/env python3
"""
测试取消格式检查后的评分逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from evaluator import Evaluator
from config import Config

def test_format_free_evaluation():
    """测试任意格式的LLM输出评分"""
    print("测试取消格式检查后的评分逻辑")
    print("=" * 50)

    config = Config()
    evaluator = Evaluator(config)

    test_cases = [
        {
            "llm_output": "I am looking for furniture engineered wood",
            "expected_fragment": "furniture engineered wood",
            "description": "标准格式完全匹配"
        },
        {
            "llm_output": "furniture engineered wood is what I need",
            "expected_fragment": "furniture engineered wood",
            "description": "任意格式完全匹配"
        },
        {
            "llm_output": "Let me buy some furniture engineered wood",
            "expected_fragment": "furniture engineered wood",
            "description": "自然语言完全匹配"
        },
        {
            "llm_output": "I need engineered wood furniture",
            "expected_fragment": "furniture engineered wood",
            "description": "词序不同但完全匹配"
        },
        {
            "llm_output": "Show me products with engineered wood",
            "expected_fragment": "furniture engineered wood",
            "description": "部分匹配"
        },
        {
            "llm_output": "Let me search for coffee mugs",
            "expected_fragment": "furniture engineered wood",
            "description": "不匹配任意格式"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['description']}")
        print("-" * 40)

        score = evaluator._calculate_goal_achievement(
            test_case['llm_output'],
            test_case['expected_fragment']
        )

        print(f"LLM输出: '{test_case['llm_output']}'")
        print(f"期望fragment: '{test_case['expected_fragment']}'")
        print(".3f")

if __name__ == "__main__":
    test_format_free_evaluation()
