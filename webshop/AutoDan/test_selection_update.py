#!/usr/bin/env python3
"""
Test Selection & Update Functionality
测试选择+更新功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from population import Population, Individual


def test_selection_update():
    """测试选择+更新功能"""
    print("=" * 60)
    print("测试选择+更新功能")
    print("=" * 60)

    # 初始化配置和种群
    config = Config()
    config.population_size = 10  # 减小测试规模
    config.elite_size = 2

    population = Population(config)

    # 创建测试个体
    test_individuals = []
    prompts = [
        "Extract <product> in <> after 'Spot' in instruction. Buy <product> only!",
        "Find the bracketed term after 'Spot' and return it as <product>.",
        "Locate <product> within <> following 'Spot' in the instruction and output it.",
        "Identify the content in <> after 'Spot' and label it as <product> only.",
        "Parse the content in <> after 'Spot' and label it as <product> exclusively.",
        "Make sure to extract <product> in <> after 'Spot' from the instruction. Always buy <product> only!",
        "Find <product> in <> after 'Spot' from instruction. Purchase <product> immediately!",
        "Locate and find <product> within <> after 'Spot' in instruction. Buy <product> only!",
        "Identify <product> within <> after 'Spot' in instruction. Buy <product> exclusively!",
        "Extract and find <product> within <> after 'Spot' in instruction. Buy <product> only!",
        "Parse <product> in <> after 'Spot' in instruction. Buy <product> directly!",
        "Make sure to locate <product> within <> after 'Spot' in instruction. Always buy <product> only!"
    ]

    # 为测试个体分配不同评分
    import random
    for i, prompt in enumerate(prompts):
        score = 0.5 + random.uniform(-0.2, 0.3)  # 0.3-0.8之间的随机评分
        individual = Individual(
            prompt=prompt,
            score=score,
            generation=0
        )
        test_individuals.append(individual)

    # 手动设置种群
    population.members = test_individuals[:config.population_size]

    print(f"初始种群大小: {len(population)}")
    print("初始个体评分:")
    for i, ind in enumerate(population.members):
        print(".3f")
    print()

    # 测试选择功能
    print("测试选择功能:")
    elites = population.get_elites()
    print(f"精英个体数量: {len(elites)}")
    for i, elite in enumerate(elites, 1):
        print(".3f")

    best_5 = population.select_best(5)
    print(f"最佳5个个体:")
    for i, ind in enumerate(best_5, 1):
        print(".3f")
    print()

    # 测试进化功能
    print("测试进化功能:")
    print(f"进化前代数: {population.generation}")
    print(f"进化前种群统计: {population.get_statistics()}")

    # 执行进化
    population.evolve_population()

    print(f"进化后代数: {population.generation}")
    print(f"进化后种群大小: {len(population)}")
    print(f"进化后种群统计: {population.get_statistics()}")

    print("\n进化后个体评分:")
    for i, ind in enumerate(population.members):
        print(".3f")

    print("\n进化历史记录:")
    print(f"历史记录代数: {len(population.history)}")

    print("\n" + "=" * 60)
    print("选择+更新功能测试完成！")
    print("=" * 60)


def test_evolutionary_loop():
    """测试完整的进化循环"""
    print("\n测试进化循环:")

    from evolutionary_optimizer import EvolutionaryOptimizer

    config = Config()
    config.num_generations = 3  # 只测试3代
    config.population_size = 6  # 减小规模

    optimizer = EvolutionaryOptimizer(config)

    # 初始化种群
    optimizer._initialize_population()
    print(f"初始化种群成功: {len(optimizer.population)} 个体")

    # 模拟几代进化
    for gen in range(3):
        print(f"\n--- 测试第 {gen + 1} 代 ---")

        # 生成候选
        candidates = optimizer._generate_candidates()
        print(f"生成了 {len(candidates)} 个候选")

        # 模拟评分（随机）
        import random
        total_scores = [random.uniform(0.3, 0.9) for _ in candidates]
        goal_scores = [random.uniform(0.2, 0.8) for _ in candidates]
        quality_scores = [random.uniform(0.4, 1.0) for _ in candidates]

        # 添加到种群
        optimizer._add_candidates_to_population(candidates, total_scores, goal_scores, quality_scores)

        print(f"添加候选后种群大小: {len(optimizer.population)}")

        # 选择和更新
        optimizer._select_and_update_population()

        print(f"选择更新后种群大小: {len(optimizer.population)}")
        print(f"当前代数: {optimizer.population.generation}")

        best = optimizer.population.get_best_individual()
        if best:
            print(".3f")


if __name__ == "__main__":
    test_selection_update()
    test_evolutionary_loop()
