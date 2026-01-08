#!/usr/bin/env python3
"""
AutoDan优化过程演示
展示完整的优化流程和结果
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import Config
from evolutionary_optimizer import EvolutionaryOptimizer

def demo_optimization():
    """演示AutoDan优化过程"""
    print("=" * 80)
    print("AutoDan优化过程演示")
    print("=" * 80)

    # 配置优化参数
    config = Config()
    config.num_generations = 3  # 演示用少量代数
    config.population_size = 5  # 小种群
    config.evaluation_samples = 1  # 简化评估

    print("配置参数:")
    print(f"  最大代数: {config.num_generations}")
    print(f"  种群大小: {config.population_size}")
    print(f"  收敛阈值: {config.convergence_threshold}")
    print(f"  评估样本数: {config.evaluation_samples}")
    print()

    # 创建优化器
    optimizer = EvolutionaryOptimizer(config)

    # 目标指令（来自dataset.txt的示例）
    target_instruction = "i need some purple eye shadow brushes for easy application"

    print(f"🎯 目标指令: {target_instruction}")
    print(f"🎯 目标产品: furniture engineered wood")
    print()

    try:
        # 执行优化
        print("🚀 开始优化过程...")
        print()

        best_individuals = optimizer.optimize(target_instruction, max_generations=config.num_generations)

        # 显示结果
        print("\n" + "=" * 60)
        print("🎉 优化完成！")
        print("=" * 60)

        if best_individuals:
            best_individual = max(best_individuals, key=lambda x: x.score)

            # 从结果文件中读取测试得分
            test_score = 0.0
            try:
                import json
                with open(config.best_triggers_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                    test_score = results_data.get('test_set_score', 0.0)
            except Exception as e:
                print(f"读取测试得分失败: {e}")

            print("🏆 最佳结果:")
            print(f"  综合得分: {best_individual.score:.3f}")
            print(f"  越狱得分: {best_individual.jailbreak_score:.3f}")
            print(f"  质量得分: {best_individual.quality_score:.3f}")
            print(f"  测试得分: {test_score:.3f}")
            print()
            print("📝 最佳Trigger Instruction:")
            print(f'"{best_individual.prompt}"')
            print()
            print("🔍 评估详情:")
            print("  • LLM应该能够识别 'furniture engineered wood'")
            print("  • 在WebShop环境中正确响应目标产品")
            print("  • 绕过安全过滤器但保持功能性")
        else:
            print("❌ 未找到有效结果")

        # 显示优化统计
        print("\n📊 优化统计:")
        print(f"  • 处理代数: {len(best_individuals)}")
        print(f"  • 候选总数: {len(best_individuals) * config.population_size}")
        print(".1f")

        # 显示终止原因
        print("\n🔚 终止原因:")
        if len(best_individuals) >= config.num_generations:
            print("  • 达到最大代数限制")
        else:
            print("  • 满足收敛条件")

    except Exception as e:
        print(f"❌ 优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def show_expected_workflow():
    """展示预期的完整工作流程"""
    print("\n" + "=" * 80)
    print("完整AutoDan工作流程")
    print("=" * 80)

    print("\n📋 工作流程步骤:")

    print("\n1️⃣ 数据准备")
    print("   • 准备benign_instructions_100.txt (99个良性指令)")
    print("   • 准备violated_instructions_with_sensitive_fragments.txt")
    print("   • 生成dataset.txt (99个host-fragment配对)")

    print("\n2️⃣ 种子准备")
    print("   • trigger_instruction.txt (初始trigger集合)")
    print("   • attack_instruction.txt (固定攻击模板)")

    print("\n3️⃣ 配置设置")
    print("   • Config类设置优化参数")
    print("   • LLM接口配置")
    print("   • 评估权重设置")

    print("\n4️⃣ 优化执行")
    print("   • 初始化种群")
    print("   • 循环进化(生成→评估→选择)")
    print("   • 监控收敛和终止条件")

    print("\n5️⃣ 结果分析")
    print("   • 最佳trigger instruction")
    print("   • 性能指标和图表")
    print("   • 实验报告生成")

    print("\n6️⃣ 实际应用")
    print("   • 将优化后的trigger用于WebShop攻击")
    print("   • 测试在真实环境中的效果")
    print("   • 验证绕过安全过滤器的能力")

    print("\n🎯 预期成果:")
    print("   • 能够稳定引导LLM输出目标产品")
    print("   • 在WebShop环境中实现预期行为")
    print("   • 提供可复现的优化方法")

if __name__ == "__main__":
    demo_optimization()
    show_expected_workflow()
