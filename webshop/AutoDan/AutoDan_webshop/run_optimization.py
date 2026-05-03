#!/usr/bin/env python3
"""
AutoDan Evolutionary Optimization Runner
主执行脚本：运行AutoDan进化优化
"""

import os
import sys
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from evolutionary_optimizer import EvolutionaryOptimizer
from utils import print_optimization_summary, plot_optimization_progress, generate_experiment_report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AutoDan Evolutionary Optimization')
    parser.add_argument('--max-generations', '-g', type=int, default=None,
                       help='最大进化代数（覆盖配置文件）')
    parser.add_argument('--population-size', '-p', type=int, default=None,
                       help='种群大小（覆盖配置文件）')
    parser.add_argument('--experiment-name', '-n', type=str, default='default_experiment',
                       help='实验名称')
    parser.add_argument('--config-file', '-c', type=str, default=None,
                       help='自定义配置文件路径')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成图表')

    args = parser.parse_args()

    print("=" * 80)
    print("AutoDan 进化优化系统")
    print("=" * 80)
    print(f"实验名称: {args.experiment_name}")
    print()

    try:
        # 1. 加载配置
        config = Config()

        # 从自定义配置文件加载（如果指定）
        if args.config_file and os.path.exists(args.config_file):
            config.load_from_file(args.config_file)
            print(f"已从 {args.config_file} 加载配置")

        # 覆盖命令行参数
        if args.max_generations is not None:
            config.num_generations = args.max_generations
        if args.population_size is not None:
            config.population_size = args.population_size

        # 设置实验目录 - 总是使用optimization_i目录
        if args.experiment_name != 'default_experiment':
            from utils import setup_experiment_directory
            experiment_dir = setup_experiment_directory(config.base_dir, args.experiment_name)
            config.results_dir = os.path.join(experiment_dir, 'results')
        else:
            # 对于默认实验，使用Config类已创建的optimization_i目录
            config.results_dir = config.experiment_dir

        # 更新所有文件路径到results_dir
        config.best_triggers_file = os.path.join(config.results_dir, 'best_triggers.json')
        config.optimization_log_file = os.path.join(config.results_dir, 'optimization_log.txt')
        config.optimization_log_full_file = os.path.join(config.results_dir, 'optimization_log_full.txt')
        config.population_history_file = os.path.join(config.results_dir, 'population_history.json')

        # 确保结果目录存在
        os.makedirs(config.results_dir, exist_ok=True)

        print("配置信息:")
        print(f"  种群大小: {config.population_size}")
        print(f"  最大代数: {config.num_generations}")
        print(f"  精英大小: {config.elite_size}")
        print(f"  评分机制: score = jailbreak_score (直接使用越狱成功率)")
        print(f"  结果目录: {config.results_dir}")
        print()

        # 2. 创建优化器
        optimizer = EvolutionaryOptimizer(config)

        # 3. 执行优化
        best_individuals = optimizer.optimize()

        # 4. 输出结果摘要
        if best_individuals:
            print("\n优化完成！发现的最佳individuals:")
            for i, ind in enumerate(sorted(best_individuals, key=lambda x: x.score, reverse=True)[:5], 1):
                print(f"{i}. 评分: {ind.score:.3f} | 代数: {ind.generation}")
                print(f"   Prompt: {ind.prompt}")
                print()

        # 5. 生成可视化（如果未禁用）
        if not args.no_plots:
            try:
                plot_file = os.path.join(config.results_dir, 'optimization_progress.png')
                plot_optimization_progress(config.optimization_log_file, plot_file)
                print(f"图表已保存到: {plot_file}")
            except Exception as e:
                print(f"生成图表失败: {e}")

        print("\n" + "=" * 80)
        print("实验完成！")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n用户中断优化过程")

    except Exception as e:
        print(f"\n优化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    # 无论优化是否成功，都保存配置和报告
    try:
        # 保存配置（用于重现实验）
        config_file = os.path.join(config.results_dir, 'config_used.json')
        print(f"正在保存配置到: {config_file}")
        config.save_to_file(config_file)
        print(f"配置已保存到: {config_file}")

        # 生成实验报告
        try:
            report_file = generate_experiment_report(config.results_dir, output_file=os.path.join(config.results_dir, 'experiment_report.md'))
            print(f"实验报告已生成: {report_file}")
        except Exception as e:
            print(f"生成实验报告失败: {e}")
            import traceback
            traceback.print_exc()

        # 最终摘要（如果有结果文件）
        if os.path.exists(config.best_triggers_file):
            print_optimization_summary(config.best_triggers_file)

    except Exception as e:
        print(f"保存配置和报告时发生错误: {e}")

    return 0 if 'best_individuals' in locals() and best_individuals else 1


def test_basic_functionality():
    """测试基本功能（用于调试）"""
    print("测试基本功能...")

    try:
        # 测试配置加载
        config = Config()
        print(f"✓ 配置加载成功: {config}")

        # 测试种群初始化
        from population import Population
        from evaluator import Evaluator
        population = Population(config)
        evaluator = Evaluator(config)
        population.initialize_from_seeds(evaluator=evaluator)
        print(f"✓ 种群初始化成功: {population}")

        # 测试提案生成器
        from proposer import Proposer
        proposer = Proposer(config)
        candidates = proposer.generate_candidates([ind.prompt for ind in population.members[:3]])
        print(f"✓ 提案生成成功: 生成 {len(candidates)} 个候选")

        # 测试评价器
        if candidates:
            scores, jb_scores, interaction_histories = evaluator.evaluate_population(
                candidates[:3], memory_examples=[]
            )
            print(f"✓ 评价器测试成功: 评估了 {len(scores)} 个候选")

        print("所有基本功能测试通过！")
        return True

    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_help():
    """显示帮助信息"""
    help_text = """
AutoDan 进化优化系统使用指南

基本用法:
    python run_optimization.py

高级用法:
    python run_optimization.py --max-generations 100 -n my_experiment

参数说明:
    --max-generations, -g: 最大进化代数
    --population-size, -p: 种群大小
    --experiment-name, -n: 实验名称（用于组织结果）
    --config-file, -c: 自定义配置文件
    --no-plots: 不生成图表

示例:
    # 基本优化
    python run_optimization.py

    # 大规模优化
    python run_optimization.py -g 200 -p 50 -n "large_scale_test"

    # 测试基本功能
    python run_optimization.py --test

文件结构:
    AutoDan/
    ├── config.py              # 配置管理
    ├── population.py          # 种群管理
    ├── proposer.py            # 提案生成器
    ├── evaluator.py           # 评价器
    ├── evolutionary_optimizer.py  # 主优化器
    ├── utils.py               # 工具函数
    ├── run_optimization.py    # 主执行脚本（本文件）
    ├── trigger_instruction.txt # 种子prompts
    └── results/                # 输出结果目录
        ├── best_triggers.json
        ├── optimization_log.txt
        └── population_history.json
"""
    print(help_text)


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 运行基本功能测试
        success = test_basic_functionality()
        sys.exit(0 if success else 1)

    else:
        # 运行主优化
        sys.exit(main())
