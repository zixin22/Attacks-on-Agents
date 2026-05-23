#!/usr/bin/env python3
"""
AutoDan Evolutionary Optimization Runner
：AutoDan
"""

import os
import sys
import argparse

# Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from evolutionary_optimizer import EvolutionaryOptimizer
from utils import print_optimization_summary, plot_optimization_progress, generate_experiment_report


def main():
    """"""
    parser = argparse.ArgumentParser(description='AutoDan Evolutionary Optimization')
    parser.add_argument('--max-generations', '-g', type=int, default=None,
                       help='（）')
    parser.add_argument('--population-size', '-p', type=int, default=None,
                       help='（）')
    parser.add_argument('--experiment-name', '-n', type=str, default='default_experiment',
                       help='')
    parser.add_argument('--config-file', '-c', type=str, default=None,
                       help='')
    parser.add_argument('--no-plots', action='store_true',
                       help='')

    args = parser.parse_args()

    print("=" * 80)
    print("AutoDan ")
    print("=" * 80)
    print(f": {args.experiment_name}")
    print()

    try:
        # 1. 
        config = Config()

        # （）
        if args.config_file and os.path.exists(args.config_file):
            config.load_from_file(args.config_file)
            print(f" {args.config_file} ")

        # 
        if args.max_generations is not None:
            config.num_generations = args.max_generations
        if args.population_size is not None:
            config.population_size = args.population_size

        #  - optimization_i
        if args.experiment_name != 'default_experiment':
            from utils import setup_experiment_directory
            experiment_dir = setup_experiment_directory(config.base_dir, args.experiment_name)
            config.results_dir = os.path.join(experiment_dir, 'results')
        else:
            # ，Configoptimization_i
            config.results_dir = config.experiment_dir

        # results_dir
        config.best_triggers_file = os.path.join(config.results_dir, 'best_triggers.json')
        config.optimization_log_file = os.path.join(config.results_dir, 'optimization_log.txt')
        config.optimization_log_full_file = os.path.join(config.results_dir, 'optimization_log_full.txt')
        config.population_history_file = os.path.join(config.results_dir, 'population_history.json')

        # 
        os.makedirs(config.results_dir, exist_ok=True)

        print(":")
        print(f"  : {config.population_size}")
        print(f"  : {config.num_generations}")
        print(f"  : {config.elite_size}")
        print(f"  : score = jailbreak_score ()")
        print(f"  : {config.results_dir}")
        print()

        # 2. 
        optimizer = EvolutionaryOptimizer(config)

        # 3. 
        best_individuals = optimizer.optimize()

        # 4. 
        if best_individuals:
            print("\n！individuals:")
            for i, ind in enumerate(sorted(best_individuals, key=lambda x: x.score, reverse=True)[:5], 1):
                print(f"{i}. : {ind.score:.3f} | : {ind.generation}")
                print(f"   Prompt: {ind.prompt}")
                print()

        # 5. （）
        if not args.no_plots:
            try:
                plot_file = os.path.join(config.results_dir, 'optimization_progress.png')
                plot_optimization_progress(config.optimization_log_file, plot_file)
                print(f": {plot_file}")
            except Exception as e:
                print(f": {e}")

        print("\n" + "=" * 80)
        print("！")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n")

    except Exception as e:
        print(f"\n: {e}")
        import traceback
        traceback.print_exc()

    # ，
    try:
        # （）
        config_file = os.path.join(config.results_dir, 'config_used.json')
        print(f": {config_file}")
        config.save_to_file(config_file)
        print(f": {config_file}")

        # 
        try:
            report_file = generate_experiment_report(config.results_dir, output_file=os.path.join(config.results_dir, 'experiment_report.md'))
            print(f": {report_file}")
        except Exception as e:
            print(f": {e}")
            import traceback
            traceback.print_exc()

        # （）
        if os.path.exists(config.best_triggers_file):
            print_optimization_summary(config.best_triggers_file)

    except Exception as e:
        print(f": {e}")

    return 0 if 'best_individuals' in locals() and best_individuals else 1


def test_basic_functionality():
    """（）"""
    print("...")

    try:
        # 
        config = Config()
        print(f"✓ : {config}")

        # 
        from population import Population
        from evaluator import Evaluator
        population = Population(config)
        evaluator = Evaluator(config)
        population.initialize_from_seeds(evaluator=evaluator)
        print(f"✓ : {population}")

        # 
        from proposer import Proposer
        proposer = Proposer(config)
        candidates = proposer.generate_candidates([ind.prompt for ind in population.members[:3]])
        print(f"✓ :  {len(candidates)} ")

        # 
        if candidates:
            scores, jb_scores, interaction_histories = evaluator.evaluate_population(
                candidates[:3], memory_examples=[]
            )
            print(f"✓ :  {len(scores)} ")

        print("！")
        return True

    except Exception as e:
        print(f"✗ : {e}")
        import traceback
        traceback.print_exc()
        return False


def show_help():
    """"""
    help_text = """
AutoDan 

:
    python run_optimization.py

:
    python run_optimization.py --max-generations 100 -n my_experiment

:
    --max-generations, -g: 
    --population-size, -p: 
    --experiment-name, -n: （）
    --config-file, -c: 
    --no-plots: 

:
    # 
    python run_optimization.py

    # 
    python run_optimization.py -g 200 -p 50 -n "large_scale_test"

    # 
    python run_optimization.py --test

:
    AutoDan/
    ├── config.py              # 
    ├── population.py          # 
    ├── proposer.py            # 
    ├── evaluator.py           # 
    ├── evolutionary_optimizer.py  # 
    ├── utils.py               # 
    ├── run_optimization.py    # （）
    ├── trigger_instruction.txt # prompts
    └── results/                # 
        ├── best_triggers.json
        ├── optimization_log.txt
        └── population_history.json
"""
    print(help_text)


if __name__ == "__main__":
    # 
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 
        success = test_basic_functionality()
        sys.exit(0 if success else 1)

    else:
        # 
        sys.exit(main())
