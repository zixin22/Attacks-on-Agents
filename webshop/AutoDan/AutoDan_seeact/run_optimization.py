#!/usr/bin/env python3
"""CLI entry: run AutoDan evolutionary optimization (SeeAct)."""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from evolutionary_optimizer import EvolutionaryOptimizer
from utils import print_optimization_summary, plot_optimization_progress, generate_experiment_report


def main():
    parser = argparse.ArgumentParser(description="AutoDan SeeAct evolutionary optimization")
    parser.add_argument("--max-generations", "-g", type=int, default=None, help="Override max generations")
    parser.add_argument("--population-size", "-p", type=int, default=None, help="Override population size")
    parser.add_argument(
        "--experiment-name",
        "-n",
        type=str,
        default="default_experiment",
        help="Named run under experiments/<name>/results (default keeps results/optimization_<id>/)",
    )
    parser.add_argument("--resume-from", "-r", type=str, default=None, help="Resume from checkpoint JSON")
    parser.add_argument("--config-file", "-c", type=str, default=None, help="Load JSON overrides for Config")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib progress plot")

    args = parser.parse_args()

    print("=" * 80)
    print("AutoDan (SeeAct)")
    print("=" * 80)
    print(f"Experiment name: {args.experiment_name}")
    print()

    config = None
    optimizer = None
    best_individuals = None

    try:
        config = Config()

        if args.config_file and os.path.exists(args.config_file):
            config.load_from_file(args.config_file)
            print(f"Loaded config from {args.config_file}")

        if args.max_generations is not None:
            config.num_generations = args.max_generations
        if args.population_size is not None:
            config.population_size = args.population_size

        if args.experiment_name != "default_experiment":
            from utils import setup_experiment_directory

            experiment_dir = setup_experiment_directory(config.base_dir, args.experiment_name)
            config.results_dir = os.path.join(experiment_dir, "results")
        else:
            config.results_dir = config.experiment_dir

        config.best_triggers_file = os.path.join(config.results_dir, "best_triggers.json")
        config.optimization_log_file = os.path.join(config.results_dir, "optimization_log.json")
        config.population_history_file = os.path.join(config.results_dir, "population_history.json")

        os.makedirs(config.results_dir, exist_ok=True)

        print("Config:")
        print(f"  population_size: {config.population_size}")
        print(f"  num_generations: {config.num_generations}")
        print(f"  elite_size: {config.elite_size}")
        print(f"  score: same as goal / jailbreak similarity metric")
        print(f"  results_dir: {config.results_dir}")
        print()

        optimizer = EvolutionaryOptimizer(config)

        if args.resume_from:
            print(f"Resuming from {args.resume_from}")
            best_individuals = optimizer.resume_optimization(args.resume_from)
        else:
            best_individuals = optimizer.optimize()

        if best_individuals:
            print("\nDone. Top individuals:")
            for i, ind in enumerate(sorted(best_individuals, key=lambda x: x.score, reverse=True)[:5], 1):
                print(f"{i}. score={ind.score:.3f} gen={ind.generation}")
                print(f"   {ind.prompt}")
                print()

        if not args.no_plots:
            try:
                plot_file = os.path.join(config.results_dir, "optimization_progress.png")
                plot_optimization_progress(config.optimization_log_file, plot_file)
                print(f"Plot: {plot_file}")
            except Exception as e:
                print(f"Plot failed: {e}")

        print("\n" + "=" * 80)
        print("Run finished")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if optimizer is not None and config is not None:
            try:
                checkpoint_file = os.path.join(config.results_dir, "checkpoint_interrupt.json")
                optimizer.save_checkpoint(checkpoint_file)
                print(f"Checkpoint: {checkpoint_file}")
                print("Resume with: python run_optimization.py --resume-from <path>")
            except Exception as e:
                print(f"Checkpoint save failed: {e}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    if config is None:
        return 1

    try:
        config_file = os.path.join(config.results_dir, "config_used.json")
        print(f"Writing {config_file}")
        config.save_to_file(config_file)

        try:
            report_file = generate_experiment_report(
                config.results_dir, output_file=os.path.join(config.results_dir, "experiment_report.md")
            )
            print(f"Report: {report_file}")
        except Exception as e:
            print(f"Report failed: {e}")
            import traceback

            traceback.print_exc()

        if os.path.exists(config.best_triggers_file):
            print_optimization_summary(config.best_triggers_file)

    except Exception as e:
        print(f"Post-run save failed: {e}")

    return 0 if best_individuals else 1


def test_basic_functionality():
    print("Smoke test...")

    try:
        config = Config()
        print(f"OK config: {config}")

        from population import Population
        from evaluator import Evaluator

        population = Population(config)
        evaluator = Evaluator(config)
        population.initialize_from_seeds(evaluator=evaluator)
        print(f"OK population: {population}")

        from proposer import Proposer

        proposer = Proposer(config)
        candidates = proposer.generate_candidates([ind.prompt for ind in population.members[:3]])
        print(f"OK proposer: {len(candidates)} candidate(s)")

        if candidates:
            scores, jb_scores, interaction_histories = evaluator.evaluate_population(candidates[:3], memory_examples=[])
            print(f"OK evaluator: {len(scores)} score(s)")

        print("Smoke test passed")
        return True

    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_help():
    print(
        """
AutoDan SeeAct — quick reference

Default run:
    python run_optimization.py

Custom:
    python run_optimization.py -g 20 -p 15 -n my_run

Named experiment outputs:
    experiments/<name>/results/{best_triggers.json,optimization_log.json,...}

Default outputs:
    results/optimization_<id>/...

Resume:
    python run_optimization.py -r results/checkpoint_interrupt.json

API key:
    Put the key in openai_key.txt (same folder as this script) or set OPENAI_API_KEY.

Smoke test:
    python run_optimization.py --test
"""
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_help()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.exit(0 if test_basic_functionality() else 1)

    sys.exit(main())
