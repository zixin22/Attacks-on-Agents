"""Coordinates evolutionary search over trigger templates."""

import os
import time
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import Config
from population import Population, Individual
from proposer import Proposer
from evaluator import Evaluator


class EvolutionaryOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.population = Population(config)
        self.proposer = Proposer(config)
        self.evaluator = Evaluator(config)

        self.current_generation = 0
        self.best_individual: Optional[Individual] = None
        self.start_time: Optional[float] = None

        self.optimization_log: List[Dict[str, Any]] = []

    def optimize(self, max_generations: Optional[int] = None) -> List[Individual]:
        if max_generations is None:
            max_generations = self.config.num_generations

        print("=" * 80)
        print("AutoDan evolutionary optimization")
        print("=" * 80)
        print(f"Max generations: {max_generations}")
        print(f"Population size: {self.config.population_size}")
        print()

        self.start_time = time.time()

        self._initialize_population()

        best_individuals = []
        no_improvement_count = 0
        previous_best_score = 0.0

        for generation in range(max_generations):
            self.current_generation = generation
            print(f"\n--- Generation {generation + 1} ---")

            candidates = self._generate_candidates()
            if not candidates:
                print("Warning: no new candidates")
                continue

            print(f"Candidates this gen: {len(candidates)}")

            memory_examples: List = []
            total_scores, goal_scores, interaction_histories = self.evaluator.evaluate_population(
                candidates, memory_examples=memory_examples
            )
            jailbreak_scores = goal_scores

            self._add_candidates_to_population(candidates, total_scores, jailbreak_scores, interaction_histories)

            self._select_and_update_population()

            self.population.history.append([ind.to_dict() for ind in self.population.members])

            current_best = self.population.get_best_individual()
            if current_best:
                best_individuals.append(current_best)
                print(f"Best score so far: {current_best.score:.3f}")

            self._log_generation_info(generation, current_best)

            if self._check_termination_conditions(current_best, previous_best_score, no_improvement_count):
                print("Stopping: termination condition met")
                break

            if current_best and current_best.score > previous_best_score:
                previous_best_score = current_best.score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        self._finalize_optimization(best_individuals)
        return best_individuals

    def _initialize_population(self) -> None:
        print("Initializing population...")
        try:
            self.population.initialize_from_seeds(evaluator=self.evaluator)
            print(f"Initialized {len(self.population)} seed individual(s)")
        except Exception as e:
            print(f"Population init failed: {e}")
            raise

    def _generate_candidates(self) -> List[str]:
        current_prompts = [ind.prompt for ind in self.population.members]
        elites = self.population.get_elites()
        elite_indices = [i for i, ind in enumerate(self.population.members) if ind in elites]
        return self.proposer.generate_candidates(current_prompts, elite_indices)

    def _add_candidates_to_population(
        self,
        candidates: List[str],
        total_scores: List[float],
        goal_scores: List[float],
        interaction_histories: List[List[Dict[str, str]]],
    ) -> None:
        parent_ids = [[i] for i in range(len(candidates))]
        current_gen = self.current_generation

        self.population.add_candidates(
            candidates,
            total_scores,
            goal_scores,
            parent_ids,
            interaction_histories,
            generation=current_gen,
        )

    def _select_and_update_population(self) -> None:
        self.population.evolve_population()

    def _check_termination_conditions(
        self, _current_best: Optional[Individual], _previous_best_score: float, no_improvement_count: int
    ) -> bool:
        if no_improvement_count >= self.config.no_improvement_generations:
            print(f"No improvement for {no_improvement_count} generation(s); stopping")
            return True
        return False

    def _log_generation_info(self, generation: int, current_best: Optional[Individual]) -> None:
        stats = self.population.get_statistics()

        log_entry = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(self.population),
            "statistics": stats,
            "best_individual": current_best.to_dict() if current_best else None,
            "diversity": stats.get("diversity", 0.0),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
        }

        self.optimization_log.append(log_entry)

        self._save_optimization_log()
        self._save_generation_snapshot(generation, current_best)

    def _finalize_optimization(self, best_individuals: List[Individual]) -> None:
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 80)
        print("Optimization finished")
        print("=" * 80)
        print(f"Elapsed: {elapsed_time:.2f}s")
        print(f"Generations run: {self.current_generation + 1}")
        print(f"Tracked best snapshots: {len(best_individuals)}")

        if best_individuals:
            final_best = max(best_individuals, key=lambda x: x.score)
            print("\nFinal best (by tracked snapshots):")
            print(f"  Prompt: {final_best.prompt}")
            print(f"  Train score: {final_best.score:.3f}")
            print(f"  jailbreak_score: {final_best.jailbreak_score:.3f}")
            print(f"  Born at generation: {final_best.generation}")

        self._save_final_results(best_individuals)

    def _save_optimization_log(self) -> None:
        try:
            with open(self.config.optimization_log_file, "w", encoding="utf-8") as f:
                json.dump(self.optimization_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to write optimization log: {e}")

    def _save_generation_snapshot(self, generation: int, current_best: Optional[Individual]) -> None:
        try:
            best_individuals = self.population.get_elites()[:5]

            def clean_individual_dict(ind):
                d = ind.to_dict()
                d.pop("interaction_history", None)
                return d

            best_data = {
                "generation_snapshot_at": datetime.now().isoformat(),
                "current_generation": generation,
                "best_individuals_so_far": [clean_individual_dict(ind) for ind in best_individuals],
                "current_best": clean_individual_dict(current_best) if current_best else None,
                "population_size": len(self.population),
                "statistics": self.population.get_statistics(),
                "config": self.config.to_dict(),
            }

            with open(self.config.best_triggers_file, "w", encoding="utf-8") as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)

            if hasattr(self.population, "history") and self.population.history:
                with open(self.config.population_history_file, "w", encoding="utf-8") as f:
                    json.dump(self.population.history, f, indent=2, ensure_ascii=False)

            with open(self.config.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

            print(f"[checkpoint] generation {generation} saved")

        except Exception as e:
            print(f"Generation snapshot save failed: {e}")

    def _save_final_results(self, best_individuals: List[Individual]) -> None:
        try:
            final_best = max(best_individuals, key=lambda x: x.score) if best_individuals else None

            test_score = 0.0
            test_interaction_history = []
            if final_best:
                try:
                    test_score, test_interaction_history = self.evaluator.evaluate_on_test_set(final_best.prompt)
                    print(f"Held-out test mean score: {test_score:.4f}")

                    if test_interaction_history:
                        test_log_entry = {
                            "generation": "test_evaluation",
                            "timestamp": datetime.now().isoformat(),
                            "test_score": test_score,
                            "test_interactions": test_interaction_history,
                            "total_test_pairs": len(test_interaction_history),
                        }
                        self.optimization_log.append(test_log_entry)

                except Exception as e:
                    print(f"Test-set eval failed: {e}")

            def clean_individual_dict(ind):
                d = ind.to_dict()
                d.pop("interaction_history", None)
                return d

            best_data = {
                "optimization_completed_at": datetime.now().isoformat(),
                "total_generations": self.current_generation + 1,
                "best_individuals": [clean_individual_dict(ind) for ind in best_individuals],
                "final_best": clean_individual_dict(final_best) if final_best else None,
                "test_set_score": test_score,
                "config": self.config.to_dict(),
            }

            with open(self.config.best_triggers_file, "w", encoding="utf-8") as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)

            if test_interaction_history:
                test_generation_data = {
                    "population_test_evaluation": [
                        {
                            "prompt": final_best.prompt if final_best else "",
                            "score": test_score,
                            "jailbreak_score": test_score,
                            "generation": "test_evaluation",
                            "parent_ids": [],
                            "interaction_history": test_interaction_history,
                        }
                    ]
                }

                try:
                    if os.path.exists(self.config.population_history_file):
                        with open(self.config.population_history_file, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    else:
                        existing_data = {}

                    existing_data.update(test_generation_data)

                    with open(self.config.population_history_file, "w", encoding="utf-8") as f:
                        json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Failed merging test history: {e}")

            self.population.save_history(self.config.population_history_file)

            self._save_optimization_log()

            print(f"Artifacts directory: {self.config.experiment_dir}")

        except Exception as e:
            print(f"Final save failed: {e}")
            traceback.print_exc()

    def get_optimization_summary(self) -> Dict[str, Any]:
        if not self.optimization_log:
            return {}

        final_log = self.optimization_log[-1]
        best_individuals = [log["best_individual"] for log in self.optimization_log if log["best_individual"] is not None]

        return {
            "total_generations": len(self.optimization_log),
            "final_population_size": final_log["population_size"],
            "final_statistics": final_log["statistics"],
            "best_score_progression": [ind["score"] for ind in best_individuals] if best_individuals else [],
            "optimization_time": final_log.get("elapsed_time", 0),
        }

    def resume_optimization(self, checkpoint_file: str) -> List[Individual]:
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            self.current_generation = checkpoint.get("current_generation", 0)
            self.optimization_log = checkpoint.get("optimization_log", [])

            if "population" in checkpoint:
                self.population = Population(self.config)
                population_data = checkpoint["population"]
                self.population.members = [
                    Individual.from_dict(ind_data) for ind_data in population_data.get("members", [])
                ]
                self.population.generation = population_data.get("generation", 0)

            print(f"Resumed at generation {self.current_generation + 1}")

            remaining_generations = self.config.num_generations - self.current_generation - 1
            return self.optimize(remaining_generations)

        except Exception as e:
            print(f"Resume failed: {e}")
            return []

    def save_checkpoint(self, checkpoint_file: str) -> None:
        try:
            checkpoint = {
                "current_generation": self.current_generation,
                "optimization_log": self.optimization_log,
                "population": {
                    "generation": self.population.generation,
                    "members": [ind.to_dict() for ind in self.population.members],
                },
                "config": self.config.to_dict(),
                "timestamp": datetime.now().isoformat(),
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Checkpoint save failed: {e}")

    def __str__(self) -> str:
        summary = self.get_optimization_summary()
        return (
            f"EvolutionaryOptimizer(generations={summary.get('total_generations', 0)}, "
            f"pop_size={summary.get('final_population_size', 0)}, "
            f"time_s={summary.get('optimization_time', 0):.1f})"
        )
