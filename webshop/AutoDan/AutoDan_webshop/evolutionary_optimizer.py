"""
Evolutionary Optimizer Module
：
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import Config
from population import Population, Individual
from proposer import Proposer
from evaluator import Evaluator
from symbol_proposer import SymbolProposer


class EvolutionaryOptimizer:
    """AutoDan"""

    def __init__(self, config: Config):
        self.config = config
        self.population = Population(config, template_kind="trigger")
        self.attack_population = Population(config, template_kind="attack")
        self.proposer = Proposer(config)
        self.symbol_proposer = SymbolProposer(config)
        self.evaluator = Evaluator(config)

        # 
        self.current_generation = 0
        self.best_individual: Optional[Individual] = None
        self.start_time: Optional[float] = None

        # 
        self.optimization_log: List[Dict[str, Any]] = []
        self.optimization_log_full: List[Dict[str, Any]] = []

    def optimize(self, max_generations: Optional[int] = None) -> List[Individual]:
        """
        （： trigger， attack ）
        :  trigger 
        """
        if max_generations is None:
            max_generations = self.config.num_generations

        print("=" * 80)
        print("AutoDan（ trigger / attack ）")
        print("=" * 80)
        print(f"（ trigger+attack ）: {max_generations}")
        print(f": {self.config.population_size}")
        print()

        self.start_time = time.time()

        self._initialize_population()
        self._initialize_attack_population()

        best_individuals: List[Individual] = []
        no_improvement_count = 0
        previous_best_joint_score = -1.0
        subgeneration_counter = 0

        for macro_generation in range(max_generations):
            print(f"\n{'=' * 60}\n {macro_generation + 1}/{max_generations}\n{'=' * 60}")

            # ----- Trigger phase (fix attack template) -----
            self.current_generation = subgeneration_counter
            fix_attack = self._current_best_attack_template()
            print(f"\n--- Trigger  {subgeneration_counter + 1} ( attack ) ---")
            print(f"[Fixed attack template] {fix_attack}")

            candidates = self._generate_candidates()
            if not candidates:
                print(":  trigger ")
            else:
                print(f" {len(candidates)}  trigger ")
                memory_examples: List = []
                total_scores, goal_scores, interaction_histories = self.evaluator.evaluate_population(
                    candidates, memory_examples=memory_examples, attack_template=fix_attack
                )
                self._add_candidates_to_population(
                    candidates, total_scores, goal_scores, interaction_histories, self.population
                )
                self._select_and_update_population(self.population)

            self.population.history.append([ind.to_dict() for ind in self.population.members])
            current_best_trigger = self.population.get_best_individual()
            if current_best_trigger:
                best_individuals.append(current_best_trigger)
                print(f" trigger : {current_best_trigger.score:.3f}")

            self._log_generation_info(
                subgeneration_counter,
                current_best_trigger,
                phase="trigger",
                attack_best=self.attack_population.get_best_individual(),
                fixed_attack_template=fix_attack,
            )
            subgeneration_counter += 1

            # ----- Attack template phase (fix trigger) -----
            self.current_generation = subgeneration_counter
            fix_trigger = self._current_best_trigger_prompt()
            print(f"\n--- Attack  {subgeneration_counter + 1} ( trigger) ---")
            trigger_preview = (
                fix_trigger[:200] + "..." if fix_trigger and len(fix_trigger) > 200 else fix_trigger
            )
            print(f"[Fixed trigger prompt] {trigger_preview}")

            attack_candidates = self._generate_attack_candidates()
            if not attack_candidates:
                print(":  attack ")
            else:
                print(f" {len(attack_candidates)}  attack ")
                memory_examples = []
                at_total, at_goal, at_hist = self.evaluator.evaluate_attack_templates(
                    fix_trigger, attack_candidates, memory_examples=memory_examples
                )
                self._add_candidates_to_population(
                    attack_candidates, at_total, at_goal, at_hist, self.attack_population
                )
                self._select_and_update_population(self.attack_population)

            self.attack_population.history.append([ind.to_dict() for ind in self.attack_population.members])
            current_best_attack = self.attack_population.get_best_individual()
            print(
                f" attack : {current_best_attack.score:.3f}"
                if current_best_attack
                else " attack "
            )

            self._log_generation_info(
                subgeneration_counter,
                current_best_trigger,
                phase="attack",
                attack_best=current_best_attack,
                fixed_trigger=fix_trigger,
            )
            subgeneration_counter += 1

            # Joint score (for early stop): best trigger + best attack together
            joint_score = self._evaluate_joint_best()
            print(f" ( trigger +  attack ): {joint_score:.4f}")

            if self._check_termination_conditions_joint(no_improvement_count):
                print("，")
                break

            if joint_score > previous_best_joint_score:
                previous_best_joint_score = joint_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        self._finalize_optimization(best_individuals)
        return best_individuals

    def _current_best_attack_template(self) -> str:
        ind = self.attack_population.get_best_individual()
        if ind:
            return ind.prompt
        return self.evaluator._attack_template

    def _current_best_trigger_prompt(self) -> str:
        ind = self.population.get_best_individual()
        if ind:
            return ind.prompt
        return ""

    def _evaluate_joint_best(self) -> float:
        bt = self.population.get_best_individual()
        ba = self.attack_population.get_best_individual()
        if not bt or not ba:
            return 0.0
        score, _ = self.evaluator.evaluate_goal_achievement(
            bt.prompt, [], attack_template=ba.prompt
        )
        return float(score)

    def _check_termination_conditions_joint(self, no_improvement_count: int) -> bool:
        if no_improvement_count >= self.config.no_improvement_generations:
            print(f" {no_improvement_count} ，")
            return True
        return False

    def _initialize_attack_population(self) -> None:
        print(" attack ...")
        ref = self._current_best_trigger_prompt()
        if not ref:
            raise RuntimeError(" trigger  attack ")
        try:
            self.attack_population.initialize_attack_templates_from_file(
                self.evaluator, reference_trigger=ref
            )
            print(f"Attack : {len(self.attack_population)}")
        except Exception as e:
            print(f" attack : {e}")
            raise

    def _generate_attack_candidates(self) -> List[str]:
        current = [ind.prompt for ind in self.attack_population.members]
        return self.symbol_proposer.generate_candidates(current)

    def _initialize_population(self) -> None:
        """"""
        print("...")
        try:
            self.population.initialize_from_seeds(evaluator=self.evaluator)
            print(f" {len(self.population)} ")
        except Exception as e:
            print(f": {e}")
            raise

    def _generate_candidates(self) -> List[str]:
        """"""
        current_prompts = [ind.prompt for ind in self.population.members]
        return self.proposer.generate_candidates(current_prompts)

    def _add_candidates_to_population(
        self,
        candidates: List[str],
        total_scores: List[float],
        goal_scores: List[float],
        interaction_histories: List[List[Dict[str, str]]],
        population: Optional[Population] = None,
    ) -> None:
        """"""
        pop = population if population is not None else self.population
        parent_ids = [[i] for i in range(len(candidates))]
        current_gen = self.current_generation

        pop.add_candidates(
            candidates, total_scores, goal_scores, parent_ids, interaction_histories, generation=current_gen
        )

    def _select_and_update_population(self, population: Optional[Population] = None) -> None:
        """"""
        pop = population if population is not None else self.population
        pop.evolve_population()

    def _log_generation_info(
        self,
        generation: int,
        current_best: Optional[Individual],
        phase: str,
        attack_best: Optional[Individual] = None,
        fixed_attack_template: Optional[str] = None,
        fixed_trigger: Optional[str] = None,
    ) -> None:
        """"""
        def _clean_individual(ind: Optional[Individual]) -> Optional[Dict[str, Any]]:
            if not ind:
                return None
            d = ind.to_dict()
            d.pop("interaction_history", None)
            return d

        def _full_individual(ind: Optional[Individual]) -> Optional[Dict[str, Any]]:
            if not ind:
                return None
            return ind.to_dict()

        stats_trigger = self.population.get_statistics()
        stats_attack = self.attack_population.get_statistics()

        log_entry = {
            'generation': generation,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'population_size_trigger': len(self.population),
            'population_size_attack': len(self.attack_population),
            'statistics_trigger': stats_trigger,
            'statistics_attack': stats_attack,
            'best_individual': _clean_individual(current_best),
            'best_attack_template_individual': _clean_individual(attack_best),
            'fixed_attack_template': fixed_attack_template,
            'fixed_attack_template_full': fixed_attack_template,
            'fixed_trigger_full': fixed_trigger,
            'fixed_trigger_preview': (fixed_trigger[:120] + "...") if fixed_trigger and len(fixed_trigger) > 120 else fixed_trigger,
            'diversity': stats_trigger.get('diversity', 0.0),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
        }
        log_entry_full = {
            'generation': generation,
            'phase': phase,
            'timestamp': log_entry['timestamp'],
            'population_size_trigger': len(self.population),
            'population_size_attack': len(self.attack_population),
            'statistics_trigger': stats_trigger,
            'statistics_attack': stats_attack,
            'best_individual': _full_individual(current_best),
            'best_attack_template_individual': _full_individual(attack_best),
            'fixed_attack_template': fixed_attack_template,
            'fixed_attack_template_full': fixed_attack_template,
            'fixed_trigger_full': fixed_trigger,
            'fixed_trigger_preview': log_entry['fixed_trigger_preview'],
            'diversity': stats_trigger.get('diversity', 0.0),
            'elapsed_time': log_entry['elapsed_time'],
        }

        self.optimization_log.append(log_entry)
        self.optimization_log_full.append(log_entry_full)

        # 
        self._save_optimization_log()

    def _finalize_optimization(self, best_individuals: List[Individual]) -> None:
        """"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 80)
        print("")
        print("=" * 80)
        print(f": {elapsed_time:.2f} ")
        print(f": {self.current_generation + 1}")
        print(f": {len(best_individuals)}")

        if best_individuals:
            final_best = max(best_individuals, key=lambda x: x.score)
            print("\n trigger:")
            print(f"  Prompt: {final_best.prompt}")
            print(f"  : {final_best.score:.3f}")
            print(f"  jailbreak_score: {final_best.jailbreak_score:.3f}")
            print(f"  : {final_best.generation}")

        ab = self.attack_population.get_best_individual()
        if ab:
            print("\n attack :")
            print(f"  {ab.prompt}")
            print(f"  : {ab.score:.3f}")

        # 
        self._save_final_results(best_individuals)

    def _save_optimization_log(self) -> None:
        """"""
        try:
            with open(self.config.optimization_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_log, f, indent=2, ensure_ascii=False)
            full_file = getattr(self.config, "optimization_log_full_file", None)
            if full_file:
                with open(full_file, 'w', encoding='utf-8') as f:
                    json.dump(self.optimization_log_full, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f": {e}")

    def _save_final_results(self, best_individuals: List[Individual]) -> None:
        """"""
        try:
            # 
            final_best = max(best_individuals, key=lambda x: x.score) if best_individuals else None

            # 
            test_score = 0.0
            test_interaction_history = []
            final_best_attack = self.attack_population.get_best_individual()
            attack_for_test = final_best_attack.prompt if final_best_attack else None

            if final_best:
                try:
                    test_score, test_interaction_history = self.evaluator.evaluate_on_test_set(
                        final_best.prompt, attack_template=attack_for_test
                    )
                    print(f": {test_score:.4f}")

                    # optimization_log
                    if test_interaction_history:
                        test_log_entry = {
                            'generation': 'test_evaluation',
                            'timestamp': datetime.now().isoformat(),
                            'test_score': test_score,
                            'total_test_pairs': len(test_interaction_history),
                            'attack_template_used': attack_for_test,
                        }
                        self.optimization_log.append(test_log_entry)
                        test_log_entry_full = dict(test_log_entry)
                        test_log_entry_full['test_interactions'] = test_interaction_history
                        self.optimization_log_full.append(test_log_entry_full)

                except Exception as e:
                    print(f": {e}")

            # 
            # best_triggers.json（interaction_history）
            def clean_individual_dict(ind):
                d = ind.to_dict()
                d.pop('interaction_history', None)  # interaction_history
                return d

            best_data = {
                'optimization_completed_at': datetime.now().isoformat(),
                'total_generations': self.current_generation + 1,
                'best_individuals': [clean_individual_dict(ind) for ind in best_individuals],
                'final_best': clean_individual_dict(final_best) if final_best else None,
                'final_best_attack_template': clean_individual_dict(final_best_attack)
                if final_best_attack
                else None,
                'test_set_score': test_score,  # 
                'config': self.config.to_dict(),
            }

            with open(self.config.best_triggers_file, 'w', encoding='utf-8') as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)

            # population_history
            if test_interaction_history:
                # population_history
                test_generation_data = {
                    f'population_test_evaluation': [{
                        'prompt': final_best.prompt if final_best else '',
                        'attack_template': attack_for_test or '',
                        'score': test_score,
                        'jailbreak_score': test_score,  # test_score
                        'generation': 'test_evaluation',
                        'parent_ids': [],
                        'interaction_history': test_interaction_history
                    }]
                }

                # population_history，，
                try:
                    if os.path.exists(self.config.population_history_file):
                        with open(self.config.population_history_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    else:
                        existing_data = {}

                    existing_data.update(test_generation_data)

                    with open(self.config.population_history_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f": {e}")

            # 
            self.population.save_history(self.config.population_history_file)
            attack_hist = os.path.join(
                os.path.dirname(self.config.population_history_file),
                "population_history_attack.json",
            )
            self.attack_population.save_history(attack_hist)

            # 
            self._save_optimization_log()

            print(f": {self.config.experiment_dir}")

        except Exception as e:
            print(f": {e}")
            import traceback
            traceback.print_exc()  # 

    def get_optimization_summary(self) -> Dict[str, Any]:
        """"""
        if not self.optimization_log:
            return {}

        final_log = self.optimization_log[-1]
        best_individuals = [log['best_individual'] for log in self.optimization_log
                           if log.get('best_individual') is not None]

        pop_sz = final_log.get('population_size_trigger', final_log.get('population_size', 0))

        stats_tr = final_log.get('statistics_trigger', final_log.get('statistics', {}))

        return {
            'total_generations': len(self.optimization_log),
            'final_population_size': pop_sz,
            'final_statistics': stats_tr,
            'best_score_progression': [ind['score'] for ind in best_individuals] if best_individuals else [],
            'optimization_time': final_log.get('elapsed_time', 0),
        }

    def __str__(self) -> str:
        """"""
        summary = self.get_optimization_summary()
        return f"EvolutionaryOptimizer(={summary.get('total_generations', 0)}, " \
               f"={summary.get('final_population_size', 0)}, " \
               f"={summary.get('optimization_time', 0):.1f}s)"
