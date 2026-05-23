"""
Population Management Module
：prompt
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Individual:
    """"""
    prompt: str  # trigger instruction
    score: float = 0.0  # 
    jailbreak_score: float = 0.0  # （score）
    generation: int = 0  # 
    parent_ids: List[int] = None  # ID
    interaction_history: List[Dict[str, str]] = None  # LLM [{"prompt": "...", "response": "..."}]

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.interaction_history is None:
            self.interaction_history = []

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            'prompt': self.prompt,
            'score': self.score,
            'jailbreak_score': self.jailbreak_score,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'interaction_history': self.interaction_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """"""
        return cls(
            prompt=data['prompt'],
            score=data.get('score', 0.0),
            jailbreak_score=data.get('jailbreak_score', 0.0),
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            interaction_history=data.get('interaction_history', [])
        )


class Population:
    """prompt"""

    def __init__(self, config, template_kind: str = "trigger"):
        self.config = config
        self.template_kind = template_kind  # "trigger" | "attack"
        self.size = config.population_size
        self.members: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[List[Individual]] = []  # 

    def initialize_from_seeds(self, evaluator=None, trigger_file: str = None) -> None:
        """
        ：trigger，
        ：5pair，3
        """
        if trigger_file is None:
            trigger_file = os.path.join(self.config.base_dir, 'data_webshop', 'trigger_instruction.txt')

        if evaluator is None:
            raise ValueError("evaluator")

        # 1. trigger instruction
        if not os.path.exists(trigger_file):
            raise FileNotFoundError(f"Trigger instruction: {trigger_file}")

        trigger_templates = []
        with open(trigger_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 
            template_blocks = [p.strip() for p in content.split('\n\n') if p.strip()]
            for template_block in template_blocks:
                lines = [line.strip() for line in template_block.split('\n') if line.strip()]
                if lines:
                    template = '\n'.join(lines)
                    # f""
                    if template.startswith('f"') and template.endswith('"'):
                        template = template[2:-1]
                    trigger_templates.append(template)

        print(f"trigger {len(trigger_templates)} trigger")

        # 2. 
        # evaluator
        training_pairs = evaluator._train_pairs
        if not training_pairs:
            evaluator._load_and_split_dataset()
            training_pairs = evaluator._train_pairs

        if not training_pairs:
            raise ValueError("")

        print(f" {len(training_pairs)} pair")

        # 3. ：pair（）
        template_scores = []
        for template_idx, template in enumerate(trigger_templates):
            print(f" {template_idx + 1}/{len(trigger_templates)}...")

            # ，
            avg_score, interaction_history = evaluator.evaluate_goal_achievement(template, [])

            template_scores.append({
                'template': template,
                'avg_score': avg_score,
                'template_idx': template_idx,
                'interaction_history': interaction_history
            })

            print(f"   {template_idx + 1}:  {avg_score:.4f}")

        # 4.  elite_size 
        template_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        elite_n = min(self.config.elite_size, len(template_scores))
        elite_templates = template_scores[:elite_n]

        print("\n===  ===")
        for i, elite in enumerate(elite_templates, 1):
            print(f" {i}:  {elite['avg_score']:.4f}")
            print(f"  : {elite['template'][:50]}...")

        # 5. Individual
        self.members = []
        for elite in elite_templates:
            # 
            individual = Individual(
                prompt=elite['template'],
                score=elite['avg_score'],  # 
                generation=0,
                parent_ids=[elite['template_idx']],  # 
                interaction_history=elite['interaction_history']  # 
            )
            self.members.append(individual)

        print(f"\n: {len(self.members)} ({len(elite_templates)})")
        self._update_best_individual()

    def initialize_attack_templates_from_file(
        self,
        evaluator,
        reference_trigger: str,
        attack_file: str = None,
    ) -> None:
        """
        Load attack_instruction = ... lines from data_webshop/attack_instruction_template.txt,
        score each template with a fixed trigger, keep top elites (prompt = full template string).
        """
        if attack_file is None:
            attack_file = os.path.join(self.config.base_dir, "data_webshop", "attack_instruction_template.txt")
        if not os.path.exists(attack_file):
            raise FileNotFoundError(f"Attack instruction file not found: {attack_file}")

        from attack_template_utils import load_attack_instruction_lines

        templates_all = load_attack_instruction_lines(attack_file)
        if not templates_all:
            raise ValueError(f"No attack templates parsed from {attack_file}")
        # Attack template initialization uses only the first seed line.
        # Additional lines can be rotated manually across runs.
        templates = [templates_all[0]]

        print(f"Loaded first attack template seed from {attack_file} (1 of {len(templates_all)} lines)")

        training_pairs = evaluator._train_pairs
        if not training_pairs:
            evaluator._load_and_split_dataset()
            training_pairs = evaluator._train_pairs
        if not training_pairs:
            raise ValueError("No training pairs for attack template evaluation")

        template_scores = []
        for template_idx, template in enumerate(templates):
            print(f"Evaluating attack template {template_idx + 1}/{len(templates)}...")
            avg_score, interaction_history = evaluator.evaluate_goal_achievement(
                reference_trigger, [], attack_template=template
            )
            template_scores.append(
                {
                    "template": template,
                    "avg_score": avg_score,
                    "template_idx": template_idx,
                    "interaction_history": interaction_history,
                }
            )
            print(f"  Template {template_idx + 1}: avg score {avg_score:.4f}")

        template_scores.sort(key=lambda x: x["avg_score"], reverse=True)
        elite_n = min(self.config.elite_size, len(template_scores))
        elite_templates = template_scores[:elite_n]

        print("\n=== Elite attack templates ===")
        for i, elite in enumerate(elite_templates, 1):
            print(f"Elite {i}: score {elite['avg_score']:.4f}")
            print(f"  {elite['template'][:80]}...")

        self.members = []
        for elite in elite_templates:
            self.members.append(
                Individual(
                    prompt=elite["template"],
                    score=elite["avg_score"],
                    generation=0,
                    parent_ids=[elite["template_idx"]],
                    interaction_history=elite["interaction_history"],
                )
            )

        print(f"\nInitial attack population size: {len(self.members)}")
        self._update_best_individual()

    def _simple_mutate(self, prompt: str) -> str:
        """（，[MASK]）"""
        if self.template_kind == "attack":
            return prompt
        # [MASK]prompt，
        if '[MASK]' in prompt or '[mask]' in prompt:
            return prompt

        mutations = [
            lambda x: x.replace("Extract", "Find"),
            lambda x: x.replace("Output", "Return"),
            lambda x: x.replace("Locate", "Identify"),
            lambda x: x.replace("<product>", "[product]"),
            lambda x: x.replace("Buy", "Purchase"),
        ]

        mutation = random.choice(mutations)
        return mutation(prompt)

    def add_candidates(self, new_candidates: List[str], scores: List[float],
                      jailbreak_scores: List[float], parent_ids: List[List[int]] = None,
                      interaction_histories: List[List[Dict[str, str]]] = None,
                      generation: int = None) -> None:
        """"""
        if len(new_candidates) != len(scores):
            raise ValueError("")

        if parent_ids is None:
            parent_ids = [[] for _ in range(len(new_candidates))]
        if interaction_histories is None:
            interaction_histories = [[] for _ in range(len(new_candidates))]

        # 
        new_individuals = []
        for i, (prompt, score, jb_score, parents, interactions) in enumerate(
            zip(new_candidates, scores, jailbreak_scores, parent_ids, interaction_histories)):

            individual = Individual(
                prompt=prompt,
                score=score,
                jailbreak_score=jb_score,
                generation=generation if generation is not None else self.generation,
                parent_ids=parents,
                interaction_history=interactions
            )
            new_individuals.append(individual)

        # 
        self.members.extend(new_individuals)

        # 
        self._update_best_individual()

    def select_best(self, num_select: int) -> List[Individual]:
        """"""
        # 
        sorted_members = sorted(self.members, key=lambda x: x.score, reverse=True)
        return sorted_members[:num_select]

    def get_elites(self) -> List[Individual]:
        """"""
        return self.select_best(self.config.elite_size)

    def evolve_population(self) -> None:
        """"""

        # 
        elites = self.get_elites()
        remaining_slots = self.size - len(elites)

        # （）
        non_elites = [ind for ind in self.members if ind not in elites]
        if non_elites:
            # ，
            selected_non_elites = sorted(non_elites, key=lambda x: x.score, reverse=True)[:remaining_slots]
        else:
            selected_non_elites = []

        # 
        self.members = elites + selected_non_elites

        # ，
        while len(self.members) < self.size and elites:
            elite = random.choice(elites)
            mutated_prompt = self._simple_mutate(elite.prompt)
            mutated_individual = Individual(
                prompt=mutated_prompt,
                score=elite.score * 0.9,  # 
                generation=self.generation,
                parent_ids=[id(self)]  # ID
            )
            self.members.append(mutated_individual)

        self.generation += 1

    def _update_best_individual(self) -> None:
        """"""
        if self.members:
            best = max(self.members, key=lambda x: x.score)
            if self.best_individual is None or best.score > self.best_individual.score:
                self.best_individual = best

    def get_best_individual(self) -> Optional[Individual]:
        """"""
        return self.best_individual

    def get_statistics(self) -> Dict[str, float]:
        """"""
        if not self.members:
            return {}

        scores = [ind.score for ind in self.members]
        jb_scores = [ind.jailbreak_score for ind in self.members]

        return {
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'avg_jailbreak_score': sum(jb_scores) / len(jb_scores),
            'diversity': self._calculate_diversity()
        }

    def _calculate_diversity(self) -> float:
        """（prompt）"""
        if len(self.members) <= 1:
            return 0.0

        # ：prompt
        unique_prompts = set(ind.prompt for ind in self.members)
        return len(unique_prompts) / len(self.members)

    def save_history(self, file_path: str) -> None:
        """"""
        # 
        history_dict = {}
        for i, generation_population in enumerate(self.history):
            cleaned_generation = []
            for ind in generation_population:
                if isinstance(ind, dict):
                    d = dict(ind)
                    d.pop("interaction_history", None)
                    cleaned_generation.append(d)
                else:
                    d = ind.to_dict()
                    d.pop("interaction_history", None)
                    cleaned_generation.append(d)
            history_dict[f"population_generation_{i}"] = cleaned_generation

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        """"""
        return len(self.members)

    def __str__(self) -> str:
        """"""
        stats = self.get_statistics()
        return f"Population(={len(self)}, ={self.generation}, " \
               f"={stats.get('avg_score', 0):.3f}, " \
               f"={stats.get('max_score', 0):.3f}, " \
               f"={stats.get('diversity', 0):.3f})"
