"""Population of candidate trigger instructions."""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Individual:
    prompt: str
    score: float = 0.0
    jailbreak_score: float = 0.0
    generation: int = 0
    parent_ids: List[int] = None
    interaction_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.interaction_history is None:
            self.interaction_history = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "score": self.score,
            "jailbreak_score": self.jailbreak_score,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "interaction_history": self.interaction_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Individual":
        return cls(
            prompt=data["prompt"],
            score=data.get("score", 0.0),
            jailbreak_score=data.get("jailbreak_score", 0.0),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            interaction_history=data.get("interaction_history", []),
        )


class Population:
    def __init__(self, config):
        self.config = config
        self.size = config.population_size
        self.members: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[List[Individual]] = []

    def initialize_from_seeds(self, evaluator=None, attack_file: str = None, trigger_file: str = None) -> None:
        """Pick top-3 seed templates by mean train score (same eval as individuals)."""
        if trigger_file is None:
            trigger_file = os.path.join(self.config.base_dir, "data_seeact", "trigger_instruction_short_seed.txt")

        if evaluator is None:
            raise ValueError("evaluator is required for seed scoring")

        if not os.path.exists(trigger_file):
            raise FileNotFoundError(f"Trigger seed file not found: {trigger_file}")

        trigger_templates = []
        with open(trigger_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            template_blocks = [p.strip() for p in content.split("\n\n") if p.strip()]
            for template_block in template_blocks:
                lines = [line.strip() for line in template_block.split("\n") if line.strip()]
                if lines:
                    template = "\n".join(lines)
                    if template.startswith('f"') and template.endswith('"'):
                        template = template[2:-1]
                    trigger_templates.append(template)

        print(f"Loaded {len(trigger_templates)} seed trigger(s) from {trigger_file}")

        training_pairs = evaluator._train_pairs
        if not training_pairs:
            evaluator._load_and_split_dataset()
            training_pairs = evaluator._train_pairs

        if not training_pairs:
            raise ValueError("No training pairs for template scoring")

        print(f"Scoring seeds on {len(training_pairs)} training pair(s)")

        template_scores = []
        for template_idx, template in enumerate(trigger_templates):
            print(f"Scoring template {template_idx + 1}/{len(trigger_templates)}...")
            avg_score, interaction_history = evaluator.evaluate_goal_achievement(template, [])
            template_scores.append(
                {
                    "template": template,
                    "avg_score": avg_score,
                    "template_idx": template_idx,
                    "interaction_history": interaction_history,
                }
            )
            print(f"  template {template_idx + 1}: mean score {avg_score:.4f}")

        template_scores.sort(key=lambda x: x["avg_score"], reverse=True)
        elite_templates = template_scores[:3]

        print("\n=== Top seed templates ===")
        for i, elite in enumerate(elite_templates, 1):
            print(f"  {i}: score {elite['avg_score']:.4f}")
            print(f"     {elite['template'][:50]}...")

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

        print(f"\nInitial population size: {len(self.members)} (top {len(elite_templates)} seeds)")
        self._update_best_individual()

    def _load_dataset_combinations(self, dataset_file: str) -> List[Dict[str, str]]:
        combinations = []

        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                content = f.read()

            import re

            pair_pattern = r"Pair \d+:(.*?)(?=Pair \d+:|$)"
            pair_blocks = re.findall(pair_pattern, content, re.DOTALL)

            for block in pair_blocks:
                host_match = re.search(r"Host Instruction:\s*(.+?)(?=\n|$)", block, re.MULTILINE)
                frag_match = re.search(r"Sensitive Fragment:\s*(.+?)(?=\n|$)", block, re.MULTILINE)

                if host_match and frag_match:
                    combinations.append(
                        {
                            "host_instruction": host_match.group(1).strip(),
                            "fragment": frag_match.group(1).strip(),
                        }
                    )

        except Exception as e:
            print(f"Dataset parse error: {e}")

        return combinations

    def _simple_mutate(self, prompt: str) -> str:
        if "[MASK]" in prompt or "[mask]" in prompt:
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

    def add_candidates(
        self,
        new_candidates: List[str],
        scores: List[float],
        jailbreak_scores: List[float],
        parent_ids: List[List[int]] = None,
        interaction_histories: List[List[Dict[str, str]]] = None,
        generation: int = None,
    ) -> None:
        if len(new_candidates) != len(scores):
            raise ValueError("candidates and scores length mismatch")

        if parent_ids is None:
            parent_ids = [[] for _ in range(len(new_candidates))]
        if interaction_histories is None:
            interaction_histories = [[] for _ in range(len(new_candidates))]

        new_individuals = []
        for i, (prompt, score, jb_score, parents, interactions) in enumerate(
            zip(new_candidates, scores, jailbreak_scores, parent_ids, interaction_histories)
        ):
            new_individuals.append(
                Individual(
                    prompt=prompt,
                    score=score,
                    jailbreak_score=jb_score,
                    generation=generation if generation is not None else self.generation,
                    parent_ids=parents,
                    interaction_history=interactions,
                )
            )

        self.members.extend(new_individuals)
        self._update_best_individual()

    def select_best(self, num_select: int) -> List[Individual]:
        sorted_members = sorted(self.members, key=lambda x: x.score, reverse=True)
        return sorted_members[:num_select]

    def get_elites(self) -> List[Individual]:
        return self.select_best(self.config.elite_size)

    def evolve_population(self) -> None:
        elites = self.get_elites()
        remaining_slots = self.size - len(elites)

        non_elites = [ind for ind in self.members if ind not in elites]
        if non_elites:
            selected_non_elites = sorted(non_elites, key=lambda x: x.score, reverse=True)[:remaining_slots]
        else:
            selected_non_elites = []

        self.members = elites + selected_non_elites

        while len(self.members) < self.size and elites:
            elite = random.choice(elites)
            mutated_prompt = self._simple_mutate(elite.prompt)
            self.members.append(
                Individual(
                    prompt=mutated_prompt,
                    score=elite.score * 0.9,
                    generation=self.generation,
                    parent_ids=[id(self)],
                )
            )

        self.generation += 1

    def _update_best_individual(self) -> None:
        if self.members:
            best = max(self.members, key=lambda x: x.score)
            if self.best_individual is None or best.score > self.best_individual.score:
                self.best_individual = best

    def get_best_individual(self) -> Optional[Individual]:
        return self.best_individual

    def get_statistics(self) -> Dict[str, float]:
        if not self.members:
            return {}

        scores = [ind.score for ind in self.members]
        jb_scores = [ind.jailbreak_score for ind in self.members]

        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_jailbreak_score": sum(jb_scores) / len(jb_scores),
            "diversity": self._calculate_diversity(),
        }

    def _calculate_diversity(self) -> float:
        if len(self.members) <= 1:
            return 0.0

        unique_prompts = set(ind.prompt for ind in self.members)
        return len(unique_prompts) / len(self.members)

    def save_to_file(self, file_path: str) -> None:
        data = {
            "generation": self.generation,
            "size": len(self.members),
            "members": [ind.to_dict() for ind in self.members],
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "statistics": self.get_statistics(),
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            return

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.generation = data.get("generation", 0)
        self.members = [Individual.from_dict(ind_data) for ind_data in data.get("members", [])]

        if data.get("best_individual"):
            self.best_individual = Individual.from_dict(data["best_individual"])

    def save_history(self, file_path: str) -> None:
        history_dict = {}
        for i, generation_population in enumerate(self.history):
            history_dict[f"population_generation_{i}"] = generation_population

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.members)

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (
            f"Population(n={len(self)}, gen={self.generation}, "
            f"mean_score={stats.get('avg_score', 0):.3f}, "
            f"max_score={stats.get('max_score', 0):.3f}, "
            f"diversity={stats.get('diversity', 0):.3f})"
        )
