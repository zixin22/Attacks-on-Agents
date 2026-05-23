"""
AutoDan Evolutionary Optimization Configuration
：
"""

import os
from typing import Dict, Any


class Config:
    """AutoDan"""

    def __init__(self):
        # ===  ===
        self.population_size = 20  # 
        self.num_generations = 50  # 
        self.elite_size = 3  # （）

        # ===  ===
        self.llm_rewrite_variants = 5  # promptLLM
        self.symbol_proposer_variants_per_template = 3  # Symbol proposer: LLM delimiter pairs per attack template
        self.crossover_rate = 0.3  # 
        self.mutation_rate = 0.1  # 
        self.max_prompt_length = 200  # prompt

        # === LLM ===
        self.llm_config = {
            'model': 'gpt-4o',  # GPT-4o
            'temperature': 0.8,  # 
            'max_tokens': 150,  # token
            'api_base': "http://152.53.53.64:3000/v1"  # APIURL
        }

        # ===  ===
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.base_dir, 'results')

        # ===  ===
        self.experiment_id = self._get_next_experiment_id()
        self.experiment_dir = os.path.join(self.results_dir, f'optimization_{self.experiment_id}')

        # 
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 
        self.best_triggers_file = os.path.join(self.experiment_dir, 'best_triggers.json')
        self.optimization_log_file = os.path.join(self.experiment_dir, 'optimization_log.txt')
        self.optimization_log_full_file = os.path.join(self.experiment_dir, 'optimization_log_full.txt')
        self.population_history_file = os.path.join(self.experiment_dir, 'population_history.json')

        # === API ===
        self.request_interval = 0.2  # LLM API（），

        # ===  ===
        self.no_improvement_generations = 10  # 

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            'experiment_id': self.experiment_id,
            'population_size': self.population_size,
            'num_generations': self.num_generations,
            'elite_size': self.elite_size,
            'llm_rewrite_variants': self.llm_rewrite_variants,
            'symbol_proposer_variants_per_template': self.symbol_proposer_variants_per_template,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'max_prompt_length': self.max_prompt_length,
            'llm_config': self.llm_config,
            'no_improvement_generations': self.no_improvement_generations
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, file_path: str):
        """"""
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: str):
        """"""
        import json
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.update_from_dict(config_dict)

    def _get_next_experiment_id(self) -> int:
        """"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
            return 1

        # optimization_
        existing_experiments = []
        for item in os.listdir(self.results_dir):
            if os.path.isdir(os.path.join(self.results_dir, item)) and item.startswith('optimization_'):
                try:
                    exp_id = int(item.split('_')[1])
                    existing_experiments.append(exp_id)
                except (ValueError, IndexError):
                    continue

        # ID
        return max(existing_experiments) + 1 if existing_experiments else 1

    def __str__(self) -> str:
        """"""
        return f"AutoDan Config:\n" \
               f"  Experiment ID: {self.experiment_id}\n" \
               f"  Population Size: {self.population_size}\n" \
               f"  Generations: {self.num_generations}\n" \
               f"  Elite Size: {self.elite_size}\n" \
               f"  LLM Model: {self.llm_config['model']}\n" \
               f"  Scoring: score = jailbreak_score (direct)"
