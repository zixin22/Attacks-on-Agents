"""
AutoDan Evolutionary Optimization Configuration
配置文件：管理进化算法的所有参数
"""

import os
from typing import Dict, Any


class Config:
    """AutoDan进化优化配置类"""

    def __init__(self):
        # === 种群参数 ===
        self.population_size = 20  # 种群大小
        self.num_generations = 50  # 最大进化代数
        self.elite_size = 3  # 精英个体数量（直接保留到下一代）

        # === 提案生成参数 ===
        self.llm_rewrite_variants = 5  # 每个prompt的LLM改写变体数
        self.crossover_rate = 0.3  # 交叉概率
        self.mutation_rate = 0.1  # 变异概率
        self.max_prompt_length = 200  # 最大prompt长度限制

        # === 评价参数 ===
        self.jailbreak_weight = 0.7  # 越狱成功权重
        self.quality_weight = 0.3  # 语言质量权重
        self.evaluation_samples = 3  # 每次评价的样本数

        # === LLM配置 ===
        self.llm_config = {
            'model': 'gpt-4o',  # 默认使用GPT-4o
            'temperature': 0.8,  # 创造性温度
            'max_tokens': 150,  # 最大token数
            'api_base': "http://152.53.53.64:3000/v1"  # API基础URL
        }

        # === 评价LLM配置 (用于质量评估) ===
        self.judge_config = {
            'model': 'gpt-4o',
            'temperature': 0.1,  # 较低温度以获得一致的评估
            'max_tokens': 50,
            'api_base': "http://152.53.53.64:3000/v1"
        }

        # === 文件路径 ===
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.seed_file = os.path.join(self.base_dir, 'data', 'trigger_instruction.txt')
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.best_triggers_file = os.path.join(self.results_dir, 'best_triggers.json')
        self.optimization_log_file = os.path.join(self.results_dir, 'optimization_log.txt')
        self.population_history_file = os.path.join(self.results_dir, 'population_history.json')

        # === RuleChecker配置 ===
        self.rule_checker_config = {
            'model': 'gpt-4o',
            'verbose': False,
            'api_base': "http://152.53.53.64:3000/v1"
        }

        # === 终止条件 ===
        self.convergence_threshold = 0.95  # 收敛阈值（当最佳分数超过此值时停止）
        self.no_improvement_generations = 10  # 无改进代数上限

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'population_size': self.population_size,
            'num_generations': self.num_generations,
            'elite_size': self.elite_size,
            'llm_rewrite_variants': self.llm_rewrite_variants,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'max_prompt_length': self.max_prompt_length,
            'jailbreak_weight': self.jailbreak_weight,
            'quality_weight': self.quality_weight,
            'evaluation_samples': self.evaluation_samples,
            'llm_config': self.llm_config,
            'judge_config': self.judge_config,
            'rule_checker_config': self.rule_checker_config,
            'convergence_threshold': self.convergence_threshold,
            'no_improvement_generations': self.no_improvement_generations
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: str):
        """从文件加载配置"""
        import json
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self.update_from_dict(config_dict)

    def __str__(self) -> str:
        """字符串表示"""
        return f"AutoDan Config:\n" \
               f"  Population Size: {self.population_size}\n" \
               f"  Generations: {self.num_generations}\n" \
               f"  Elite Size: {self.elite_size}\n" \
               f"  LLM Model: {self.llm_config['model']}\n" \
               f"  Jailbreak Weight: {self.jailbreak_weight}\n" \
               f"  Quality Weight: {self.quality_weight}"
