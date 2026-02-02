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
        self.num_generations = 10  # 最大进化代数
        self.elite_size = 5  # 精英个体数量（直接保留到下一代）

        # === 提案生成参数 ===
        self.llm_rewrite_variants = 5  # 每个prompt的LLM改写变体数
        self.crossover_rate = 0.3  # 交叉概率
        self.mutation_rate = 0.1  # 变异概率
        self.max_prompt_length = 200  # 最大prompt长度限制

        # === 评价参数 ===
        # 直接使用jailbreak_score作为score，不再需要权重配置
        self.evaluation_samples = 3  # 每次评价的样本数

        # === 数据集划分参数 ===
        self.dataset_total_size = 99  # 数据集总大小
        self.train_ratio = 5/99       # 训练集比例 (5个，前5个)
        self.val_ratio = 0.0          # 验证集比例 (取消验证集)
        self.test_ratio = 79/99       # 测试集比例 (79个)

        # 计算具体数量
        self.train_size = int(self.dataset_total_size * self.train_ratio)  # 20
        self.val_size = int(self.dataset_total_size * self.val_ratio)      # 0
        self.test_size = self.dataset_total_size - self.train_size - self.val_size  # 79

        # === LLM配置 ===
        self.llm_config = {
            'model': 'gpt-4o-2024-08-06',  # 使用指定的GPT-4o模型
            'temperature': 0.8,  # 创造性温度
            'max_tokens': 150,  # 最大token数
            'api_base': "http://152.53.53.64:3000/v1"  # API基础URL
        }

        # === 评价LLM配置 (用于质量评估) ===
        self.judge_config = {
            'model': 'gpt-4o-2024-08-06',  # 使用指定的GPT-4o模型
            'temperature': 0.1,  # 较低温度以获得一致的评估
            'max_tokens': 50,
            'api_base': "http://152.53.53.64:3000/v1"
        }

        # === 文件路径 ===
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.seed_file = os.path.join(self.base_dir, 'data_seeact', 'trigger_instruction_short_seed.txt')
        self.results_dir = os.path.join(self.base_dir, 'results')

        # === 实验管理 ===
        self.experiment_id = self._get_next_experiment_id()
        self.experiment_dir = os.path.join(self.results_dir, f'optimization_{self.experiment_id}')

        # 创建实验目录
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 实验内文件路径
        self.best_triggers_file = os.path.join(self.experiment_dir, 'best_triggers.json')
        self.optimization_log_file = os.path.join(self.experiment_dir, 'optimization_log.txt')
        self.population_history_file = os.path.join(self.experiment_dir, 'population_history.json')
        self.trigger_host_similarity_file = os.path.join(self.experiment_dir, 'trigger_host_similarity.json')

        # === RuleChecker配置 ===
        self.rule_checker_config = {
            'model': 'gpt-4o',
            'verbose': False,
            'api_base': "http://152.53.53.64:3000/v1"
        }

        # === API和性能配置 ===
        self.request_interval = 0.2  # LLM API请求间隔（秒），配合重试机制防止速率限制

        # === 终止条件 ===
        self.convergence_threshold = 0.95  # 收敛阈值（当最佳分数超过此值时停止）
        self.no_improvement_generations = 10  # 无改进代数上限

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'experiment_id': self.experiment_id,
            'population_size': self.population_size,
            'num_generations': self.num_generations,
            'elite_size': self.elite_size,
            'llm_rewrite_variants': self.llm_rewrite_variants,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'max_prompt_length': self.max_prompt_length,
            'evaluation_samples': self.evaluation_samples,
            'dataset_total_size': self.dataset_total_size,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
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

    def _get_next_experiment_id(self) -> int:
        """获取下一个实验编号"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
            return 1

        # 查找所有optimization_开头的文件夹
        existing_experiments = []
        for item in os.listdir(self.results_dir):
            if os.path.isdir(os.path.join(self.results_dir, item)) and item.startswith('optimization_'):
                try:
                    exp_id = int(item.split('_')[1])
                    existing_experiments.append(exp_id)
                except (ValueError, IndexError):
                    continue

        # 返回下一个可用的ID
        return max(existing_experiments) + 1 if existing_experiments else 1

    def __str__(self) -> str:
        """字符串表示"""
        return f"AutoDan Config:\n" \
               f"  Experiment ID: {self.experiment_id}\n" \
               f"  Population Size: {self.population_size}\n" \
               f"  Generations: {self.num_generations}\n" \
               f"  Elite Size: {self.elite_size}\n" \
               f"  LLM Model: {self.llm_config['model']}\n" \
               f"  Scoring: score = jailbreak_score (direct)"
