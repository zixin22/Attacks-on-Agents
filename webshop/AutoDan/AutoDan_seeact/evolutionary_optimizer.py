"""
Evolutionary Optimizer Module
主进化优化器模块：协调整个进化过程
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


class EvolutionaryOptimizer:
    """AutoDan进化优化器主控制器"""

    def __init__(self, config: Config):
        self.config = config
        self.population = Population(config)
        self.proposer = Proposer(config)
        self.evaluator = Evaluator(config)

        # 优化状态跟踪
        self.current_generation = 0
        self.best_individual: Optional[Individual] = None
        self.convergence_history: List[float] = []
        self.start_time: Optional[float] = None

        # 结果存储
        self.optimization_log: List[Dict[str, Any]] = []

    def optimize(self, target_instruction: str, max_generations: Optional[int] = None) -> List[Individual]:
        """
        执行进化优化
        返回: 优化过程中发现的最佳个体列表
        """
        if max_generations is None:
            max_generations = self.config.num_generations

        print("=" * 80)
        print("开始AutoDan进化优化")
        print("=" * 80)
        print(f"目标指令: {target_instruction}")
        print(f"最大代数: {max_generations}")
        print(f"种群大小: {self.config.population_size}")
        print(f"收敛阈值: {self.config.convergence_threshold}")
        print()

        self.start_time = time.time()

        # 初始化种群
        self._initialize_population()

        # 优化循环
        best_individuals = []
        no_improvement_count = 0
        previous_best_score = 0.0

        for generation in range(max_generations):
            self.current_generation = generation
            print(f"\n--- 第 {generation + 1} 代 ---")

            # 1. 生成候选
            candidates = self._generate_candidates()
            if not candidates:
                print("警告: 未能生成新的候选个体")
                continue

            print(f"生成了 {len(candidates)} 个候选个体")

            # 2. 评价候选trigger instructions
            # 传入memory examples（如果有的话）
            memory_examples = []  # 暂时为空，可以后续扩展
            total_scores, goal_scores, interaction_histories = self.evaluator.evaluate_population(
                candidates, memory_examples=memory_examples
            )
            # score = jailbreak_score，直接使用goal_scores
            jailbreak_scores = goal_scores

            # 3. 添加到种群
            self._add_candidates_to_population(candidates, total_scores, jailbreak_scores, interaction_histories)

            # 4. 选择和更新
            self._select_and_update_population()

            # 保存当前代的历史（在选择和更新后）
            self.population.history.append([ind.to_dict() for ind in self.population.members])

            # 5. 记录最佳个体
            current_best = self.population.get_best_individual()
            if current_best:
                best_individuals.append(current_best)
                print(".3f")

            # 记录这一代的信息（在终止检查前记录，确保最后一代也被记录）
            self._log_generation_info(generation, current_best)

            # 6. 检查终止条件
            if self._check_termination_conditions(current_best, previous_best_score, no_improvement_count):
                print("达到终止条件，停止优化")
                break

            # 更新无改进计数器
            if current_best and current_best.score > previous_best_score:
                previous_best_score = current_best.score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        # 优化完成
        self._finalize_optimization(best_individuals)
        return best_individuals

    def _initialize_population(self) -> None:
        """初始化种群"""
        print("正在初始化种群...")
        try:
            self.population.initialize_from_seeds(evaluator=self.evaluator)
            print(f"成功初始化了 {len(self.population)} 个精英模板个体")
        except Exception as e:
            print(f"初始化种群失败: {e}")
            raise

    def _generate_candidates(self) -> List[str]:
        """生成候选个体"""
        current_prompts = [ind.prompt for ind in self.population.members]

        # 获取精英个体的索引
        elites = self.population.get_elites()
        elite_indices = [i for i, ind in enumerate(self.population.members) if ind in elites]

        return self.proposer.generate_candidates(current_prompts, elite_indices)

    def _add_candidates_to_population(self, candidates: List[str],
                                    total_scores: List[float],
                                    goal_scores: List[float],
                                    interaction_histories: List[List[Dict[str, str]]]) -> None:
        """将候选添加到种群"""
        # 为每个候选创建父代ID列表（模拟）
        parent_ids = [[i] for i in range(len(candidates))]

        # 新候选属于当前代数
        current_gen = self.current_generation

        self.population.add_candidates(
            candidates, total_scores, goal_scores, parent_ids, interaction_histories, generation=current_gen
        )

    def _select_and_update_population(self) -> None:
        """选择和更新种群"""
        self.population.evolve_population()

    def _check_termination_conditions(self, current_best: Optional[Individual],
                                    previous_best_score: float,
                                    no_improvement_count: int) -> bool:
        """检查终止条件"""
        # 只检查无改进情况，不再检查收敛阈值（让进化完整运行指定代数）
        if no_improvement_count >= self.config.no_improvement_generations:
            print(f"连续 {no_improvement_count} 代无改进，停止优化")
            return True

        return False

    def _log_generation_info(self, generation: int, current_best: Optional[Individual]) -> None:
        """记录一代的信息"""
        stats = self.population.get_statistics()

        log_entry = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'population_size': len(self.population),
            'statistics': stats,
            'best_individual': current_best.to_dict() if current_best else None,
            'diversity': stats.get('diversity', 0.0),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }

        self.optimization_log.append(log_entry)

        # 实时保存所有文件
        self._save_optimization_log()
        self._save_generation_snapshot(generation, current_best)

    def _finalize_optimization(self, best_individuals: List[Individual]) -> None:
        """完成优化过程"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 80)
        print("进化优化完成")
        print("=" * 80)
        print(".2f")
        print(f"总代数: {self.current_generation + 1}")
        print(f"最佳个体数量: {len(best_individuals)}")

        if best_individuals:
            final_best = max(best_individuals, key=lambda x: x.score)
            print("\n最终最佳个体:")
            print(f"  Prompt: {final_best.prompt}")
            print(".3f")
            print(".3f")
            print(f"  出生代数: {final_best.generation}")

        # 保存最终结果
        self._save_final_results(best_individuals)

    def _save_optimization_log(self) -> None:
        """保存优化日志"""
        try:
            with open(self.config.optimization_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存优化日志失败: {e}")

    def _save_generation_snapshot(self, generation: int, current_best: Optional[Individual]) -> None:
        """每代结束时保存当前状态快照"""
        try:
            # 保存当前最佳结果
            best_individuals = self.population.get_elites()[:5]  # 保存当前Top 5

            def clean_individual_dict(ind):
                d = ind.to_dict()
                d.pop('interaction_history', None)  # 移除interaction_history节省空间
                return d

            best_data = {
                'generation_snapshot_at': datetime.now().isoformat(),
                'current_generation': generation,
                'best_individuals_so_far': [clean_individual_dict(ind) for ind in best_individuals],
                'current_best': clean_individual_dict(current_best) if current_best else None,
                'population_size': len(self.population),
                'statistics': self.population.get_statistics(),
                'config': self.config.to_dict()
            }

            with open(self.config.best_triggers_file, 'w', encoding='utf-8') as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)

            # 保存种群历史快照
            if hasattr(self.population, 'history') and self.population.history:
                with open(self.config.population_history_file, 'w', encoding='utf-8') as f:
                    json.dump(self.population.history, f, indent=2, ensure_ascii=False)

            # 保存配置快照
            with open(self.config.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

            print(f"[快照] 第{generation}代状态已保存")

        except Exception as e:
            print(f"保存代快照失败: {e}")

    def _save_final_results(self, best_individuals: List[Individual]) -> None:
        """保存最终结果"""
        try:
            # 获取最终最佳个体
            final_best = max(best_individuals, key=lambda x: x.score) if best_individuals else None

            # 在测试集上评估最终最佳个体
            test_score = 0.0
            test_interaction_history = []
            if final_best:
                try:
                    test_score, test_interaction_history = self.evaluator.evaluate_on_test_set(final_best.prompt)
                    print(f"测试集评估得分: {test_score:.4f}")

                    # 在optimization_log中记录测试集评估详情
                    if test_interaction_history:
                        test_log_entry = {
                            'generation': 'test_evaluation',
                            'timestamp': datetime.now().isoformat(),
                            'test_score': test_score,
                            'test_interactions': test_interaction_history,  # 记录所有测试交互
                            'total_test_pairs': len(test_interaction_history)
                        }
                        self.optimization_log.append(test_log_entry)

                except Exception as e:
                    print(f"测试集评估失败: {e}")

            # 保存最佳个体
            # 为best_triggers.json创建干净的数据（不包含interaction_history）
            def clean_individual_dict(ind):
                d = ind.to_dict()
                d.pop('interaction_history', None)  # 移除interaction_history
                return d

            best_data = {
                'optimization_completed_at': datetime.now().isoformat(),
                'total_generations': self.current_generation + 1,
                'best_individuals': [clean_individual_dict(ind) for ind in best_individuals],
                'final_best': clean_individual_dict(final_best) if final_best else None,
                'test_set_score': test_score,  # 添加测试集得分
                'config': self.config.to_dict()
            }

            with open(self.config.best_triggers_file, 'w', encoding='utf-8') as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)

            # 在population_history中记录测试集评估
            if test_interaction_history:
                # 临时添加测试集数据到population_history
                test_generation_data = {
                    f'population_test_evaluation': [{
                        'prompt': final_best.prompt if final_best else '',
                        'score': test_score,
                        'jailbreak_score': test_score,  # 直接使用test_score
                        'generation': 'test_evaluation',
                        'parent_ids': [],
                        'interaction_history': test_interaction_history
                    }]
                }

                # 读取现有的population_history，添加测试数据，然后重新保存
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
                    print(f"保存测试集历史失败: {e}")

            # 保存种群历史
            self.population.save_history(self.config.population_history_file)

            # 确保优化日志也被保存
            self._save_optimization_log()

            print(f"结果已保存到: {self.config.experiment_dir}")

        except Exception as e:
            print(f"保存最终结果失败: {e}")
            import traceback
            traceback.print_exc()  # 打印完整堆栈跟踪

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self.optimization_log:
            return {}

        final_log = self.optimization_log[-1]
        best_individuals = [log['best_individual'] for log in self.optimization_log
                           if log['best_individual'] is not None]

        return {
            'total_generations': len(self.optimization_log),
            'final_population_size': final_log['population_size'],
            'final_statistics': final_log['statistics'],
            'best_score_progression': [ind['score'] for ind in best_individuals] if best_individuals else [],
            'optimization_time': final_log.get('elapsed_time', 0),
            'convergence_achieved': any(ind['score'] >= self.config.convergence_threshold
                                      for ind in best_individuals) if best_individuals else False
        }

    def resume_optimization(self, checkpoint_file: str, target_instruction: str) -> List[Individual]:
        """从检查点恢复优化"""
        try:
            # 加载检查点
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            # 恢复状态
            self.current_generation = checkpoint.get('current_generation', 0)
            self.optimization_log = checkpoint.get('optimization_log', [])

            # 恢复种群
            if 'population' in checkpoint:
                self.population = Population(self.config)
                population_data = checkpoint['population']
                self.population.members = [Individual.from_dict(ind_data)
                                         for ind_data in population_data.get('members', [])]
                self.population.generation = population_data.get('generation', 0)

            print(f"从检查点恢复优化: 第 {self.current_generation + 1} 代")

            # 继续优化
            remaining_generations = self.config.num_generations - self.current_generation - 1
            return self.optimize(target_instruction, remaining_generations)

        except Exception as e:
            print(f"恢复优化失败: {e}")
            return []

    def save_checkpoint(self, checkpoint_file: str) -> None:
        """保存检查点"""
        try:
            checkpoint = {
                'current_generation': self.current_generation,
                'optimization_log': self.optimization_log,
                'population': {
                    'generation': self.population.generation,
                    'members': [ind.to_dict() for ind in self.population.members]
                },
                'config': self.config.to_dict(),
                'timestamp': datetime.now().isoformat()
            }

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"保存检查点失败: {e}")

    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_optimization_summary()
        return f"EvolutionaryOptimizer(代数={summary.get('total_generations', 0)}, " \
               f"种群大小={summary.get('final_population_size', 0)}, " \
               f"优化时间={summary.get('optimization_time', 0):.1f}s)"
