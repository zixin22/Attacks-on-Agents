"""
Population Management Module
种群管理模块：负责管理候选prompt池
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Individual:
    """种群中的个体"""
    prompt: str  # trigger instruction文本
    score: float = 0.0  # 越狱成功评分
    jailbreak_score: float = 0.0  # 越狱成功评分（与score相同）
    generation: int = 0  # 出生代数
    parent_ids: List[int] = None  # 父代ID列表
    interaction_history: List[Dict[str, str]] = None  # 与LLM的交互历史 [{"prompt": "...", "response": "..."}]

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.interaction_history is None:
            self.interaction_history = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
        """从字典创建实例"""
        return cls(
            prompt=data['prompt'],
            score=data.get('score', 0.0),
            jailbreak_score=data.get('jailbreak_score', 0.0),
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            interaction_history=data.get('interaction_history', [])
        )


class Population:
    """候选prompt池管理类"""

    def __init__(self, config):
        self.config = config
        self.size = config.population_size
        self.members: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[List[Individual]] = []  # 记录每一代的种群

    def initialize_from_seeds(self, evaluator=None, trigger_file: str = None) -> None:
        """
        模板级别初始化：评估所有trigger模板，选择表现最好的作为精英
        新的初始化逻辑：每个模板在5个随机采样训练pair上评估，选出平均得分最高的3个模板
        """
        if trigger_file is None:
            trigger_file = os.path.join(self.config.base_dir, 'data_osagent', 'trigger_instruction_short_seed.txt')

        if evaluator is None:
            raise ValueError("需要提供evaluator来进行模板评估")

        # 1. 读取trigger instruction模板变体
        if not os.path.exists(trigger_file):
            raise FileNotFoundError(f"Trigger instruction文件不存在: {trigger_file}")

        trigger_templates = []
        with open(trigger_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 按空行分割不同的模板
            template_blocks = [p.strip() for p in content.split('\n\n') if p.strip()]
            for template_block in template_blocks:
                lines = [line.strip() for line in template_block.split('\n') if line.strip()]
                if lines:
                    template = '\n'.join(lines)
                    # 移除可能的f""包装
                    if template.startswith('f"') and template.endswith('"'):
                        template = template[2:-1]
                    trigger_templates.append(template)

        print(f"从trigger文件加载了 {len(trigger_templates)} 个种子trigger模板")

        # 2. 获取训练集数据用于评估
        # evaluator应该已经加载并划分了数据集
        training_pairs = evaluator._train_pairs
        if not training_pairs:
            evaluator._load_and_split_dataset()
            training_pairs = evaluator._train_pairs

        if not training_pairs:
            raise ValueError("无法获取训练集数据进行模板评估")

        print(f"使用 {len(training_pairs)} 个训练pair进行模板评估")

        # 3. 评估每个模板：在所有训练pair上测试（与后续个体评估一致）
        template_scores = []
        for template_idx, template in enumerate(trigger_templates):
            print(f"评估模板 {template_idx + 1}/{len(trigger_templates)}...")

            # 使用完整的评估方法，与后续个体评估一致
            avg_score, interaction_history = evaluator.evaluate_goal_achievement(template, [])

            template_scores.append({
                'template': template,
                'avg_score': avg_score,
                'template_idx': template_idx,
                'interaction_history': interaction_history
            })

            print(f"  模板 {template_idx + 1}: 平均得分 {avg_score:.4f}")

        # 4. 选择平均得分最高的3个模板作为精英
        template_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        elite_templates = template_scores[:3]  # 选择Top 3

        print("\n=== 精英模板选择结果 ===")
        for i, elite in enumerate(elite_templates, 1):
            print(f"精英 {i}: 得分 {elite['avg_score']:.4f}")
            print(f"  模板: {elite['template'][:50]}...")

        # 5. 为每个精英模板创建Individual对象
        self.members = []
        for elite in elite_templates:
            # 为每个精英模板创建一个对应的个体
            individual = Individual(
                prompt=elite['template'],
                score=elite['avg_score'],  # 使用评估得到的平均得分
                generation=0,
                parent_ids=[elite['template_idx']],  # 记录原始模板索引
                interaction_history=elite['interaction_history']  # 记录交互历史
            )
            self.members.append(individual)

        print(f"\n初始种群大小: {len(self.members)} (选择了{len(elite_templates)}个精英模板)")
        self._update_best_individual()

    def _simple_mutate(self, prompt: str) -> str:
        """简单的变异操作（用于初始化扩展，保护[MASK]）"""
        # 跳过包含[MASK]的prompt，避免破坏结构
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
        """添加新候选到种群"""
        if len(new_candidates) != len(scores):
            raise ValueError("候选数量与评分数量不匹配")

        if parent_ids is None:
            parent_ids = [[] for _ in range(len(new_candidates))]
        if interaction_histories is None:
            interaction_histories = [[] for _ in range(len(new_candidates))]

        # 创建新个体
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

        # 更新当前种群
        self.members.extend(new_individuals)

        # 更新最佳个体
        self._update_best_individual()

    def select_best(self, num_select: int) -> List[Individual]:
        """选择最优个体"""
        # 按综合评分排序
        sorted_members = sorted(self.members, key=lambda x: x.score, reverse=True)
        return sorted_members[:num_select]

    def get_elites(self) -> List[Individual]:
        """获取精英个体"""
        return self.select_best(self.config.elite_size)

    def evolve_population(self) -> None:
        """进化到下一代"""

        # 选择下一代
        elites = self.get_elites()
        remaining_slots = self.size - len(elites)

        # 从剩余个体中选择（给予一定生存机会）
        non_elites = [ind for ind in self.members if ind not in elites]
        if non_elites:
            # 按评分排序，选择剩余槽位
            selected_non_elites = sorted(non_elites, key=lambda x: x.score, reverse=True)[:remaining_slots]
        else:
            selected_non_elites = []

        # 合并精英和选择出的个体
        self.members = elites + selected_non_elites

        # 如果还有空位，用精英的变体填充
        while len(self.members) < self.size and elites:
            elite = random.choice(elites)
            mutated_prompt = self._simple_mutate(elite.prompt)
            mutated_individual = Individual(
                prompt=mutated_prompt,
                score=elite.score * 0.9,  # 轻微降低评分
                generation=self.generation,
                parent_ids=[id(self)]  # 使用对象ID作为父代标识
            )
            self.members.append(mutated_individual)

        self.generation += 1

    def _update_best_individual(self) -> None:
        """更新最佳个体"""
        if self.members:
            best = max(self.members, key=lambda x: x.score)
            if self.best_individual is None or best.score > self.best_individual.score:
                self.best_individual = best

    def get_best_individual(self) -> Optional[Individual]:
        """获取当前最佳个体"""
        return self.best_individual

    def get_statistics(self) -> Dict[str, float]:
        """获取种群统计信息"""
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
        """计算种群多样性（基于prompt相似度）"""
        if len(self.members) <= 1:
            return 0.0

        # 简单的多样性度量：不同prompt的比例
        unique_prompts = set(ind.prompt for ind in self.members)
        return len(unique_prompts) / len(self.members)

    def save_history(self, file_path: str) -> None:
        """保存进化历史"""
        # 转换为更清晰的对象格式
        history_dict = {}
        for i, generation_population in enumerate(self.history):
            history_dict[f"population_generation_{i}"] = generation_population

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        """返回种群大小"""
        return len(self.members)

    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return f"Population(大小={len(self)}, 代数={self.generation}, " \
               f"平均评分={stats.get('avg_score', 0):.3f}, " \
               f"最佳评分={stats.get('max_score', 0):.3f}, " \
               f"多样性={stats.get('diversity', 0):.3f})"
