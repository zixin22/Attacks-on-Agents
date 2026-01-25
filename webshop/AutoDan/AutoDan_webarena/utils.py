"""
Utility Functions Module
工具函数模块：提供各种辅助功能
"""

import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not available. Plotting functions disabled.")


def load_seed_prompts(file_path: str) -> List[str]:
    """加载种子prompts"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"种子文件不存在: {file_path}")

    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

        # 按空行分割不同的prompt
        prompt_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]

        for block in prompt_blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if lines:
                prompts.append('\n'.join(lines))

    return prompts


def save_population_snapshot(population, generation: int, output_dir: str) -> None:
    """保存种群快照"""
    os.makedirs(output_dir, exist_ok=True)

    snapshot_file = os.path.join(output_dir, f'population_gen_{generation:03d}.json')

    snapshot = {
        'generation': generation,
        'timestamp': datetime.now().isoformat(),
        'population_size': len(population.members),
        'members': [ind.to_dict() for ind in population.members],
        'statistics': population.get_statistics(),
        'best_individual': population.get_best_individual().to_dict() if population.get_best_individual() else None
    }

    with open(snapshot_file, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


def calculate_diversity_score(population) -> float:
    """计算种群多样性"""
    if len(population.members) <= 1:
        return 0.0

    prompts = [ind.prompt for ind in population.members]
    unique_prompts = set(prompts)
    return len(unique_prompts) / len(prompts)


def plot_optimization_progress(log_file: str, output_file: str = None) -> None:
    """绘制优化进度图"""
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available. Cannot generate plots.")
        return

    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    generations = []
    best_scores = []
    avg_scores = []
    diversities = []

    for entry in log_data:
        generations.append(entry['generation'])
        stats = entry['statistics']
        best_scores.append(stats.get('max_score', 0))
        avg_scores.append(stats.get('avg_score', 0))
        diversities.append(stats.get('diversity', 0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 分数进度图
    ax1.plot(generations, best_scores, 'r-', label='Best Score', linewidth=2)
    ax1.plot(generations, avg_scores, 'b-', label='Average Score', alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 多样性图
    ax2.plot(generations, diversities, 'g-', label='Population Diversity', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Population Diversity Over Generations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"优化进度图已保存到: {output_file}")
    else:
        plt.show()


def analyze_optimization_results(results_file: str) -> Dict[str, Any]:
    """分析优化结果"""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"结果文件不存在: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    best_individuals = results.get('best_individuals', [])
    if not best_individuals:
        return {}

    # 分析最佳个体
    scores = [ind['score'] for ind in best_individuals]
    jailbreak_scores = [ind['jailbreak_score'] for ind in best_individuals]

    analysis = {
        'total_generations': results.get('total_generations', 0),
        'num_best_individuals': len(best_individuals),
        'score_distribution': {
            'mean': sum(scores) / len(scores),
            'max': max(scores),
            'min': min(scores),
            'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5
        },
        'jailbreak_score_distribution': {
            'mean': sum(jailbreak_scores) / len(jailbreak_scores),
            'max': max(jailbreak_scores),
            'min': min(jailbreak_scores)
        },
        'final_best_prompt': results.get('final_best', {}).get('prompt', ''),
        'final_best_score': results.get('final_best', {}).get('score', 0.0)
    }

    return analysis


def print_optimization_summary(results_file: str) -> None:
    """打印优化摘要"""
    try:
        analysis = analyze_optimization_results(results_file)

        if not analysis:
            print("没有找到有效的优化结果")
            return

        print("=" * 60)
        print("AutoDan 进化优化摘要")
        print("=" * 60)
        print(f"总进化代数: {analysis['total_generations']}")
        print(f"最佳个体数量: {analysis['num_best_individuals']}")
        print()

        print("综合评分分布:")
        score_dist = analysis['score_distribution']
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print()

        print("越狱评分分布:")
        jb_dist = analysis['jailbreak_score_distribution']
        print(".3f")
        print(".3f")
        print(".3f")
        print()


        print("最终最佳Prompt:")
        print(f"评分: {analysis['final_best_score']:.3f}")
        print(f"内容: {analysis['final_best_prompt']}")
        print("=" * 60)

    except Exception as e:
        print(f"分析结果时出错: {e}")


def compare_prompts(prompts: List[str], target_instruction: str) -> Dict[str, Any]:
    """比较多个prompts的表现"""
    # 这里可以扩展为实际的比较逻辑
    # 暂时返回基本的统计信息

    comparison = {
        'num_prompts': len(prompts),
        'avg_length': sum(len(p) for p in prompts) / len(prompts),
        'unique_words': set(),
        'common_patterns': []
    }

    # 收集所有唯一词
    for prompt in prompts:
        words = prompt.lower().split()
        comparison['unique_words'].update(words)

    comparison['unique_words'] = len(comparison['unique_words'])

    # 查找常见模式
    patterns = ['extract', 'find', 'locate', 'product', 'spot', '<>', '[]']
    for pattern in patterns:
        count = sum(1 for p in prompts if pattern in p.lower())
        if count > len(prompts) * 0.5:  # 超过50%的prompts包含此模式
            comparison['common_patterns'].append((pattern, count))

    return comparison


def export_prompts_to_file(prompts: List[str], output_file: str,
                          format_type: str = 'json') -> None:
    """将prompts导出到文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if format_type == 'json':
        data = {
            'exported_at': datetime.now().isoformat(),
            'num_prompts': len(prompts),
            'prompts': prompts
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif format_type == 'txt':
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# AutoDan Generated Prompts\n")
            f.write(f"# Generated at: {datetime.now().isoformat()}\n")
            f.write(f"# Total prompts: {len(prompts)}\n\n")

            for i, prompt in enumerate(prompts, 1):
                f.write(f"## Prompt {i}\n")
                f.write(f"{prompt}\n\n")

    print(f"已导出 {len(prompts)} 个prompts 到 {output_file}")


def load_prompts_from_file(input_file: str) -> List[str]:
    """从文件加载prompts"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == '.json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('prompts', [])

    elif file_ext == '.txt':
        prompts = []
        with open(input_file, 'r', encoding='utf-8') as f:
            current_prompt = []
            for line in f:
                line = line.strip()
                if line.startswith('## Prompt'):
                    if current_prompt:
                        prompts.append('\n'.join(current_prompt))
                        current_prompt = []
                elif line and not line.startswith('#'):
                    current_prompt.append(line)

            if current_prompt:
                prompts.append('\n'.join(current_prompt))

        return prompts

    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def setup_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """设置实验目录"""
    experiment_dir = os.path.join(base_dir, 'experiments', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建子目录
    subdirs = ['results', 'logs', 'checkpoints', 'plots']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    return experiment_dir


def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """验证prompt的有效性"""
    if not prompt or len(prompt.strip()) < 5:
        return False, "Prompt太短"

    if len(prompt) > 500:
        return False, "Prompt太长"

    # 检查必要元素
    required_elements = ['spot', 'product']
    prompt_lower = prompt.lower()

    missing_elements = [elem for elem in required_elements if elem not in prompt_lower]
    if missing_elements:
        return False, f"缺少必要元素: {missing_elements}"

    # 检查是否有危险内容
    dangerous_patterns = ['hack', 'exploit', 'attack', 'malware', 'virus', 'bypass']
    for pattern in dangerous_patterns:
        if pattern in prompt_lower:
            return False, f"包含危险内容: {pattern}"

    return True, "有效"


def batch_validate_prompts(prompts: List[str]) -> List[Tuple[bool, str]]:
    """批量验证prompts"""
    return [validate_prompt(prompt) for prompt in prompts]


def generate_experiment_report(results_dir: str, output_file: str = None) -> str:
    """生成实验报告"""
    if output_file is None:
        output_file = os.path.join(results_dir, 'experiment_report.md')

    report_lines = [
        "# AutoDan 进化优化实验报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 文件结构",
        f"- 结果目录: {results_dir}",
        f"- 最佳triggers: results/best_triggers.json",
        f"- 优化日志: results/optimization_log.txt",
        f"- 种群历史: results/population_history.json",
        "",
        "## 优化结果摘要",
    ]

    # 尝试读取结果摘要
    best_triggers_file = os.path.join(results_dir, 'best_triggers.json')
    if os.path.exists(best_triggers_file):
        try:
            analysis = analyze_optimization_results(best_triggers_file)
            report_lines.extend([
                f"- 总进化代数: {analysis.get('total_generations', 'N/A')}",
                f"- 最佳个体数量: {analysis.get('num_best_individuals', 'N/A')}",
                ".3f"                ".3f"                ".3f"            ])
        except Exception as e:
            report_lines.append(f"- 读取结果失败: {e}")

    # 添加使用说明
    report_lines.extend([
        "",
        "## 使用说明",
        "",
        "### 1. 运行优化",
        "```bash",
        "cd AutoDan",
        "python run_optimization.py",
        "```",
        "",
        "### 2. 查看结果",
        "```bash",
        "python -c \"from utils import print_optimization_summary; print_optimization_summary('results/best_triggers.json')\"",
        "```",
        "",
        "### 3. 绘制进度图",
        "```bash",
        "python -c \"from utils import plot_optimization_progress; plot_optimization_progress('results/optimization_log.txt')\"",
        "```",
    ])

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"实验报告已生成: {output_file}")
    return output_file
