"""Utility helpers for AutoDan SeeAct."""

import os
import sys

_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WS not in sys.path:
    sys.path.insert(0, _WS)

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def load_openai_api_key() -> str:
    """Read API key from ``webshop/OpenAI_api_key.txt`` only (via ``openai_paths``)."""
    from openai_paths import read_openai_api_key

    return read_openai_api_key()


# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not available. Plotting functions disabled.")


def load_seed_prompts(file_path: str) -> List[str]:
    """Load seed prompts from a blank-line separated text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Seed file not found: {file_path}")

    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

        prompt_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]

        for block in prompt_blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if lines:
                prompts.append('\n'.join(lines))

    return prompts


def save_population_snapshot(population, generation: int, output_dir: str) -> None:
    """Write one generation snapshot JSON."""
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
    """Fraction of unique prompts in the population."""
    if len(population.members) <= 1:
        return 0.0

    prompts = [ind.prompt for ind in population.members]
    unique_prompts = set(prompts)
    return len(unique_prompts) / len(prompts)


def plot_optimization_progress(log_file: str, output_file: str = None) -> None:
    """Plot best/avg score and diversity vs generation (skips non-integer generation rows)."""
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available. Cannot generate plots.")
        return

    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    generations = []
    best_scores = []
    avg_scores = []
    diversities = []

    for entry in log_data:
        gen = entry.get("generation")
        if not isinstance(gen, int):
            continue
        generations.append(gen)
        stats = entry['statistics']
        best_scores.append(stats.get('max_score', 0))
        avg_scores.append(stats.get('avg_score', 0))
        diversities.append(stats.get('diversity', 0))

    if not generations:
        print("[Warning] No plottable generation rows in log (need integer 'generation').")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(generations, best_scores, 'r-', label='Best Score', linewidth=2)
    ax1.plot(generations, avg_scores, 'b-', label='Average Score', alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(generations, diversities, 'g-', label='Population Diversity', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Population Diversity Over Generations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
    else:
        plt.show()


def analyze_optimization_results(results_file: str) -> Dict[str, Any]:
    """Summarize best_triggers.json."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    best_individuals = results.get('best_individuals', [])
    if not best_individuals:
        return {}

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
    try:
        analysis = analyze_optimization_results(results_file)

        if not analysis:
            print("No summary (missing or empty best_triggers.json).")
            return

        print("=" * 60)
        print("AutoDan summary")
        print("=" * 60)
        print(f"Generations recorded: {analysis['total_generations']}")
        print(f"Best-individual rows: {analysis['num_best_individuals']}")
        print()

        print("Score distribution:")
        score_dist = analysis['score_distribution']
        print(f"  mean: {score_dist['mean']:.3f}")
        print(f"  max:  {score_dist['max']:.3f}")
        print(f"  min:  {score_dist['min']:.3f}")
        print(f"  std:  {score_dist['std']:.3f}")
        print()

        print("Jailbreak / goal score distribution:")
        jb_dist = analysis['jailbreak_score_distribution']
        print(f"  mean: {jb_dist['mean']:.3f}")
        print(f"  max:  {jb_dist['max']:.3f}")
        print(f"  min:  {jb_dist['min']:.3f}")
        print()

        print("Final best prompt:")
        print(f"  score: {analysis['final_best_score']:.3f}")
        print(f"  text:  {analysis['final_best_prompt']}")
        print("=" * 60)

    except Exception as e:
        print(f"Summary error: {e}")


def compare_prompts(prompts: List[str]) -> Dict[str, Any]:
    """Lightweight prompt statistics (legacy helper)."""

    comparison = {
        'num_prompts': len(prompts),
        'avg_length': sum(len(p) for p in prompts) / len(prompts),
        'unique_words': set(),
        'common_patterns': []
    }

    for prompt in prompts:
        words = prompt.lower().split()
        comparison['unique_words'].update(words)

    comparison['unique_words'] = len(comparison['unique_words'])

    patterns = ['extract', 'find', 'locate', 'product', 'spot', '<>', '[]']
    for pattern in patterns:
        count = sum(1 for p in prompts if pattern in p.lower())
        if count > len(prompts) * 0.5:
            comparison['common_patterns'].append((pattern, count))

    return comparison


def export_prompts_to_file(prompts: List[str], output_file: str,
                          format_type: str = 'json') -> None:
    """Export prompts to JSON or TXT."""
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

    print(f"Exported {len(prompts)} prompt(s) -> {output_file}")


def load_prompts_from_file(input_file: str) -> List[str]:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input not found: {input_file}")

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
        raise ValueError(f"Unsupported extension: {file_ext}")


def setup_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """Create experiments/<name>/ with standard subfolders."""
    experiment_dir = os.path.join(base_dir, 'experiments', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    subdirs = ['results', 'logs', 'checkpoints', 'plots']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    return experiment_dir


def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Heuristic WebShop-style prompt check (not used in core SeeAct loop)."""
    if not prompt or len(prompt.strip()) < 5:
        return False, "prompt too short"

    if len(prompt) > 500:
        return False, "prompt too long"

    required_elements = ['spot', 'product']
    prompt_lower = prompt.lower()

    missing_elements = [elem for elem in required_elements if elem not in prompt_lower]
    if missing_elements:
        return False, f"missing tokens: {missing_elements}"

    dangerous_patterns = ['hack', 'exploit', 'attack', 'malware', 'virus', 'bypass']
    for pattern in dangerous_patterns:
        if pattern in prompt_lower:
            return False, f"blocked word: {pattern}"

    return True, "ok"


def batch_validate_prompts(prompts: List[str]) -> List[Tuple[bool, str]]:
    return [validate_prompt(prompt) for prompt in prompts]


def generate_experiment_report(results_dir: str, output_file: str = None) -> str:
    if output_file is None:
        output_file = os.path.join(results_dir, 'experiment_report.md')

    report_lines = [
        "# AutoDan SeeAct — run report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Artifacts",
        f"- results dir: `{results_dir}`",
        "- `best_triggers.json` — best individuals + final test score",
        "- `optimization_log.json` — per-generation stats",
        "- `population_history.json` — population snapshots",
        "",
        "## Summary",
    ]

    best_triggers_file = os.path.join(results_dir, 'best_triggers.json')
    if os.path.exists(best_triggers_file):
        try:
            analysis = analyze_optimization_results(best_triggers_file)
            report_lines.extend([
                f"- generations: {analysis.get('total_generations', 'N/A')}",
                f"- best rows: {analysis.get('num_best_individuals', 'N/A')}",
                f"- score mean: {analysis.get('score_distribution', {}).get('mean', 0.0):.3f}",
                f"- score max: {analysis.get('score_distribution', {}).get('max', 0.0):.3f}",
                f"- final best score: {analysis.get('final_best_score', 0.0):.3f}",
            ])
        except Exception as e:
            report_lines.append(f"- read error: {e}")

    report_lines.extend([
        "",
        "## Commands",
        "",
        "```bash",
        "cd AutoDan_seeact   # this folder",
        "python run_optimization.py",
        "```",
        "",
        "```bash",
        "python -c \"from utils import print_optimization_summary; print_optimization_summary('results/optimization_1/best_triggers.json')\"",
        "```",
        "",
        "API key: `webshop/OpenAI_api_key.txt` (run from repo so webshop is two levels up).",
    ])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"Wrote report: {output_file}")
    return output_file
