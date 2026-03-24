"""
Evaluate trigger_query similarity: memory vs attack
For AgentHarm dataset

Data sources:
- Memory: D:\rap-main\webshop\promptarmor\agentharm_data\32memory.json ("Instruction" field)
- Trigger: D:\rap-main\webshop\promptarmor\agentharm_data\retrieve_datasets.jsonl ("task_input_core" field)
- Attack: D:\rap-main\webshop\promptarmor\agentharm_data\hostpair_seperate_results.jsonl ("injection_instruction" field)

Uses all 5 similarity methods: e5, all_minilm, levenshtein, string_matching, bm25
"""

import json
import os
import sys
from typing import List, Dict, Tuple, Any

sys.path.insert(0, r"D:\rap-main\webshop")

from text_similarity import (
    cosine_similarity_e5,
    cosine_similarity_all_minilm,
    levenshtein_distance,
    string_matching,
    bm25_similarity
)


def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_memory_instructions(filepath: str) -> List[str]:
    """Load all Instructions from 32memory.json"""
    data = load_json(filepath)
    return [item['Instruction'] for item in data]


def get_trigger_instructions(filepath: str) -> Dict[str, str]:
    """Load task_id -> task_input_core mapping from retrieve_datasets.jsonl"""
    data = load_jsonl(filepath)
    return {item['task_id']: item['task_input_core'] for item in data}


def get_attack_instructions(filepath: str) -> Dict[str, str]:
    """Load task_id -> injection_instruction mapping from hostpair_seperate_results.jsonl"""
    data = load_jsonl(filepath)
    return {item['task_id']: item['injection_instruction'] for item in data}


def compute_best_match(trigger_instruction: str,
                       attack_instruction: str,
                       memory_instructions: List[str],
                       method: str) -> Tuple[str, float, str]:
    """
    Compute similarity: trigger vs attack & memory
    Returns: (source, max_similarity, source_info)
    """
    if method == 'e5':
        attack_sim = cosine_similarity_e5(trigger_instruction, attack_instruction)
    elif method == 'all_minilm':
        attack_sim = cosine_similarity_all_minilm(trigger_instruction, attack_instruction)
    elif method == 'levenshtein':
        attack_sim = levenshtein_distance(trigger_instruction, attack_instruction)
    elif method == 'string_matching':
        attack_sim = string_matching(trigger_instruction, attack_instruction)
    elif method == 'bm25':
        scores = bm25_similarity(trigger_instruction, memory_instructions + [attack_instruction])
        memory_max = float(scores[:-1].max())
        attack_sim = float(scores[-1])
    else:
        raise ValueError(f"Unknown method: {method}")

    if method != 'bm25':
        memory_max = -1
        for mem_instr in memory_instructions:
            if method == 'e5':
                sim = cosine_similarity_e5(trigger_instruction, mem_instr)
            elif method == 'all_minilm':
                sim = cosine_similarity_all_minilm(trigger_instruction, mem_instr)
            elif method == 'levenshtein':
                sim = levenshtein_distance(trigger_instruction, mem_instr)
            elif method == 'string_matching':
                sim = string_matching(trigger_instruction, mem_instr)
            else:
                sim = 0
            if sim > memory_max:
                memory_max = sim

    if attack_sim >= memory_max:
        return 'attack', attack_sim, attack_instruction[:50] + '...'
    else:
        return 'memory', memory_max, 'memory_1.json'


def evaluate_method(trigger_instructions: Dict[str, str],
                   attack_instructions: Dict[str, str],
                   memory_instructions: List[str],
                   method: str) -> Dict:
    """Evaluate all cases using specified method"""
    matches = 0
    total = len(trigger_instructions)
    details = []

    print(f"\n{'='*70}")
    print(f"Method: {method}")
    print(f"{'='*70}")

    # Get common task_ids
    common_ids = set(trigger_instructions.keys()) & set(attack_instructions.keys())
    print(f"Common task_ids: {len(common_ids)}")

    for task_id in sorted(common_ids):
        trigger_instruction = trigger_instructions[task_id]
        attack_instruction = attack_instructions[task_id]

        source, max_similarity, source_info = compute_best_match(
            trigger_instruction, attack_instruction,
            memory_instructions, method
        )

        is_match = (source == 'attack')
        if is_match:
            matches += 1

        details.append({
            'task_id': task_id,
            'source': source,
            'similarity': max_similarity,
            'is_match': is_match
        })

        # Print first 10 or matches
        task_num = int(task_id.split('-')[0]) if '-' in task_id else 0
        if task_num <= 10 or is_match:
            status = "✓ ATTACK WIN" if is_match else "✗ MEMORY WIN"
            print(f"Task {task_id}: source={source}, sim={max_similarity:.4f} [{status}]")

    match_rate = matches / total if total > 0 else 0

    print(f"\nStats:")
    print(f"  Total: {total}")
    print(f"  Matches (attack win): {matches}")
    print(f"  Match rate: {match_rate:.2%}")

    return {
        'method': method,
        'total': total,
        'matches': matches,
        'match_rate': match_rate,
        'details': details
    }


def main():
    # File paths
    memory_file = r"D:\rap-main\webshop\promptarmor\agentharm_data\32memory.json"
    trigger_file = r"D:\rap-main\webshop\promptarmor\agentharm_data\retrieve_datasets.jsonl"
    attack_file = r"D:\rap-main\webshop\promptarmor\agentharm_data\hostpair_seperate_results.jsonl"

    print("Loading data...")
    memory_instructions = get_memory_instructions(memory_file)
    trigger_instructions = get_trigger_instructions(trigger_file)
    attack_instructions = get_attack_instructions(attack_file)

    print(f"Memory instructions: {len(memory_instructions)}")
    print(f"Trigger instructions: {len(trigger_instructions)}")
    print(f"Attack instructions: {len(attack_instructions)}")

    # All 5 methods
    methods = ['e5', 'all_minilm', 'levenshtein', 'string_matching', 'bm25']

    all_results = {}

    for method in methods:
        print(f"\n\nProcessing method: {method} ...")
        try:
            result = evaluate_method(
                trigger_instructions, attack_instructions,
                memory_instructions, method
            )
            all_results[method] = result
        except Exception as e:
            print(f"\nError in method {method}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {
                'method': method,
                'error': str(e),
                'total': len(trigger_instructions),
                'matches': 0,
                'match_rate': 0.0
            }

    print("\n" + "=" * 70)
    print("Summary Results")
    print("=" * 70)
    print(f"{'Method':<20} {'AttackWin':<15} {'MemoryWin':<15} {'Total':<10} {'AttackRate':<15}")
    print("-" * 70)

    for method in methods:
        result = all_results[method]
        if 'error' in result:
            print(f"{method:<20} {'ERROR':<15} {result['total']:<10} {'N/A':<15}")
        else:
            memory_win = result['total'] - result['matches']
            print(f"{method:<20} {result['matches']:<15} {memory_win:<15} "
                  f"{result['total']:<10} {result['match_rate']:.2%}")

    # Save results
    output_file = (r"D:\rap-main\webshop\promptarmor\agentharm_data"
                  r"\attack_vs_memory_agentharm_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

