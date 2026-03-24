"""
Evaluate trigger_query similarity: memory vs attack
Verify if trigger_query is more similar to original attack_query
"""

import json
import os
import sys
from typing import List, Dict, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def get_memory_instructions(filepath: str) -> List[str]:
    """Load all Instructions from memory_1.json"""
    data = load_json(filepath)
    return [item['Instruction'] for item in data]


def get_attack_instructions_by_id(filepath: str) -> Dict[str, str]:
    """Load case_id -> instruction mapping from attack_query_results.json"""
    data = load_json(filepath)
    return {item['case_id']: item['instruction'] for item in data['results']}


def get_trigger_instructions(filepath: str) -> List[Dict]:
    """Load all cases from trigger_query_results.json"""
    data = load_json(filepath)
    return data['results']


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


def evaluate_method(trigger_results: List[Dict],
                    attack_instructions: Dict[str, str],
                    memory_instructions: List[str],
                    method: str) -> Dict:
    """Evaluate all cases using specified method"""
    matches = 0
    total = len(trigger_results)
    details = []

    print(f"\n{'='*70}")
    print(f"Method: {method}")
    print(f"{'='*70}")

    for trigger_item in trigger_results:
        trigger_case_id = trigger_item['case_id']
        trigger_instruction = trigger_item['instruction']

        attack_instruction = attack_instructions.get(trigger_case_id, '')
        if not attack_instruction:
            print(f"Warning: case_id {trigger_case_id} not found in attack")
            continue

        source, max_similarity, source_info = compute_best_match(
            trigger_instruction, attack_instruction,
            memory_instructions, method
        )

        is_match = (source == 'attack')
        if is_match:
            matches += 1

        details.append({
            'trigger_case_id': trigger_case_id,
            'source': source,
            'similarity': max_similarity,
            'is_match': is_match
        })

        case_num = int(trigger_case_id)
        if case_num <= 10 or is_match:
            status = "✓ ATTACK MATCH" if is_match else "✗ MEMORY WIN"
            print(f"Case {trigger_case_id}: source={source}, sim={max_similarity:.4f} [{status}]")

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
    memory_file = (r"D:\rap-main\webshop\batch_attack\initail_memory_32\memory_1.json")
    attack_file = (r"D:\rap-main\webshop\promptarmor\webshop"
                   r"\attack_query_results.json")
    trigger_file = (r"D:\rap-main\webshop\promptarmor\webshop"
                    r"\trigger_query_results.json")

    print("Loading data...")
    memory_instructions = get_memory_instructions(memory_file)
    attack_instructions = get_attack_instructions_by_id(attack_file)
    trigger_results = get_trigger_instructions(trigger_file)

    print(f"Memory instructions: {len(memory_instructions)}")
    print(f"Attack instructions: {len(attack_instructions)}")
    print(f"Trigger cases: {len(trigger_results)}")

    methods = ['e5', 'all_minilm', 'levenshtein', 'string_matching', 'bm25']

    all_results = {}

    for method in methods:
        print(f"\n\nProcessing method: {method} ...")
        try:
            result = evaluate_method(
                trigger_results, attack_instructions,
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
                'total': len(trigger_results),
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

    output_file = (r"D:\rap-main\webshop\promptarmor\webshop"
                   r"\attack_vs_memory_evaluation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
