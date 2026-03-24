import sys
sys.path.insert(0, r"D:\rap-main\webshop")

import json
from typing import List, Dict
from text_similarity import bm25_similarity

def get_memory_instructions(file_path: str) -> List[str]:
    """Extract confirmed_task from memory JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item.get('confirmed_task', '') for item in data]

def get_attack_instructions(file_path: str) -> List[str]:
    """Extract confirmed_task from attack JSON (spltting half)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item.get('confirmed_task', '') for item in data]

def get_carrier_instructions(file_path: str) -> List[str]:
    """Extract confirmed_task from carrier JSON (insert fragment half)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item.get('confirmed_task', '') for item in data]

def evaluate_bm25(
    memory_instructions: List[str],
    carrier_instructions: List[str],
    attack_instructions: List[str]
) -> Dict:
    """
    Evaluate BM25 similarity between carrier and (memory + attack)
    """
    total = len(carrier_instructions)
    matches = 0
    details = []

    print(f"\nEvaluating BM25 for {total} cases...")
    print("-" * 60)

    for idx, carrier in enumerate(carrier_instructions):
        case_id = f"case_{idx + 1}"
        attack = attack_instructions[idx]

        # Compute BM25 scores
        all_docs = memory_instructions + [attack]
        scores = bm25_similarity(carrier, all_docs)

        attack_score = float(scores[-1])
        memory_scores = scores[:-1]
        memory_max = float(memory_scores.max()) if len(memory_scores) > 0 else 0.0

        # Determine source
        is_match = attack_score >= memory_max
        source = 'attack' if is_match else 'memory'

        if is_match:
            matches += 1

        # Print all cases
        status = "ATTACK MATCH" if is_match else "MEMORY WIN"
        print(f"Case {case_id}: source={source}, attack_score={attack_score:.4f}, memory_max={memory_max:.4f} [{status}]")

        details.append({
            'task_id': case_id,
            'source': source,
            'attack_score': attack_score,
            'memory_max': memory_max,
            'is_match': is_match
        })

    match_rate = matches / total if total > 0 else 0

    print(f"\nStats:")
    print(f"  Total: {total}")
    print(f"  Matches (attack win): {matches}")
    print(f"  Match rate: {match_rate:.2%}")

    return {
        'method': 'bm25',
        'total': total,
        'matches': matches,
        'match_rate': match_rate,
        'details': details
    }

def main():
    # File paths
    memory_file = r"D:\rap-main\webshop\promptarmor\seeact\32memory.json"
    attack_file = r"D:\rap-main\webshop\promptarmor\seeact\1-splitted_half.json"
    carrier_file = r"D:\rap-main\webshop\promptarmor\seeact\1-splitted_insert_fragment_half.json"
    output_file = r"D:\rap-main\webshop\promptarmor\seeact\seeact_evaluation_results.json"

    print("Loading data...")
    memory_instructions = get_memory_instructions(memory_file)
    attack_instructions = get_attack_instructions(attack_file)
    carrier_instructions = get_carrier_instructions(carrier_file)

    print(f"Memory instructions: {len(memory_instructions)}")
    print(f"Carrier cases: {len(carrier_instructions)}")
    print(f"Attack cases: {len(attack_instructions)}")

    # Evaluate BM25
    results = evaluate_bm25(
        memory_instructions,
        carrier_instructions,
        attack_instructions
    )

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()






