"""
Generate random profile-goal pairs and check violation rates using hardcoded rules
"""

import json
import os
import sys
import random
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rule_and_profile.user_profile import UserProfile
from rule_and_profile.analyze_safe_goals_profiles import HardcodedRuleValidator


def load_profiles(profiles_file: str, limit: int = 2000) -> List[UserProfile]:
    """Load user profiles from generated_profiles.json"""
    with open(profiles_file, 'r', encoding='utf-8') as f:
        profiles_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(profiles_data, dict):
        if 'profiles' in profiles_data:
            profiles_list = profiles_data['profiles']
        else:
            profiles_list = list(profiles_data.values())
    else:
        profiles_list = profiles_data
    
    profiles = []
    for i, profile_dict in enumerate(profiles_list[:limit]):
        profile = UserProfile(
            profile_id=f"profile_{i}",
            age=profile_dict.get('age', 25),
            country=profile_dict.get('country', 'allowed'),
            is_verified=profile_dict.get('is_verified', True),
            payment_method=profile_dict.get('payment_method', 'Visa'),
            failed_payment_attempts=profile_dict.get('failed_payment_attempts', 0),
            credit_score=profile_dict.get('credit_score', 700),
            account_age_days=profile_dict.get('account_age_days', 365),
            account_status=profile_dict.get('account_status', 'active'),
            return_rate=profile_dict.get('return_rate', 0.0),
            total_purchase_amount=profile_dict.get('total_purchase_amount', 0.0)
        )
        profiles.append(profile)
    
    return profiles


def load_goals(goals_file: str, limit: int = 2000) -> Dict[str, str]:
    """Load goals from extracted_goals.json"""
    with open(goals_file, 'r', encoding='utf-8') as f:
        goals = json.load(f)

    # Get first limit goals (sorted by key as integer)
    goal_items = sorted(goals.items(), key=lambda x: int(x[0]))[:limit]
    return dict(goal_items)


def load_queries_from_goal_query_map(goals: Dict[str, str], goal_query_map_file: str = None) -> Dict[str, str]:
    """
    Load queries from goal_query_map.json
    Returns dict mapping goal_id -> query
    """
    queries = {}
    
    if goal_query_map_file is None:
        goal_query_map_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "WebShop-master", "baseline_models", "data", "goal_query_map.json"
        )
    
    if os.path.exists(goal_query_map_file):
        try:
            with open(goal_query_map_file, 'r', encoding='utf-8') as f:
                goal_query_map = json.load(f)
            
            # Match goals by instruction text
            for goal_id, instruction in goals.items():
                instruction_lower = instruction.lower()
                for query_instruction, query in goal_query_map.items():
                    if instruction_lower == query_instruction.lower():
                        # Ensure query is a string
                        if isinstance(query, list):
                            query = " ".join(query) if query else ""
                        elif not isinstance(query, str):
                            query = str(query) if query else ""
                        queries[goal_id] = query
                        break
        except Exception as e:
            print(f"Warning: Could not load queries from {goal_query_map_file}: {e}")
    
    return queries


def check_violations(profile: UserProfile, instruction: str, query: str) -> List[str]:
    """
    Check which rules are violated for a profile-goal pair using hardcoded logic
    Returns list of violated rule names
    """
    violated_rules = []
    
    for rule_name, rule_info in HardcodedRuleValidator.RULES.items():
        # Check if goal triggers this rule
        if HardcodedRuleValidator.check_goal_has_trigger_category(instruction, query, rule_info['trigger_categories']):
            # Check if profile violates the condition
            if rule_info['condition'](profile):
                violated_rules.append(rule_name)
    
    return violated_rules


def generate_random_pairs(profiles: List[UserProfile], goals: Dict[str, str], 
                          queries: Dict[str, str], seed: int = 42) -> List[Dict]:
    """
    Generate random profile-goal pairs
    """
    random.seed(seed)
    
    pairs = []
    goal_ids = list(goals.keys())
    
    for i, profile in enumerate(profiles):
        # Randomly select a goal
        goal_id = random.choice(goal_ids)
        instruction = goals[goal_id]
        query = queries.get(goal_id, "")
        
        # Check violations
        violated_rules = check_violations(profile, instruction, query)
        
        pair = {
            'pair_id': i,
            'profile_id': profile.profile_id,
            'goal_id': goal_id,
            'instruction': instruction,
            'query': query,
            'violated_rules': violated_rules,
            'has_violation': len(violated_rules) > 0,
            'profile': profile.to_dict()
        }
        
        pairs.append(pair)
    
    return pairs


def analyze_violation_statistics(pairs: List[Dict]) -> Dict:
    """Analyze violation statistics"""
    total_pairs = len(pairs)
    pairs_with_violations = sum(1 for p in pairs if p['has_violation'])
    violation_rate = pairs_with_violations / total_pairs if total_pairs > 0 else 0
    
    # Count violations by rule
    violations_by_rule = defaultdict(int)
    for pair in pairs:
        for rule in pair['violated_rules']:
            violations_by_rule[rule] += 1
    
    # Count pairs with multiple violations
    pairs_with_multiple = sum(1 for p in pairs if len(p['violated_rules']) > 1)
    
    return {
        'total_pairs': total_pairs,
        'pairs_with_violations': pairs_with_violations,
        'pairs_without_violations': total_pairs - pairs_with_violations,
        'violation_rate': violation_rate,
        'violation_percentage': violation_rate * 100,
        'pairs_with_multiple_violations': pairs_with_multiple,
        'violations_by_rule': dict(violations_by_rule),
        'average_violations_per_pair': sum(len(p['violated_rules']) for p in pairs) / total_pairs if total_pairs > 0 else 0
    }


def main():
    print("="*80)
    print("Profile-Goal Pair Generation and Violation Analysis")
    print("="*80)
    
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Try generated_profiles_2000.json first, fallback to generated_profiles.json
    profiles_file_2000 = os.path.join(base_dir, "generated_profiles_2000.json")
    profiles_file = os.path.join(base_dir, "generated_profiles.json")
    if os.path.exists(profiles_file_2000):
        profiles_file = profiles_file_2000
    goals_file = os.path.join(base_dir, "data", "selected_reward_0.5", "extracted_goals.json")
    output_file = os.path.join(base_dir, "data", "groundtruth", "profile_goal_pairs_2000.json")
    
    # Load data
    print("\n1. Loading profiles...")
    profiles = load_profiles(profiles_file, limit=2000)
    print(f"   Loaded {len(profiles)} profiles")
    
    print("\n2. Loading goals...")
    goals = load_goals(goals_file, limit=2000)
    print(f"   Loaded {len(goals)} goals")
    
    print("\n3. Loading queries...")
    queries = load_queries_from_goal_query_map(goals)
    print(f"   Loaded {len(queries)} queries")
    
    # Generate random pairs
    print("\n4. Generating random profile-goal pairs...")
    pairs = generate_random_pairs(profiles, goals, queries, seed=42)
    print(f"   Generated {len(pairs)} pairs")
    
    # Analyze violations
    print("\n5. Analyzing violations...")
    stats = analyze_violation_statistics(pairs)
    
    print("\n" + "="*80)
    print("VIOLATION STATISTICS")
    print("="*80)
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Pairs with violations: {stats['pairs_with_violations']}")
    print(f"Pairs without violations: {stats['pairs_without_violations']}")
    print(f"Violation rate: {stats['violation_rate']:.4f} ({stats['violation_percentage']:.2f}%)")
    print(f"Pairs with multiple violations: {stats['pairs_with_multiple_violations']}")
    print(f"Average violations per pair: {stats['average_violations_per_pair']:.2f}")
    
    print("\nViolations by rule:")
    for rule, count in sorted(stats['violations_by_rule'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {rule}: {count}")
    
    # Save results
    print(f"\n6. Saving results to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    result = {
        'metadata': {
            'num_profiles': len(profiles),
            'num_goals': len(goals),
            'num_pairs': len(pairs),
            'random_seed': 42
        },
        'statistics': stats,
        'pairs': pairs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"   Results saved successfully!")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

