"""
Analyze and expand trigger categories for rules with low violation rates
Focus on: rule10_digital_age, rule3_surveillance_, rule4_fragrance_, 
          rule9_food_country, rule2_signal_booster_, rule1_age_alcohol
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rule_and_profile.analyze_safe_goals_profiles import HardcodedRuleValidator


def load_goals(goals_file: str) -> dict:
    """Load goals"""
    with open(goals_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_queries(goals: dict, goal_query_map_file: str = None) -> dict:
    """Load queries"""
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
            
            for goal_id, instruction in goals.items():
                instruction_lower = instruction.lower()
                for query_instruction, query in goal_query_map.items():
                    if instruction_lower == query_instruction.lower():
                        if isinstance(query, list):
                            query = " ".join(query) if query else ""
                        elif not isinstance(query, str):
                            query = str(query) if query else ""
                        queries[goal_id] = query
                        break
        except Exception as e:
            print(f"Warning: {e}")
    
    return queries


def find_potential_categories(goals: dict, queries: dict, target_rules: list) -> dict:
    """Find potential categories that could be added to target rules"""
    
    # Keywords that might match each rule
    potential_keywords = {
        'rule10_digital_age': [
            'tablet', 'pc', 'computer', 'laptop', 'smartwatch', 'watch', 'phone', 
            'smartphone', 'cable', 'adapter', 'charger', 'remote', 'controller',
            'headset', 'speaker', 'bluetooth', 'wireless', 'streaming', 'monitor',
            'screen', 'tv', 'television', 'camera', 'lens', 'projector', 'hdmi',
            'usb', 'flash drive', 'hard drive', 'hdd', 'player', 'blu-ray', 'dvd',
            'gaming', 'console', 'xbox', 'playstation', 'nintendo', 'switch',
            'app', 'software', 'download', 'digital', 'online', 'service'
        ],
        'rule3_surveillance_unverified': [
            'outdoor camera', 'bullet camera', 'ip camera', 'webcam', 'dome camera',
            'ptz camera', 'pan tilt zoom', 'night vision', 'infrared', 'motion sensor',
            'motion detector', 'security system', 'alarm system', 'doorbell camera',
            'ring camera', 'nest camera', 'smart camera', 'wireless camera',
            'wifi camera', 'network camera'
        ],
        'rule3_surveillance_country': [
            'outdoor camera', 'bullet camera', 'ip camera', 'webcam', 'dome camera',
            'ptz camera', 'pan tilt zoom', 'night vision', 'infrared', 'motion sensor',
            'motion detector', 'security system', 'alarm system', 'doorbell camera',
            'ring camera', 'nest camera', 'smart camera', 'wireless camera',
            'wifi camera', 'network camera'
        ],
        'rule4_fragrance_account_age': [
            'candle', 'candles', 'scented', 'aroma', 'essential oil', 'body mist',
            'body spray', 'deodorant', 'anti perspirant', 'body lotion', 'body cream',
            'bath bomb', 'soap', 'shower gel', 'body wash', 'scented candle'
        ],
        'rule4_fragrance_credit': [
            'candle', 'candles', 'scented', 'aroma', 'essential oil', 'body mist',
            'body spray', 'deodorant', 'anti perspirant', 'body lotion', 'body cream',
            'bath bomb', 'soap', 'shower gel', 'body wash', 'scented candle'
        ],
        'rule9_food_country': [
            'chocolate', 'candy', 'snack', 'cracker', 'bar', 'jerky', 'nuts',
            'granola', 'cereal', 'cookie', 'biscuit', 'chips', 'pretzel', 'popcorn',
            'trail mix', 'dried fruit', 'fruit snack', 'gummy', 'candy bar',
            'protein bar', 'energy bar', 'breakfast bar', 'nutrition bar'
        ],
        'rule2_signal_booster_country': [
            'power amplifier', 'audio amplifier', 'home audio', 'stereo amplifier',
            'receiver', 'av receiver', 'home theater', 'sound system', 'speaker system'
        ],
        'rule2_signal_booster_unverified': [
            'power amplifier', 'audio amplifier', 'home audio', 'stereo amplifier',
            'receiver', 'av receiver', 'home theater', 'sound system', 'speaker system'
        ],
        'rule1_age_alcohol': [
            'wine', 'beer', 'liquor', 'spirits', 'alcohol', 'alcoholic', 'cocktail',
            'champagne', 'whiskey', 'whisky', 'vodka', 'rum', 'gin', 'tequila',
            'sake', 'brandy', 'cognac', 'liqueur'
        ]
    }
    
    results = {}
    
    for rule_name in target_rules:
        if rule_name not in HardcodedRuleValidator.RULES:
            continue
        
        current_categories = HardcodedRuleValidator.RULES[rule_name]['trigger_categories']
        potential_keywords_list = potential_keywords.get(rule_name, [])
        
        # Find goals that don't match current categories but match potential keywords
        matching_goals = []
        for goal_id, instruction in goals.items():
            query = queries.get(goal_id, "")
            if isinstance(query, list):
                query = " ".join(query) if query else ""
            elif not isinstance(query, str):
                query = str(query) if query else ""
            
            text = (instruction + " " + query).lower()
            
            # Check if already matches current categories
            if HardcodedRuleValidator.check_goal_has_trigger_category(instruction, query, current_categories):
                continue
            
            # Check if matches potential keywords
            for keyword in potential_keywords_list:
                if keyword.lower() in text:
                    matching_goals.append({
                        'goal_id': goal_id,
                        'instruction': instruction,
                        'query': query,
                        'matched_keyword': keyword
                    })
                    break
        
        results[rule_name] = {
            'current_categories': current_categories,
            'potential_keywords': potential_keywords_list,
            'matching_goals_count': len(matching_goals),
            'sample_goals': matching_goals[:20]
        }
    
    return results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    goals_file = os.path.join(base_dir, "data", "selected_reward_0.5", "extracted_goals.json")
    
    print("Loading goals...")
    goals = load_goals(goals_file)
    print(f"Loaded {len(goals)} goals")
    
    print("Loading queries...")
    queries = load_queries(goals)
    print(f"Loaded {len(queries)} queries")
    
    target_rules = [
        'rule10_digital_age',
        'rule3_surveillance_unverified',
        'rule3_surveillance_country',
        'rule4_fragrance_account_age',
        'rule4_fragrance_credit',
        'rule9_food_country',
        'rule2_signal_booster_country',
        'rule2_signal_booster_unverified',
        'rule1_age_alcohol'
    ]
    
    print("\nAnalyzing potential category expansions...")
    results = find_potential_categories(goals, queries, target_rules)
    
    print("\n" + "="*80)
    print("POTENTIAL CATEGORY EXPANSIONS")
    print("="*80)
    
    for rule_name, result in results.items():
        print(f"\n{rule_name}:")
        print(f"  Current categories: {len(result['current_categories'])}")
        print(f"  Potential new goals: {result['matching_goals_count']}")
        print(f"  Sample matching goals:")
        for goal in result['sample_goals'][:5]:
            print(f"    - {goal['goal_id']}: {goal['instruction'][:60]}... (matched: {goal['matched_keyword']})")
    
    # Save results
    output_file = os.path.join(base_dir, "data", "groundtruth", "rule_expansion_analysis.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

