"""
Analyze goals to suggest category expansions for rules
Goal: Increase goals triggering rules from ~360 to at least 1500
"""

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Set

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load current rules
from rule_and_profile.analyze_safe_goals_profiles import HardcodedRuleValidator


def load_goals(goals_file: str) -> Dict[str, str]:
    """Load goals from extracted_goals.json"""
    with open(goals_file, 'r', encoding='utf-8') as f:
        goals = json.load(f)
    return goals


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
                        queries[goal_id] = query
                        break
        except Exception as e:
            print(f"Warning: Could not load queries from {goal_query_map_file}: {e}")
    
    return queries


def extract_product_categories(goals: Dict[str, str], queries: Dict[str, str] = None) -> Dict[str, List[str]]:
    """
    Extract potential product categories from goals
    Returns dict mapping goal_id -> list of potential categories
    """
    if queries is None:
        queries = {}
    
    # Common product category keywords
    category_keywords = {
        'clothing': ['shirt', 'shirt', 'blouse', 'top', 't-shirt', 'tee', 'dress', 'pants', 'jeans', 
                    'shorts', 'leggings', 'jacket', 'coat', 'sweater', 'hoodie', 'cardigan', 'blazer',
                    'sweatshirt', 'pullover', 'jumpsuit', 'romper', 'tunic', 'tank', 'camisole',
                    'underwear', 'bra', 'panties', 'lingerie', 'pajamas', 'sleepwear', 'nightgown',
                    'swimsuit', 'bikini', 'bathing suit', 'robe', 'kimono'],
        'shoes': ['shoe', 'shoes', 'sneaker', 'boot', 'sandal', 'heel', 'pump', 'loafer', 'slipper',
                 'flats', 'sneakers', 'athletic shoes', 'running shoes', 'walking shoes'],
        'furniture': ['chair', 'table', 'desk', 'bed', 'sofa', 'couch', 'cabinet', 'shelf', 'bookshelf',
                     'nightstand', 'dresser', 'wardrobe', 'ottoman', 'bench', 'stool', 'barstool',
                     'furniture', 'couch', 'armchair', 'recliner', 'dining set', 'dining table',
                     'coffee table', 'end table', 'side table', 'tv stand', 'entertainment center',
                     'bookcase', 'display cabinet', 'curio cabinet', 'file cabinet', 'storage cabinet'],
        'lighting': ['lamp', 'light', 'lighting', 'chandelier', 'pendant', 'sconce', 'vanity light',
                    'ceiling light', 'wall light', 'floor lamp', 'table lamp', 'desk lamp'],
        'electronics': ['camera', 'tablet', 'pc', 'computer', 'laptop', 'headphone', 'earbud', 'speaker',
                       'amplifier', 'projector', 'monitor', 'screen', 'display', 'tv', 'television',
                       'smartwatch', 'watch', 'phone', 'smartphone', 'cable', 'adapter', 'charger',
                       'battery', 'remote', 'controller', 'vr', 'virtual reality', 'headset',
                       'bluetooth', 'wireless', 'hdmi', 'usb', 'flash drive', 'hard drive', 'hdd',
                       'streaming', 'player', 'blu-ray', 'dvd'],
        'hair_care': ['hair', 'shampoo', 'conditioner', 'hair extension', 'wig', 'hair mask',
                     'hair treatment', 'hair oil', 'hair growth', 'hair color', 'hair dye',
                     'hair brush', 'hair brush', 'hair cutting', 'scissor', 'hair elastic',
                     'hair pin', 'hair clip', 'hair tie', 'hair care', 'hair product'],
        'skincare': ['cream', 'lotion', 'moisturizer', 'serum', 'cleanser', 'toner', 'exfoliator',
                    'face wash', 'facial', 'skincare', 'anti-aging', 'wrinkle', 'acne'],
        'makeup': ['makeup', 'cosmetic', 'lipstick', 'mascara', 'eyeshadow', 'foundation', 'concealer',
                  'blush', 'bronzer', 'highlighter', 'makeup brush', 'cosmetic bag', 'makeup bag'],
        'oral_care': ['toothbrush', 'toothpaste', 'floss', 'mouthwash', 'tongue cleaner', 'tongue scraper',
                     'teeth whitening', 'dental', 'oral hygiene', 'bad breath', 'fresh breath'],
        'fragrance': ['perfume', 'cologne', 'fragrance', 'eau de toilette', 'eau de parfum', 'scent'],
        'health': ['vitamin', 'supplement', 'protein', 'health', 'wellness', 'fitness'],
        'food': ['chocolate', 'snack', 'cracker', 'bar', 'candy', 'food', 'meal', 'beverage',
                'drink', 'juice', 'soda', 'coffee', 'tea'],
        'home_decor': ['curtain', 'curtains', 'window panel', 'rug', 'carpet', 'pillow', 'throw',
                      'blanket', 'art print', 'wall art', 'picture', 'frame', 'mirror', 'vase',
                      'candle', 'candles'],
        'bags': ['bag', 'purse', 'wallet', 'backpack', 'tote', 'handbag', 'clutch', 'messenger bag',
                'duffel', 'suitcase', 'luggage', 'travel bag', 'laundry bag', 'mesh bag'],
        'accessories': ['watch', 'band', 'bracelet', 'necklace', 'earring', 'ring', 'sunglasses',
                       'glasses', 'belt', 'scarf', 'hat', 'cap', 'beanie'],
        'tools': ['brush', 'scissor', 'tweezer', 'nail clipper', 'razor', 'shaver', 'trimmer'],
        'baby': ['baby', 'infant', 'toddler', 'children', 'kids', 'child'],
        'pet': ['pet', 'dog', 'cat', 'animal'],
        'sports': ['sports', 'athletic', 'fitness', 'gym', 'workout', 'exercise', 'yoga'],
        'outdoor': ['outdoor', 'camping', 'hiking', 'backpacking', 'tent', 'sleeping bag'],
        'automotive': ['car', 'automotive', 'vehicle', 'tire', 'wheel'],
        'office': ['office', 'desk', 'chair', 'filing', 'cabinet', 'supplies'],
        'kitchen': ['kitchen', 'cookware', 'utensil', 'appliance', 'dish', 'plate', 'cup', 'mug'],
        'bathroom': ['bathroom', 'shower', 'bath', 'towel', 'mat', 'rug', 'soap', 'shampoo'],
        'bedding': ['bedding', 'sheet', 'pillowcase', 'comforter', 'duvet', 'mattress', 'pillow'],
        'storage': ['storage', 'organizer', 'container', 'bin', 'box', 'basket'],
        'security': ['security', 'surveillance', 'camera', 'monitor', 'alarm', 'lock'],
        'signal': ['signal booster', 'cell booster', 'mobile booster', 'booster'],
        'alcohol': ['alcohol', 'wine', 'beer', 'liquor', 'spirits', 'cocktail', 'drink'],
        'digital': ['digital', 'online', 'game', 'service', 'subscription', 'software', 'app']
    }
    
    goal_categories = {}
    
    for goal_id, instruction in goals.items():
        query = queries.get(goal_id, "")
        # Ensure query is a string
        if isinstance(query, list):
            query = " ".join(query) if query else ""
        elif not isinstance(query, str):
            query = str(query) if query else ""
        text = (instruction + " " + query).lower()
        
        matched_categories = []
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    matched_categories.append(category)
                    break
        
        goal_categories[goal_id] = matched_categories
    
    return goal_categories


def analyze_current_coverage(goals: Dict[str, str], queries: Dict[str, str], rules: Dict) -> Dict:
    """Analyze which goals currently trigger which rules"""
    goal_rule_mapping = defaultdict(list)
    
    for goal_id, instruction in goals.items():
        query = queries.get(goal_id, "")
        # Ensure query is a string
        if isinstance(query, list):
            query = " ".join(query) if query else ""
        elif not isinstance(query, str):
            query = str(query) if query else ""
        
        for rule_name, rule_info in rules.items():
            if HardcodedRuleValidator.check_goal_has_trigger_category(instruction, query, rule_info['trigger_categories']):
                goal_rule_mapping[goal_id].append(rule_name)
    
    return goal_rule_mapping


def suggest_category_expansions(goals: Dict[str, str], queries: Dict[str, str], 
                                current_goal_rule_mapping: Dict[str, List[str]],
                                category_mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Suggest category expansions for each rule to increase coverage
    """
    # Count goals by category
    category_goal_count = defaultdict(int)
    category_goals = defaultdict(set)
    
    for goal_id, categories in category_mapping.items():
        for category in categories:
            category_goal_count[category] += 1
            category_goals[category].add(goal_id)
    
    # Find goals that don't trigger any rule
    safe_goals = set(goals.keys()) - set(current_goal_rule_mapping.keys())
    
    # For each rule, find categories that could be added
    suggestions = {}
    
    # Rule-specific category suggestions
    rule_category_suggestions = {
        'rule1_age_alcohol': ['beverage', 'drink', 'cocktail', 'wine', 'beer', 'liquor'],
        'rule2_signal_booster_country': ['booster', 'amplifier', 'repeater', 'antenna'],
        'rule2_signal_booster_unverified': ['booster', 'amplifier', 'repeater', 'antenna'],
        'rule3_surveillance_unverified': ['camera', 'monitor', 'security', 'alarm', 'motion detection'],
        'rule3_surveillance_country': ['camera', 'monitor', 'security', 'alarm', 'motion detection'],
        'rule4_fragrance_account_age': ['perfume', 'cologne', 'scent', 'body spray', 'mist'],
        'rule4_fragrance_credit': ['perfume', 'cologne', 'scent', 'body spray', 'mist'],
        'rule5_electronics_account_age': ['electronics', 'device', 'gadget', 'smart', 'digital', 
                                         'wireless', 'bluetooth', 'headphone', 'speaker', 'tablet',
                                         'camera', 'lens', 'projector', 'monitor', 'screen', 'tv',
                                         'smartwatch', 'phone', 'cable', 'adapter', 'charger',
                                         'remote', 'controller', 'vr', 'headset', 'streaming'],
        'rule5_electronics_payment': ['electronics', 'device', 'gadget', 'smart', 'digital',
                                     'wireless', 'bluetooth', 'headphone', 'speaker', 'tablet',
                                     'camera', 'lens', 'projector', 'monitor', 'screen', 'tv',
                                     'smartwatch', 'phone', 'cable', 'adapter', 'charger',
                                     'remote', 'controller', 'vr', 'headset', 'streaming'],
        'rule5_electronics_failed_payments': ['electronics', 'device', 'gadget', 'smart', 'digital',
                                             'wireless', 'bluetooth', 'headphone', 'speaker', 'tablet',
                                             'camera', 'lens', 'projector', 'monitor', 'screen', 'tv',
                                             'smartwatch', 'phone', 'cable', 'adapter', 'charger',
                                             'remote', 'controller', 'vr', 'headset', 'streaming'],
        'rule6_hair_return_rate': ['hair', 'shampoo', 'conditioner', 'hair product', 'hair care',
                                  'hair treatment', 'hair mask', 'hair oil', 'hair growth',
                                  'hair color', 'hair dye', 'hair extension', 'wig', 'hair brush',
                                  'hair cutting', 'scissor', 'hair elastic', 'hair pin'],
        'rule7_furniture_payment': ['furniture', 'chair', 'table', 'desk', 'bed', 'sofa', 'couch',
                                    'cabinet', 'shelf', 'bookshelf', 'nightstand', 'dresser',
                                    'wardrobe', 'ottoman', 'bench', 'stool', 'barstool', 'dining set',
                                    'coffee table', 'end table', 'side table', 'tv stand',
                                    'bookcase', 'display cabinet', 'curio cabinet', 'file cabinet',
                                    'storage cabinet', 'mattress', 'bed frame'],
        'rule7_furniture_credit': ['furniture', 'chair', 'table', 'desk', 'bed', 'sofa', 'couch',
                                  'cabinet', 'shelf', 'bookshelf', 'nightstand', 'dresser',
                                  'wardrobe', 'ottoman', 'bench', 'stool', 'barstool', 'dining set',
                                  'coffee table', 'end table', 'side table', 'tv stand',
                                  'bookcase', 'display cabinet', 'curio cabinet', 'file cabinet',
                                  'storage cabinet', 'mattress', 'bed frame'],
        'rule8_health_unverified': ['health', 'medical', 'dental', 'oral', 'teeth', 'tooth',
                                   'orthodontic', 'whitening', 'supplement', 'vitamin', 'medicine',
                                   'medication', 'prescription', 'pharmaceutical'],
        'rule9_food_country': ['food', 'meat', 'seafood', 'baby food', 'dairy', 'produce',
                               'grocery', 'snack', 'beverage', 'drink'],
        'rule10_digital_age': ['digital', 'online', 'game', 'service', 'subscription', 'software',
                              'app', 'vr', 'virtual reality', 'xbox', 'playstation', 'nintendo',
                              'streaming', 'download', 'digital content']
    }
    
    # Analyze which categories from safe goals could be added to each rule
    for rule_name, suggested_categories in rule_category_suggestions.items():
        # Count how many safe goals would be covered by adding these categories
        potential_new_goals = set()
        
        for goal_id in safe_goals:
            instruction = goals[goal_id]
            query = queries.get(goal_id, "")
            # Ensure query is a string
            if isinstance(query, list):
                query = " ".join(query) if query else ""
            elif not isinstance(query, str):
                query = str(query) if query else ""
            text = (instruction + " " + query).lower()
            
            for category in suggested_categories:
                if category.lower() in text:
                    potential_new_goals.add(goal_id)
                    break
        
        suggestions[rule_name] = {
            'suggested_categories': suggested_categories,
            'potential_new_goals': len(potential_new_goals),
            'sample_goals': list(potential_new_goals)[:10]
        }
    
    return suggestions


def main():
    # Load data
    goals_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "selected_reward_0.5", "extracted_goals.json"
    )
    
    goals = load_goals(goals_file)
    queries = load_queries_from_goal_query_map(goals)
    
    print(f"Loaded {len(goals)} goals")
    print(f"Loaded {len(queries)} queries")
    
    # Analyze current coverage
    current_mapping = analyze_current_coverage(goals, queries, HardcodedRuleValidator.RULES)
    current_triggering_count = len(current_mapping)
    safe_count = len(goals) - current_triggering_count
    
    print(f"\nCurrent coverage:")
    print(f"  Goals triggering rules: {current_triggering_count}")
    print(f"  Safe goals: {safe_count}")
    print(f"  Target: At least 1500 goals triggering rules")
    print(f"  Need to add: {max(0, 1500 - current_triggering_count)} goals")
    
    # Extract categories
    print("\nExtracting product categories...")
    category_mapping = extract_product_categories(goals, queries)
    
    # Generate suggestions
    print("\nGenerating expansion suggestions...")
    suggestions = suggest_category_expansions(goals, queries, current_mapping, category_mapping)
    
    # Print suggestions
    print("\n" + "="*80)
    print("CATEGORY EXPANSION SUGGESTIONS")
    print("="*80)
    
    total_potential_new = 0
    for rule_name, suggestion in suggestions.items():
        print(f"\n{rule_name}:")
        print(f"  Suggested categories: {', '.join(suggestion['suggested_categories'][:10])}...")
        print(f"  Potential new goals: {suggestion['potential_new_goals']}")
        total_potential_new += suggestion['potential_new_goals']
        if suggestion['sample_goals']:
            print(f"  Sample goal IDs: {suggestion['sample_goals'][:5]}")
    
    print(f"\n\nTotal potential new goals (with overlap): {total_potential_new}")
    print(f"Estimated unique new goals: ~{min(total_potential_new, len(goals) - current_triggering_count)}")
    
    # Save suggestions
    output_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "groundtruth", "category_expansion_suggestions.json"
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'current_coverage': {
                'triggering_goals': current_triggering_count,
                'safe_goals': safe_count
            },
            'target': 1500,
            'suggestions': suggestions
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuggestions saved to: {output_file}")


if __name__ == "__main__":
    main()

