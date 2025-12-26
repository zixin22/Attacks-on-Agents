"""
Analyze query distribution in goals to identify which product categories
should be added to rules to increase rule violations
"""

import json
import os
from collections import Counter
from typing import Dict, List

def analyze_query_distribution():
    """Analyze which queries appear most frequently in goals"""
    
    # Load goals and queries
    goals_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "selected_reward_0.5", "extracted_goals.json"
    )
    query_map_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "WebShop-master", "baseline_models", "data", "goal_query_map.json"
    )
    
    with open(goals_file, 'r', encoding='utf-8') as f:
        goals = json.load(f)
    
    with open(query_map_file, 'r', encoding='utf-8') as f:
        query_map = json.load(f)
    
    # Map goal IDs to queries
    goal_to_query = {}
    for goal_id, instruction in goals.items():
        # Find query for this instruction
        queries = query_map.get(instruction, [])
        if queries:
            # Use the first query (main query)
            goal_to_query[goal_id] = queries[0].lower()
        else:
            goal_to_query[goal_id] = ""
    
    # Count query frequencies
    query_counts = Counter(goal_to_query.values())
    
    print("=" * 80)
    print("TOP 100 MOST FREQUENT QUERIES IN GOALS")
    print("=" * 80)
    for query, count in query_counts.most_common(100):
        if query:  # Skip empty queries
            print(f"{count:4d} | {query}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total goals: {len(goals)}")
    print(f"Total unique queries: {len([q for q in query_counts.keys() if q])}")
    
    return query_counts

if __name__ == "__main__":
    analyze_query_distribution()

