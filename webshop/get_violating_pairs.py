#!/usr/bin/env python3
"""
Script to extract all pair_ids that violate the four specified rules from profile_goal_pairs_2000.json

Target rules:
- rule7_credit_restriction_furniture
- rule1_age_minor_restriction
- rule6_return_rate_restriction_hair
- rule4_credit_restriction_fragrance
"""

import json
import os

def get_violating_pairs(json_file_path):
    """
    Extract all pair_ids that violate any of the four target rules.

    Args:
        json_file_path: Path to the profile_goal_pairs_2000.json file

    Returns:
        dict: Dictionary with rule names as keys and list of pair_ids as values
    """

    # Target rules to check for
    target_rules = {
        'rule7_credit_restriction_furniture',
        'rule1_age_minor_restriction',
        'rule6_return_rate_restriction_hair',
        'rule4_credit_restriction_fragrance'
    }

    # Dictionary to store violating pairs for each rule
    violating_pairs = {
        'rule7_credit_restriction_furniture': [],
        'rule1_age_minor_restriction': [],
        'rule6_return_rate_restriction_hair': [],
        'rule4_credit_restriction_fragrance': [],
        'any_rule': []  # Pairs that violate any of the four rules
    }

    print(f"Loading {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data['pairs']
    total_pairs = len(pairs)

    print(f"Processing {total_pairs} pairs...")

    for i, pair in enumerate(pairs):
        if i % 200 == 0:  # Progress indicator
            print(f"Processed {i}/{total_pairs} pairs...")

        pair_id = pair['pair_id']
        violated_rules = pair.get('violated_rules', [])

        # Check if this pair violates any of our target rules
        violates_any_target = False

        for rule in violated_rules:
            if rule in target_rules:
                violating_pairs[rule].append(pair_id)
                violates_any_target = True

        if violates_any_target:
            violating_pairs['any_rule'].append(pair_id)

    print(f"Completed processing {total_pairs} pairs.")
    return violating_pairs

def main():
    """Main function"""
    json_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\groundtruth\profile_goal_pairs_2000.json"
    output_file = r"C:\Users\22749\Desktop\rap-main\webshop\violating_pairs_analysis.json"

    # Get violating pairs
    violating_pairs = get_violating_pairs(json_file)

    # Print summary
    print("\n" + "="*80)
    print("VIOLATING PAIRS ANALYSIS SUMMARY")
    print("="*80)

    for rule, pair_ids in violating_pairs.items():
        if rule != 'any_rule':
            print("20")

    print(f"\nTotal pairs violating any of the four rules: {len(violating_pairs['any_rule'])}")

    # Print some sample pair_ids for each rule
    print("\nSAMPLE PAIR_IDS FOR EACH RULE:")
    for rule, pair_ids in violating_pairs.items():
        if rule != 'any_rule' and pair_ids:
            sample_ids = pair_ids[:10]  # Show first 10
            remaining = len(pair_ids) - 10
            print("15"
                  f"{' and ' + str(remaining) + ' more' if remaining > 0 else ''}")

    # Save detailed results to JSON file
    print(f"\nSaving detailed results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(violating_pairs, f, indent=2, ensure_ascii=False)

    print("✅ Analysis complete!")
    print(f"📄 Detailed results saved to: {output_file}")

    # Also save just the list of all violating pair_ids for easy reference
    simple_output_file = r"C:\Users\22749\Desktop\rap-main\webshop\violating_pair_ids.txt"
    print(f"📝 Saving simple pair_id list to: {simple_output_file}")
    with open(simple_output_file, 'w', encoding='utf-8') as f:
        f.write("# Pair IDs that violate any of the four target rules\n")
        f.write("# Rules: rule7_credit_restriction_furniture, rule1_age_minor_restriction, rule6_return_rate_restriction_hair, rule4_credit_restriction_fragrance\n")
        f.write("# Total: " + str(len(violating_pairs['any_rule'])) + " pairs\n\n")
        for pair_id in sorted(violating_pairs['any_rule']):
            f.write(f"{pair_id}\n")

    print("✅ All files saved successfully!")

if __name__ == "__main__":
    main()
