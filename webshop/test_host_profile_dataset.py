#!/usr/bin/env python3
"""
Test script to validate host-profile pairs using RuleCheckerDetailed
"""

import json
import os
import sys
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def test_with_rule_checker_detailed(dataset_file: str):
    """Test all pairs in the dataset with RuleCheckerDetailed"""

    print("Loading dataset...")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    pairs = dataset['pairs']
    print(f"Loaded {len(pairs)} pairs to test")

    # Import RuleCheckerDetailed
    try:
        from main import RuleCheckerDetailed
        from rule_and_profile import UserProfile
    except ImportError as e:
        print(f"Import error: {e}")
        return

    # Initialize RuleChecker
    print("Initializing RuleCheckerDetailed...")
    rule_checker = RuleCheckerDetailed(verbose=False, model="gpt-4o")

    results = []
    valid_count = 0
    invalid_count = 0

    print("\nTesting pairs...")
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"Testing pair {i}/{len(pairs)}...")

        # Create UserProfile from profile dict
        profile_dict = pair['profile']
        profile = UserProfile(
            profile_id=pair['profile_id'],
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

        instruction = pair['instruction']

        # Test with RuleCheckerDetailed
        try:
            is_valid, violated_rules = rule_checker.check_all_rules(profile, instruction, "")

            result = {
                'pair_id': pair['pair_id'],
                'original_instruction_pair_id': pair['original_instruction_pair_id'],
                'instruction': instruction,
                'profile_id': pair['profile_id'],
                'is_valid': is_valid,
                'violated_rules': violated_rules,
                'original_violated_rules': pair.get('original_violated_rules', []),
                'original_has_violation': pair.get('original_has_violation', False)
            }
            results.append(result)

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

        except Exception as e:
            print(f"Error testing pair {pair['pair_id']}: {e}")
            result = {
                'pair_id': pair['pair_id'],
                'instruction': instruction,
                'profile_id': pair['profile_id'],
                'error': str(e)
            }
            results.append(result)

    # Save results
    output_file = dataset_file.replace('.json', '_validation_results.json')
    output_data = {
        'metadata': {
            'source_dataset': dataset_file,
            'total_pairs_tested': len(pairs),
            'valid_pairs': valid_count,
            'invalid_pairs': invalid_count,
            'validation_rate': valid_count / len(pairs) if pairs else 0
        },
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nValidation completed!")
    print(f"Total pairs tested: {len(pairs)}")
    print(f"Valid pairs: {valid_count}")
    print(f"Invalid pairs: {invalid_count}")
    print(f"Validation rate: {valid_count / len(pairs) * 100:.2f}%")
    print(f"Results saved to: {output_file}")

    return output_data

if __name__ == "__main__":
    # Set random seed for reproducibility
    import random
    random.seed(42)

    dataset_file = "host-profile-dataset.json"
    if os.path.exists(dataset_file):
        test_with_rule_checker_detailed(dataset_file)
    else:
        print(f"Dataset file not found: {dataset_file}")
        print("Please run create_host_profile_dataset.py first")
