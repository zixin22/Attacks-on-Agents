#!/usr/bin/env python3
"""
Script to process simple_attack_dataset_100_goals.json:
1. Read all original_instruction
2. Use NER method to decompose each instruction into fragments
3. Use MASK method to detect sensitive fragments
4. Save details to fragment_100.json
5. Print processing progress in terminal
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from attack.attack import FragmentAttackGenerator
from attack.mask_check import MaskChecker
from rule_and_profile.rule_checker import RuleChecker
from rule_and_profile.user_profile import UserProfile


def log_rule_checker_call(log_file, pair_id, instruction_idx, check_type, prompt, response, result):
    """Log RuleChecker call details"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n================================================================================\n")
        f.write(f"RuleChecker Call - {timestamp}\n")
        f.write(f"================================================================================\n")
        f.write(f"Pair ID: {pair_id}\n")
        f.write(f"Instruction Index: {instruction_idx}\n")
        f.write(f"Check Type: {check_type}\n")
        f.write(f"\nRuleChecker Prompt:\n")
        f.write(f"============================================================\n")
        f.write(f"{prompt}\n")
        f.write(f"============================================================\n")
        f.write(f"\nRuleChecker Response:\n")
        f.write(f"============================================================\n")
        f.write(f"{response}\n")
        f.write(f"============================================================\n")
        f.write(f"\nResult: {result}\n")
        f.write(f"================================================================================\n\n")


def process_instructions():
    """Main function to process all instructions"""

    # File paths
    input_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\groundtruth\profile_goal_pairs_2000.json"
    output_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\fragment_100_3.json"
    log_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\rulechecker_mask_log_3.txt"

    # Initialize log file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"================================================================================\n")
        f.write(f"RuleChecker MASK Detection Log\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Input File: {input_file} (first 100 pairs)\n")
        f.write(f"Profile: Dynamic profiles from pairs\n")
        f.write(f"================================================================================\n\n")

    # Initialize components
    print("Initializing components...")

    # Initialize rule checker
    rule_checker = RuleChecker(verbose=False, model="gpt-4o")
    print(f"DEBUG: RuleChecker class: {rule_checker.__class__}")
    print(f"DEBUG: RuleChecker module: {rule_checker.__class__.__module__}")

    # Initialize attack generator with NER capability
    attack_generator = FragmentAttackGenerator(verbose=False)

    # Load input data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract first 100 pairs
    pairs_data = data["pairs"][:]
    results = []
    total_instructions = len(pairs_data)

    print(f"Processing {total_instructions} pairs...")

    # For testing, only process first 5 pairs
    pairs_data = pairs_data[:100]
    total_instructions = len(pairs_data)

    for idx, pair in enumerate(pairs_data, 1):
        original_instruction = pair['instruction']
        pair_id = pair['pair_id']
        profile_data = pair['profile']

        # Create UserProfile from pair data
        profile = UserProfile(
            profile_id=profile_data['profile_id'],
            age=profile_data['age'],
            country=profile_data['country'],
            is_verified=profile_data['is_verified'],
            payment_method=profile_data['payment_method'],
            failed_payment_attempts=profile_data['failed_payment_attempts'],
            credit_score=profile_data['credit_score'],
            account_age_days=profile_data['account_age_days'],
            account_status=profile_data['account_status'],
            return_rate=profile_data['return_rate'],
            total_purchase_amount=profile_data['total_purchase_amount']
        )

        print(f"\n[{idx}/{total_instructions}] Processing pair_id: {pair_id} (profile: {profile.profile_id}):")
        print(f"  Original: {original_instruction}")
        print(f"  Profile: age={profile.age}, verified={profile.is_verified}, credit={profile.credit_score}, country={profile.country}")

        try:
            # Step 1: Use NER to decompose instruction into fragments
            print("  Step 1: NER decomposition...")
            fragments = attack_generator.decompose_target_instruction_ner(original_instruction)

            print("  Fragments:")
            for i, fragment in enumerate(fragments, 1):
                print(f"    F{i}: '{fragment}'")

            # Step 2: Use MASK to detect sensitive fragments
            print("  Step 2: MASK detection...")
            mask_checker = MaskChecker(
                rule_checker=rule_checker,
                profile=profile,
                host_instruction=original_instruction,
                query="",
                verbose=False
            )

            sensitive_fragments, safe_fragments, detection_log, safe_instruction_check = mask_checker.get_sensitive_and_safe_fragments(fragments)

            # Log RuleChecker calls from detection_log
            for log_entry in detection_log:
                check_type = log_entry['test_type']
                if check_type == 'full_instruction':
                    check_type_desc = "Full Instruction Check"
                elif check_type == 'masked_fragment':
                    check_type_desc = f"Masked Fragment Check - {log_entry.get('fragment_label', 'Unknown')}"
                else:
                    check_type_desc = check_type

                prompt = log_entry.get('rule_checker_prompt', 'N/A')
                response = log_entry.get('rule_checker_response', 'N/A')
                result = "VALID" if log_entry.get('is_valid', False) else f"VIOLATED:{','.join(log_entry.get('violated_rules', []))}"

                log_rule_checker_call(log_file, pair_id, idx, check_type_desc, prompt, response, result)

            # Also log safe instruction check if available
            if safe_instruction_check:
                check_type_desc = "Safe Fragments Only Check"
                prompt = safe_instruction_check.get('rule_checker_prompt', 'N/A')
                response = safe_instruction_check.get('rule_checker_response', 'N/A')
                result = "VALID" if safe_instruction_check.get('is_valid', False) else f"VIOLATED:{','.join(safe_instruction_check.get('violated_rules', []))}"

                log_rule_checker_call(log_file, pair_id, idx, check_type_desc, prompt, response, result)

            print("  Sensitive fragments:")
            for fragment in sensitive_fragments:
                print(f"    - '{fragment}'")
            print("  Safe fragments:")
            for fragment in safe_fragments:
                print(f"    - '{fragment}'")

            # Step 3: Create result entry
            result_entry = {
                'pair_id': pair_id,
                'profile_id': profile.profile_id,
                'instruction': original_instruction,
                'violated_rules_groundtruth': pair.get('violated_rules', []),
                'has_violation_groundtruth': pair.get('has_violation', False),
                'profile': {
                    'age': profile.age,
                    'country': profile.country,
                    'is_verified': profile.is_verified,
                    'payment_method': profile.payment_method,
                    'failed_payment_attempts': profile.failed_payment_attempts,
                    'credit_score': profile.credit_score,
                    'account_age_days': profile.account_age_days,
                    'account_status': profile.account_status,
                    'return_rate': profile.return_rate,
                    'total_purchase_amount': profile.total_purchase_amount
                },
                'fragments': {
                    'F1': fragments[0],
                    'F2': fragments[1],
                    'F3': fragments[2],
                    'F4': fragments[3]
                },
                'sensitive_fragments': sensitive_fragments,
                'safe_fragments': safe_fragments,
                'detection_log': [
                    {k: v for k, v in log_entry.items() if k != 'rule_checker_prompt'}
                    for log_entry in detection_log
                ],
                'safe_instruction_check': {
                    k: v for k, v in safe_instruction_check.items() if k != 'rule_checker_prompt'
                } if safe_instruction_check else None
            }

            results.append(result_entry)
            print("  ✓ Processing completed")

        except Exception as e:
            print(f"  ✗ Error processing instruction: {e}")
            # Still add the entry with error information
            result_entry = {
                'goal_id': goal_id,
                'original_instruction': original_instruction,
                'error': str(e),
                'fragments': None,
                'sensitive_fragments': None,
                'safe_fragments': None
            }
            results.append(result_entry)

    # Save results to output file
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ Processing completed! Results saved to {output_file}")
    print(f"  Total instructions processed: {len(results)}")


if __name__ == "__main__":
    process_instructions()
