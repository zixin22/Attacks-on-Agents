#!/usr/bin/env python3
"""
Generate Attack Dataset Script

This script reads the first 100 goals from extracted_goals.json and generates
attack instructions and trigger instructions using the fragment-based attack system.
It also identifies sensitive fragments using the mask method with NER decomposition.

Output: A JSON dataset containing:
- Original goals
- Generated fragments (F1, F2, F3, F4)
- Sensitive fragments identified by mask check
- Fragment attack instructions
- Trigger attack instruction
"""

import json
import os
import sys
from typing import Dict, List, Any

_WEBSHOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WEBSHOP_ROOT not in sys.path:
    sys.path.insert(0, _WEBSHOP_ROOT)

from attack.attack import FragmentAttackGenerator
from ner_mask.live_attack_plan import generate_live_attack_plan

# Import rule system components (for mask checking)
from rule_and_profile import RuleChecker, UserProfile


def load_first_100_pairs() -> List[Dict]:
    """
    Load the first 100 pairs from profile_goal_pairs_2000.json

    Returns:
        List of pair dictionaries, each containing profile, instruction, etc.
    """
    pairs_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\groundtruth\profile_goal_pairs_2000.json"

    try:
        with open(pairs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get first 100 pairs
        pairs = data["pairs"][:100]

        print(f"Loaded {len(pairs)} pairs from {pairs_file}")
        return pairs

    except FileNotFoundError:
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    except Exception as e:
        raise Exception(f"Error loading pairs: {e}")


def setup_attack_system() -> FragmentAttackGenerator:
    """
    Setup the fragment attack generator

    Returns:
        Configured FragmentAttackGenerator instance
    """
    return FragmentAttackGenerator(verbose=False)


def setup_rule_system() -> RuleChecker:
    """
    Setup rule checker for mask checking

    Returns:
        RuleChecker instance, or None if setup fails
    """
    try:
        # Initialize RuleChecker
        rule_checker = RuleChecker(verbose=False, model="gpt-4o")
        return rule_checker
    except Exception as e:
        print(f"Warning: Failed to setup rule system: {e}")
        print("Proceeding without mask checking...")
        return None


def generate_attack_data_for_goal(
    pair: Dict,
    attack_generator: FragmentAttackGenerator,
    rule_checker: RuleChecker
) -> Dict[str, Any]:
    """
    Generate attack data for a single pair

    Args:
        pair: Pair dictionary containing profile, instruction, etc.
        attack_generator: FragmentAttackGenerator instance
        rule_checker: RuleChecker instance

    Returns:
        Dictionary containing attack data for this pair
    """
    pair_id = pair["pair_id"]
    goal_id = pair["goal_id"]
    instruction = pair["instruction"]

    print(f"Processing pair {pair_id} (goal {goal_id}): {instruction[:50]}...")

    # Create UserProfile from pair data
    profile_data = pair["profile"]
    profile = UserProfile(
        profile_id=profile_data["profile_id"],
        age=profile_data["age"],
        country=profile_data["country"],
        is_verified=profile_data["is_verified"],
        payment_method=profile_data["payment_method"],
        failed_payment_attempts=profile_data["failed_payment_attempts"],
        credit_score=profile_data["credit_score"],
        account_age_days=profile_data["account_age_days"],
        account_status=profile_data["account_status"],
        return_rate=profile_data["return_rate"],
        total_purchase_amount=profile_data["total_purchase_amount"]
    )

    # For this analysis, we decompose each host instruction to analyze its fragments
    # and check which parts might trigger rules for the given user profile
    target_instruction = instruction  # Use the host instruction itself as target for analysis

    try:
        # Generate attack plan with mask checking (only if rule_checker is available)
        if rule_checker is not None:
            attack_plan = generate_live_attack_plan(
                attack_generator,
                host_instruction=instruction,
                target_instruction=target_instruction,
                rule_checker=rule_checker,
                profile=profile,
                query=pair.get("query", ""),
                attack_log_file=None,
            )
        else:
            attack_plan = generate_live_attack_plan(
                attack_generator,
                host_instruction=instruction,
                target_instruction=target_instruction,
                rule_checker=None,
                profile=None,
                query=pair.get("query", ""),
                attack_log_file=None,
            )

        # Display violation status for each fragment
        if rule_checker is not None:
            print(f"  Profile: {profile.profile_id} (age={profile.age}, credit_score={profile.credit_score}, return_rate={profile.return_rate})")

            # Show instruction fragments and their sensitivity analysis
            fragments = attack_plan["fragments"]
            sensitive_fragments = attack_plan.get("sensitive_fragments", [])
            fragment_attacks = attack_plan.get("fragment_attacks", [])

            print(f"  Instruction fragments analysis:")
            for i, fragment in enumerate(fragments):
                if fragment.strip():  # Only show non-empty fragments
                    label = f"F{i+1}"
                    is_sensitive = fragment in sensitive_fragments

                    # Also check from fragment_attacks if available
                    if fragment_attacks and i < len(fragment_attacks):
                        attack_info = fragment_attacks[i]
                        is_sensitive = attack_info.get("is_sensitive", is_sensitive)

                    status = "❌ SENSITIVE" if is_sensitive else "✅ SAFE"
                    print(f"    {label}: '{fragment}' -> {status}")

            print(f"    Sensitive fragments: {len(sensitive_fragments)}")
            print(f"    Safe fragments: {len(fragments) - len(sensitive_fragments)}")

        # Extract relevant data
        attack_data = {
            "pair_id": pair_id,
            "goal_id": goal_id,
            "instruction": instruction,
            "profile": profile_data,  # Include original profile data
            "fragments": {  # Fragments from instruction analysis
                "F1": attack_plan["fragments"][0] if len(attack_plan["fragments"]) > 0 else "",
                "F2": attack_plan["fragments"][1] if len(attack_plan["fragments"]) > 1 else "",
                "F3": attack_plan["fragments"][2] if len(attack_plan["fragments"]) > 2 else "",
                "F4": attack_plan["fragments"][3] if len(attack_plan["fragments"]) > 3 else ""
            },
            "sensitive_fragments": attack_plan.get("sensitive_fragments", []),
            "safe_fragments": attack_plan.get("safe_fragments", []),
            "fragment_attacks": attack_plan.get("fragment_attacks", []),
            "trigger_instruction": attack_plan.get("trigger_instruction", ""),
            "mask_detection_performed": rule_checker is not None and profile is not None and profile is not None
        }

    except Exception as e:
        print(f"Error generating attack data for goal {goal_id}: {e}")
        # Return minimal data on error
        attack_data = {
            "goal_id": goal_id,
            "goal_text": goal_text,
            "target_instruction": target_instruction,
            "fragments": {"F1": "", "F2": "", "F3": "", "F4": ""},
            "sensitive_fragments": [],
            "safe_fragments": [],
            "fragment_attacks": [],
            "trigger_instruction": "",
            "mask_detection_performed": False,
            "error": str(e)
        }

    return attack_data


def save_attack_dataset(attack_dataset: List[Dict[str, Any]], output_file: str):
    """
    Save the attack dataset to JSON file

    Args:
        attack_dataset: List of attack data dictionaries
        output_file: Output file path
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(attack_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved attack dataset to {output_file} ({len(attack_dataset)} goals)")


def main():
    """Main function to generate the attack dataset"""
    print("="*80)
    print("GENERATING ATTACK DATASET")
    print("="*80)

    try:
        # Step 1: Load first 100 pairs
        print("\nStep 1: Loading pairs...")
        pairs = load_first_100_pairs()

        # Step 2: Setup attack system
        print("\nStep 2: Setting up attack system...")
        attack_generator = setup_attack_system()

        # Step 3: Setup rule system for mask checking
        print("\nStep 3: Setting up rule system for mask checking...")
        rule_checker = setup_rule_system()

        # Step 4: Generate attack data for each pair
        print("\nStep 4: Generating attack data...")
        attack_dataset = []
        for pair in pairs:
            attack_data = generate_attack_data_for_goal(
                pair, attack_generator, rule_checker
            )
            attack_dataset.append(attack_data)

        # Step 5: Save the dataset
        print("\nStep 5: Saving dataset...")
        output_file = r"C:\Users\22749\Desktop\rap-main\webshop\data\groundtruth\attack_dataset_100_goals.json"
        save_attack_dataset(attack_dataset, output_file)

        print("\n" + "="*80)
        print("ATTACK DATASET GENERATION COMPLETE")
        print("="*80)
        print(f"Generated data for {len(attack_dataset)} goals")
        print(f"Output file: {output_file}")

        # Print summary statistics
        successful_generations = sum(1 for item in attack_dataset if item.get("mask_detection_performed", False))
        total_sensitive_fragments = sum(len(item.get("sensitive_fragments", [])) for item in attack_dataset)
        total_fragment_attacks = sum(len(item.get("fragment_attacks", [])) for item in attack_dataset)

        print("\nSummary:")
        print(f"- Successful generations: {successful_generations}/{len(attack_dataset)}")
        print(f"- Total sensitive fragments identified: {total_sensitive_fragments}")
        print(f"- Total fragment attack instructions generated: {total_fragment_attacks}")
        print(f"- Pairs processed: {len(attack_dataset)}")
        print(f"- Using real profiles from profile_goal_pairs_2000.json")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
