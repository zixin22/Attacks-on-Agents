#!/usr/bin/env python3
"""
Test script for GuardAgent functionality
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def main():
    parser = argparse.ArgumentParser(description='Test GuardAgent with attack parameters')
    parser.add_argument("--attack", action="store_true", help="Enable fragment-based attack mode")
    parser.add_argument("--attack_fixed_number", type=int, default=None, help="Fixed number of the case to attack (required if --attack is set)")
    parser.add_argument("--attack_target_instruction", type=str, default=None, help="Target instruction to inject in attack")
    parser.add_argument("--defense_mode", type=str, default="rule_checker",
                        choices=["rule_checker", "guard_agent", "none"],
                        help="Defense mechanism")
    parser.add_argument("--enable_guard_agent", action="store_true", help="Enable GuardAgent")
    parser.add_argument("--guard_agent_llm", type=str, default="gpt-4o",
                        choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo"],
                        help="LLM model for GuardAgent")
    parser.add_argument("--guard_agent_shots", type=int, default=3,
                        choices=[1, 2, 3],
                        help="Number of few-shot examples for GuardAgent")
    parser.add_argument("--output", type=str, default="output", help="Output folder")

    args = parser.parse_args()

    print("=== GuardAgent Test Parameters ===")
    print(f"attack: {args.attack}")
    print(f"attack_fixed_number: {args.attack_fixed_number}")
    print(f"attack_target_instruction: {args.attack_target_instruction}")
    print(f"defense_mode: {args.defense_mode}")
    print(f"enable_guard_agent: {args.enable_guard_agent}")
    print(f"guard_agent_llm: {args.guard_agent_llm}")
    print(f"guard_agent_shots: {args.guard_agent_shots}")
    print(f"output: {args.output}")

    # Test GuardAgent import
    print("\n=== Testing GuardAgent Import ===")
    try:
        from rule_and_profile.webshop_guard_agent import WebShopGuardAgent
        print("✓ WebShopGuardAgent import successful")
    except ImportError as e:
        print(f"✗ WebShopGuardAgent import failed: {e}")
        return 1

    # Test GuardAgent initialization
    if args.defense_mode == 'guard_agent' and args.enable_guard_agent:
        print("\n=== Testing GuardAgent Initialization ===")
        try:
            guard_agent = WebShopGuardAgent(
                verbose=True,
                model=args.guard_agent_llm,
                num_shots=args.guard_agent_shots,
                seed=42
            )
            print("✓ GuardAgent initialization successful")
        except Exception as e:
            print(f"✗ GuardAgent initialization failed: {e}")
            return 1

    print("\n=== All tests passed! ===")
    print("You can now run the GuardAgent attack with the following command:")
    print(f"python main.py --attack --attack_fixed_number {args.attack_fixed_number} --attack_target_instruction \"{args.attack_target_instruction}\" --defense_mode guard_agent --enable_guard_agent --guard_agent_llm {args.guard_agent_llm} --output {args.output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
