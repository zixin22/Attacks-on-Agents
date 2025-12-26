"""
Generate groundtruth dataset for profile-goal pairs
For each profile-goal pair, check if it violates any rules and record the results
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Try to import tqdm for progress bar, fallback to simple counter if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress bar replacement
    class tqdm:
        def __init__(self, total=None, desc=""):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\r{self.desc}: {self.n}/{self.total}", end="", flush=True)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            print()  # New line after progress

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rule_and_profile.rule_checker import RuleChecker
from rule_and_profile.user_profile import UserProfile


def load_profiles(profiles_file: str) -> List[UserProfile]:
    """Load user profiles from JSON file"""
    with open(profiles_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    profiles_list = data.get('profiles', [])
    profiles = []
    for profile_dict in profiles_list:
        profile = UserProfile(
            profile_id=profile_dict.get('profile_id', 'unknown'),
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


def load_goals(goals_file: str) -> Dict[str, str]:
    """Load goals from extracted_goals.json"""
    with open(goals_file, 'r', encoding='utf-8') as f:
        goals = json.load(f)
    return goals


def get_query_from_memory(fixed_number: int, memory_dir: str = None) -> str:
    """
    Try to get query from memory files if available
    If not found, return empty string (RuleChecker can work with empty query)
    """
    if memory_dir is None:
        # Try to find query from normal_output memory files
        memory_dir = Path(__file__).parent.parent / "data" / "normal_output_0_12000"
    
    # Try multiple possible memory files
    possible_files = [
        memory_dir / "output_100sample" / "output_gpt4o" / "memory_1.json",
        memory_dir / "output_0_500" / "memory_1.json",
    ]
    
    for mem_file in possible_files:
        if mem_file.exists():
            try:
                with open(mem_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                
                # Find entry with matching fixed_number
                for entry in memory:
                    entry_id = entry.get('Id', '')
                    if f'fixed_{fixed_number}' == entry_id:
                        query = entry.get('Query', '')
                        if query:
                            return query
            except:
                continue
    
    # If not found, return empty string
    return ""


def generate_groundtruth(
    profiles_file: str,
    goals_file: str,
    output_file: str,
    memory_dir: str = None,
    verbose: bool = False,
    model: str = "gpt-4o",
    sample_limit: int = None
):
    """
    Generate groundtruth dataset by checking all profile-goal pairs
    
    Args:
        profiles_file: Path to generated_profiles.json
        goals_file: Path to extracted_goals.json
        output_file: Path to save groundtruth dataset
        memory_dir: Optional directory to search for query information
        verbose: Whether to print detailed information
        model: LLM model to use for rule checking
        sample_limit: Limit number of pairs to check (for testing)
    """
    print("="*80)
    print("Generating Groundtruth Dataset")
    print("="*80)
    
    # Load profiles and goals
    print(f"\n1. Loading profiles from {profiles_file}...")
    profiles = load_profiles(profiles_file)
    print(f"   Loaded {len(profiles)} profiles")
    
    print(f"\n2. Loading goals from {goals_file}...")
    goals = load_goals(goals_file)
    print(f"   Loaded {len(goals)} goals")
    
    # Initialize RuleChecker
    print(f"\n3. Initializing RuleChecker (model: {model})...")
    rule_checker = RuleChecker(verbose=verbose, model=model)
    
    # Prepare results
    total_pairs = len(profiles) * len(goals)
    if sample_limit:
        total_pairs = min(total_pairs, sample_limit)
        print(f"\n   Limiting to {sample_limit} pairs for testing")
    
    print(f"\n4. Checking {total_pairs} profile-goal pairs...")
    print("   This may take a while...\n")
    
    results = []
    violated_count = 0
    valid_count = 0
    
    # Create progress bar
    pair_count = 0
    with tqdm(total=total_pairs, desc="Checking pairs") as pbar:
        for profile in profiles:
            if pair_count >= total_pairs:
                break
            
            for fixed_num_str, instruction in goals.items():
                if pair_count >= total_pairs:
                    break
                
                fixed_num = int(fixed_num_str)
                
                # Try to get query from memory, otherwise use empty string
                query = get_query_from_memory(fixed_num, memory_dir)
                
                # Check rules
                try:
                    is_valid, violated_rules = rule_checker.check_all_rules(
                        profile, instruction, query
                    )
                    
                    result = {
                        "profile_id": profile.profile_id,
                        "fixed_number": fixed_num,
                        "instruction": instruction,
                        "query": query,
                        "is_valid": is_valid,
                        "violated_rules": violated_rules,
                        "profile_attributes": {
                            "age": profile.age,
                            "country": profile.country,
                            "is_verified": profile.is_verified,
                            "payment_method": profile.payment_method,
                            "failed_payment_attempts": profile.failed_payment_attempts,
                            "credit_score": profile.credit_score,
                            "account_age_days": profile.account_age_days,
                            "account_status": profile.account_status,
                            "return_rate": profile.return_rate,
                            "total_purchase_amount": profile.total_purchase_amount
                        }
                    }
                    
                    results.append(result)
                    
                    if is_valid:
                        valid_count += 1
                    else:
                        violated_count += 1
                    
                    pair_count += 1
                    pbar.update(1)
                    
                    # Print progress every 100 pairs
                    if verbose and pair_count % 100 == 0:
                        print(f"\n   Processed {pair_count}/{total_pairs} pairs")
                        print(f"   Valid: {valid_count}, Violated: {violated_count}")
                
                except Exception as e:
                    print(f"\n   Error checking pair (profile={profile.profile_id}, fixed={fixed_num}): {e}")
                    # Still record the result with error
                    result = {
                        "profile_id": profile.profile_id,
                        "fixed_number": fixed_num,
                        "instruction": instruction,
                        "query": query,
                        "is_valid": None,
                        "violated_rules": [],
                        "error": str(e),
                        "profile_attributes": {
                            "age": profile.age,
                            "country": profile.country,
                            "is_verified": profile.is_verified,
                            "payment_method": profile.payment_method,
                            "failed_payment_attempts": profile.failed_payment_attempts,
                            "credit_score": profile.credit_score,
                            "account_age_days": profile.account_age_days,
                            "account_status": profile.account_status,
                            "return_rate": profile.return_rate,
                            "total_purchase_amount": profile.total_purchase_amount
                        }
                    }
                    results.append(result)
                    pair_count += 1
                    pbar.update(1)
    
    # Create summary statistics
    rule_violation_counts = {}
    for result in results:
        if result.get('is_valid') == False:
            for rule in result.get('violated_rules', []):
                rule_violation_counts[rule] = rule_violation_counts.get(rule, 0) + 1
    
    # Create output structure
    output = {
        "metadata": {
            "total_profiles": len(profiles),
            "total_goals": len(goals),
            "total_pairs_checked": len(results),
            "valid_pairs": valid_count,
            "violated_pairs": violated_count,
            "model_used": model,
            "generation_timestamp": None  # Can add timestamp if needed
        },
        "statistics": {
            "violation_rate": violated_count / len(results) if results else 0,
            "rule_violation_counts": rule_violation_counts
        },
        "results": results
    }
    
    # Save results
    print(f"\n5. Saving results to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total pairs checked: {len(results)}")
    print(f"Valid pairs: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"Violated pairs: {violated_count} ({violated_count/len(results)*100:.1f}%)")
    print(f"\nRule violation breakdown:")
    for rule, count in sorted(rule_violation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rule}: {count} violations")
    print(f"\nResults saved to: {output_file}")
    print("="*80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate groundtruth dataset for profile-goal pairs")
    parser.add_argument(
        "--profiles_file",
        type=str,
        default=r"C:\Users\22749\Desktop\rap-main\webshop\generated_profiles.json",
        help="Path to generated_profiles.json"
    )
    parser.add_argument(
        "--goals_file",
        type=str,
        default=r"C:\Users\22749\Desktop\rap-main\webshop\data\selected_reward_0.5\extracted_goals.json",
        help="Path to extracted_goals.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=r"C:\Users\22749\Desktop\rap-main\webshop\data\groundtruth\profile_goal_groundtruth.json",
        help="Path to save groundtruth dataset"
    )
    parser.add_argument(
        "--memory_dir",
        type=str,
        default=None,
        help="Optional directory to search for query information"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4-0613", "gpt-3.5-turbo-instruct"],
        help="LLM model to use"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit number of pairs to check (for testing)"
    )
    
    args = parser.parse_args()
    
    generate_groundtruth(
        profiles_file=args.profiles_file,
        goals_file=args.goals_file,
        output_file=args.output_file,
        memory_dir=args.memory_dir,
        verbose=args.verbose,
        model=args.model,
        sample_limit=args.sample_limit
    )


if __name__ == "__main__":
    main()

