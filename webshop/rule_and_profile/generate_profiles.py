"""
Generate 100 user profiles with random attribute distributions
Each profile's attributes are randomly distributed across all possible values
"""

import json
import random
import os
import sys
from typing import List, Dict

# Import constants from user_profile module
# Support both relative import (when used as module) and absolute import (when run directly)
try:
    from .user_profile import PAYMENT_METHODS, ACCOUNT_STATUSES, COUNTRIES
except ImportError:
    # If relative import fails, add parent directory to path and use absolute import
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rule_and_profile.user_profile import PAYMENT_METHODS, ACCOUNT_STATUSES, COUNTRIES


def generate_random_profile(profile_id: int) -> Dict:
    """
    Generate a single user profile with random attributes
    
    Args:
        profile_id: Unique identifier for the profile
        
    Returns:
        Dictionary containing profile attributes
    """
    # Boolean attributes: true/false 对半分
    is_verified = random.choice([True, False])
    
    # Country: allowed/not_allowed 随机
    country = random.choice(COUNTRIES)
    
    # Age: random between 1 and 100 (reasonable range)
    age = random.randint(1, 100)
    
    # Credit score: 0-1000 随机
    credit_score = random.randint(0, 1000)
    
    # Return rate: 0-100 随机 (percentage)
    return_rate = round(random.uniform(0.0, 100.0), 1)
    
    # Payment method: 随机选择
    payment_method = random.choice(PAYMENT_METHODS)
    
    # Failed payment attempts: 0-10 随机
    failed_payment_attempts = random.randint(0, 10)
    
    # Account age in days: 1-3650 随机 (1 day to 10 years)
    account_age_days = random.randint(1, 3650)
    
    # Account status: 随机选择
    account_status = random.choice(ACCOUNT_STATUSES)
    
    # Total purchase amount: 0-50000 随机
    total_purchase_amount = round(random.uniform(0.0, 50000.0), 2)
    
    return {
        "profile_id": f"profile_{profile_id}",
        "age": age,
        "country": country,
        "is_verified": is_verified,
        "payment_method": payment_method,
        "failed_payment_attempts": failed_payment_attempts,
        "credit_score": credit_score,
        "account_age_days": account_age_days,
        "account_status": account_status,
        "return_rate": return_rate,
        "total_purchase_amount": total_purchase_amount
    }


def generate_profiles(num_profiles: int = 100) -> List[Dict]:
    """
    Generate multiple user profiles
    
    Args:
        num_profiles: Number of profiles to generate
        
    Returns:
        List of profile dictionaries
    """
    profiles = []
    for i in range(num_profiles):
        profile = generate_random_profile(i)
        profiles.append(profile)
    
    return profiles


def calculate_distribution_stats(profiles: List[Dict]) -> Dict:
    """
    Calculate distribution statistics for the generated profiles
    
    Args:
        profiles: List of profile dictionaries
        
    Returns:
        Dictionary containing distribution statistics
    """
    stats = {
        "is_verified_true": sum(1 for p in profiles if p["is_verified"]),
        "is_verified_false": sum(1 for p in profiles if not p["is_verified"]),
        "country_allowed": sum(1 for p in profiles if p["country"] == "allowed"),
        "country_not_allowed": sum(1 for p in profiles if p["country"] == "not_allowed"),
        "payment_methods": {},
        "account_statuses": {},
        "credit_score_min": min(p["credit_score"] for p in profiles),
        "credit_score_max": max(p["credit_score"] for p in profiles),
        "credit_score_avg": sum(p["credit_score"] for p in profiles) / len(profiles),
        "return_rate_min": min(p["return_rate"] for p in profiles),
        "return_rate_max": max(p["return_rate"] for p in profiles),
        "return_rate_avg": sum(p["return_rate"] for p in profiles) / len(profiles),
        "age_min": min(p["age"] for p in profiles),
        "age_max": max(p["age"] for p in profiles),
        "age_avg": sum(p["age"] for p in profiles) / len(profiles),
    }
    
    # Count payment methods
    for method in PAYMENT_METHODS:
        stats["payment_methods"][method] = sum(1 for p in profiles if p["payment_method"] == method)
    
    # Count account statuses
    for status in ACCOUNT_STATUSES:
        stats["account_statuses"][status] = sum(1 for p in profiles if p["account_status"] == status)
    
    return stats


def main():
    """Main function to generate profiles and save to JSON file"""
    num_profiles = 100
    
    print(f"Generating {num_profiles} user profiles with random attribute distributions...")
    
    # Generate profiles
    profiles = generate_profiles(num_profiles)
    
    # Calculate statistics
    stats = calculate_distribution_stats(profiles)
    
    # Create output structure
    output = {
        "total_profiles": num_profiles,
        "generation_config": {
            "num_profiles": num_profiles,
            "random_distribution": True,
            "description": "All attributes are randomly distributed across their possible values"
        },
        "distribution_stats": stats,
        "profiles": profiles
    }
    
    # Save to JSON file in the webshop directory (parent directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    webshop_dir = os.path.dirname(script_dir)
    output_file = os.path.join(webshop_dir, "generated_profiles.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully generated {num_profiles} profiles")
    print(f"✓ Saved to {output_file}")
    print(f"\nDistribution Statistics:")
    print(f"  - Verified (True): {stats['is_verified_true']} ({stats['is_verified_true']/num_profiles*100:.1f}%)")
    print(f"  - Verified (False): {stats['is_verified_false']} ({stats['is_verified_false']/num_profiles*100:.1f}%)")
    print(f"  - Country (allowed): {stats['country_allowed']} ({stats['country_allowed']/num_profiles*100:.1f}%)")
    print(f"  - Country (not_allowed): {stats['country_not_allowed']} ({stats['country_not_allowed']/num_profiles*100:.1f}%)")
    print(f"  - Credit Score: {stats['credit_score_min']}-{stats['credit_score_max']} (avg: {stats['credit_score_avg']:.1f})")
    print(f"  - Return Rate: {stats['return_rate_min']:.1f}%-{stats['return_rate_max']:.1f}% (avg: {stats['return_rate_avg']:.1f}%)")
    print(f"  - Age: {stats['age_min']}-{stats['age_max']} (avg: {stats['age_avg']:.1f})")
    print(f"\nPayment Methods:")
    for method, count in stats['payment_methods'].items():
        print(f"  - {method}: {count} ({count/num_profiles*100:.1f}%)")
    print(f"\nAccount Statuses:")
    for status, count in stats['account_statuses'].items():
        print(f"  - {status}: {count} ({count/num_profiles*100:.1f}%)")


if __name__ == "__main__":
    main()

