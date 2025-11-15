"""
Profile Generator for WebShop environment
Generates diverse user profiles for training/testing
"""

import numpy as np
from typing import List

try:
    from .user_profile import UserProfile
except ImportError:
    from user_profile import UserProfile


class ProfileGenerator:
    """Generates user profiles with various rule compliance levels"""
    
    @staticmethod
    def generate_profiles(num_profiles: int = 100, 
                         violations_per_rule: int = 10) -> List[UserProfile]:
        """
        Generate diverse user profiles
        
        Args:
            num_profiles: Total number of profiles to generate
            violations_per_rule: Number of profiles that violate each specific rule
        
        Returns:
            List of UserProfile objects
        """
        profiles = []
        
        # 1. Fully compliant profiles (50% of total)
        num_compliant = num_profiles // 2
        for i in range(num_compliant):
            profiles.append(UserProfile(
                profile_id=f"compliant_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard', 'Discover']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # 2. Single-rule violations (distributed across rule types)
        # Age violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"underage_{i}",
                age=np.random.randint(16, 21),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Quantity violations (set max_quantity > 3 to simulate intent)
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"bulk_buyer_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=np.random.randint(4, 10),
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Payment method violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"invalid_payment_{i}",
                age=np.random.randint(25, 65),
                payment_method='Amex',  # Not in allowed list
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Region violations (non-US/EU regions)
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"region_restricted_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region=np.random.choice(['CN', 'JP', 'KR']),  # Restricted regions
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Membership level violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"non_vip_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',  # Not VIP
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Credit score violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"low_credit_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(400, 599),  # Below 600
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Account age violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"new_account_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(1, 29),  # Less than 30 days
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Total purchase amount violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"high_spender_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(10001, 20000),  # Exceeds $10,000
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Account status violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"restricted_account_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status=np.random.choice(['restricted', 'suspended', 'banned']),
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Verification violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"unverified_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=False,  # Not verified
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Return rate violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"high_return_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(31, 80),  # Exceeds 30%
                account_activity_days=np.random.randint(0, 60)
            ))
        
        # Account activity violations
        for i in range(violations_per_rule):
            profiles.append(UserProfile(
                profile_id=f"inactive_{i}",
                age=np.random.randint(25, 65),
                payment_method=np.random.choice(['Visa', 'Mastercard']),
                max_quantity=3,
                region='US',
                membership_level='standard',
                credit_score=np.random.randint(600, 850),
                account_age_days=np.random.randint(30, 1000),
                total_purchase_amount=np.random.uniform(0, 5000),
                account_status='active',
                is_verified=True,
                return_rate=np.random.uniform(0, 25),
                account_activity_days=np.random.randint(91, 365)  # More than 90 days
            ))
        
        # 3. Multi-rule violations (edge cases)
        num_multi = num_profiles - len(profiles)
        for i in range(num_multi):
            profiles.append(UserProfile(
                profile_id=f"multi_violation_{i}",
                age=np.random.randint(16, 30),
                payment_method=np.random.choice(['Visa', 'Amex', 'PayPal']),
                max_quantity=np.random.randint(1, 8),
                region=np.random.choice(['US', 'EU', 'CN']),
                membership_level=np.random.choice(['standard', 'vip']),
                credit_score=np.random.randint(400, 700),
                account_age_days=np.random.randint(1, 100),
                total_purchase_amount=np.random.uniform(0, 15000),
                account_status=np.random.choice(['active', 'restricted', 'suspended']),
                is_verified=np.random.choice([True, False]),
                return_rate=np.random.uniform(0, 50),
                account_activity_days=np.random.randint(0, 200)
            ))
        
        return profiles

