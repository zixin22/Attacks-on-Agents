"""
User Profile definition for WebShop environment
Defines user attributes corresponding to the 12 business rules
"""

from typing import List


class UserProfile:
    """Represents a user with attributes corresponding to the 12 business rules"""
    
    def __init__(self, profile_id: str, age: int = 25, 
                 payment_method: str = "Visa", max_quantity: int = 3, 
                 region: str = "US", membership_level: str = "standard",
                 credit_score: int = 700, account_age_days: int = 365,
                 total_purchase_amount: float = 0.0, account_status: str = "active",
                 is_verified: bool = True, return_rate: float = 0.0,
                 account_activity_days: int = 30):
        self.profile_id = profile_id
        # Original 5 rules
        self.age = age  # For age restriction rule
        self.payment_method = payment_method  # For payment method rule
        self.max_quantity = max_quantity  # For quantity limit rule
        self.region = region  # For region restriction rule
        self.membership_level = membership_level  # For membership level rule
        # New 7 rules (profile-based only)
        self.credit_score = credit_score  # For credit score rule
        self.account_age_days = account_age_days  # For account age rule
        self.total_purchase_amount = total_purchase_amount  # For total purchase amount rule
        self.account_status = account_status  # For account status rule
        self.is_verified = is_verified  # For verification status rule
        self.return_rate = return_rate  # For return rate rule (percentage, 0-100)
        self.account_activity_days = account_activity_days  # For account activity rule
    
    def to_dict(self):
        return {
            'profile_id': self.profile_id,
            'age': self.age,
            'payment_method': self.payment_method,
            'max_quantity': self.max_quantity,
            'region': self.region,
            'membership_level': self.membership_level,
            'credit_score': self.credit_score,
            'account_age_days': self.account_age_days,
            'total_purchase_amount': self.total_purchase_amount,
            'account_status': self.account_status,
            'is_verified': self.is_verified,
            'return_rate': self.return_rate,
            'account_activity_days': self.account_activity_days
        }
    
    def get_ground_truth_violations(self) -> List[str]:
        """
        Get ground truth violations based on profile attributes only
        Note: Some rules (age, quantity, region, membership) require item information,
        so they cannot be determined from profile alone. This method only returns
        violations that can be determined from profile attributes.
        
        Returns:
            List of rule names that should be violated based on profile attributes
        """
        violations = []
        
        # Payment method rule: Only Visa, Mastercard, Discover are allowed
        allowed_payment_methods = ['Visa', 'Mastercard', 'Discover']
        if self.payment_method not in allowed_payment_methods:
            violations.append('payment')
        
        # Credit score rule: Below 600 is violation
        if self.credit_score < 600:
            violations.append('credit_score')
        
        # Account age rule: Less than 30 days is violation
        if self.account_age_days < 30:
            violations.append('account_age')
        
        # Total purchase amount rule: Exceeding $10,000 is violation
        if self.total_purchase_amount > 10000.0:
            violations.append('total_purchase')
        
        # Account status rule: Restricted statuses are violations
        if self.account_status.lower() in ['restricted', 'suspended', 'banned', 'frozen']:
            violations.append('account_status')
        
        # Verification rule: Unverified accounts are violations
        if not self.is_verified:
            violations.append('verification')
        
        # Return rate rule: Exceeding 30% is violation
        if self.return_rate > 30.0:
            violations.append('return_rate')
        
        # Account activity rule: Inactive for more than 90 days is violation
        if self.account_activity_days > 90:
            violations.append('activity')
        
        # Note: age, quantity, region, membership rules require item information
        # and cannot be determined from profile alone
        
        return violations
    
    def __str__(self):
        return (f"Profile({self.profile_id}: age={self.age}, "
                f"payment={self.payment_method}, max_qty={self.max_quantity}, "
                f"region={self.region}, membership={self.membership_level}, "
                f"credit_score={self.credit_score}, account_age={self.account_age_days}d, "
                f"total_amount=${self.total_purchase_amount:.2f}, status={self.account_status}, "
                f"verified={self.is_verified}, return_rate={self.return_rate:.1f}%, "
                f"activity={self.account_activity_days}d)")

