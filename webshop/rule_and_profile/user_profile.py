"""
User Profile definition for WebShop environment
Defines user attributes corresponding to the 10 business rules
"""

from typing import List

# Payment method options
PAYMENT_METHODS = ["Visa", "MasterCard", "PayPal", "Prepaid", "Gift Card"]

# Account status options
ACCOUNT_STATUSES = ["active", "restricted"]

# Country options
COUNTRIES = ["allowed", "not_allowed"]


class UserProfile:
    """Represents a user with attributes corresponding to the 10 business rules"""
    
    def __init__(self, profile_id: str, age: int = 25, 
                 country: str = "allowed", is_verified: bool = True,
                 payment_method: str = "Visa", failed_payment_attempts: int = 0,
                 credit_score: int = 700, account_age_days: int = 365,
                 account_status: str = "active", return_rate: float = 0.0,
                 total_purchase_amount: float = 0.0):
        self.profile_id = profile_id
        # 基础身份字段
        self.age = age  # 用户年龄
        self.country = country  # 用户国家是否允许购买某类商品 (allowed/not_allowed)
        self.is_verified = is_verified  # 用户是否已完成验证
        
        # 支付相关字段
        self.payment_method = payment_method  # 支付方式 (Visa, MasterCard, PayPal, Prepaid, Gift Card)
        self.failed_payment_attempts = failed_payment_attempts  # 失败支付尝试次数
        
        # 信用、账户行为字段
        self.credit_score = credit_score  # 信用分 (0-850)
        self.account_age_days = account_age_days  # 账户注册天数
        self.account_status = account_status  # 账号状态 (active/restricted)
        self.return_rate = return_rate  # 退货率 (0-100%)
        self.total_purchase_amount = total_purchase_amount  # 历史累计消费
    
    def to_dict(self):
        return {
            'profile_id': self.profile_id,
            'age': self.age,
            'country': self.country,
            'is_verified': self.is_verified,
            'payment_method': self.payment_method,
            'failed_payment_attempts': self.failed_payment_attempts,
            'credit_score': self.credit_score,
            'account_age_days': self.account_age_days,
            'account_status': self.account_status,
            'return_rate': self.return_rate,
            'total_purchase_amount': self.total_purchase_amount
        }
    
    def __str__(self):
        return (f"Profile({self.profile_id}: age={self.age}, "
                f"country={self.country}, verified={self.is_verified}, "
                f"payment={self.payment_method}, failed_payments={self.failed_payment_attempts}, "
                f"credit_score={self.credit_score}, account_age={self.account_age_days}d, "
                f"status={self.account_status}, return_rate={self.return_rate:.1f}%, "
                f"total_amount=${self.total_purchase_amount:.2f})")

