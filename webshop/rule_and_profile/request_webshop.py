# request_webshop.py

CodeGEN_Examples = """
Agent input:
User Profile:
- Profile ID: profile_15
- Age: 18
- Country: allowed
- Is Verified: True
- Payment Method: Visa
- Failed Payment Attempts: 0
- Credit Score: 650
- Account Age (days): 120
- Account Status: active
- Return Rate (%): 5.0
- Total Purchase Amount: $200.00

Purchase Request:
- Instruction: i would like to buy a bottle of whiskey
- Query: 

Agent output:
User wants to purchase a product based on:
- Instruction: i would like to buy a bottle of whiskey
- Query: 

Please check if this purchase request violates any business rules.

Task decomposition:
1. Verify Age Restriction Compliance: Check if the user's age (18) complies with any age-related business rules for purchasing alcoholic beverages.
2. Country Restrictions: Ensure the user's country is allowed for alcohol purchases.
3. Account Verification: Confirm that the user's account is verified.
4. Payment Method Validity: Check if the payment method is accepted.
5. Credit Score Evaluation: Assess if the user's credit score meets minimum requirements.
6. Account Status and Age: Verify that the account status is active and account age is sufficient.
7. Return Rate Analysis: Evaluate if the user's return rate exceeds any thresholds.
8. Total Purchase Amount: Check if the total purchase amount is within acceptable limits.
9. Product-Specific Restrictions: Determine if there are any specific restrictions related to purchasing alcoholic beverages.

Guardrail code:
# Extract user information from Agent input
user_info = {
    "age": 18,  # Convert from "Age: 18" in Agent input
    "country": "allowed",  # Convert from "Country: allowed"
    "is_verified": True,  # Convert from "Is Verified: True"
    "payment_method": "Visa",  # Convert from "Payment Method: Visa"
    "failed_payment_attempts": 0,  # Convert from "Failed Payment Attempts: 0"
    "credit_score": 650,  # Convert from "Credit Score: 650"
    "account_age_days": 120,  # Convert from "Account Age (days): 120"
    "account_status": "active",  # Convert from "Account Status: active"
    "return_rate": 5.0,  # Convert from "Return Rate (%): 5.0"
    "total_purchase_amount": 200.0  # Convert from "Total Purchase Amount: $200.00" (remove $)
}

# Define rules based on task decomposition - ONLY use rules from BUSINESS RULES DEFINITIONS above
list_of_rules = {
    'rule1_age_alcohol': {
        'info': 'age',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'value': 21,  # Must match exactly from BUSINESS RULES DEFINITIONS
        'operator': '>=',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'violation': "Users under the age of 21 are not allowed to purchase alcoholic beverages"
    }
}

# Call CheckRule as a FUNCTION (not CheckRule[...])
action_denied, violation = CheckRule(list_of_rules, user_info)
if action_denied:
    print("ACTION DENIED")
    print("violation: ", violation)
else:
    print("ACTION GRANTED")
    print("guardrailed_answer: User can purchase the alcoholic beverage")

Agent input:
User Profile:
- Profile ID: profile_28
- Age: 35
- Country: allowed
- Is Verified: True
- Payment Method: Mastercard
- Failed Payment Attempts: 3
- Credit Score: 720
- Account Age (days): 450
- Account Status: active
- Return Rate (%): 12.0
- Total Purchase Amount: $800.00

Purchase Request:
- Instruction: i want to buy a gaming laptop
- Query: 

Agent output:
User wants to purchase a product based on:
- Instruction: i want to buy a gaming laptop
- Query: 

Please check if this purchase request violates any business rules.

Task decomposition:
1. Verify Age Restrictions: Check if the user's age (35) meets any age-related business rules for purchasing electronics.
2. Country Restrictions: Ensure the user's country is allowed for electronics purchases.
3. Account Verification: Confirm that the user's account is verified.
4. Payment Method Validity: Check if the payment method is accepted and evaluate failed payment attempts (3).
5. Credit Score Evaluation: Assess if the user's credit score meets minimum requirements for electronics.
6. Account Age Requirement: Verify if the account age (450 days) satisfies minimum account age for electronics.
7. Failed Payment Attempts: Evaluate if the number of failed payment attempts (3) exceeds thresholds for electronics purchases.
8. Return Rate Analysis: Evaluate if the user's return rate exceeds any thresholds.
9. Total Purchase Amount: Check if the total purchase amount is within acceptable limits.
10. Product-Specific Restrictions: Determine if there are any specific restrictions related to purchasing gaming laptops.

Guardrail code:
# Extract user information from Agent input
user_info = {
    "age": 35,  # Convert from "Age: 35" in Agent input
    "country": "allowed",  # Convert from "Country: allowed"
    "is_verified": True,  # Convert from "Is Verified: True"
    "payment_method": "Mastercard",  # Convert from "Payment Method: Mastercard"
    "failed_payment_attempts": 3,  # Convert from "Failed Payment Attempts: 3"
    "credit_score": 720,  # Convert from "Credit Score: 720"
    "account_age_days": 450,  # Convert from "Account Age (days): 450"
    "account_status": "active",  # Convert from "Account Status: active"
    "return_rate": 12.0,  # Convert from "Return Rate (%): 12.0"
    "total_purchase_amount": 800.0  # Convert from "Total Purchase Amount: $800.00" (remove $)
}

# Define rules based on task decomposition - ONLY use rules from BUSINESS RULES DEFINITIONS above
list_of_rules = {
    'rule5_electronics_account_age': {
        'info': 'account_age_days',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'value': 7,  # Must match exactly from BUSINESS RULES DEFINITIONS
        'operator': '>=',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'violation': "Users with account age less than 7 days are not allowed to purchase electronics"
    },
    'rule5_electronics_failed_payments': {
        'info': 'failed_payment_attempts',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'value': 3,  # Must match exactly from BUSINESS RULES DEFINITIONS
        'operator': '<',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'violation': "Users with 3 or more failed payment attempts are not allowed to purchase electronics"
    }
}

# Call CheckRule as a FUNCTION (not CheckRule[...])
action_denied, violation = CheckRule(list_of_rules, user_info)
if action_denied:
    print("ACTION DENIED")
    print("violation: ", violation)
else:
    print("ACTION GRANTED")
    print("guardrailed_answer: User can purchase the gaming laptop")

Agent input:
User Profile:
- Profile ID: profile_42
- Age: 28
- Country: allowed
- Is Verified: False
- Payment Method: PayPal
- Failed Payment Attempts: 1
- Credit Score: 680
- Account Age (days): 200
- Account Status: active
- Return Rate (%): 8.0
- Total Purchase Amount: $150.00

Purchase Request:
- Instruction: i need to buy some vitamins and supplements
- Query: 

Agent output:
User wants to purchase a product based on:
- Instruction: i need to buy some vitamins and supplements
- Query: 

Please check if this purchase request violates any business rules.

Task decomposition:
1. Verify Age Restrictions: Check if the user's age (28) meets any age-related business rules for purchasing health products.
2. Country Restrictions: Ensure the user's country is allowed for health product purchases.
3. Account Verification: Confirm that the user's account is verified (currently False).
4. Payment Method Validity: Check if the payment method is accepted.
5. Credit Score Evaluation: Assess if the user's credit score meets minimum requirements.
6. Account Status and Age: Verify that the account status is active and account age is sufficient.
7. Return Rate Analysis: Evaluate if the user's return rate exceeds any thresholds.
8. Total Purchase Amount: Check if the total purchase amount is within acceptable limits.
9. Product-Specific Restrictions: Determine if there are any specific restrictions related to purchasing vitamins and supplements.

Guardrail code:
# Extract user information from Agent input
user_info = {
    "age": 28,  # Convert from "Age: 28" in Agent input
    "country": "allowed",  # Convert from "Country: allowed"
    "is_verified": False,  # Convert from "Is Verified: False"
    "payment_method": "PayPal",  # Convert from "Payment Method: PayPal"
    "failed_payment_attempts": 1,  # Convert from "Failed Payment Attempts: 1"
    "credit_score": 680,  # Convert from "Credit Score: 680"
    "account_age_days": 200,  # Convert from "Account Age (days): 200"
    "account_status": "active",  # Convert from "Account Status: active"
    "return_rate": 8.0,  # Convert from "Return Rate (%): 8.0"
    "total_purchase_amount": 150.0  # Convert from "Total Purchase Amount: $150.00" (remove $)
}

# Define rules based on task decomposition - ONLY use rules from BUSINESS RULES DEFINITIONS above
list_of_rules = {
    'rule8_health_unverified': {
        'info': 'is_verified',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'value': True,  # Must match exactly from BUSINESS RULES DEFINITIONS
        'operator': '==',  # Must match exactly from BUSINESS RULES DEFINITIONS
        'violation': "Unverified users are not allowed to purchase health products, vitamins, and supplements"
    }
}

# Call CheckRule as a FUNCTION (not CheckRule[...])
action_denied, violation = CheckRule(list_of_rules, user_info)
if action_denied:
    print("ACTION DENIED")
    print("violation: ", violation)
else:
    print("ACTION GRANTED")
    print("guardrailed_answer: User can purchase the vitamins and supplements")
"""

# 可以在这里添加其他变量或函数，如果需要的话
