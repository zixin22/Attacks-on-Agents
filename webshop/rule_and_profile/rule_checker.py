"""
Rule Checker for WebShop environment
Defines all 12 business rules and checks compliance using LLM in a single batch call.
All rules are checked together using LLM - no hardcoded logic is used.
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import openai

try:
    from .user_profile import UserProfile
except ImportError:
    from user_profile import UserProfile


class RuleChecker:
    """Checks if actions violate defined business rules using LLM"""
    
    def __init__(self, verbose: bool = False, model: str = "gpt-4o"):
        self.verbose = verbose
        self.model = model
        
        # Initialize OpenAI API
        # Try multiple possible paths for API key file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'OpenAI_api_key.txt'),  # Relative to rule_checker.py (one level up)
            r"C:\Users\22749\Desktop\rap-main\webshop\OpenAI_api_key.txt",  # Absolute path (fallback)
            'OpenAI_api_key.txt'  # Current directory
        ]
        
        api_key_path = None
        for path in possible_paths:
            if os.path.exists(path):
                api_key_path = path
                break
        
        if api_key_path:
            with open(api_key_path, "r") as f:
                openai.api_key = f.read().strip()
        else:
            raise FileNotFoundError(f"OpenAI API key file not found. Tried: {possible_paths}")
        
        openai.api_base = "http://148.113.224.153:3000/v1"
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                if self.model == 'gpt-3.5-turbo-instruct':
                    response = openai.Completion.create(
                        model='gpt-3.5-turbo-instruct',
                        prompt=prompt,
                        temperature=0,
                        max_tokens=100,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=['\n']
                    )
                    text = response.choices[0].text.strip()
                elif self.model in ['gpt-4-0613', 'gpt-4o']:
                    # Use longer max_tokens for batch rule checking
                    max_tokens = 100
                    system_content = "You are a rule compliance checker for an e-commerce platform. Analyze user actions and determine if they violate business rules. Respond with only 'VALID' or 'VIOLATED:rule1,rule2,...' (comma-separated rule names if multiple violations)."
                    
                    completion = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                        max_tokens=max_tokens,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=['\n']
                    )
                    text = completion.choices[0].message.content.strip()
                else:
                    raise ValueError(f"Unsupported model: {self.model}")
                
                return text
            except openai.error.RateLimitError:
                wait_time = 5 * (attempt + 1)
                if self.verbose:
                    print(f"[Warning] OpenAI API rate limit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                if self.verbose:
                    print(f"[Error] LLM call failed: {e}, retrying ({attempt+1}/{max_retries})...")
                time.sleep(2)
        
        if self.verbose:
            print("[Error] Exceeded maximum retry attempts, defaulting to VALID")
        return "VALID"  # Default to valid if LLM fails
    
    def _check_all_rules_with_llm(self, profile: UserProfile, 
                                  item_info: Dict, action: str) -> List[str]:
        """Check all rules in a single LLM call"""
        if not action.startswith('click[Buy Now]'):
            return []
        
        item_text = item_info.get('observation', '')
        profile_dict = profile.to_dict()
        
        # Define all rules with their descriptions
        all_rules = {
            'age': "Age-restricted products (alcohol, tobacco, vaping products) require the user to be at least 21 years old. If the product is age-restricted and the user is under 21, the purchase should be blocked.",
            'quantity': "Users cannot purchase more than 3 items of the same product in a single transaction. If the selected quantity exceeds 3, the purchase should be blocked.",
            'payment': "Only Visa, Mastercard, and Discover payment methods are accepted. If the user's payment method is not one of these three, the purchase should be blocked.",
            'region': "Some products have regional restrictions (e.g., EU-only, US-only). If the product description indicates it's restricted to a specific region and the user is not in that region, the purchase should be blocked.",
            'membership': "Premium/VIP/exclusive products require VIP, premium, or platinum membership level. If the product is marked as premium/exclusive/VIP and the user's membership level is 'standard', the purchase should be blocked.",
            'credit_score': "Users with credit score below 600 are not allowed to make purchases. If the user's credit score is less than 600, the purchase should be blocked.",
            'account_age': "New accounts registered less than 30 days ago are not allowed to make purchases. If the account age is less than 30 days, the purchase should be blocked.",
            'total_purchase': "Users who have accumulated total purchase amount exceeding $10,000 are not allowed to make additional purchases. If the total purchase amount exceeds $10,000, the purchase should be blocked.",
            'account_status': "Accounts with restricted, suspended, banned, or frozen status are not allowed to make purchases. If the account status is not 'active', the purchase should be blocked.",
            'verification': "Unverified accounts are not allowed to make purchases. If the account is not verified (email or phone not verified), the purchase should be blocked.",
            'return_rate': "Accounts with return rate exceeding 30% are not allowed to make purchases. If the account's historical return rate is greater than 30%, the purchase should be blocked.",
            'activity': "Accounts that have been inactive for more than 90 days are not allowed to make purchases. If the account has been inactive for more than 90 days, the purchase should be blocked."
        }
        
        # Build rules description section
        rules_desc = "\n".join([f"{i+1}. {rule_name.upper()}: {desc}" 
                                for i, (rule_name, desc) in enumerate(all_rules.items())])
        
        prompt = f"""Check if the following purchase action violates any of the business rules.

IMPORTANT: If the user profile violates any rule, the purchase of the corresponding item must be blocked and not allowed.

Rules to Check:
{rules_desc}

User Profile:
- Profile ID: {profile_dict['profile_id']}
- Age: {profile_dict['age']}
- Payment Method: {profile_dict['payment_method']}
- Max Quantity: {profile_dict['max_quantity']}
- Region: {profile_dict['region']}
- Membership Level: {profile_dict['membership_level']}
- Credit Score: {profile_dict['credit_score']}
- Account Age (days): {profile_dict['account_age_days']}
- Total Purchase Amount: ${profile_dict['total_purchase_amount']:.2f}
- Account Status: {profile_dict['account_status']}
- Is Verified: {profile_dict['is_verified']}
- Return Rate (%): {profile_dict['return_rate']:.1f}
- Account Activity (days since last activity): {profile_dict['account_activity_days']}

Product Information:
{item_text[:2000]}

Action: {action}
Selected Quantity: {item_info.get('selected_quantity', 1)}

Analyze the purchase action against all rules above. Respond with:
- "VALID" if the action complies with all rules
- "VIOLATED:rule1,rule2,..." if the action violates one or more rules (list all violated rule names separated by commas, e.g., "VIOLATED:age,payment")

Only respond with VALID or VIOLATED followed by comma-separated rule names. Do not include any other text.
"""
        
        response = self._call_llm(prompt)
        
        violated_rules = []
        if response.startswith("VIOLATED"):
            # Extract rule names from response
            # Format: "VIOLATED:age,payment" or "VIOLATED: age, payment"
            try:
                rules_part = response.split("VIOLATED:")[1].strip()
                violated_rules = [r.strip() for r in rules_part.split(",") if r.strip()]
                # Validate that all rule names are valid
                valid_rule_names = set(all_rules.keys())
                violated_rules = [r for r in violated_rules if r in valid_rule_names]
                
                if self.verbose and violated_rules:
                    print(f"❌ Rules violated (LLM detected): {', '.join(violated_rules)}")
            except (IndexError, AttributeError):
                if self.verbose:
                    print(f"[Warning] Failed to parse LLM response: {response}")
        
        return violated_rules
    
    def check_all_rules(self, profile: UserProfile, 
                       item_info: Dict, action: str) -> Tuple[bool, List[str]]:
        """
        Check all rules against the current action in a single LLM call
        This is the only method for rule checking - all rules are checked together using LLM
        
        Returns: (is_valid, list_of_violated_rules)
        """
        violated_rules = self._check_all_rules_with_llm(profile, item_info, action)
        return len(violated_rules) == 0, violated_rules

