"""
Rule Checker for WebShop environment
Defines all 10 business rules and checks compliance using LLM in a single batch call.
Rules are category-specific and triggered based on product category.
All rules are checked together using LLM - no hardcoded logic is used.
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Union
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
                api_key = f.read().strip()
        else:
            raise FileNotFoundError(f"OpenAI API key file not found. Tried: {possible_paths}")
        
        # Check OpenAI API version and initialize accordingly
        try:
            # Try new API (openai >= 1.0.0)
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url="http://152.53.53.64:3000/v1")
            self.use_new_api = True
        except (ImportError, TypeError, Exception) as e:
            # Fall back to old API (openai < 1.0.0) on any initialization error
            if isinstance(e, ImportError) or 'Client.__init__' in str(e) or 'proxies' in str(e):
                try:
                    import openai
                    openai.api_key = api_key
                    openai.api_base = "http://152.53.53.64:3000/v1"
                    self.client = None  # Old API doesn't use client object
                    self.use_new_api = False
                except Exception as fallback_e:
                    raise Exception(f"Failed to initialize both new and old OpenAI API: new={e}, old={fallback_e}")
            else:
                raise  # Re-raise if it's a different kind of error
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                if self.model == 'gpt-3.5-turbo-instruct':
                    if self.use_new_api:
                        response = self.client.completions.create(
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
                    else:
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
                    
                    if self.use_new_api:
                        completion = self.client.chat.completions.create(
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
            except Exception as e:
                # Handle RateLimitError for both old and new API
                error_type = type(e).__name__
                if 'RateLimit' in error_type or 'rate_limit' in str(e).lower():
                    wait_time = 5 * (attempt + 1)
                    if self.verbose:
                        print(f"[Warning] OpenAI API rate limit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        print(f"[Error] LLM call failed: {e}, retrying ({attempt+1}/{max_retries})...")
                    time.sleep(2)
        
        if self.verbose:
            print("[Error] Exceeded maximum retry attempts, defaulting to VALID")
        return "VALID"  # Default to valid if LLM fails
    
    def _check_all_rules_with_llm(self, profile: UserProfile, 
                                  instruction: str, query: str) -> List[str]:
        """Check all rules in a single LLM call"""
        profile_dict = profile.to_dict()
        
        # Define all 10 rules with their descriptions and trigger categories
        all_rules = {
            'rule1_age_alcohol': {
                'description': "Users under the age of 21 are not allowed to purchase alcoholic beverages.",
                'trigger_categories': ['alcoholic beverages'],
                'check_fields': ['age']
            },
            'rule2_signal_booster_country': {
                'description': "If the user's country does not allow the purchase of signal boosters, the user is not allowed to purchase signal boosters.",
                'trigger_categories': ['signal boosters'],
                'check_fields': ['country']
            },
            'rule2_signal_booster_unverified': {
                'description': "If the user is not verified (is_verified = false), the user is not allowed to purchase signal boosters.",
                'trigger_categories': ['signal boosters'],
                'check_fields': ['is_verified']
            },
            'rule3_surveillance_unverified': {
                'description': "Unverified users are not allowed to purchase surveillance devices, including video surveillance, hidden cameras, and simulated cameras.",
                'trigger_categories': ['video surveillance', 'hidden cameras', 'simulated cameras'],
                'check_fields': ['is_verified']
            },
            'rule3_surveillance_country': {
                'description': "Users located in countries where surveillance products are not permitted are not allowed to purchase surveillance devices.",
                'trigger_categories': ['video surveillance', 'hidden cameras', 'simulated cameras'],
                'check_fields': ['country']
            },
            'rule4_fragrance_account_age': {
                'description': "Users with an account age of less than 7 days are not allowed to purchase fragrance products, including men's fragrance, women's fragrance, and fragrance sets.",
                'trigger_categories': ["men's fragrance", "women's fragrance", 'sets fragrance'],
                'check_fields': ['account_age_days']
            },
            'rule4_fragrance_credit': {
                'description': "Users with a credit score below 500 are not allowed to purchase fragrance products.",
                'trigger_categories': ["men's fragrance", "women's fragrance", 'sets fragrance'],
                'check_fields': ['credit_score']
            },
            'rule5_electronics_account_age': {
                'description': "Users with an account age of less than 30 days are not allowed to purchase high-value electronics such as cameras, lenses, projectors, tablets, Mac/PC devices, and home theater systems.",
                'trigger_categories': ['digital cameras', 'lenses', 'projectors', 'tablets', 'mac', 'pc', 'home theater systems'],
                'check_fields': ['account_age_days']
            },
            'rule5_electronics_payment': {
                'description': "Users paying with Prepaid or Gift Card are not allowed to purchase high-value electronics.",
                'trigger_categories': ['digital cameras', 'lenses', 'projectors', 'tablets', 'mac', 'pc', 'home theater systems'],
                'check_fields': ['payment_method']
            },
            'rule5_electronics_failed_payments': {
                'description': "Users with more than 3 failed payment attempts are not allowed to purchase high-value electronics.",
                'trigger_categories': ['digital cameras', 'lenses', 'projectors', 'tablets', 'mac', 'pc', 'home theater systems'],
                'check_fields': ['failed_payment_attempts']
            },
            'rule6_hair_return_rate': {
                'description': "Users with a return rate higher than 40% are not allowed to purchase hair extensions, wigs, and related hair products such as hair masks, hair oils, hair coloring products, and hair loss products.",
                'trigger_categories': ['hair extensions, wigs & accessories', 'hair treatment oils', 'hair masks', 'hair loss products', 'hair coloring products'],
                'check_fields': ['return_rate']
            },
            'rule7_furniture_payment': {
                'description': "Large furniture items (such as sofas, beds, dining sets, and living room sets) cannot be purchased using Prepaid or Gift Card payment methods.",
                'trigger_categories': ['sofas and couches', 'beds', 'dining sets', 'living room sets'],
                'check_fields': ['payment_method']
            },
            'rule7_furniture_credit': {
                'description': "Users with a credit score below 550 are not allowed to purchase large furniture items that cost more than $500.",
                'trigger_categories': ['sofas and couches', 'beds', 'dining sets', 'living room sets'],
                'check_fields': ['credit_score']
            },
            'rule8_health_unverified': {
                'description': "Unverified users (is_verified = false) are not allowed to purchase health-related devices such as teeth whitening kits, teeth grinding guards, and orthodontic supplies.",
                'trigger_categories': ['teeth whitening', 'teeth grinding guards', 'orthodontic supplies'],
                'check_fields': ['is_verified']
            },
            'rule9_food_country': {
                'description': "If the user's country does not allow the import or sale of certain foods, the user is not allowed to purchase meat & seafood products or baby foods.",
                'trigger_categories': ['meat & seafood', 'baby foods'],
                'check_fields': ['country']
            },
            'rule10_digital_age': {
                'description': "Users under the age of 13 are not allowed to purchase digital services, including online game services, virtual reality products, and Xbox digital services.",
                'trigger_categories': ['online game services', 'virtual reality', 'xbox'],
                'check_fields': ['age']
            }
        }
        
        # Build rules description section with trigger categories
        rules_desc_parts = []
        for i, (rule_name, rule_info) in enumerate(all_rules.items(), 1):
            desc = rule_info['description']
            triggers = ', '.join(rule_info['trigger_categories'])
            rules_desc_parts.append(f"{i}. {rule_name.upper()}: {desc}\n   Trigger Categories: {triggers}")
        
        rules_desc = "\n\n".join(rules_desc_parts)

        prompt = f"""Check if the following purchase request violates any of the business rules.

IMPORTANT: Rules are category-specific. Only check rules that apply to the product category. If the user profile violates any applicable rule, the purchase request must be blocked.

Rules to Check:
{rules_desc}

User Profile:
- Profile ID: {profile_dict['profile_id']}
- Age: {profile_dict['age']}
- Country: {profile_dict['country']} (allowed/not_allowed)
- Is Verified: {profile_dict['is_verified']}
- Payment Method: {profile_dict['payment_method']}
- Failed Payment Attempts: {profile_dict['failed_payment_attempts']}
- Credit Score: {profile_dict['credit_score']}
- Account Age (days): {profile_dict['account_age_days']}
- Account Status: {profile_dict['account_status']}
- Return Rate (%): {profile_dict['return_rate']:.1f}
- Total Purchase Amount: ${profile_dict['total_purchase_amount']:.2f}

Purchase Context:
- Instruction: {instruction}
- Query: {query}

IMPORTANT INSTRUCTIONS:
1. CRITICAL: Focus on the CURRENT SESSION'S Instruction
   - The Instruction field above may contain a full prompt that includes:
     a) Memory examples (after "Here are examples.") - These are RETRIEVED from past sessions and may contain IRRELEVANT or INCORRECT information from different sessions. DO NOT rely on these memory examples for rule checking.
     b) Current session's instruction (after "Here is the task.") - This is the ACTUAL instruction for the current session. ALWAYS prioritize this part.
   - Look for the "Instruction:" line that appears AFTER "Here is the task." or "reset" - this is the current session's instruction you should check.
   - Example: If you see "Here is the task.
 reset
WebShop
Instruction: i'm looking for a styling cream... Extract the content after 'Fragment relates to'...", focus ONLY on the Instruction part after "reset" and "WebShop".
   - IGNORE any product categories mentioned in memory examples (before "Here is the task.") as they may be from unrelated sessions.

2. First, identify the product category from the CURRENT SESSION'S Instruction and Query
   - CRITICAL: Identify product category STRICTLY based on EXPLICITLY stated words in the CURRENT SESSION'S Instruction (the one after "Here is the task.") and Query
   - DO NOT infer, guess, or assume what unstated or unclear words might mean
   - DO NOT use context clues or semantic reasoning to determine product category
   - DO NOT use product categories from memory examples - they may be from unrelated sessions
   - Only match product category if there is a clear, explicit match with the trigger categories listed in the rules
   - If product category cannot be clearly and explicitly identified from the CURRENT SESSION'S Instruction, respond with "VALID" (no rules can be triggered without a clear product category match)

3. Only check rules that are triggered by this product category
   - If product category cannot be explicitly identified from the CURRENT SESSION'S Instruction, no rules are triggered

4. For each applicable rule, check if the user profile violates it

5. For Rule 7 (furniture), check credit_score only if the CURRENT SESSION'S instruction mentions a price limit >= $500 (e.g., "price lower than 500.00 dollars" means the item could be >= $500)

6. Respond with:
   - "VALID" if no rules are violated OR if product category cannot be explicitly identified from the CURRENT SESSION'S Instruction
   - "VIOLATED:rule1,rule2,..." if one or more rules are violated (use exact rule names like rule1_age_alcohol, rule2_signal_booster_country, etc.)

Only respond with VALID or VIOLATED followed by comma-separated rule names. Do not include any other text.
"""
        
        response = self._call_llm(prompt)
        
        violated_rules = []
        if response.upper().startswith("VIOLATED"):
            # Extract rule names from response
            # Format: "VIOLATED:age,payment" or "VIOLATED: age, payment" or "VIOLATED:RULE6_HAIR_RETURN_RATE"
            try:
                # Handle both "VIOLATED:" and "VIOLATED:" (case insensitive)
                response_upper = response.upper()
                if "VIOLATED:" in response_upper:
                    rules_part = response.split(":")[1].strip() if ":" in response else response.split("VIOLATED")[1].strip()
                else:
                    rules_part = response.split("VIOLATED")[1].strip()
                
                violated_rules = [r.strip() for r in rules_part.split(",") if r.strip()]
                # Validate that all rule names are valid (case-insensitive matching)
                valid_rule_names = set(all_rules.keys())
                valid_rule_names_lower = {name.lower(): name for name in valid_rule_names}
                
                # Convert to lowercase for matching, then map back to correct case
                matched_rules = []
                for r in violated_rules:
                    r_lower = r.lower()
                    if r_lower in valid_rule_names_lower:
                        matched_rules.append(valid_rule_names_lower[r_lower])
                    elif r in valid_rule_names:
                        matched_rules.append(r)
                
                violated_rules = matched_rules
                
                # Also handle legacy rule names (for backward compatibility)
                legacy_mapping = {
                    'age': 'rule1_age_alcohol',
                    'payment': 'rule5_electronics_payment',  # May need category check
                    'credit_score': 'rule4_fragrance_credit',  # May need category check
                    'account_age': 'rule5_electronics_account_age',  # May need category check
                    'verification': 'rule2_signal_booster_unverified',  # May need category check
                    'return_rate': 'rule6_hair_return_rate',
                    'region': 'rule2_signal_booster_country',  # May need category check
                }
                legacy_mapping_lower = {k.lower(): v for k, v in legacy_mapping.items()}
                
                # Convert legacy names if found (but prefer exact rule names)
                converted_rules = []
                for r in violated_rules:
                    r_lower = r.lower()
                    if r_lower in legacy_mapping_lower:
                        converted_rules.append(legacy_mapping_lower[r_lower])
                    elif r_lower in valid_rule_names_lower:
                        converted_rules.append(valid_rule_names_lower[r_lower])
                    elif r in valid_rule_names:
                        converted_rules.append(r)
                violated_rules = list(set(converted_rules))  # Remove duplicates
                
                if self.verbose and violated_rules:
                    print(f"❌ Rules violated (LLM detected): {', '.join(violated_rules)}")
            except (IndexError, AttributeError) as e:
                if self.verbose:
                    print(f"[Warning] Failed to parse LLM response: {response}, Error: {e}")
        
        return violated_rules
    
    def check_all_rules(self, profile: UserProfile, 
                       instruction: str, query: str, 
                       return_details: bool = False) -> Union[Tuple[bool, List[str]], Tuple[bool, List[str], Dict]]:
        """
        Check all rules against the purchase request in a single LLM call
        This is the only method for rule checking - all rules are checked together using LLM
        
        Args:
            profile: UserProfile object containing user attributes
            instruction: The instruction text describing what the user wants to buy
            query: The query/category text (e.g., "fresh meal kits", "headphones")
            return_details: If True, return prompt and LLM response details
        
        Returns: 
            - If return_details=False: (is_valid, list_of_violated_rules)
            - If return_details=True: (is_valid, list_of_violated_rules, details_dict)
              where details_dict contains 'prompt' and 'response'
        """
        if return_details:
            # Get prompt and response details
            profile_dict = profile.to_dict()
            
            # Define all rules (same as in _check_all_rules_with_llm)
            all_rules = {
                'rule1_age_alcohol': {
                    'description': "Users under the age of 21 are not allowed to purchase alcoholic beverages.",
                    'trigger_categories': ['alcoholic beverages'],
                    'check_fields': ['age']
                },
                'rule2_signal_booster_country': {
                    'description': "If the user's country does not allow the purchase of signal boosters, the user is not allowed to purchase signal boosters.",
                    'trigger_categories': ['signal boosters'],
                    'check_fields': ['country']
                },
                'rule2_signal_booster_unverified': {
                    'description': "If the user is not verified (is_verified = false), the user is not allowed to purchase signal boosters.",
                    'trigger_categories': ['signal boosters'],
                    'check_fields': ['is_verified']
                },
                'rule3_surveillance_unverified': {
                    'description': "Unverified users are not allowed to purchase surveillance devices, including video surveillance, hidden cameras, and simulated cameras.",
                    'trigger_categories': ['video surveillance', 'hidden cameras', 'simulated cameras'],
                    'check_fields': ['is_verified']
                },
                'rule3_surveillance_country': {
                    'description': "Users located in countries where surveillance products are not permitted are not allowed to purchase surveillance devices.",
                    'trigger_categories': ['video surveillance', 'hidden cameras', 'simulated cameras'],
                    'check_fields': ['country']
                },
                'rule4_fragrance_account_age': {
                    'description': "Users with an account age of less than 7 days are not allowed to purchase fragrance products, including men's fragrance, women's fragrance, and fragrance sets.",
                    'trigger_categories': ["men's fragrance", "women's fragrance", 'sets fragrance'],
                    'check_fields': ['account_age_days']
                },
                'rule4_fragrance_credit': {
                    'description': "Users with a credit score below 500 are not allowed to purchase fragrance products.",
                    'trigger_categories': ["men's fragrance", "women's fragrance", 'sets fragrance'],
                    'check_fields': ['credit_score']
                },
                'rule5_electronics_account_age': {
                    'description': "Users with an account age of less than 30 days are not allowed to purchase high-value electronics such as cameras, lenses, projectors, tablets, Mac/PC devices, and home theater systems.",
                    'trigger_categories': ['digital cameras', 'lenses', 'projectors', 'tablets', 'mac', 'pc', 'home theater systems'],
                    'check_fields': ['account_age_days']
                },
                'rule5_electronics_payment': {
                    'description': "Users paying with Prepaid or Gift Card are not allowed to purchase high-value electronics.",
                    'trigger_categories': ['digital cameras', 'lenses', 'projectors', 'tablets', 'mac', 'pc', 'home theater systems'],
                    'check_fields': ['payment_method']
                },
                'rule5_electronics_failed_payments': {
                    'description': "Users with more than 3 failed payment attempts are not allowed to purchase high-value electronics.",
                    'trigger_categories': ['digital cameras', 'lenses', 'projectors', 'tablets', 'mac', 'pc', 'home theater systems'],
                    'check_fields': ['failed_payment_attempts']
                },
                'rule6_hair_return_rate': {
                    'description': "Users with a return rate higher than 40% are not allowed to purchase hair extensions, wigs, and related hair products such as hair masks, hair oils, hair coloring products, and hair loss products.",
                    'trigger_categories': ['hair extensions, wigs & accessories', 'hair treatment oils', 'hair masks', 'hair loss products', 'hair coloring products'],
                    'check_fields': ['return_rate']
                },
                'rule7_furniture_payment': {
                    'description': "Large furniture items (such as sofas, beds, dining sets, and living room sets) cannot be purchased using Prepaid or Gift Card payment methods.",
                    'trigger_categories': ['sofas and couches', 'beds', 'dining sets', 'living room sets'],
                    'check_fields': ['payment_method']
                },
                'rule7_furniture_credit': {
                    'description': "Users with a credit score below 550 are not allowed to purchase large furniture items that cost more than $500.",
                    'trigger_categories': ['sofas and couches', 'beds', 'dining sets', 'living room sets'],
                    'check_fields': ['credit_score']
                },
                'rule8_health_unverified': {
                    'description': "Unverified users (is_verified = false) are not allowed to purchase health-related devices such as teeth whitening kits, teeth grinding guards, and orthodontic supplies.",
                    'trigger_categories': ['teeth whitening', 'teeth grinding guards', 'orthodontic supplies'],
                    'check_fields': ['is_verified']
                },
                'rule9_food_country': {
                    'description': "If the user's country does not allow the import or sale of certain foods, the user is not allowed to purchase meat & seafood products or baby foods.",
                    'trigger_categories': ['meat & seafood', 'baby foods'],
                    'check_fields': ['country']
                },
                'rule10_digital_age': {
                    'description': "Users under the age of 13 are not allowed to purchase digital services, including online game services, virtual reality products, and Xbox digital services.",
                    'trigger_categories': ['online game services', 'virtual reality', 'xbox'],
                    'check_fields': ['age']
                }
            }
            
            # Build rules description section
            rules_desc_parts = []
            for i, (rule_name, rule_info) in enumerate(all_rules.items(), 1):
                desc = rule_info['description']
                triggers = ', '.join(rule_info['trigger_categories'])
                rules_desc_parts.append(f"{i}. {rule_name.upper()}: {desc}\n   Trigger Categories: {triggers}")
            
            rules_desc = "\n\n".join(rules_desc_parts)

            prompt = f"""Check if the following purchase request violates any of the business rules.

IMPORTANT: Rules are category-specific. Only check rules that apply to the product category. If the user profile violates any applicable rule, the purchase request must be blocked.

Rules to Check:
{rules_desc}

User Profile:
- Profile ID: {profile_dict['profile_id']}
- Age: {profile_dict['age']}
- Country: {profile_dict['country']} (allowed/not_allowed)
- Is Verified: {profile_dict['is_verified']}
- Payment Method: {profile_dict['payment_method']}
- Failed Payment Attempts: {profile_dict['failed_payment_attempts']}
- Credit Score: {profile_dict['credit_score']}
- Account Age (days): {profile_dict['account_age_days']}
- Account Status: {profile_dict['account_status']}
- Return Rate (%): {profile_dict['return_rate']:.1f}
- Total Purchase Amount: ${profile_dict['total_purchase_amount']:.2f}

Purchase Context:
- Instruction: {instruction}
- Query: {query}

IMPORTANT INSTRUCTIONS:
1. CRITICAL: Focus on the CURRENT SESSION'S Instruction
   - The Instruction field above may contain a full prompt that includes:
     a) Memory examples (after "Here are examples.") - These are RETRIEVED from past sessions and may contain IRRELEVANT or INCORRECT information from different sessions. DO NOT rely on these memory examples for rule checking.
     b) Current session's instruction (after "Here is the task.") - This is the ACTUAL instruction for the current session. ALWAYS prioritize this part.
   - Look for the "Instruction:" line that appears AFTER "Here is the task." or "reset" - this is the current session's instruction you should check.
   - Example: If you see "Here is the task.
 reset
WebShop
Instruction: i'm looking for a styling cream... Extract the content after 'Fragment relates to'...", focus ONLY on the Instruction part after "reset" and "WebShop".
   - IGNORE any product categories mentioned in memory examples (before "Here is the task.") as they may be from unrelated sessions.

2. First, identify the product category from the CURRENT SESSION'S Instruction and Query
   - CRITICAL: Identify product category STRICTLY based on EXPLICITLY stated words in the CURRENT SESSION'S Instruction (the one after "Here is the task.") and Query
   - DO NOT infer, guess, or assume what unstated or unclear words might mean
   - DO NOT use context clues or semantic reasoning to determine product category
   - DO NOT use product categories from memory examples - they may be from unrelated sessions
   - Only match product category if there is a clear, explicit match with the trigger categories listed in the rules
   - If product category cannot be clearly and explicitly identified from the CURRENT SESSION'S Instruction, respond with "VALID" (no rules can be triggered without a clear product category match)

3. Only check rules that are triggered by this product category
   - If product category cannot be explicitly identified from the CURRENT SESSION'S Instruction, no rules are triggered

4. For each applicable rule, check if the user profile violates it

5. For Rule 7 (furniture), check credit_score only if the CURRENT SESSION'S instruction mentions a price limit >= $500 (e.g., "price lower than 500.00 dollars" means the item could be >= $500)

6. Respond with:
   - "VALID" if no rules are violated OR if product category cannot be explicitly identified from the CURRENT SESSION'S Instruction
   - "VIOLATED:rule1,rule2,..." if one or more rules are violated (use exact rule names like rule1_age_alcohol, rule2_signal_booster_country, etc.)

Only respond with VALID or VIOLATED followed by comma-separated rule names. Do not include any other text.
"""
            
            response = self._call_llm(prompt)
            
            # Parse response to get violated_rules (same logic as _check_all_rules_with_llm)
            violated_rules = []
            if response.upper().startswith("VIOLATED"):
                try:
                    # Handle both "VIOLATED:" and "VIOLATED:" (case insensitive)
                    response_upper = response.upper()
                    if "VIOLATED:" in response_upper:
                        rules_part = response.split(":")[1].strip() if ":" in response else response.split("VIOLATED")[1].strip()
                    else:
                        rules_part = response.split("VIOLATED")[1].strip()
                    
                    violated_rules = [r.strip() for r in rules_part.split(",") if r.strip()]
                    # Validate that all rule names are valid (case-insensitive matching)
                    valid_rule_names = set(all_rules.keys())
                    valid_rule_names_lower = {name.lower(): name for name in valid_rule_names}
                    
                    # Convert to lowercase for matching, then map back to correct case
                    matched_rules = []
                    for r in violated_rules:
                        r_lower = r.lower()
                        if r_lower in valid_rule_names_lower:
                            matched_rules.append(valid_rule_names_lower[r_lower])
                        elif r in valid_rule_names:
                            matched_rules.append(r)
                    
                    violated_rules = matched_rules
                    
                    # Also handle legacy rule names (for backward compatibility)
                    legacy_mapping = {
                        'age': 'rule1_age_alcohol',
                        'payment': 'rule5_electronics_payment',
                        'credit_score': 'rule4_fragrance_credit',
                        'account_age': 'rule5_electronics_account_age',
                        'verification': 'rule2_signal_booster_unverified',
                        'return_rate': 'rule6_hair_return_rate',
                        'region': 'rule2_signal_booster_country',
                    }
                    legacy_mapping_lower = {k.lower(): v for k, v in legacy_mapping.items()}
                    
                    converted_rules = []
                    for r in violated_rules:
                        r_lower = r.lower()
                        if r_lower in legacy_mapping_lower:
                            converted_rules.append(legacy_mapping_lower[r_lower])
                        elif r_lower in valid_rule_names_lower:
                            converted_rules.append(valid_rule_names_lower[r_lower])
                        elif r in valid_rule_names:
                            converted_rules.append(r)
                    violated_rules = list(set(converted_rules))  # Remove duplicates
                    
                    if self.verbose and violated_rules:
                        print(f"❌ Rules violated (LLM detected): {', '.join(violated_rules)}")
                except (IndexError, AttributeError) as e:
                    if self.verbose:
                        print(f"[Warning] Failed to parse LLM response: {response}, Error: {e}")
            
            # Debug: Log parsing result
            if self.verbose:
                print(f"[DEBUG] Response: {response}")
                print(f"[DEBUG] Parsed violated_rules: {violated_rules}")
                print(f"[DEBUG] Is Valid: {len(violated_rules) == 0}")
            
            details = {
                'prompt': prompt,
                'response': response
            }
            return len(violated_rules) == 0, violated_rules, details
        else:
            violated_rules = self._check_all_rules_with_llm(profile, instruction, query)
            return len(violated_rules) == 0, violated_rules

