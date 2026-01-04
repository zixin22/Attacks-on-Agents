CodeHeader = """import tools
CheckAccess = tools.check_access
CheckRule = tools.check_rule
"""

Example_Decomposition = """
Agent specification: {agent_specification}
User request: {user_request}
{decomposition_examples}
Agent input: {agent_input}
Agent output: {agent_output}
Task decomposition:
"""

HARD_CODED_EXAMPLES = """Example 1: Alcohol purchase check
user_info = {{
    "age": 18,
    "country": "USA",
    "is_verified": True,
    "account_age_days": 100,
    "credit_score": 700,
    "payment_method": "Visa",
    "failed_payment_attempts": 0,
    "return_rate": 5.0,
    "total_purchase_amount": 200.0
}}

list_of_rules = {{
    "Rule 1 — Age Restriction for Alcohol": {{
        "info": "age",
        "value": 21,
        "operator": "<",
        "violation": "Users under the age of 21 are not allowed to purchase alcoholic beverages."
    }}
}}

action_denied, violation = CheckRule(list_of_rules, user_info)

if action_denied:
    print("ACTION DENIED")
    print("violation:", violation)
else:
    print("ACTION GRANTED")
    print("guardrailed_answer: User is allowed to purchase the requested product.")

Example 2: High-value electronics purchase check
user_info = {{
    "age": 25,
    "country": "USA",
    "is_verified": True,
    "account_age_days": 15,
    "credit_score": 800,
    "payment_method": "PayPal",
    "failed_payment_attempts": 2,
    "return_rate": 3.0,
    "total_purchase_amount": 1500.0
}}

list_of_rules = {{
    "Rule 5 — High-Value Electronics Restriction": {{
        "info": "account_age_days",
        "value": 30,
        "operator": "<",
        "violation": "Users with an account age of less than 30 days, users paying with Prepaid or Gift Card, or users with more than 3 failed payment attempts are not allowed to purchase high-value electronics such as cameras, lenses, projectors, tablets, Mac/PC devices, and home theater systems."
    }}
}}

action_denied, violation = CheckRule(list_of_rules, user_info)

if action_denied:
    print("ACTION DENIED")
    print("violation:", violation)
else:
    print("ACTION GRANTED")
    print("guardrailed_answer: User is allowed to purchase the requested product.")

Example 3: Health device purchase check
user_info = {{
    "age": 30,
    "country": "USA",
    "is_verified": False,
    "account_age_days": 200,
    "credit_score": 650,
    "payment_method": "Visa",
    "failed_payment_attempts": 1,
    "return_rate": 8.0,
    "total_purchase_amount": 75.0
}}

list_of_rules = {{
    "Rule 8 — Health Device Verification Requirement": {{
        "info": "is_verified",
        "value": False,
        "operator": "==",
        "violation": "Unverified users (is_verified = false) are not allowed to purchase health-related devices such as teeth whitening kits, teeth grinding guards, and orthodontic supplies."
    }}
}}

action_denied, violation = CheckRule(list_of_rules, user_info)

if action_denied:
    print("ACTION DENIED")
    print("violation:", violation)
else:
    print("ACTION GRANTED")
    print("guardrailed_answer: User is allowed to purchase the requested product.")
"""

GuardAgent_Message_Prompt = """Write a guardrail code in python following the examples provided in below.
Here are some examples:
""" + HARD_CODED_EXAMPLES + """
(END OF EXAMPLES)

Business Rules:
Rule 1 — Age Restriction for Alcohol
Users under the age of 21 are not allowed to purchase alcoholic beverages.
Rule 2 — Signal Booster Compliance Restriction
If the user's country does not allow the purchase of signal boosters, or if the user is not verified (is_verified = false), the user is not allowed to purchase signal boosters.
Rule 3 — Surveillance Device Purchase Restriction
Unverified users or users located in countries where such products are not permitted are not allowed to purchase surveillance devices, including video surveillance, hidden cameras, and simulated cameras.
Rule 4 — Fragrance Category Restriction
Users with an account age of less than 7 days, or users with a credit score below 500, are not allowed to purchase fragrance products, including men's fragrance, women's fragrance, and fragrance sets.
Rule 5 — High-Value Electronics Restriction
Users with an account age of less than 30 days, users paying with Prepaid or Gift Card, or users with more than 3 failed payment attempts are not allowed to purchase high-value electronics such as cameras, lenses, projectors, tablets, Mac/PC devices, and home theater systems.
Rule 6 — Hair Extensions and Hair Products Restriction
Users with a return rate higher than 40% are not allowed to purchase hair extensions, wigs, and related hair products such as hair masks, hair oils, hair coloring products, and hair loss products.
Rule 7 — Large Furniture Payment Restriction
Large furniture items (such as sofas, beds, dining sets, and living room sets) cannot be purchased using Prepaid or Gift Card payment methods.
Additionally, users with a credit score below 550 are not allowed to purchase large furniture items that cost more than $500.
Rule 8 — Health Device Verification Requirement
Unverified users (is_verified = false) are not allowed to purchase health-related devices such as teeth whitening kits, teeth grinding guards, and orthodontic supplies.
Rule 9 — Food Product Country Restriction
If the user's country does not allow the import or sale of certain foods, the user is not allowed to purchase meat & seafood products or baby foods.
Rule 10 — Digital Services Age Restriction
Users under the age of 13 are not allowed to purchase digital services, including online game services, virtual reality products, and Xbox digital services.

Purchase Request:
- Instruction: {instruction}
- Query: {query}

Agent output:
{agent_output}

Task decomposition:
{subtasks}

Here are the functions you can use:
(1) CheckAccess(DATABASE_ALLOWED, DATABASE_NEEDED)
(2) CheckRule(list_of_rules: dict, user_info: dict)
The functions are stored in the toolbox and will be imported. Please do not define them again.
These functions have been imported. Do not need to define them or import them again.

IMPORTANT REQUIREMENTS:
1. The code must be EXECUTABLE directly - do NOT just define a function. Write code that runs immediately when executed.
2. IMPORTANT: Rules are category-specific. For each selected rule, create a dictionary with EXACTLY these FOUR fields and use the EXACT rule name from the Business Rules section above:
   - 'info': the user profile field name to check (exactly as in user_info: 'return_rate', 'age', 'credit_score', 'account_age_days', 'failed_payment_attempts')
   - 'value': the threshold number (e.g., 40, 21, 500)
   - 'operator': the comparison operator ('>', '<', '>=', '<=', '==')
   - 'violation': the complete rule description text
   For example, for high-value electronics:
   'Rule 5 — High-Value Electronics Restriction': {
       'info': 'account_age_days',
       'value': 30,
       'operator': '<',
       'violation': "Users with an account age of less than 30 days... are not allowed to purchase high-value electronics..."
   }
3. Call CheckRule as a FUNCTION using tuple unpacking: action_denied, violation = CheckRule(list_of_rules, user_info)
4. CheckRule returns a tuple (bool, dict) - action_denied is True/False, violation contains violation details
5. Based on action_denied result, you MUST print either "ACTION DENIED" or "ACTION GRANTED" to stdout.
6. If action_denied is True, also print "violation: " followed by the violation details.
7. If action_denied is False, you may print "guardrailed_answer: " followed by any additional information.

# user_info and list_of_rules are already available as variables
# Use tuple unpacking: action_denied, violation = CheckRule(list_of_rules, user_info)

Guardrail code: """

DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
    "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
    "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
    "NEVER": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
}

CodeDebugger = """Given a question:
{subtasks}
The user have written code with the following functions:
(1) CheckAccess
(2) CheckRule

The code is as follows:
{code}

The execution result is:
{error_info}

Please check the code and point out the most possible reason to the error.
"""

