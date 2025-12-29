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

SYSTEM_PROMPT_DECOMPOSITION = """You are the planner of a helpful AI assistant designed to provide guardrail to protect AI agents.
You will be provided with the following information.
1) Agent specification: A description of the agent to be protected, including the input and output to the agent and how the agent works.
2) User request: A description of the desired agent behavior by the user.
3) Agent input: An inference-time input to the agent to be protected.
4) Agent output: An inference-time output by the agent to be protected.
You need to first read the "user request" carefully and then generate a few subtasks for checking whether the "agent input" and the "agent output" satisfy the user request.
You will be provided several related examples.
Follow the examples to complete the "task decomposition" for the given "agent input" and "agent output".
"""

GuardAgent_Message_Prompt = """Write a guardrail code in python following the examples provided in below.
Here are some examples:
{examples}
(END OF EXAMPLES)
Agent input:
{agent_input}
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
2. Extract user information from the Agent input and create a user_info dictionary.
   - Keys MUST be lowercase and match exactly: "age", "country", "is_verified", "payment_method", "failed_payment_attempts", "credit_score", "account_age_days", "account_status", "return_rate", "total_purchase_amount"
   - Convert values appropriately: Age -> int, Country -> str, Is Verified -> bool, Payment Method -> str, Failed Payment Attempts -> int, Credit Score -> int, Account Age (days) -> int, Account Status -> str, Return Rate (%) -> float, Total Purchase Amount -> float (remove $ sign)
3. Create a list_of_rules dictionary based on the task decomposition and business rules.
   - Each key is a unique rule identifier (e.g., "rule1_age_alcohol", "rule5a_electronics_account_age")
   - Each value is a dict with keys: "info", "value", "operator", "violation"
   - "info" must match a key in user_info (e.g., "age", "account_age_days")
   - "operator" can be: ">=", "<=", ">", "<", "==", "!=", "in", "not in"
   - "value" is the threshold or value to compare against
4. Call CheckRule(list_of_rules, user_info) as a FUNCTION - use CheckRule(...) NOT CheckRule[...]
5. Based on the result, you MUST print either "ACTION DENIED" or "ACTION GRANTED" to stdout.
6. If ACTION DENIED, also print "violation: " followed by the violation message.
7. If ACTION GRANTED, you may print "guardrailed_answer: " followed by any additional information.

The code should look like this structure:
# Extract user information from Agent input
user_info = {{
    "age": 20,  # Convert from "Age: 20" in Agent input
    "country": "allowed",  # Convert from "Country: allowed"
    "is_verified": True,  # Convert from "Is Verified: True"
    "payment_method": "Visa",  # Convert from "Payment Method: Visa"
    "failed_payment_attempts": 0,  # Convert from "Failed Payment Attempts: 0"
    "credit_score": 700,  # Convert from "Credit Score: 700"
    "account_age_days": 365,  # Convert from "Account Age (days): 365"
    "account_status": "active",  # Convert from "Account Status: active"
    "return_rate": 10.0,  # Convert from "Return Rate (%): 10.0"
    "total_purchase_amount": 500.0  # Convert from "Total Purchase Amount: $500.00" (remove $)
}}

# Define rules based on task decomposition
list_of_rules = {{
    'rule1_age_alcohol': {{
        'info': 'age',  # Must match a key in user_info
        'value': 21,
        'operator': '>=',
        'violation': "Users under the age of 21 are not allowed to purchase alcoholic beverages"
    }}
}}

# Call CheckRule as a FUNCTION (not CheckRule[...])
action_denied, violation = CheckRule(list_of_rules, user_info)
if action_denied:
    print("ACTION DENIED")
    print("violation: ", violation)
else:
    print("ACTION GRANTED")
    print("guardrailed_answer: ", ...)

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

