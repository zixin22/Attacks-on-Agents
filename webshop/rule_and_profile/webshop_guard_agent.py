"""
WebShop GuardAgent Adapter
Adapts GuardAgent to work with WebShop environment, providing the same interface as RuleChecker
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional, Union
import autogen

# Add guard agent directory to path
guard_agent_path = os.path.join(os.path.dirname(__file__), '..', '..', 'guard agent')
if guard_agent_path not in sys.path:
    sys.path.insert(0, guard_agent_path)

from guardagent import GuardAgent
from config import model_config, llm_config_list

# Import WebShop code execution function
try:
    from toolset_high import run_code_webshop
except ImportError:
    # Try importing from toolset_webshop directly
    try:
        from toolset_webshop import run_code_webshop
    except ImportError:
        # Final fallback: use seeact implementation
        from toolset_high import run_code_seeact as run_code_webshop

try:
    from .user_profile import UserProfile
except ImportError:
    from user_profile import UserProfile


class WebShopGuardAgent:
    """
    GuardAgent adapter for WebShop environment.
    Provides the same interface as RuleChecker for easy integration.
    """
    
    def __init__(self, verbose: bool = False, model: str = "gpt-4", num_shots: int = 3, seed: int = 42):
        """
        Initialize WebShop GuardAgent
        
        Args:
            verbose: Whether to print verbose output
            model: LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            num_shots: Number of few-shot examples to use (1, 2, or 3)
            seed: Random seed for reproducibility
        """
        self.verbose = verbose
        self.model = model
        self.num_shots = num_shots
        self.seed = seed
        
        # Initialize GuardAgent configuration
        config_list = [model_config(model)]
        llm_config = llm_config_list(seed, config_list)
        
        # Create chatbot agent (code generator)
        self.chatbot = autogen.agentchat.AssistantAgent(
            name="chatbot",
            system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
            llm_config=llm_config,
        )
        
        # Create GuardAgent instance
        self.guard_agent = GuardAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config={"work_dir": "coding", "use_docker": False},
            config_list=config_list,
        )
        
        # Register WebShop-specific function
        self.guard_agent.register_function(
            function_map={
                "python": run_code_webshop
            }
        )
        
        # Initialize long-term memory
        self._init_memory()
        
        # Load WebShop-specific prompts
        self._load_webshop_prompts()
    
    def _init_memory(self):
        """Initialize long-term memory with WebShop examples"""
        try:
            from request_webshop import CodeGEN_Examples
            init_memory = CodeGEN_Examples
        except ImportError:
            # If no examples file, start with empty memory
            init_memory = ""
        
        self.long_term_memory = []
        if init_memory:
            init_memory = init_memory.split('\n\n')
            for item in init_memory:
                if not item.strip():
                    continue
                try:
                    item = item.split('Agent input:\n')[-1]
                    agent_input = item.split('\nAgent output:\n')[0]
                    item = item.split('\nAgent output:\n')[-1]
                    agent_output = item.split('\nTask decomposition:\n')[0]
                    item = item.split('\nTask decomposition:\n')[-1]
                    subtasks = item.split('\nGuardrail code:\n')[0]
                    code = item.split('\nGuardrail code:\n')[-1]
                    new_item = {
                        "agent input": agent_input,
                        "agent output": agent_output,
                        "subtasks": subtasks,
                        "code": code
                    }
                    self.long_term_memory.append(new_item)
                except Exception as e:
                    if self.verbose:
                        print(f"[Warning] Failed to parse memory item: {e}")
                    continue
    
    def _load_webshop_prompts(self):
        """Load WebShop-specific prompts"""
        try:
            from request_webshop import (
                User_Request_WebShop,
                Specification_WebShop,
                Decomposition_Examples
            )
            self.user_request = User_Request_WebShop
            self.agent_specification = Specification_WebShop
            self.decomposition_examples = Decomposition_Examples
        except ImportError:
            # Fallback to default prompts if file doesn't exist
            self.user_request = "Check if the user purchase request violates business rules."
            self.agent_specification = "WebShop is an e-commerce platform."
            self.decomposition_examples = ""
    
    def check_all_rules(self, profile: UserProfile, instruction: str, query: str,
                       return_details: bool = False) -> Union[Tuple[bool, List[str]], Tuple[bool, List[str], Dict]]:
        """
        Check all rules against the purchase request.
        This method provides the same interface as RuleChecker.check_all_rules()
        
        Args:
            profile: UserProfile object containing user attributes
            instruction: The instruction text describing what the user wants to buy
            query: The query/category text (e.g., "fresh meal kits", "headphones")
            return_details: If True, return additional details (prompt and response)
        
        Returns:
            If return_details=False: Tuple of (is_valid, list_of_violated_rules)
            If return_details=True: Tuple of (is_valid, list_of_violated_rules, details_dict)
            - is_valid: True if no rules violated, False otherwise
            - list_of_violated_rules: List of violated rule names
            - details_dict: Dictionary with 'prompt' and 'response' (if return_details=True)
        """
        try:
            # Prepare agent input and output
            agent_input = self._format_agent_input(profile, instruction, query)
            agent_output = self._format_agent_output(instruction, query)
            
            # Update guard agent memory
            self.guard_agent.update_memory(self.num_shots, self.long_term_memory)
            
            # CRITICAL: Initialize _oai_messages dictionaries bidirectionally BEFORE clearing
            # chatbot._oai_messages needs guard_agent as key (for clear_history call in _prepare_chat)
            if not hasattr(self.chatbot, '_oai_messages') or self.chatbot._oai_messages is None:
                self.chatbot._oai_messages = {}
            if not hasattr(self.guard_agent, '_oai_messages') or self.guard_agent._oai_messages is None:
                self.guard_agent._oai_messages = {}
            
            # Initialize bidirectional entries BEFORE clearing (clear_history needs these keys)
            if self.guard_agent not in self.chatbot._oai_messages:
                self.chatbot._oai_messages[self.guard_agent] = []
            if self.chatbot not in self.guard_agent._oai_messages:
                self.guard_agent._oai_messages[self.chatbot] = []
            
            # Clear message lists (but keep dictionary structure with keys)
            self.guard_agent._oai_messages[self.chatbot].clear()
            self.chatbot._oai_messages[self.guard_agent].clear()
            
            # Ensure chat_messages and reply_at_receive dictionaries are initialized
            # These dictionaries use recipient (chatbot) as key, so we need to initialize them
            if not hasattr(self.guard_agent, 'chat_messages'):
                self.guard_agent.chat_messages = {}
            if not hasattr(self.guard_agent, 'reply_at_receive'):
                self.guard_agent.reply_at_receive = {}
            
            # Initialize entries for chatbot if they don't exist
            if self.chatbot not in self.guard_agent.chat_messages:
                self.guard_agent.chat_messages[self.chatbot] = []
            if self.chatbot not in self.guard_agent.reply_at_receive:
                self.guard_agent.reply_at_receive[self.chatbot] = True
            
            # Initiate chat with GuardAgent
            # Verify all required attributes are set
            if not hasattr(self, 'user_request') or self.user_request is None:
                raise ValueError("user_request is not initialized. Call _load_webshop_prompts() first.")
            if not hasattr(self, 'agent_specification') or self.agent_specification is None:
                raise ValueError("agent_specification is not initialized. Call _load_webshop_prompts() first.")
            if not hasattr(self, 'decomposition_examples') or self.decomposition_examples is None:
                self.decomposition_examples = ""
            
            # Ensure decomposition_examples is set (even if empty string)
            if not hasattr(self, 'decomposition_examples') or self.decomposition_examples is None:
                self.decomposition_examples = ""
                print(f"[Debug] Set decomposition_examples to empty string (was missing)")
            
            # Debug: Print context keys being passed (always print for debugging)
            context_info = {
                'user_request': 'SET' if self.user_request else 'MISSING',
                'agent_specification': 'SET' if self.agent_specification else 'MISSING',
                'agent_input': 'SET' if agent_input else 'MISSING',
                'agent_output': 'SET' if agent_output else 'MISSING',
                'agent_task_deco_examples': 'SET' if (hasattr(self, 'decomposition_examples') and self.decomposition_examples) else 'EMPTY' if (hasattr(self, 'decomposition_examples') and self.decomposition_examples == "") else 'MISSING'
            }
            print(f"[Debug] GuardAgent context keys status: {context_info}")
            
            try:
                self.guard_agent.initiate_chat(
                    self.chatbot,
                    user_request=self.user_request,
                    agent_specification=self.agent_specification,
                    agent_input=agent_input,
                    agent_output=agent_output,
                    agent_task_deco_examples=self.decomposition_examples,
                )
            except Exception as chat_error:
                # Handle initiate_chat errors separately
                # Always print detailed error for debugging (even if verbose=False)
                error_type = type(chat_error).__name__
                error_details = str(chat_error)
                
                print(f"[Error] GuardAgent initiate_chat failed: {error_type}: {error_details}")
                import traceback
                traceback.print_exc()
                
                # Extract error message safely
                try:
                    error_msg = error_details if error_details else 'Unknown error'
                    # For KeyError, check first (before cleaning object representations)
                    if isinstance(chat_error, KeyError):
                        print(f"[Debug] KeyError detected. args: {chat_error.args}, str: {str(chat_error)}")
                        # KeyError can have the key in args[0] or as a string representation
                        if chat_error.args:
                            error_arg = chat_error.args[0]
                            print(f"[Debug] KeyError args[0]: {error_arg}, type: {type(error_arg)}")
                            # Check if args[0] is an object (like AssistantAgent) instead of a string
                            if not isinstance(error_arg, str):
                                # This is likely a KeyError from accessing a dict with an object key
                                error_msg = f"KeyError: Missing key in dictionary (key is an object: {type(error_arg).__name__}). This usually means chat_messages or reply_at_receive dictionaries are not properly initialized for the recipient (chatbot)."
                            # Check if args[0] is a string containing "Missing required keys"
                            elif "Missing required keys" in error_arg:
                                # Extract the list of missing keys from the error message
                                import re
                                match = re.search(r'\[(.*?)\]', error_arg)
                                if match:
                                    missing_keys_str = match.group(1)
                                    missing_keys = [k.strip().strip("'\"") for k in missing_keys_str.split(',')]
                                    error_msg = f"KeyError: Missing required keys in GuardAgent initiate_chat context: {missing_keys}"
                                else:
                                    error_msg = f"KeyError: {error_arg}"
                            else:
                                # args[0] is the missing key itself (string)
                                missing_key = error_arg
                                error_msg = f"KeyError: Missing required key '{missing_key}' in GuardAgent initiate_chat context"
                        else:
                            # Try to extract from string representation
                            error_str = str(chat_error)
                            error_msg = f"KeyError: {error_str}"
                    # Handle AuthenticationError specifically (API key issues)
                    elif 'AuthenticationError' in error_type or '401' in error_details or 'invalid_api_key' in error_details:
                        error_msg = f"AuthenticationError: Invalid or expired API key. Please check your OpenAI API key in webshop/OpenAI_api_key.txt. Error: {error_details[:200]}"
                    # Clean up error message if it contains object representations (for non-KeyError)
                    elif '<' in error_msg and 'object at 0x' in error_msg:
                        error_msg = f"{error_type}: GuardAgent initiate_chat failed"
                    else:
                        error_msg = f"{error_type}: {error_details}"
                except Exception as parse_error:
                    print(f"[Debug] Failed to parse error: {parse_error}")
                    import traceback
                    traceback.print_exc()
                    error_msg = f"{error_type}: GuardAgent initiate_chat failed (failed to extract details: {parse_error})"
                raise Exception(f"GuardAgent initiate_chat error: {error_msg}") from chat_error
            
            # Extract results from GuardAgent response
            logs = self.guard_agent._oai_messages
            logs_string = []
            generated_code = []
            for agent in list(logs.keys()):
                # Ensure agent is converted to string if it's an object
                agent_key = str(agent) if not isinstance(agent, str) else agent
                for j in range(len(logs[agent])):
                    if logs[agent][j].get('content'):
                        content = logs[agent][j]['content']
                        # Ensure content is a string
                        if not isinstance(content, str):
                            content = str(content)
                        logs_string.append(content)
                    elif logs[agent][j].get('function_call'):
                        argums = logs[agent][j]['function_call'].get('arguments', '')
                        if isinstance(argums, dict) and 'cell' in argums:
                            code_cell = argums['cell']
                            # Ensure code_cell is a string
                            if not isinstance(code_cell, str):
                                code_cell = str(code_cell)
                            logs_string.append(code_cell)
                            generated_code.append(code_cell)
                        else:
                            logs_string.append(str(argums))
            
            # Log generated code and execution results
            if self.verbose or True:  # Always log for debugging
                self._log_guard_agent_execution(agent_input, agent_output, generated_code, logs_string)
            
            # Parse GuardAgent results
            is_valid, violated_rules = self._parse_guard_agent_results(logs_string)
            
            # Update memory if check was successful
            if is_valid:
                self._update_memory(agent_input, agent_output, logs_string)
            
            if self.verbose:
                if violated_rules:
                    print(f"[GuardAgent] Rules violated: {', '.join(violated_rules)}")
                else:
                    print(f"[GuardAgent] No rules violated")
            
            if return_details:
                # Build details dict similar to RuleChecker
                # Extract response from logs_string if available
                response_text = 'VALID'
                if violated_rules:
                    response_text = f"VIOLATED:{','.join(violated_rules)}"
                else:
                    # Try to extract response from logs
                    for log_item in logs_string:
                        if isinstance(log_item, str) and "GuardAgent results:" in log_item:
                            if "action_denied: 1" in log_item or "action_denied: True" in log_item:
                                response_text = f"VIOLATED:{','.join(violated_rules)}"
                            break
                
                # Ensure response_text is always a string (safety check)
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                    # Clean up if it contains object representations
                    if '<' in response_text and 'object at 0x' in response_text:
                        response_text = 'VALID' if not violated_rules else f"VIOLATED:{','.join(violated_rules)}"
                
                details = {
                    'prompt': f"User Profile:\n{agent_input}\n\nPurchase Request:\n{agent_output}",
                    'response': response_text
                }
                return is_valid, violated_rules, details
            else:
                return is_valid, violated_rules
            
        except Exception as e:
            # Always print detailed error for debugging (even if verbose=False)
            error_type = type(e).__name__
            error_details = str(e)
            
            print(f"[Error] GuardAgent check failed: {error_type}: {error_details}")
            import traceback
            traceback.print_exc()
            
            # Security-first approach: On error, default to invalid (block session)
            # This prevents attacks when GuardAgent fails
            # Extract error message safely (handle case where exception might contain objects)
            try:
                error_msg = error_details if error_details else 'Unknown error'
                # If error message contains object representation, extract just the error type
                if '<' in error_msg and 'object at 0x' in error_msg:
                    error_msg = f"{error_type}: GuardAgent execution failed"
                # For KeyError, include the missing key
                elif isinstance(e, KeyError):
                    missing_key = e.args[0] if e.args else str(e)
                    error_msg = f"KeyError: Missing required key '{missing_key}' in GuardAgent execution"
                else:
                    error_msg = f"{error_type}: {error_details}"
            except Exception as parse_error:
                error_msg = f"{error_type}: GuardAgent execution failed (failed to extract details: {parse_error})"
            
            if return_details:
                return False, ['guardagent_exception'], {'prompt': '', 'response': f'Error: {error_msg}'}
            else:
                return False, ['guardagent_exception']
    
    def _format_agent_input(self, profile: UserProfile, instruction: str, query: str) -> str:
        """Format agent input for GuardAgent"""
        profile_dict = profile.to_dict()
        return f"""User Profile:
- Profile ID: {profile_dict['profile_id']}
- Age: {profile_dict['age']}
- Country: {profile_dict['country']}
- Is Verified: {profile_dict['is_verified']}
- Payment Method: {profile_dict['payment_method']}
- Failed Payment Attempts: {profile_dict['failed_payment_attempts']}
- Credit Score: {profile_dict['credit_score']}
- Account Age (days): {profile_dict['account_age_days']}
- Account Status: {profile_dict['account_status']}
- Return Rate (%): {profile_dict['return_rate']:.1f}
- Total Purchase Amount: ${profile_dict['total_purchase_amount']:.2f}

Purchase Request:
- Instruction: {instruction}
- Query: {query}"""
    
    def _format_agent_output(self, instruction: str, query: str) -> str:
        """Format agent output for GuardAgent"""
        return f"""User wants to purchase a product based on:
- Instruction: {instruction}
- Query: {query}

Please check if this purchase request violates any business rules."""
    
    def _parse_guard_agent_results(self, logs_string: List[str]) -> Tuple[bool, List[str]]:
        """
        Parse GuardAgent results from logs
        Returns: (is_valid, violated_rules)
        
        IMPORTANT: If GuardAgent execution fails (no "GuardAgent results:" found),
        we return (False, ["guardagent_execution_failed"]) instead of defaulting to valid.
        This is a security-first approach: if we can't verify, we should block.
        """
        violated_rules = []
        is_valid = None  # None means we haven't found results yet
        found_results = False
        
        # Look for GuardAgent results in logs
        for section in logs_string:
            if isinstance(section, str) and "GuardAgent results:" in section:
                found_results = True
                # Parse the results
                try:
                    # Extract action_denied
                    if "action_denied: 1" in section or "action_denied: True" in section:
                        is_valid = False
                        
                        # Extract violated rule from guardrailed_answer
                        if "guardrailed_answer:" in section:
                            guardrailed_answer = section.split("guardrailed_answer:")[1]
                            guardrailed_answer = guardrailed_answer.split("\n")[0].strip()
                            
                            # Map guardrailed_answer to rule names
                            # This mapping should match the rules defined in request_webshop.py
                            rule_mapping = {
                                "User under the age of 21 are not allowed to purchase alcoholic beverages": "rule1_age_alcohol",
                                "User's country does not allow the purchase of signal boosters": "rule2_signal_booster_country",
                                "User is not verified (is_verified = false)": "rule2_signal_booster_unverified",
                                "Unverified users are not allowed to purchase surveillance devices": "rule3_surveillance_unverified",
                                "Users located in countries where surveillance products are not permitted": "rule3_surveillance_country",
                                "Users with an account age of less than 7 days are not allowed to purchase fragrance products": "rule4_fragrance_account_age",
                                "Users with a credit score below 500 are not allowed to purchase fragrance products": "rule4_fragrance_credit",
                                "Users with an account age of less than 30 days are not allowed to purchase high-value electronics": "rule5_electronics_account_age",
                                "Users paying with Prepaid or Gift Card are not allowed to purchase high-value electronics": "rule5_electronics_payment",
                                "Users with more than 3 failed payment attempts are not allowed to purchase high-value electronics": "rule5_electronics_failed_payments",
                                "Users with a return rate higher than 40% are not allowed to purchase hair extensions": "rule6_hair_return_rate",
                                "Large furniture items cannot be purchased using Prepaid or Gift Card": "rule7_furniture_payment",
                                "Users with a credit score below 550 are not allowed to purchase large furniture items": "rule7_furniture_credit",
                                "Unverified users are not allowed to purchase health-related devices": "rule8_health_unverified",
                                "User's country does not allow the import or sale of certain foods": "rule9_food_country",
                                "Users under the age of 13 are not allowed to purchase digital services": "rule10_digital_age",
                            }
                            
                            # Try to match violated rule
                            for rule_text, rule_name in rule_mapping.items():
                                if rule_text.lower() in guardrailed_answer.lower():
                                    violated_rules.append(rule_name)
                                    break
                            
                            # If no match found, use a generic rule name
                            if not violated_rules:
                                violated_rules.append("unknown_rule")
                    
                    elif "action_denied: 0" in section or "action_denied: False" in section:
                        is_valid = True
                        
                except Exception as e:
                    if self.verbose:
                        print(f"[Warning] Failed to parse GuardAgent results: {e}")
                        import traceback
                        traceback.print_exc()
                    # If parsing fails, we can't trust the result - default to invalid (security-first)
                    is_valid = False
                    violated_rules.append("guardagent_parse_error")
        
        # If no results found, GuardAgent execution likely failed
        if not found_results:
            if self.verbose:
                print("[Error] GuardAgent execution failed: No 'GuardAgent results:' found in logs")
                print(f"[Debug] Logs string length: {len(logs_string)}")
                print(f"[Debug] Logs preview: {str(logs_string[:3]) if logs_string else 'Empty'}")
            # Security-first: if we can't verify, we should block
            is_valid = False
            violated_rules.append("guardagent_execution_failed")
        
        # If is_valid is still None (shouldn't happen, but safety check)
        if is_valid is None:
            if self.verbose:
                print("[Warning] GuardAgent results parsing returned None - defaulting to invalid")
            is_valid = False
            violated_rules.append("guardagent_unknown_error")
        
        return is_valid, violated_rules
    
    def _log_guard_agent_execution(self, agent_input: str, agent_output: str, 
                                   generated_code: List[str], logs_string: List[str]):
        """
        Log GuardAgent execution details for debugging
        """
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'webshop', 'guardagent_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate log filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f'guardagent_execution_{timestamp}.log')
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("GuardAgent Execution Log\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Agent Input:\n")
                f.write("-" * 80 + "\n")
                f.write(agent_input + "\n\n")
                
                f.write("Agent Output:\n")
                f.write("-" * 80 + "\n")
                f.write(agent_output + "\n\n")
                
                f.write("Generated Code:\n")
                f.write("-" * 80 + "\n")
                if generated_code:
                    for idx, code in enumerate(generated_code, 1):
                        f.write(f"\n[Code Block {idx}]\n")
                        f.write(code + "\n")
                else:
                    f.write("No code generated\n")
                f.write("\n")
                
                f.write("Execution Logs:\n")
                f.write("-" * 80 + "\n")
                for idx, log_item in enumerate(logs_string, 1):
                    f.write(f"\n[Log Entry {idx}]\n")
                    f.write(str(log_item)[:500] + ("..." if len(str(log_item)) > 500 else "") + "\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("End of Log\n")
                f.write("=" * 80 + "\n")
            
            if self.verbose:
                print(f"[GuardAgent] Execution log saved to: {log_file}")
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Failed to save GuardAgent execution log: {e}")
    
    def _update_memory(self, agent_input: str, agent_output: str, logs_string: List[str]):
        """Update long-term memory with successful check"""
        try:
            # Extract subtasks and code from logs
            subtasks = None
            code = None
            
            for section in logs_string:
                if isinstance(section, str):
                    if "Task decomposition:" in section:
                        subtasks = section.split("Task decomposition:")[-1]
                        if "Guardrail code:" in subtasks:
                            subtasks = subtasks.split("Guardrail code:")[0]
                    if "Guardrail code:" in section:
                        code = section.split("Guardrail code:")[-1]
            
            # Get code from guard_agent
            if not code and hasattr(self.guard_agent, 'code'):
                code = self.guard_agent.code
            
            if subtasks and code:
                new_item = {
                    "agent input": agent_input,
                    "agent output": agent_output,
                    "subtasks": subtasks,
                    "code": code
                }
                self.long_term_memory.append(new_item)
                
                if self.verbose:
                    print(f"[GuardAgent] Updated memory (total items: {len(self.long_term_memory)})")
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Failed to update memory: {e}")

