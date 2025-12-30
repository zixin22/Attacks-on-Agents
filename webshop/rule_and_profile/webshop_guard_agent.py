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
    
    def __init__(self, verbose: bool = False, model: str = "gpt-4", num_shots: int = 3, seed: int = 42, output_dir: str = None):
        """
        Initialize WebShop GuardAgent

        Args:
            verbose: Whether to print verbose output
            model: LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            num_shots: Number of few-shot examples to use (1, 2, or 3)
            seed: Random seed for reproducibility
            output_dir: Output directory for detailed logs (optional)
        """
        self.verbose = verbose
        self.model = model
        self.num_shots = num_shots
        self.seed = seed
        self.output_dir = output_dir

        # Counter for log entries
        self.log_entry_counter = 0

        # Use default OpenAI API (not custom proxy)
        # Comment out custom base URL to use official OpenAI API
        # import os
        # base_url = "http://152.53.53.64:3000/v1"
        # os.environ["OPENAI_BASE_URL"] = base_url
        
        # Initialize GuardAgent configuration
        config_list = [model_config(model)]
        llm_config = llm_config_list(seed, config_list)
        
        # Create chatbot agent (code generator)
        # Create a custom AssistantAgent class that filters None content messages
        class FilteredAssistantAgent(autogen.agentchat.AssistantAgent):
            def _clean_message(self, msg):
                """Helper function to clean a single message"""
                if isinstance(msg, dict):
                    msg_copy = msg.copy()
                    # Ensure content is not None - ALWAYS set content to string for OpenAI API
                    has_function_call = 'function_call' in msg_copy and msg_copy.get('function_call') is not None
                    has_tool_calls = 'tool_calls' in msg_copy and msg_copy.get('tool_calls') is not None

                    # For OpenAI API, content must always be a string (never None)
                    if 'content' not in msg_copy or msg_copy.get('content') is None:
                        if not has_function_call and not has_tool_calls:
                            msg_copy['content'] = ""
                        else:
                            # For function/tool call messages, content can be None but we set it to empty string anyway
                            msg_copy['content'] = ""
                    elif not isinstance(msg_copy['content'], str):
                        # Convert content to string if it's not already
                        msg_copy['content'] = str(msg_copy['content'])

                    # Ensure role is present (required by OpenAI API)
                    if 'role' not in msg_copy:
                        # Try to infer role from context, default to 'user'
                        msg_copy['role'] = 'user'

                    return msg_copy
                elif isinstance(msg, str):
                    # If message is a string, ensure it's not None
                    return msg if msg is not None else ""
                else:
                    # For other types, convert to string
                    return str(msg) if msg is not None else ""
            
            def _append_oai_message(self, message, conversation_id, role='assistant', name=None, is_sending=None):
                """
                Override _append_oai_message to clean messages before they're stored.
                This ensures all messages in internal storage are clean.
                """
                # Clean message before appending
                cleaned_message = self._clean_message(message)
                # Final safety check: ensure cleaned message has valid content
                if isinstance(cleaned_message, dict):
                    if cleaned_message.get('content') is None:
                        has_function_call = 'function_call' in cleaned_message and cleaned_message.get('function_call') is not None
                        if not has_function_call:
                            cleaned_message = cleaned_message.copy()
                            cleaned_message['content'] = ""
                # Call parent's _append_oai_message with cleaned message
                # Handle different AutoGen versions
                try:
                    # Try new API with is_sending parameter
                    return super()._append_oai_message(cleaned_message, conversation_id, role, name, is_sending)
                except TypeError:
                    # Fall back to old API without is_sending parameter
                    return super()._append_oai_message(cleaned_message, conversation_id, role, name)
            
            def _process_received_message(self, message, sender, silent):
                """
                Override _process_received_message to clean messages before processing.
                This ensures messages are cleaned when received.
                """
                # Clean message before processing
                cleaned_message = self._clean_message(message)
                # Final safety check: ensure cleaned message has valid content
                if isinstance(cleaned_message, dict):
                    if cleaned_message.get('content') is None:
                        has_function_call = 'function_call' in cleaned_message and cleaned_message.get('function_call') is not None
                        if not has_function_call:
                            cleaned_message = cleaned_message.copy()
                            cleaned_message['content'] = ""
                # Call parent's _process_received_message with cleaned message
                return super()._process_received_message(cleaned_message, sender, silent)
            
            def generate_oai_reply(self, messages=None, sender=None, config=None):
                """
                Override generate_oai_reply to clean messages before calling OpenAI API.
                This is CRITICAL because autogen calls this method through reply_func,
                and we need to ensure messages are cleaned here.
                """
                # CRITICAL: Clean messages in internal storage FIRST before autogen accesses them
                if sender is not None:
                    # Clean chat_messages[sender] directly (autogen uses this internally)
                    if hasattr(self, 'chat_messages') and sender in self.chat_messages:
                        self.chat_messages[sender] = [self._clean_message(msg) for msg in self.chat_messages[sender]]
                    
                    # Clean _oai_messages[sender] if it exists (autogen may use these)
                    if hasattr(self, '_oai_messages') and self._oai_messages is not None:
                        if sender in self._oai_messages:
                            self._oai_messages[sender] = [self._clean_message(msg) for msg in self._oai_messages[sender]]
                
                # Clean messages parameter if provided
                if messages is not None:
                    messages = [self._clean_message(msg) for msg in messages]
                elif sender is not None and sender in self.chat_messages:
                    # If messages is None, use and clean self.chat_messages[sender]
                    messages = [self._clean_message(msg) for msg in self.chat_messages[sender]]
                
                # Final safety check: ensure all messages have valid content
                if messages is not None:
                    cleaned_messages = []
                    for idx, msg in enumerate(messages):
                        cleaned_msg = self._clean_message(msg)
                        # Double-check: if content is still None after cleaning, set to empty string
                        if isinstance(cleaned_msg, dict):
                            # Ensure content is always a string (not None)
                            if cleaned_msg.get('content') is None:
                                has_function_call = 'function_call' in cleaned_msg and cleaned_msg.get('function_call') is not None
                                # For function role messages, content can be None, but OpenAI API requires empty string
                                if cleaned_msg.get('role') == 'function':
                                    cleaned_msg = cleaned_msg.copy()
                                    cleaned_msg['content'] = ""
                                elif not has_function_call:
                                    cleaned_msg = cleaned_msg.copy()
                                    cleaned_msg['content'] = ""
                            # Ensure content is a string type
                            elif not isinstance(cleaned_msg.get('content'), str):
                                cleaned_msg = cleaned_msg.copy()
                                cleaned_msg['content'] = str(cleaned_msg['content']) if cleaned_msg['content'] is not None else ""
                            # Ensure role is present
                            if 'role' not in cleaned_msg:
                                cleaned_msg = cleaned_msg.copy()
                                cleaned_msg['role'] = 'user'  # Default role
                        cleaned_messages.append(cleaned_msg)
                    messages = cleaned_messages
                    
                    # CRITICAL: Update internal storage with cleaned messages BEFORE calling parent
                    # This ensures autogen's internal methods see cleaned messages
                    if sender is not None:
                        if hasattr(self, 'chat_messages') and sender in self.chat_messages:
                            self.chat_messages[sender] = cleaned_messages
                        if hasattr(self, '_oai_messages') and self._oai_messages is not None:
                            if sender in self._oai_messages:
                                self._oai_messages[sender] = cleaned_messages
                
                # Call parent's generate_oai_reply with cleaned messages
                return super().generate_oai_reply(messages=messages, sender=sender, config=config)
            
            def generate_reply(self, messages=None, sender=None, exclude=None):
                # CRITICAL: autogen internally uses self.chat_messages[sender] instead of messages parameter
                # So we need to clean self.chat_messages[sender] directly, not just the messages parameter
                if sender is not None and sender in self.chat_messages:
                    # Clean messages in self.chat_messages[sender] directly
                    self.chat_messages[sender] = [self._clean_message(msg) for msg in self.chat_messages[sender]]
                
                # Also clean _oai_messages if they exist (autogen may use these)
                if sender is not None and hasattr(self, '_oai_messages') and self._oai_messages is not None:
                    if sender in self._oai_messages:
                        self._oai_messages[sender] = [self._clean_message(msg) for msg in self._oai_messages[sender]]
                
                # Also clean the messages parameter if provided
                if messages is not None:
                    messages = [self._clean_message(msg) for msg in messages]
                
                # Additional safety: ensure all messages in the messages list are properly cleaned
                if messages is not None:
                    messages = [self._clean_message(msg) for msg in messages]

                # Call parent's generate_reply with cleaned messages
                # This will internally call generate_oai_reply, which we've also overridden
                # Ensure exclude is not None
                if exclude is None:
                    exclude = []
                return super().generate_reply(messages=messages, sender=sender, exclude=exclude)
        
        self.chatbot = FilteredAssistantAgent(
            name="chatbot",
            system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
            llm_config=llm_config,
        )
        
        # CRITICAL: Monkey patch generate_oai_reply to ensure our cleaned version is always called
        # This is needed because autogen may cache function references
        original_generate_oai_reply = self.chatbot.generate_oai_reply
        def patched_generate_oai_reply(messages=None, sender=None, config=None):
            # CRITICAL: Clean messages in internal storage FIRST before autogen accesses them
            if sender is not None:
                # Clean chat_messages[sender] directly (autogen uses this internally)
                if hasattr(self.chatbot, 'chat_messages') and sender in self.chatbot.chat_messages:
                    self.chatbot.chat_messages[sender] = [self.chatbot._clean_message(msg) for msg in self.chatbot.chat_messages[sender]]
                
                # Clean _oai_messages[sender] if it exists (autogen may use these)
                if hasattr(self.chatbot, '_oai_messages') and self.chatbot._oai_messages is not None:
                    if sender in self.chatbot._oai_messages:
                        self.chatbot._oai_messages[sender] = [self.chatbot._clean_message(msg) for msg in self.chatbot._oai_messages[sender]]
            
            # Clean messages parameter if provided
            if messages is not None:
                messages = [self.chatbot._clean_message(msg) for msg in messages]
            elif sender is not None and sender in self.chatbot.chat_messages:
                messages = [self.chatbot._clean_message(msg) for msg in self.chatbot.chat_messages[sender]]
            
            # Final safety check: ensure all messages have valid content
            if messages is not None:
                cleaned_messages = []
                for idx, msg in enumerate(messages):
                    cleaned_msg = self.chatbot._clean_message(msg)
                    # Double-check: if content is still None after cleaning, set to empty string
                    if isinstance(cleaned_msg, dict):
                        # Ensure content is always a string (not None)
                        if cleaned_msg.get('content') is None:
                            has_function_call = 'function_call' in cleaned_msg and cleaned_msg.get('function_call') is not None
                            # For function role messages, content can be None, but OpenAI API requires empty string
                            if cleaned_msg.get('role') == 'function':
                                cleaned_msg = cleaned_msg.copy()
                                cleaned_msg['content'] = ""
                            elif not has_function_call:
                                cleaned_msg = cleaned_msg.copy()
                                cleaned_msg['content'] = ""
                        # Ensure content is a string type
                        elif not isinstance(cleaned_msg.get('content'), str):
                            cleaned_msg = cleaned_msg.copy()
                            cleaned_msg['content'] = str(cleaned_msg['content']) if cleaned_msg['content'] is not None else ""
                        # Ensure role is present
                        if 'role' not in cleaned_msg:
                            cleaned_msg = cleaned_msg.copy()
                            cleaned_msg['role'] = 'user'  # Default role
                    cleaned_messages.append(cleaned_msg)
                messages = cleaned_messages
                
                # CRITICAL: Update internal storage with cleaned messages BEFORE calling original
                # This ensures autogen's internal methods see cleaned messages
                if sender is not None:
                    if hasattr(self.chatbot, 'chat_messages') and sender in self.chatbot.chat_messages:
                        self.chatbot.chat_messages[sender] = cleaned_messages
                    if hasattr(self.chatbot, '_oai_messages') and self.chatbot._oai_messages is not None:
                        if sender in self.chatbot._oai_messages:
                            self.chatbot._oai_messages[sender] = cleaned_messages
            
            # Store code generation prompt for logging (messages sent to LLM)
            if hasattr(self, 'guard_agent') and messages is not None:
                # Convert messages to prompt format for logging
                prompt_text = ""
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        prompt_text += f"{role}: {content}\n"
                self.guard_agent.code_generation_prompt = prompt_text.strip()

            # Call original method with cleaned messages
            response = original_generate_oai_reply(messages=messages, sender=sender, config=config)

            # Store code generation response for logging
            if hasattr(self, 'guard_agent') and response is not None:
                if isinstance(response, dict) and 'choices' in response:
                    # This is a direct OpenAI response
                    if response['choices'] and len(response['choices']) > 0:
                        choice = response['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            self.guard_agent.code_generation_response = choice['message']['content']
                elif isinstance(response, dict) and 'content' in response:
                    # This is an autogen-formatted response
                    self.guard_agent.code_generation_response = response['content']

            return response
        
        # Replace the method
        self.chatbot.generate_oai_reply = patched_generate_oai_reply
        
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
    
    def check(self, agent_input: str, agent_output: str, user_profile: UserProfile,
              use_full_prompt: bool = False) -> Tuple[bool, List[str], Dict]:
        """
        Check method compatible with the interface expected by main.py.
        This is an alias for check_all_rules with appropriate parameter mapping.

        Args:
            agent_input: The instruction text
            agent_output: The query text
            user_profile: UserProfile object
            use_full_prompt: Whether to use full prompt (ignored, always uses instruction+query)

        Returns:
            (is_valid, violated_rules, details)
        """
        return self.check_all_rules(user_profile, agent_input, agent_output, return_details=True)

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
                full_traceback = traceback.format_exc()
                print(full_traceback)
                
                # Save full traceback to log file
                try:
                    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'webshop', 'guardagent_logs')
                    os.makedirs(log_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    error_log_file = os.path.join(log_dir, f'error_traceback_{timestamp}.log')
                    with open(error_log_file, 'w', encoding='utf-8') as f:
                        f.write("=" * 80 + "\n")
                        f.write("GuardAgent Error Traceback\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Error Type: {error_type}\n")
                        f.write(f"Error Details: {error_details}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write("Full Traceback:\n")
                        f.write("-" * 80 + "\n")
                        f.write(full_traceback)
                        f.write("\n" + "=" * 80 + "\n")
                    print(f"[Error] Full traceback saved to: {error_log_file}")
                except Exception as log_error:
                    print(f"[Warning] Failed to save error traceback: {log_error}")
                
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
            
            # Parse GuardAgent results (pass agent_input for context in rule inference)
            # Note: mapping_debug_info will be populated during parsing
            is_valid, violated_rules, mapping_debug_info = self._parse_guard_agent_results(logs_string, agent_input=agent_input)
            
            # Log generated code and execution results (include mapping debug info)
            if self.verbose or True:  # Always log for debugging
                self._log_guard_agent_execution(
                    agent_input, agent_output, generated_code, logs_string,
                    mapping_debug_info=mapping_debug_info,
                    task_decomposition_prompt=self.guard_agent.task_decomposition_prompt,
                    task_decomposition_response=self.guard_agent.task_decomposition_response,
                    code_generation_prompt=self.guard_agent.code_generation_prompt,
                    code_generation_response=self.guard_agent.code_generation_response
                )
            
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
    
    def _parse_guard_agent_results(self, logs_string: List[str], agent_input: str = None) -> Tuple[bool, List[str], List[Dict]]:
        """
        Parse GuardAgent results from logs
        Returns: (is_valid, violated_rules, mapping_debug_info)

        Args:
            logs_string: List of log entries from GuardAgent execution
            agent_input: Original agent input (user profile and purchase request) for context

        IMPORTANT: If GuardAgent execution fails (no "GuardAgent results:" found),
        we return (False, ["guardagent_execution_failed"]) instead of defaulting to valid.
        This is a security-first approach: if we can't verify, we should block.
        """
        violated_rules = []
        is_valid = None  # None means we haven't found results yet
        found_results = False
        execution_error = None

        # Debug: Collect mapping debug information (now simplified)
        mapping_debug_info = []

        # Look for GuardAgent results in logs
        for section in logs_string:
            if not isinstance(section, str):
                continue

            # Check for execution errors first
            if "Missing variables" in section or "Error:" in section or "SyntaxError" in section or "KeyError" in section or "TypeError" in section:
                execution_error = section
                if self.verbose:
                    print(f"[Debug] Found execution error in logs: {section[:200]}")

            # Look for GuardAgent results
            if "GuardAgent results:" in section:
                found_results = True
                # Parse the results
                try:
                    # Extract action_denied
                    if "action_denied: 1" in section or "action_denied: True" in section:
                        is_valid = False

                        # Extract violated rules directly from CheckRule results
                        # CheckRule returns (action_denied, inaccessible) where inaccessible is a dict
                        # with rule names as keys and violation messages as values
                        violation_text = ""
                        if "inaccessible_actions:" in section:
                            inaccessible_actions = section.split("inaccessible_actions:")[1]
                            inaccessible_actions = inaccessible_actions.split("\n")[0].strip()
                            violation_text = inaccessible_actions

                        # Parse the inaccessible_actions dict to extract rule names
                        if violation_text:
                            try:
                                # Try to parse as Python dict literal
                                import ast
                                inaccessible_dict = ast.literal_eval(violation_text)
                                if isinstance(inaccessible_dict, dict):
                                    # Extract rule names directly from the dict keys
                                    violated_rules.extend(list(inaccessible_dict.keys()))
                            except (ValueError, SyntaxError):
                                # If direct parsing fails, try regex approach
                                import re
                                # Pattern: 'rule_name': 'violation message'
                                rule_pattern = r"['\"](rule\d+_[^'\"]+)['\"]\s*:"
                                matches = re.findall(rule_pattern, violation_text)
                                violated_rules.extend(matches)

                        # Remove duplicates while preserving order
                        seen = set()
                        violated_rules = [x for x in violated_rules if not (x in seen or seen.add(x))]

                        # Debug info for simplified mapping
                        if violated_rules:
                            mapping_debug_info.append({
                                'action': 'direct_rule_extraction',
                                'violated_rules_found': violated_rules,
                                'source': 'inaccessible_actions_dict',
                                'raw_violation_text': violation_text[:200]
                            })
                    
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
                if execution_error:
                    print(f"[Debug] Execution error detected: {execution_error[:500]}")
                
                # Check if code only defines function but doesn't call it
                for log_item in logs_string:
                    if isinstance(log_item, str):
                        # Check if code defines a function but doesn't call it
                        if "def " in log_item and ("ACTION DENIED" not in log_item and "ACTION GRANTED" not in log_item):
                            # Check if function is called
                            lines = log_item.split('\n')
                            has_function_def = False
                            has_function_call = False
                            has_print_statement = False
                            for line in lines:
                                stripped = line.strip()
                                if stripped.startswith('def '):
                                    has_function_def = True
                                if '(' in stripped and not stripped.startswith('def ') and not stripped.startswith('#'):
                                    # Check if it's a function call (not a definition)
                                    if has_function_def and '(' in stripped and '=' not in stripped.split('(')[0]:
                                        has_function_call = True
                                if 'print(' in stripped or 'print "' in stripped or "print '" in stripped:
                                    has_print_statement = True
                            
                            if has_function_def and not has_function_call and not has_print_statement:
                                print("[Debug] Code analysis: Function defined but not called, and no print statements found")
                                print("[Debug] This suggests the generated code is incomplete - it defines a function but doesn't execute it")
            
            # Security-first: if we can't verify, we should block
            is_valid = False
            # Use more specific error type if we found an execution error
            if execution_error:
                if "Missing variables" in execution_error or "Code must print" in execution_error:
                    violated_rules.append("guardagent_code_missing_variables")
                elif "SyntaxError" in execution_error:
                    violated_rules.append("guardagent_code_syntax_error")
                elif "KeyError" in execution_error or "TypeError" in execution_error:
                    violated_rules.append("guardagent_code_runtime_error")
                else:
                    violated_rules.append("guardagent_execution_failed")
            else:
                violated_rules.append("guardagent_execution_failed")

        # If no results found in logs, check for execution errors
        if not found_results:
            if execution_error:
                is_valid = False
                violated_rules.append("guardagent_execution_error")
                if self.verbose:
                    print(f"[Error] GuardAgent execution failed: {execution_error[:200]}")
            else:
                # No results and no errors found - this shouldn't happen
                is_valid = False
                violated_rules.append("guardagent_no_results")
                if self.verbose:
                    print("[Warning] No GuardAgent results found in logs")

        # If is_valid is still None (shouldn't happen, but safety check)
        if is_valid is None:
            if self.verbose:
                print("[Warning] GuardAgent results parsing returned None - defaulting to invalid")
            is_valid = False
            violated_rules.append("guardagent_unknown_error")

        # Return mapping debug info along with results
        return is_valid, violated_rules, mapping_debug_info
    
    def _infer_rule_from_keywords(self, violation_msg: str, context: str, generated_rule_name: str = None) -> str:
        """
        Infer the correct rule name from violation message keywords and context.
        This is a generic method that works for all rules, not just rule_6.
        
        Args:
            violation_msg: Lowercase violation message
            context: Lowercase context (section content) containing product category info
            generated_rule_name: Rule name generated by LLM (may be incorrect)
        
        Returns:
            Inferred rule name or None if cannot infer
        """
        # Define keyword patterns for each rule type
        # Format: (keywords_in_msg, keywords_in_context, rule_name)
        rule_patterns = [
            # Rule 1 - Age restriction for alcohol
            (['age', '21', 'alcohol'], ['alcohol', 'wine', 'beer', 'liquor'], 'rule1_age_alcohol'),
            # Rule 2 - Signal boosters
            (['signal booster', 'country'], ['signal booster'], 'rule2_signal_booster_country'),
            (['signal booster', 'verified', 'verification'], ['signal booster'], 'rule2_signal_booster_unverified'),
            # Rule 3 - Surveillance devices
            (['surveillance', 'verified', 'verification'], ['surveillance', 'camera', 'hidden camera'], 'rule3_surveillance_unverified'),
            (['surveillance', 'country'], ['surveillance', 'camera', 'hidden camera'], 'rule3_surveillance_country'),
            # Rule 4 - Fragrance
            (['fragrance', 'account age', '7 days'], ['fragrance', 'perfume'], 'rule4_fragrance_account_age'),
            (['fragrance', 'credit score', '500'], ['fragrance', 'perfume'], 'rule4_fragrance_credit'),
            # Rule 5 - Electronics
            (['electronics', 'account age', '30 days'], ['camera', 'lens', 'tablet', 'mac', 'pc', 'projector', 'home theater'], 'rule5_electronics_account_age'),
            (['electronics', 'payment', 'prepaid', 'gift card'], ['camera', 'lens', 'tablet', 'mac', 'pc', 'projector', 'home theater'], 'rule5_electronics_payment'),
            (['electronics', 'failed payment', 'payment attempt'], ['camera', 'lens', 'tablet', 'mac', 'pc', 'projector', 'home theater'], 'rule5_electronics_failed_payments'),
            # Rule 6 - Hair products (return rate)
            # Pattern 1: return rate + 40 + hair context
            (['return rate', '40'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
            # Pattern 2: return rate + exceeds + hair context (exceeds can be in message, hair in context)
            (['return rate', 'exceeds'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
            # Pattern 3: return rate + limit/threshold + hair context
            (['return rate', 'limit'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
            (['return rate', 'threshold'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
            (['return rate', 'acceptable'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
            # Pattern 4: return rate + hair (both in message)
            (['return rate', 'hair'], ['hair', 'extension', 'wig', 'hair treatment', 'hair mask', 'hair loss', 'hair coloring'], 'rule6_hair_return_rate'),
            # Rule 7 - Furniture
            (['furniture', 'payment', 'prepaid', 'gift card'], ['sofa', 'bed', 'dining', 'living room', 'furniture'], 'rule7_furniture_payment'),
            (['furniture', 'credit score', '550'], ['sofa', 'bed', 'dining', 'living room', 'furniture'], 'rule7_furniture_credit'),
            # Rule 8 - Health devices
            (['health', 'verified', 'verification'], ['teeth', 'orthodontic', 'health'], 'rule8_health_unverified'),
            # Rule 9 - Food
            (['food', 'country'], ['meat', 'seafood', 'baby food', 'food'], 'rule9_food_country'),
            # Rule 10 - Digital services
            (['age', '13', 'digital', 'game', 'virtual reality', 'xbox'], ['game', 'virtual reality', 'xbox', 'digital service'], 'rule10_digital_age'),
        ]
        
        # Try to match patterns (case-insensitive)
        violation_msg_lower = violation_msg.lower() if violation_msg else ""
        context_lower = context.lower() if context else ""
        
        for msg_keywords, context_keywords, rule_name in rule_patterns:
            # Check if violation message contains relevant keywords (case-insensitive)
            msg_keywords_lower = [kw.lower() for kw in msg_keywords]
            msg_match = any(keyword in violation_msg_lower for keyword in msg_keywords_lower)
            
            # Check if context contains relevant keywords (case-insensitive)
            context_keywords_lower = [kw.lower() for kw in context_keywords]
            context_match = any(keyword in context_lower for keyword in context_keywords_lower)
            
            if msg_match and context_match:
                return rule_name
        
        # Fallback: If generated rule name looks reasonable, try to validate it
        if generated_rule_name:
            # Check if it follows pattern rule{N}_{category}_{field}
            if self._is_valid_rule_name(generated_rule_name):
                return generated_rule_name
        
        return None
    
    def _is_valid_rule_name(self, rule_name: str) -> bool:
        """
        Check if a rule name follows the expected pattern.
        Valid patterns: rule{N}_{category}_{field} or rule{N}_{description}
        Examples: rule1_age_alcohol, rule6_hair_return_rate, rule5_electronics_account_age
        """
        if not rule_name or not isinstance(rule_name, str):
            return False
        
        rule_lower = rule_name.lower()
        
        # Must start with "rule" followed by a number
        if not rule_lower.startswith('rule'):
            return False
        
        # Check if it has the pattern rule{N}_{...}
        import re
        pattern = r'^rule\d+_[a-z_]+$'
        if re.match(pattern, rule_lower):
            return True
        
        return False
    
    def _log_guard_agent_execution(self, agent_input: str, agent_output: str,
                                   generated_code: List[str], logs_string: List[str],
                                   mapping_debug_info: List[Dict] = None,
                                   task_decomposition_prompt: str = None, task_decomposition_response: str = None,
                                   code_generation_prompt: str = None, code_generation_response: str = None):
        """
        Log GuardAgent execution details for debugging

        Args:
            agent_input: Original agent input
            agent_output: Agent output
            generated_code: Generated code blocks
            logs_string: Execution logs
            mapping_debug_info: Debug information about rule mapping process
            task_decomposition_prompt: Task decomposition prompt sent to LLM
            task_decomposition_response: Task decomposition response from LLM
            code_generation_prompt: Code generation prompt sent to LLM
            code_generation_response: Code generation response from LLM
        """
        try:
            # Create a single detailed log file in output directory
            if self.output_dir:
                log_file = os.path.join(self.output_dir, 'guard agent_detailed.txt')
            else:
                # Fallback to default location
                log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'webshop', 'guardagent_logs')
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, 'guard agent_detailed.txt')
            
            # Increment counter for this execution
            self.log_entry_counter += 1

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write(f"GuardAgent Execution #{self.log_entry_counter} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 100 + "\n\n")

                # Task Decomposition Section
                if task_decomposition_prompt or task_decomposition_response:
                    f.write("TASK DECOMPOSITION\n")
                    f.write("=" * 80 + "\n\n")

                    if task_decomposition_prompt:
                        f.write("Task Decomposition Prompt:\n")
                        f.write("-" * 80 + "\n")
                        f.write(task_decomposition_prompt + "\n\n")

                    if task_decomposition_response:
                        f.write("Task Decomposition Response:\n")
                        f.write("-" * 80 + "\n")
                        f.write(task_decomposition_response + "\n\n")

                # Code Generation Section
                if code_generation_prompt or code_generation_response:
                    f.write("CODE GENERATION\n")
                    f.write("=" * 80 + "\n\n")

                    if code_generation_prompt:
                        f.write("Code Generation Prompt:\n")
                        f.write("-" * 80 + "\n")
                        f.write(code_generation_prompt + "\n\n")

                    if code_generation_response:
                        f.write("Code Generation Response:\n")
                        f.write("-" * 80 + "\n")
                        f.write(code_generation_response + "\n\n")

                # Agent Input/Output Section
                f.write("AGENT INPUT/OUTPUT\n")
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
                
                # Add mapping debug information
                if mapping_debug_info:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("RULE MAPPING DEBUG INFORMATION\n")
                    f.write("=" * 80 + "\n\n")
                    for idx, debug_entry in enumerate(mapping_debug_info, 1):
                        f.write(f"[Mapping Entry {idx}]\n")
                        f.write(f"Rule Name: {debug_entry['rule_name']}\n")
                        f.write(f"Violation Message: {debug_entry['violation_msg']}\n")
                        f.write(f"\nMapping Steps:\n")
                        for step_info in debug_entry['steps']:
                            f.write(f"  Step {step_info['step']}: {step_info['name']}\n")
                            f.write(f"    Matched: {step_info['matched']}\n")
                            if step_info.get('matched_pattern'):
                                f.write(f"    Matched Pattern: {step_info['matched_pattern']}\n")
                            if step_info.get('inferred_rule'):
                                f.write(f"    Inferred Rule: {step_info['inferred_rule']}\n")
                            if step_info.get('mapped_rule'):
                                f.write(f"    Mapped Rule: {step_info['mapped_rule']}\n")
                            if step_info.get('result'):
                                f.write(f"    Result: {step_info['result']}\n")
                            if step_info.get('rule_name_lower'):
                                f.write(f"    Rule Name (lowercase): {step_info['rule_name_lower']}\n")
                            if step_info.get('in_mapping_table') is not None:
                                f.write(f"    In Mapping Table: {step_info['in_mapping_table']}\n")
                            if step_info.get('context_has_hair') is not None:
                                f.write(f"    Context Has 'hair': {step_info['context_has_hair']}\n")
                            if step_info.get('context_has_extension') is not None:
                                f.write(f"    Context Has 'extension': {step_info['context_has_extension']}\n")
                            f.write("\n")
                        # Get violated_rules at the time of this entry
                        final_rules = debug_entry.get('final_violated_rules', violated_rules)
                        f.write(f"Final Violated Rules After This Entry: {final_rules}\n")
                        f.write("-" * 80 + "\n\n")
                
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

