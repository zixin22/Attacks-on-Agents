"""
WebShop-specific code execution function for GuardAgent
"""

import sys
import traceback
from io import StringIO

def run_code_webshop(cell):
    """
    Execute GuardAgent generated code for WebShop rule checking.
    Returns formatted results similar to run_code_seeact.
    """
    from prompts_guard import CodeHeader
    try:
        global_var = {
            "action_denied": None,
            "inaccessible_actions": None,
            "guardrailed_answer": None
        }
        
        # Capture print output
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        exec(CodeHeader + cell, global_var)
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        
        # Parse output to determine action_denied
        action_denied = False
        guardrailed_answer = ""
        inaccessible_actions = ""
        
        if "ACTION DENIED" in output:
            action_denied = True
            # Extract violation message
            if "violation:" in output:
                violation_part = output.split("violation:")[1]
                violation_part = violation_part.split("\n")[0].strip()
                inaccessible_actions = violation_part
        elif "ACTION GRANTED" in output:
            action_denied = False
            # Extract guardrailed_answer
            if "guardrailed_answer:" in output:
                guardrailed_answer = output.split("guardrailed_answer:")[1]
                guardrailed_answer = guardrailed_answer.split("\n")[0].strip()
        
        # Validate that we got a result
        if not ('ACTION DENIED' in output or "ACTION GRANTED" in output):
            return "Missing variables. Code must print either 'ACTION DENIED' or 'ACTION GRANTED'."
        
        return "GuardAgent results:\naction_denied: {}\ninaccessible_actions: {}\nguardrailed_answer: {}\n(End of results)".format(
            int(action_denied), inaccessible_actions, guardrailed_answer
        )
    except Exception as e:
        error_info = traceback.format_exc()
        code = CodeHeader + cell
        
        # Parse error information
        if "SyntaxError" in str(repr(e)):
            error_line = str(repr(e))
            error_type = error_line.split('(')[0]
            error_message = error_line.split(',')[0].split('(')[1]
            error_line = error_line.split('"')[1] if '"' in error_line else ""
        elif "KeyError" in str(repr(e)):
            code_lines = code.split('\n')
            key = str(repr(e)).split("'")[1]
            error_type = str(repr(e)).split('(')[0]
            error_line = ""
            for i in range(len(code_lines)):
                if key in code_lines[i]:
                    error_line = code_lines[i]
                    break
            error_message = str(repr(e))
        elif "TypeError" in str(repr(e)):
            error_type = str(repr(e)).split('(')[0]
            error_message = str(e)
            function_mapping_dict = {
                "check_access": "CheckAccess",
                "check_rule": "CheckRule"
            }
            error_key = ""
            for key in function_mapping_dict.keys():
                if key in error_message:
                    error_message = error_message.replace(key, function_mapping_dict[key])
                    error_key = function_mapping_dict[key]
            code_lines = code.split('\n')
            error_line = ""
            for i in range(len(code_lines)):
                if error_key in code_lines[i]:
                    error_line = code_lines[i]
                    break
        else:
            error_type = ""
            error_message = str(repr(e)).split("('")[-1].split("')")[0] if "('" in str(repr(e)) else str(e)
            error_line = ""
        
        # Format error information
        if error_type != "" and error_line != "":
            error_info = f'{error_type}: {error_message}. The error messages occur in the code line "{error_line}".'
        else:
            error_info = f'Error: {error_message}.'
        error_info += '\nPlease make modifications accordingly and make sure the rest code works well with the modification.'
        
        return error_info

