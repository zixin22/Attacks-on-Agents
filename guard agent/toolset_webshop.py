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
    
    # Auto-fix: If code only defines a function but doesn't call it, try to fix it
    cell_lines = cell.split('\n')
    has_function_def = False
    function_name = None
    has_function_call = False
    has_direct_execution = False
    
    # Check if code defines a function
    for line in cell_lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            has_function_def = True
            # Extract function name
            func_match = stripped.split('def ')[1].split('(')[0].strip()
            if func_match:
                function_name = func_match
        # Check if function is called
        if function_name and function_name in stripped and '(' in stripped:
            if not stripped.startswith('def ') and '=' not in stripped.split('(')[0]:
                has_function_call = True
        # Check if there's direct execution (not just function definition)
        if stripped and not stripped.startswith('def ') and not stripped.startswith('#'):
            if '=' in stripped or 'if ' in stripped or 'for ' in stripped or 'print(' in stripped:
                has_direct_execution = True
    
    # If code only defines function but doesn't call it, try to extract and execute function body
    if has_function_def and not has_function_call and not has_direct_execution:
        # Try to extract function body and execute it directly
        # This is a simple heuristic - extract code inside function definition
        try:
            import ast
            tree = ast.parse(cell)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Found function definition, try to extract body
                    # Convert function body to executable code
                    func_body_lines = []
                    for stmt in node.body:
                        func_body_lines.append(ast.unparse(stmt))
                    # Replace function definition with its body
                    cell = '\n'.join(func_body_lines)
                    break
        except Exception as e:
            # If AST parsing fails, try simple string manipulation
            # Find function definition and extract body
            lines = cell.split('\n')
            in_function = False
            indent_level = 0
            func_body = []
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    in_function = True
                    # Get indentation level
                    indent_level = len(line) - len(line.lstrip())
                    # Extract function parameters to understand what's needed
                    func_params = line.split('(')[1].split(')')[0] if '(' in line and ')' in line else ''
                    continue
                if in_function:
                    current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 1
                    if line.strip() and current_indent <= indent_level:
                        # End of function
                        break
                    # Add function body line (remove one level of indentation)
                    if line.strip():
                        func_body.append(line[indent_level:] if len(line) > indent_level else line.lstrip())
                    else:
                        func_body.append('')
            
            # If we extracted function body, check if it needs parameters
            if func_body:
                func_body_str = '\n'.join(func_body)
                # Check if function body uses user_profile or purchase_request
                needs_user_profile = 'user_profile' in func_body_str
                needs_purchase_request = 'purchase_request' in func_body_str
                
                # If function needs parameters, add placeholder extraction code before function body
                if needs_user_profile or needs_purchase_request:
                    # Add code to extract user_profile from Agent input (if available in context)
                    # This is a best-effort attempt - we can't access Agent input directly here
                    # So we'll add a comment and try to execute anyway
                    prepended_code = []
                    if needs_user_profile:
                        prepended_code.append("# Note: user_profile should be extracted from Agent input")
                        prepended_code.append("# For now, creating a placeholder - code may need manual adjustment")
                        prepended_code.append("user_profile = {}  # Placeholder - should be extracted from Agent input")
                    if needs_purchase_request:
                        prepended_code.append("# Note: purchase_request should be extracted from Agent input")
                        prepended_code.append("purchase_request = {}  # Placeholder - should be extracted from Agent input")
                    
                    cell = '\n'.join(prepended_code) + '\n' + func_body_str
                else:
                    cell = func_body_str
    
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
            # Check if code only defined a function but didn't call it
            if has_function_def and not has_function_call:
                return "Error: Code defines a function but doesn't call it. The code must be EXECUTABLE directly - do NOT just define a function. Write code that runs immediately when executed. Extract user information from Agent input and create user_info/user_profile dictionary, then call CheckRule and print 'ACTION DENIED' or 'ACTION GRANTED'."
            else:
                return "Missing variables. Code must print either 'ACTION DENIED' or 'ACTION GRANTED'. Make sure the code executes directly and prints the result."
        
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

