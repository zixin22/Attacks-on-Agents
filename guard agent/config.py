def model_config(model):
    """
    Create model configuration with API key loaded from file.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o')
    
    Returns:
        Dictionary with 'model' and 'api_key' keys
    """
    import os
    
    api_key_path = os.path.join(os.path.dirname(__file__), '..', 'webshop', 'OpenAI_api_key.txt')

    if not os.path.exists(api_key_path):
        raise FileNotFoundError(f"OpenAI API key file not found: {api_key_path}")

    with open(api_key_path, "r") as f:
        api_key = f.read().strip()
    
    # Determine model name (handle variations like 'gpt-4o')
    if 'gpt-3.5-turbo' in model.lower():
        model_name = "gpt-3.5-turbo"
    elif 'gpt-4o' in model.lower():
        model_name = "gpt-4o"
    elif 'gpt-4' in model.lower():
        model_name = "gpt-4"
    else:
        model_name = "gpt-4"  # Default
    
    # Use custom API base URL (same as RuleChecker)
    # This allows using a proxy or custom OpenAI-compatible endpoint
    # Note: autogen uses 'base_url' in config_list, not 'api_base'
    # IMPORTANT: Do NOT include 'api_base' in config_list for autogen, as it causes
    # TypeError with newer OpenAI clients. Only 'base_url' should be used.
    base_url = "http://152.53.53.64:3000/v1"
    
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,  # autogen uses 'base_url' for custom API endpoints
        # NOTE: 'api_base' is NOT included here to avoid passing it to OpenAI client
        # For direct OpenAI client calls (in task_decomposition, error_debugger),
        # we use 'api_base' from a separate variable, not from config_list
    }
    
    return config


def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list