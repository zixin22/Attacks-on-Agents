def model_config(model):
    """
    Create model configuration with API key loaded from file.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o')
    
    Returns:
        Dictionary with 'model' and 'api_key' keys
    """
    import os
    
    # Try multiple possible paths for API key file
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'webshop', 'OpenAI_api_key.txt'),  # Relative to config.py
        os.path.join(os.path.dirname(__file__), '..', '..', 'webshop', 'OpenAI_api_key.txt'),  # Alternative relative path
        r"C:\Users\22749\Desktop\rap-main\webshop\OpenAI_api_key.txt",  # Absolute path (fallback)
        'OpenAI_api_key.txt'  # Current directory
    ]
    
    api_key_path = None
    for path in possible_paths:
        if os.path.exists(path):
            api_key_path = path
            break
    
    if api_key_path:
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
        except Exception as e:
            print(f"[Warning] Failed to read API key from {api_key_path}: {e}")
            api_key = "<YOUR_API_KEY>"  # Fallback to placeholder
    else:
        print(f"[Warning] OpenAI API key file not found. Tried: {possible_paths}")
        api_key = "<YOUR_API_KEY>"  # Fallback to placeholder
    
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
    # IMPORTANT: autogen 1.0.16 with OpenAI 1.7.2 compatibility issue:
    # - autogen may internally convert 'base_url' to 'api_base' when passing to OpenAI client
    # - OpenAI 1.7.2's Completions.create() doesn't accept 'api_base' parameter
    # Solution: Set base_url via environment variable BEFORE importing autogen
    # This way autogen will use the environment variable instead of config_list
    base_url = "http://152.53.53.64:3000/v1"
    os.environ["OPENAI_BASE_URL"] = base_url
    
    config = {
        "model": model_name,
        "api_key": api_key,
        # NOTE: 'base_url' is NOT included in config_list to avoid autogen's incorrect conversion
        # Instead, we set OPENAI_BASE_URL environment variable which autogen will use
        # For direct API calls in GuardAgent (task_decomposition, error_debugger),
        # we manually create OpenAI client with base_url parameter (see guardagent.py lines 65-69, 216-220)
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