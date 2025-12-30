import sys
sys.path.insert(0, 'guard agent')
sys.path.insert(0, 'webshop')

try:
    from guard_agent_webshop import GuardAgentWebShop
    print('GuardAgentWebShop import: SUCCESS')

    # Create a test GuardAgentWebShop instance
    agent = GuardAgentWebShop(
        llm='gpt-4o',
        api_key='test_key',
        api_base='http://127.0.0.1:3000/v1'
    )
    print('GuardAgentWebShop creation: SUCCESS')

    # Test a simple check (provide required user_profile parameter)
    user_profile = {
        'profile_id': 'test_profile',
        'age': 25,
        'country': 'allowed',
        'is_verified': True,
        'payment_method': 'visa',
        'failed_payment_attempts': 0,
        'credit_score': 700,
        'account_age': 30,
        'account_status': 'active',
        'return_rate': 5.0,
        'total_purchase_amount': 500.0
    }
    result = agent.check("test instruction", user_profile)
    print(f'Check result: {result}')

except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()
