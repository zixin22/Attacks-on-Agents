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
    print('✅ AutoGen proxies error has been FIXED!')

except Exception as e:
    print(f'❌ FAILED: {e}')
    import traceback
    traceback.print_exc()
