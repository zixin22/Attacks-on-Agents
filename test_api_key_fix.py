import sys
sys.path.insert(0, 'guard agent')

from guardagent import GuardAgent

# Create test agent
agent = GuardAgent(
    name='test',
    config_list=[{
        'model': 'gpt-4o',
        'api_key': 'sk-O0g7ou2ojOXl9EI77pWKFeFfwLBzNQFmDw6EJ8MkHH74FRb9',
        'base_url': 'http://152.53.53.64:3000/v1'
    }]
)

print('GuardAgent created successfully')

# Test task_decomposition method
try:
    result = agent.task_decomposition(
        config={
            'model': 'gpt-4o',
            'api_key': 'sk-O0g7ou2ojOXl9EI77pWKFeFfwLBzNQFmDw6EJ8MkHH74FRb9',
            'base_url': 'http://152.53.53.64:3000/v1'
        },
        user_request='test',
        agent_specification='test',
        agent_input='test',
        agent_output='test',
        Decomposition_Examples='test'
    )
    print('✅ Task decomposition succeeded!')
except Exception as e:
    print(f'❌ Task decomposition failed: {e}')
    if '401' in str(e) or 'Incorrect API key' in str(e):
        print('Still API key issue')
    else:
        print('Different error')
