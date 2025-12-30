import autogen
import inspect
print('AutoGen version:', autogen.__version__)

# Check the OAI client source
try:
    from autogen.oai.client import OpenAIClient
    print('OpenAIClient source file:', inspect.getfile(OpenAIClient))

    # Try to create a client and see what parameters it passes
    import openai
    print('OpenAI OpenAI.__init__ signature:')
    sig = inspect.signature(openai.OpenAI.__init__)
    print(sig)

    # Check if AutoGen passes proxies
    source = inspect.getsource(OpenAIClient.__init__)
    if 'proxies' in source.lower():
        print('AutoGen OpenAIClient.__init__ contains proxies parameter')
        # Show the relevant lines
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'proxies' in line.lower():
                print(f'Line {i+1}: {line}')
    else:
        print('AutoGen OpenAIClient.__init__ does NOT contain proxies parameter')

except Exception as e:
    print(f'Error inspecting AutoGen: {e}')
    import traceback
    traceback.print_exc()
