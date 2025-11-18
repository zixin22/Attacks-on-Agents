import os,sys
import yaml
import json
import numpy as np
import transformers
import torch
import argparse
import re

# Import rule system components
from rule_and_profile import RuleChecker, ProfileGenerator, MetricsTracker, UserProfile


parser = argparse.ArgumentParser()
# Removed --num_trials parameter, only using memory_1.json for retrieval
parser.add_argument("--num_steps", type=int, default=15, help="The number of steps")
parser.add_argument("--model", type=str, default="gpt-4o",
                    choices=["gpt-3.5-turbo-instruct", "gpt-4-0613", "gpt-4o", "meta-llama/Llama-2-13b-chat-hf"],
                    help="The model name")
parser.add_argument("--output", type=str, default="output", help="The output folder")
parser.add_argument("--emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", choices=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"], help="The model name")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

with open('./configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

# llama2
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

if 'Llama-2' in args.model or any(map(args.model.__contains__, AutoModelForCausalLM._model_mapping._model_mapping)):
    model_name = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_4bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
#elif 'gpt' in args.model:
    #openai
    #import openai
    #from openai import OpenAI
    #os.environ["OPENAI_API_KEY"] = open('OpenAI_api_key.txt').readline()
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    #client = OpenAI()

elif 'gpt' in args.model:
    import openai
    #os.environ["OPENAI_API_KEY"] = open('OpenAI_api_key.txt').readline().strip()
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    with open(r"C:\Users\22749\Desktop\rap-main\webshop\OpenAI_api_key.txt", "r") as f:
        openai.api_key = f.read().strip()
    openai.api_base = "http://148.113.224.153:3000/v1"
else:
    print('LLM currently not supported')
    sys.exit(0)

   

import time
import openai

def llm(prompt, stop=["\n"]):
    """
    Universal LLM calling function, compatible with Llama-2, GPT-3.5-turbo-instruct, GPT-4-0613, GPT-4o.
    Automatic retry with error handling.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if 'Llama-2' in args.model:
                sequences = pipeline(
                    prompt,
                    do_sample=config['params'].get('temperature', 1) > 0,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=200,
                    temperature=config['params'].get('temperature', 1),
                    return_full_text=False,
                )
                text = sequences[0]['generated_text']

            elif args.model == 'gpt-3.5-turbo-instruct':
                response = openai.Completion.create(
                    model='gpt-3.5-turbo-instruct',
                    prompt=prompt,
                    temperature=config['params'].get('temperature', 0),
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop
                )
                text = response.choices[0].text

            elif args.model == 'gpt-4-0613':
                completion = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for household task."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop
                )
                text = completion.choices[0].message.content

            elif args.model == 'gpt-4o':
                completion = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for household task."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=config['params'].get('temperature', 0.5),
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop
                )
                text = completion.choices[0].message.content

            else:
                raise ValueError(f"Unsupported model: {args.model}")

            break  # Successfully called, exit retry loop

        except openai.error.RateLimitError:
            wait_time = 5 * (attempt + 1)
            print(f"[Warning] OpenAI API rate limit, waiting {wait_time} seconds before retry ({attempt+1}/{max_retries})...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[Error] LLM call failed: {e}, waiting 3 seconds before retry ({attempt+1}/{max_retries})...")
            time.sleep(3)
    else:
        print("[Error] Exceeded maximum retry attempts, returning empty string.")
        return ""

    

    
        
        
        
        
        
        
        
        
        

    

    if stop:
        text = text.split('\n')[0]
    if len(text) > 0 and text[0]=='>':
        text = text[1:]
    if len(text) > 0 and text[-1]=='.':
        text = text[:-1]
    return text.strip()



WEBSHOP_URL = f"http://localhost:3000/"


''' Setting up webshop environment'''
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

#def clean_str(p):
  #return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
def clean_str(p):
    try:
        return p.encode("latin1").decode("utf-8")
    except:
        return p



def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
        url = f'{WEBSHOP_URL}/{session}'
    elif page_type == 'search':
        url = f'{WEBSHOP_URL}/search_results/{session}/{query_string}/{page_num}'
    elif page_type == 'item':
        url = f'{WEBSHOP_URL}/item_page/{session}/{asin}/{query_string}/{page_num}/{options}'
    elif page_type == 'item_sub':
        url = f'{WEBSHOP_URL}/item_sub_page/{session}/{asin}/{query_string}/{page_num}/{subpage}/{options}'
    elif page_type == 'end':
        url = f'{WEBSHOP_URL}/done/{session}/{asin}/{options}'

    html = requests.get(url).text

    # Print debug information
    print("Current page type:", page_type)
    print("HTML source:\n", html)

    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))

    observation = ''
    option_type = ''
    options = {}
    asins = []
    cnt = 0
    prod_cnt = 0
    just_prod = 0

    for t in visible_texts:
        if t == '\n': continue
        if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue

        if t.parent.name == 'button':
            processed_t = f'\n[{t}] '
        elif t.parent.name == 'label':
            if f"'{t}'" in url:
                processed_t = f'[[{t}]]'
            else:
                processed_t = f'[{t}]'
            options[str(t)] = option_type
        elif t.parent.get('class') == ["product-link"]:
            processed_t = f'\n[{t}] '
            if prod_cnt >= 3:
                processed_t = ''
            prod_cnt += 1
            asins.append(str(t))
            just_prod = 0
        else:
            processed_t = '\n' + str(t) + ' '
            if cnt < 2 and page_type != 'init':
                processed_t = ''
            if just_prod <= 2 and prod_cnt >= 4:
                processed_t = ''
            option_type = str(t)
            cnt += 1
        just_prod += 1
        observation += processed_t

    # Extract ASIN from HTML if on search page
    if page_type == 'search':
        for tag in html_obj.find_all("a", class_="product-link"):
            href = tag.get("href", "")
            match = re.search(r'/product/(B0[A-Z0-9]{8})', href)
            if match:
                asins.append(match.group(1))

    info = {}
    if options:
        info['option_types'] = options
    if asins:
        info['asins'] = asins

    if 'Your score (min 0.0, max 1.0)' in visible_texts:
        idx = visible_texts.index('Your score (min 0.0, max 1.0)')
        info['reward'] = float(visible_texts[idx + 1])
        observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])

    if page_type in ['search', 'item']:
        info['img'] = list(filter(tag_visible, html_obj.findAll(lambda tag: tag.name == 'img' and tag.has_attr('src'))))

    instruction = html_obj.find(id='instruction-text')
    if instruction is not None:
        instruction = instruction.h4
        if instruction is not None:
            instruction = instruction.text
    else:
        instruction = html_obj.find(id='goal-instruction-text')
        if instruction is not None:
            instruction = instruction.pre
            if instruction is not None:
                instruction = instruction.text
    info['instruction'] = instruction

    query = html_obj.find(id='goal-query')
    if query is not None:
        query = query.pre
        if query is not None:
            query = query.text
    info['query'] = query if query is not None else ''

    category = html_obj.find(id='goal-category')
    if category is not None:
        category = category.pre
        if category is not None:
            category = category.text
    info['category'] = category if category is not None else ''

    return clean_str(observation), info



from urllib.parse import quote
class webshopEnv:
  def __init__(self, rule_checker=None):
    self.sessions = {}
    # For rule checking
    self.rule_checker = rule_checker
    self.violations = {}  # Track violations per session
  
  def step(self, session, action, profile=None):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        
        # PRE-ACTION RULE CHECK (only for Buy Now actions)
        # Note: rule_checker.check_all_rules() internally also checks for 'click[Buy Now]',
        # but we check here first to avoid unnecessary LLM calls for non-purchase actions.
        # The double check is harmless - it's a defensive programming practice.
        if self.rule_checker and profile and action.startswith('click[Buy Now]'):
            # Get current observation for context
            obs, info = webshop_text(**self.sessions[session])
            
            # Prepare item info for rule checking
            current_state = {
                'observation': obs,
                'selected_quantity': 1,
                'action': action
            }
            
            # Extract quantity from options if available
            if 'options' in self.sessions[session]:
                for opt_key, opt_val in self.sessions[session]['options'].items():
                    if 'quantity' in opt_key.lower():
                        try:
                            current_state['selected_quantity'] = int(opt_val)
                        except:
                            pass
            
            # Check all rules using LLM
            # Returns: (is_valid: bool, violated_rules: List[str])
            # - is_valid=True: No violations, purchase can proceed
            # - is_valid=False: Violations detected, purchase MUST be blocked
            is_valid, violated_rules = self.rule_checker.check_all_rules(
                profile, current_state, action
            )
            
            # If rules are violated, BLOCK the purchase immediately
            # This is a HARD BLOCK - the purchase cannot proceed under any circumstances
            # The function returns early here, preventing the purchase from happening
            if not is_valid:
                # Block the purchase and return early
                # This return statement prevents execution from reaching the normal purchase flow below
                observation = f"Purchase blocked: Rule violation ({', '.join(violated_rules)})"
                reward = 0.0
                done = True
                
                # Track violations for metrics tracking
                if session not in self.violations:
                    self.violations[session] = []
                self.violations[session].extend(violated_rules)
                
                # Return blocked purchase result
                # This prevents the purchase from proceeding - it's a hard block
                return observation, reward, done, {
                    'rule_violated': True,
                    'violated_rules': violated_rules,
                    'instruction': self.sessions[session].get('instruction', ''),
                    'query': self.sessions[session].get('query', ''),
                    'category': self.sessions[session].get('category', ''),
                    'reward': 0.0
                }
            # If is_valid == True, continue with normal purchase flow below
            # The purchase will proceed normally if no violations are detected
        
        # Help URI Encoding, as WSGI error thrown when option has '#'
        if 'options' in self.sessions[session]:
            for option_type in self.sessions[session]['options']:
                self.sessions[session]['options'][option_type] = quote(self.sessions[session]['options'][option_type])
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done, info

# Initialize rule system first
rule_checker = RuleChecker(verbose=True, model=args.model)
env = webshopEnv(rule_checker=rule_checker)  # Pass rule_checker

# Profiles will be generated inside the trial loop based on actual task count
profiles = None


# text embedding model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
model_embedding = SentenceTransformer(args.emb_model)

from prompts.webshop_prompt import *
initial_prompt = INITIAL_PROMPTS[config['params'].get('initial_prompt', 'PROMPT1')]

def generate_embeddings(memory):
    memory = [m for m in memory if m['Reward'] > 0.0]
    if config['params'].get('success', False):
      memory = [m for m in memory if m['Success']]
    print('num_retrieval',len(memory))
    if len(memory) == 0:
        return [], {}
    embeddings = {}
    for key in ['Instruction', 'Reward', 'Category', 'Query', 'Actions']:
        if key=='Actions' and 'Actions' in memory[0]:
            retrieve_info = [m[key][1:].copy() for m in memory]
            for i in range(len(retrieve_info)):
                for j in range(len(retrieve_info[i])):
                    retrieve_info[i][j] = retrieve_info[i][j].strip()
            embeddings[key] = [model_embedding.encode(r) for r in retrieve_info]
            continue
        retrieve_info = [m[key] for m in memory]
        if key=='Reward':
           embeddings[key] = retrieve_info
           continue
        # extract embeddings
        embeddings[key] = model_embedding.encode(retrieve_info)
    return memory, embeddings


def generate_examples(info, actions, memory, embeddings, reasoning='', k=3, act_len=0, use_act_obs=False):
    cos_scores=None
    # retrieve examples
    if info.get('instruction', None) is not None:
      instruction = info['instruction']
      with torch.no_grad():
        instruction_embedding = model_embedding.encode([instruction])
      cos_scores = cos_sim(instruction_embedding, embeddings['Instruction'])[0]
      if config['params'].get('query_category', False):
        cos_scores += cos_sim(instruction_embedding, embeddings['Query'])[0]
      if not config['params'].get('success', False):
        cos_scores += (torch.tensor(embeddings['Reward']) * config['params'].get('reward_weight', 1))

    if len(actions) > 2 and (actions[-2].replace('Action: ', '').startswith('think') or actions[-2].replace('Action: ', '').startswith('search')):
      reasoning = actions[-2].replace('Action: ', '')
    if cos_scores is not None:
      if act_len > 0 and reasoning != '' and 'Actions' in embeddings:
        ret_scores, ret_index, intra_scores = [], [], []
        query_embedding = model_embedding.encode([reasoning])
        for a, emb in enumerate(embeddings['Actions']):
          if use_act_obs:
            if actions[-2].replace('Action: ', '').startswith('think'):
              #print('ret word act:',actions[-2].replace('Action: ', ''))
              query_embedding = model_embedding.encode([actions[-2].replace('Action: ', '')])
              cos_scores_act = cos_sim(query_embedding, emb[::2]).numpy()
              ret_scores.append(np.max(cos_scores_act))
              ret_index.append(np.argmax(cos_scores_act)*2)
            else:
              #print('ret word obs:',actions[-1].replace('Observation: ', ''))
              query_embedding = model_embedding.encode([actions[-1].replace('Observation: ', '')])
              cos_scores_act = cos_sim(query_embedding, emb[1::2]).numpy()
              ret_scores.append(np.max(cos_scores_act))
              ret_index.append(np.argmax(cos_scores_act)*2+1)
          else:
            cos_scores_act = cos_sim(query_embedding, emb[::2]).numpy()

            ret_scores.append(np.max(cos_scores_act))
            ret_index.append(np.argmax(cos_scores_act)*2)
          if config['params'].get('intra_task', False):
            intra_scores.append(cos_sim(embeddings['Instruction'][a], emb[np.argmax(cos_scores_act)*2]).item())
        ret_scores = torch.FloatTensor(ret_scores)
        # Ensure k doesn't exceed available memory entries
        actual_k = min(k, len(memory))
        if actual_k == 0:
            return '', reasoning
        if config['params'].get('intra_task', False):
          intra_scores = torch.FloatTensor(intra_scores)
          _, hits = torch.topk(ret_scores+cos_scores+intra_scores, k=actual_k)
        else:
          _, hits = torch.topk(ret_scores+cos_scores, k=actual_k)
        init_prompt = ''
        # ret_examples = []
        for h in hits:
          part = [
            max(1, ret_index[h] - act_len + 2),
            min(len(memory[h]['Actions']), ret_index[h] + act_len + 2)
          ]

          retrieve_prompt =  memory[h]['Actions'][0] + '\n'.join(memory[h]['Actions'][part[0]:part[1]])
          if len(init_prompt) + len(retrieve_prompt) > config['params'].get('max_init_prompt_len', 6400):
            # too many retrievals, stop adding to init_prompt
            break
          init_prompt += '\n' + retrieve_prompt
          # ret_examples.append('Task:\n' + d_log[h]['actions'][0] + '\n'.join(d_log[h]['actions'][part[0]:part[1]]) + '\n')
          print(f'Retrieved from {memory[h]["Id"]}, part {part[0]} to {part[1]}')
        # init_prompt = '\n'.join(ret_examples)
      else:
        # Ensure k doesn't exceed available memory entries
        actual_k = min(k, len(memory))
        if actual_k == 0:
            return '', reasoning
        _, hits = torch.topk(cos_scores, k=actual_k)
        ret_examples = []
        for h in hits:
          ret_examples.append('\n'.join(memory[h]["Actions"]))
          if len('\n'.join(ret_examples)) > config['params'].get('max_init_prompt_len', 6400):
            ret_examples = ret_examples[:-1]
            # too many retrievals, stop adding to init_prompt
            break
          print(f'Retrieved from {memory[h]["Id"]}')
        init_prompt = '\n'.join(ret_examples)
    return init_prompt, reasoning

def webshop_run_react(idx, prompt, profile=None, to_print=True):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    actions = []

    for i in range(1, args.num_steps + 1):
        try:
            res = env.step(idx, action, profile=profile)
            observation = res[0]
        except AssertionError:
            observation = 'Invalid action!'
            # If search action fails, may be a state issue, try resetting
            if action.startswith('search[') and idx in env.sessions:
                if env.sessions[idx].get('page_type') != 'init':
                    # Auto-reset to init state
                    env.sessions[idx] = {'session': idx, 'page_type': 'init'}
                    print(f'Warning: Auto-reset session {idx} to init state due to invalid search action')

        if action.startswith('think'):
            observation = 'OK.'

        if to_print:
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()

        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
            actions.append(f'Action: {action}')
            actions.append(f'Observation: {observation}')
        else:
            prompt += f'{observation}\n\nAction:'
            actions.append(f'{observation}')
            task = observation

        # Generate new action using model
        action = llm(init_prompt + prompt[-(6400 - len(init_prompt)):], stop=['\n']).lstrip(' ')

        # Clean invalid action format with "|", e.g., click[B078GWRC1J | Buy Now]
        if "|" in action:
            # Prefer keeping the right side specific action part
            parts = [p.strip() for p in action.split("|") if p.strip()]
            if len(parts) > 1:
                # Take the last part as the actual button, e.g., "Buy Now"
                action = parts[-1]

        # Ensure click[...] syntax is valid
        action = action.replace("click", "click[") if not action.startswith("click[") else action
        if not action.endswith("]"):
            action += "]"

        # Validate action prefix
        allowed_prefixes = ['search[', 'click[', 'think[', 'reset']
        if not any(action.startswith(p) for p in allowed_prefixes):
            print(f"Invalid action generated by LLM: {action}")
            action = 'think[let me try another approach]'

        # If completed (res[2] == True), organize return data
        if res[2]:
            inv_act_idx = np.where(np.char.find(np.array(actions), 'Invalid action!') > 0)[0]
            inv_act_idx = np.append(inv_act_idx, inv_act_idx - 1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            data = {
                'Id': idx,
                'Instruction': res[3]['instruction'],
                'Actions': actions[2:-1],
                'Success': True,  # Any reward output (including 0.0) means purchase succeeded
                'Reward': res[1],
                'Category': res[3]['category'],
                'Query': res[3]['query']
            }
            return res[1], data

    return 0, ''  # No reward output = failed


def webshop_run_rap(idx, prompt, memory, embeddings, profile=None, to_print=True):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    actions = []
    reasoning = ''
    instruction = None

    for i in range(1, args.num_steps + 1):
        try:
            res = env.step(idx, action, profile=profile)
            observation = res[0]
        except AssertionError:
            observation = 'Invalid action!'
            # If search action fails, may be a state issue, try resetting
            if action.startswith('search[') and idx in env.sessions:
                if env.sessions[idx].get('page_type') != 'init':
                    # Auto-reset to init state
                    env.sessions[idx] = {'session': idx, 'page_type': 'init'}
                    print(f'Warning: Auto-reset session {idx} to init state due to invalid search action')

        if action.startswith('think'):
            observation = 'OK.'

        if to_print:
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()

        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
            actions.append(f'Action: {action}')
            actions.append(f'Observation: {observation}')
        else:
            prompt += f'{observation}\n\nAction:'
            actions.append(f'{observation}')
            task = observation

        if instruction is None and res[3].get('instruction', None) is not None:
            instruction = res[3]['instruction'].replace('Instruction: ', '')
            res[3]['instruction'] = instruction
        elif res[3].get('instruction', None) is None:
            res[3]['instruction'] = instruction

        init_prompt, reasoning = generate_examples(
            res[3], actions, memory, embeddings, reasoning,
            k=config['params'].get('num_retrieval', 1),
            act_len=config['params'].get('analogy_len', 0),
            use_act_obs=config['params'].get('act_obs', False)
        )

        full_prompt = 'Interact with a webshop application. Here are examples.\n' + init_prompt + '\nHere is the task.\n' + prompt
        full_prompt = [line for line in full_prompt.split('\n') if 'http://' not in line]
        full_prompt = '\n'.join(full_prompt).replace('Observation: \nWebShop', 'WebShop')

        action = llm(full_prompt, stop=['\n']).lstrip(' ')

        # Clean invalid action format
        if "|" in action:
            parts = [p.strip() for p in action.split("|") if p.strip()]
            if len(parts) > 1:
                action = parts[-1]  # Keep right side action (e.g., Buy Now)
            action = action.replace("click", "click[") if not action.startswith("click[") else action
            if not action.endswith("]"):
                action += "]"

        # Enforce allowed action prefixes
        allowed_prefixes = ['search[', 'click[', 'think[', 'reset']
        if not any(action.startswith(p) for p in allowed_prefixes):
            print(f"Invalid action generated by LLM: {action}")
            action = 'think[let me try another approach]'

        if res[2]:
            inv_act_idx = np.where(np.char.find(np.array(actions), 'Invalid action!') > 0)[0]
            inv_act_idx = np.append(inv_act_idx, inv_act_idx - 1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            data = {
                'Id': idx,
                'Instruction': res[3]['instruction'],
                'Actions': actions[2:-1],
                'Success': True,  # Any reward output (including 0.0) means purchase succeeded
                'Reward': res[1],
                'Category': res[3]['category'],
                'Query': res[3]['query']
            }

            if len(memory) > 0:
                prev_mem = list(filter(lambda d: d["Id"] == idx, memory))
                if len(prev_mem) > 0:
                    if prev_mem[0]["Success"]:
                        if (res[1] != 1) or (res[1] == 1 and len(prev_mem[0]["Actions"]) < len(actions[2:-1])):
                            data = prev_mem[0]
                    elif (res[1] != 1 and prev_mem[0]["Reward"] > res[1]):
                        data = prev_mem[0]
            return res[1], data

    return 0, ''  # No reward output = failed

# Initialize rule violation log file (clear previous content)
rule_violation_file = args.output + '/rule_violation.txt'
if os.path.exists(rule_violation_file):
    os.remove(rule_violation_file)  # Clear previous run's data

# Memory file path - only using memory_1.json
memory_file = args.output + '/memory_1.json'

print('### Running with real-time memory updates ###')

split = config['params']['split']

# Support reward>=0.5 mode
if split == '0-100':
    reward_file = r"C:\Users\22749\Desktop\rap-main\WebShop-master\data_generate_experiment\0_100\merged_0_100.json"  # Can be changed to config read
    with open(reward_file, "r", encoding="utf-8") as f:
        reward_data = json.load(f)
    index_list = reward_data["fixed_numbers"]
    n = len(index_list)
    start = None
    print(f"Loaded {n} fixed indices from {reward_file}")
elif split == '0-500':
    reward_file = r"C:\Users\22749\Desktop\rap-main\WebShop-master\data_generate_experiment\0_500\merged_0_500.json"  # Can be changed to config read
    with open(reward_file, "r", encoding="utf-8") as f:
        reward_data = json.load(f)
    index_list = reward_data["fixed_numbers"]
    n = len(index_list)
    start = None
    print(f"Loaded {n} fixed indices from {reward_file}")

# Original split mode
else:
    if split == 'final':
        n, start = 100, 0
    elif split == 'test':
        n, start = 500, 0
    elif split == 'eval':
        n, start = 1000, 501
    elif split == 'train':
        n, start = 10587, 1500
    else:
        n, start = 1000, 5001
    index_list = range(start, start + n)

# Generate profiles based on actual task count
# Calculate violations_per_rule: approximately 10% of total profiles should violate each rule
violations_per_rule = max(1, n // 20)  # Each rule gets ~5% violations, 12 rules = ~60% total violations
profiles = ProfileGenerator.generate_profiles(num_profiles=n, violations_per_rule=violations_per_rule)
print(f"Generated {len(profiles)} user profiles for {n} tasks (violations_per_rule={violations_per_rule})")

cnt = 0
rs = []
sr = []

# Initialize metrics tracker
metrics_tracker = MetricsTracker()

for i in index_list:
    print('-----------------')
    print(i)

    # Load memory_1.json before each case (if it exists)
    memory = []
    embeddings = None
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            if len(memory) > 0:
                memory, embeddings = generate_embeddings(memory)
                print(f"Loaded {len(memory)} entries from {memory_file} (after filtering)")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to load {memory_file}: {e}, starting with empty memory")
            memory = []
    else:
        print(f"No existing {memory_file}, starting with empty memory")

    # Select profile for this task (cycle through profiles)
    profile = profiles[i % len(profiles)]
    print(f'Using profile: {profile.profile_id}')

    # Run the case: use react for first case, rap for subsequent cases
    if len(memory) == 0:
        # First case: use react (no memory yet)
        try:
            r, mem_data = webshop_run_react(f'fixed_{i}', initial_prompt, profile=profile, to_print=True)
            # Get violations for this session
            violated_rules = env.violations.get(f'fixed_{i}', [])
            was_blocked = len(violated_rules) > 0
        except AssertionError:
            r = 0
            cnt += 1
            mem_data = ''
            violated_rules = []
            was_blocked = False
    else:
        # Subsequent cases: use rap with memory
        try:
            r, mem_data = webshop_run_rap(f'fixed_{i}', initial_prompt, memory, embeddings, 
                                         profile=profile, to_print=True)
            # Get violations for this session
            violated_rules = env.violations.get(f'fixed_{i}', [])
            was_blocked = len(violated_rules) > 0
        except AssertionError:
            r = 0
            cnt += 1
            mem_data = ''
            violated_rules = []
            was_blocked = False

    # Save result to memory_1.json immediately after each case
    if mem_data != '':
        # Append new memory data to the loaded memory
        memory.append(mem_data)
        
        # Save updated memory to file immediately
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=4, ensure_ascii=False)
        
        print(f"Saved result to {memory_file} (total entries: {len(memory)})")
        
        # Update metrics tracker
        metrics_tracker.update(profile, mem_data, violated_rules, was_blocked)

    rs.append(r)
    flag = (r == 1)
    sr.append(flag)

    if len(rs) % 1 == 0:
        r_avg = sum(rs) / len(rs)
        sr_avg = len([_ for _ in rs if _ == 1]) / len(rs)
        fr = cnt / len(rs)
        print(len(rs), r_avg, flag, sr_avg, fr)
        print('-------------\n')

# Final statistics
r = sum(rs) / len(rs)
sr_final = len([_ for _ in rs if _ == 1]) / n
fr = cnt / n
print(r, sr_final, fr)

# Print metrics summary
print("\n" + "="*60)
print("METRICS SUMMARY")
print("="*60)
metrics_tracker.print_summary()

# Save metrics to file
metrics = metrics_tracker.get_metrics()
with open(args.output + '/rule_violation.txt', 'a', encoding='utf-8') as f:
    f.write("\n" + "="*60 + "\n")
    f.write("METRICS SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"Total Episodes: {metrics_tracker.total_episodes}\n")
    f.write(f"Blocked Episodes: {metrics_tracker.blocked_episodes}\n")
    f.write(f"Allowed Episodes: {metrics_tracker.allowed_episodes}\n")
    f.write(f"Successful Purchases: {metrics_tracker.successful_purchases}\n")
    f.write(f"Failed Purchases: {metrics_tracker.failed_purchases}\n")
    f.write("\n")
    f.write(f"Block Rate: {metrics['block_rate']:.3f}\n")
    f.write("(% of episodes where purchase was blocked)\n")
    f.write(f"Detection Rate: {metrics['detection_rate']:.3f}\n")
    f.write("(% of episodes where LLM detected rule violations)\n")
    f.write(f"Purchase Success Rate: {metrics['purchase_success_rate']:.3f}\n")
    f.write("(% of allowed purchases that succeeded)\n")
    f.write("\nViolations by Rule (LLM detected):\n")
    for rule, count in metrics['violations_by_rule'].items():
        if count > 0:
            f.write(f"   {rule}: {count}\n")
    f.write("="*60 + "\n")
    
    # Add ground truth comparison
    f.write(metrics_tracker.get_comparison_summary())

# Save final statistics (single trial format)
np.savetxt(args.output + '/result_rs.txt', np.array(rs).reshape(-1, 1), fmt='%.3f')
np.savetxt(args.output + '/result_sr.txt', np.array(sr).reshape(-1, 1), fmt='%d')


    
    