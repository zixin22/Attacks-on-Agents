import os,sys
import yaml
import json
import numpy as np
import transformers
import torch
import argparse
import re
from typing import Dict, List, Tuple

# Import rule system components
from rule_and_profile import RuleChecker, MetricsTracker, UserProfile

parser = argparse.ArgumentParser()
# Removed --num_trials parameter, only using memory_1.json for retrieval
parser.add_argument("--num_steps", type=int, default=15, help="The number of steps")
parser.add_argument("--model", type=str, default="gpt-4o",
                    choices=["gpt-3.5-turbo-instruct", "gpt-4-0613", "gpt-4o", "meta-llama/Llama-2-13b-chat-hf"],
                    help="The model name")
parser.add_argument("--output", type=str, default="output", help="The output folder")
parser.add_argument("--emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", choices=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"], help="The model name")
parser.add_argument("--attack", action="store_true", help="Enable fragment-based attack mode")
parser.add_argument("--attack_fixed_number", type=int, default=None, help="Fixed number of the case to attack (required if --attack is set)")
parser.add_argument("--attack_target_instruction", type=str, default=None, help="Target instruction to inject in attack (e.g., 'i would like a bundle of hair extensions that are 20 inches')")
parser.add_argument("--split", type=str, default=None, help="Data split to use (e.g., '0-100', '0-500', 'test', 'eval', 'train'). If not specified, uses value from config file.")
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
            # Get instruction and query from session
            instruction = self.sessions[session].get('instruction', '')
            query = self.sessions[session].get('query', '')
            
            # Check all rules using LLM
            # Returns: (is_valid: bool, violated_rules: List[str])
            # - is_valid=True: No violations, purchase can proceed
            # - is_valid=False: Violations detected, purchase MUST be blocked
            is_valid, violated_rules = self.rule_checker.check_all_rules(
                profile, instruction, query, action
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

# Profiles will be generated based on actual task count
profiles = None


# text embedding model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
model_embedding = SentenceTransformer(args.emb_model)

from prompts.webshop_prompt import *
initial_prompt = INITIAL_PROMPTS[config['params'].get('initial_prompt', 'PROMPT1')]

def generate_embeddings(memory):
    # Keep all cases (both Success=True and Success=False) for retrieval
    # Reward weight will be used during retrieval to prioritize high-reward cases
    # Only filter out cases with Reward <= 0.0 (invalid cases)
    memory = [m for m in memory if m['Reward'] > 0.0]
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
      # Always use Reward weight for retrieval (prioritize high-reward cases)
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
            return '', reasoning, []
        if config['params'].get('intra_task', False):
          intra_scores = torch.FloatTensor(intra_scores)
          _, hits = torch.topk(ret_scores+cos_scores+intra_scores, k=actual_k)
        else:
          _, hits = torch.topk(ret_scores+cos_scores, k=actual_k)
        init_prompt = ''
        retrieved_ids = []  # Track retrieved memory IDs
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
          retrieved_ids.append((memory[h]["Id"], part[0], part[1]))
          print(f'Retrieved from {memory[h]["Id"]}, part {part[0]} to {part[1]}')
        # init_prompt = '\n'.join(ret_examples)
      else:
        # Ensure k doesn't exceed available memory entries
        actual_k = min(k, len(memory))
        if actual_k == 0:
            return '', reasoning, []
        _, hits = torch.topk(cos_scores, k=actual_k)
        ret_examples = []
        retrieved_ids = []  # Track retrieved memory IDs
        for h in hits:
          ret_examples.append('\n'.join(memory[h]["Actions"]))
          if len('\n'.join(ret_examples)) > config['params'].get('max_init_prompt_len', 6400):
            ret_examples = ret_examples[:-1]
            # too many retrievals, stop adding to init_prompt
            break
          retrieved_ids.append((memory[h]["Id"], None, None))  # No part info for full retrieval
          print(f'Retrieved from {memory[h]["Id"]}')
        init_prompt = '\n'.join(ret_examples)
    else:
        retrieved_ids = []  # No retrieval if no instruction
    
    return init_prompt, reasoning, retrieved_ids

def webshop_run_react(idx, prompt, profile=None, to_print=True, return_log=False, attack_instruction=None):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    actions = []
    execution_log = []  # Store all prompts and outputs
    saved_instruction = attack_instruction  # Save attack instruction for later use

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

        # Use attack_instruction if provided, replace instruction in observation BEFORE building prompt
        if saved_instruction:
            instruction = saved_instruction
            if instruction.startswith('Instruction: '):
                instruction = instruction.replace('Instruction: ', '', 1)
            # IMPORTANT: Update session's instruction so RuleChecker uses attack instruction
            if idx in env.sessions:
                env.sessions[idx]['instruction'] = instruction
            # Remove ALL instruction lines and add only our attack instruction (first step only)
            if i == 1:
                lines = observation.split('\n')
                new_lines = []
                instruction_added = False
                # Patterns that indicate an instruction line
                instruction_patterns = ['instruction:', 'i am looking', 'i need to buy', 'i need a', 
                                      'i would like', 'i am searching', 'i want', "i'm looking", "i'm searching"]
                
                for line in lines:
                    line_lower = line.strip().lower()
                    # Skip all instruction-related lines
                    if any(line_lower.startswith(pattern) for pattern in instruction_patterns):
                        # Add our attack instruction only once, replacing the first instruction line found
                        if not instruction_added:
                            new_lines.append(f'Instruction: {instruction}')
                            instruction_added = True
                        # Skip this instruction line
                        continue
                    else:
                        new_lines.append(line)
                
                # If no instruction line was found, add it after "WebShop" or "reset"
                if not instruction_added:
                    insert_idx = 0
                    for idx, line in enumerate(new_lines):
                        if 'WebShop' in line or 'reset' in line.lower():
                            insert_idx = idx + 1
                            break
                    new_lines.insert(insert_idx, f'Instruction: {instruction}')
                
                observation = '\n'.join(new_lines)

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
        full_prompt_react = init_prompt + prompt[-(6400 - len(init_prompt)):]
        
        # Log the full prompt sent to LLM
        if return_log:
            execution_log.append(f"[Step {i}] Full Prompt to LLM:")
            execution_log.append(f"{'='*60}")
            execution_log.append(full_prompt_react)
            execution_log.append(f"{'='*60}")
        
        action = llm(full_prompt_react, stop=['\n']).lstrip(' ')
        
        # Log LLM response
        if return_log:
            execution_log.append(f"[Step {i}] LLM Generated Action: {action}")

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
            
            # Use attack_instruction if provided, otherwise use environment instruction
            final_instruction = saved_instruction if saved_instruction else res[3].get('instruction', '')
            if final_instruction and final_instruction.startswith('Instruction: '):
                final_instruction = final_instruction.replace('Instruction: ', '', 1)
            
            data = {
                'Id': idx,
                'Instruction': final_instruction,
                'Actions': actions[2:-1],
                'Success': (res[1] == 1.0),  # Success=True only if Reward == 1.0
                'Reward': res[1],
                'Category': res[3].get('category', ''),
                'Query': res[3].get('query', '')
            }
            if return_log:
                return res[1], data, execution_log
            return res[1], data

    if return_log:
        return 0, '', execution_log
    return 0, ''  # No reward output = failed


def webshop_run_rap(idx, prompt, memory, embeddings, profile=None, to_print=True, return_log=False, attack_instruction=None):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    actions = []
    reasoning = ''
    instruction = None
    execution_log = []  # Store all prompts and outputs
    saved_instruction = attack_instruction  # Save attack instruction for later use

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

        # Use attack_instruction if provided, otherwise get from environment
        # IMPORTANT: Remove ALL instruction lines and add only our attack instruction (first step only)
        if saved_instruction:
            instruction = saved_instruction
            if instruction.startswith('Instruction: '):
                instruction = instruction.replace('Instruction: ', '', 1)
            res[3]['instruction'] = instruction
            # IMPORTANT: Also update session's instruction so RuleChecker uses attack instruction
            if idx in env.sessions:
                env.sessions[idx]['instruction'] = instruction
            # Remove ALL instruction lines and add only our attack instruction (first step only)
            if i == 1:
                lines = observation.split('\n')
                new_lines = []
                instruction_added = False
                # Patterns that indicate an instruction line
                instruction_patterns = ['instruction:', 'i am looking', 'i need to buy', 'i need a', 
                                      'i would like', 'i am searching', 'i want', "i'm looking", "i'm searching"]
                
                for line in lines:
                    line_lower = line.strip().lower()
                    # Skip all instruction-related lines
                    if any(line_lower.startswith(pattern) for pattern in instruction_patterns):
                        # Add our attack instruction only once, replacing the first instruction line found
                        if not instruction_added:
                            new_lines.append(f'Instruction: {instruction}')
                            instruction_added = True
                        # Skip this instruction line
                        continue
                    else:
                        new_lines.append(line)
                
                # If no instruction line was found, add it after "WebShop" or "reset"
                if not instruction_added:
                    insert_idx = 0
                    for idx, line in enumerate(new_lines):
                        if 'WebShop' in line or 'reset' in line.lower():
                            insert_idx = idx + 1
                            break
                    new_lines.insert(insert_idx, f'Instruction: {instruction}')
                
                observation = '\n'.join(new_lines)
        elif instruction is None and res[3].get('instruction', None) is not None:
            instruction = res[3]['instruction'].replace('Instruction: ', '')
            res[3]['instruction'] = instruction
        elif res[3].get('instruction', None) is None:
            res[3]['instruction'] = instruction

        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
            actions.append(f'Action: {action}')
            actions.append(f'Observation: {observation}')
        else:
            prompt += f'{observation}\n\nAction:'
            actions.append(f'{observation}')
            task = observation

        init_prompt, reasoning, retrieved_ids = generate_examples(
            res[3], actions, memory, embeddings, reasoning,
            k=config['params'].get('num_retrieval', 1),
            act_len=config['params'].get('analogy_len', 0),
            use_act_obs=config['params'].get('act_obs', False)
        )

        full_prompt = 'Interact with a webshop application. Here are examples.\n' + init_prompt + '\nHere is the task.\n' + prompt
        full_prompt = [line for line in full_prompt.split('\n') if 'http://' not in line]
        full_prompt = '\n'.join(full_prompt).replace('Observation: \nWebShop', 'WebShop')

        # Log the full prompt sent to LLM
        if return_log:
            execution_log.append(f"[Step {i}] Full Prompt to LLM:")
            execution_log.append(f"{'='*60}")
            if retrieved_ids:
                execution_log.append(f"Retrieved Memory Sessions:")
                for mem_id, part_start, part_end in retrieved_ids:
                    if part_start is not None and part_end is not None:
                        execution_log.append(f"  - {mem_id} (actions {part_start} to {part_end})")
                    else:
                        execution_log.append(f"  - {mem_id} (full actions)")
                execution_log.append(f"{'='*60}")
            execution_log.append(full_prompt)
            execution_log.append(f"{'='*60}")

        action = llm(full_prompt, stop=['\n']).lstrip(' ')
        
        # Log LLM response
        if return_log:
            execution_log.append(f"[Step {i}] LLM Generated Action: {action}")

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
            
            # Use saved attack_instruction if provided, otherwise use environment instruction
            final_instruction = saved_instruction if saved_instruction else res[3].get('instruction', '')
            if final_instruction and final_instruction.startswith('Instruction: '):
                final_instruction = final_instruction.replace('Instruction: ', '', 1)
            
            data = {
                'Id': idx,
                'Instruction': final_instruction,
                'Actions': actions[2:-1],
                'Success': (res[1] == 1.0),  # Success=True only if Reward == 1.0
                'Reward': res[1],
                'Category': res[3].get('category', ''),
                'Query': res[3].get('query', '')
            }

            if len(memory) > 0:
                prev_mem = list(filter(lambda d: d["Id"] == idx, memory))
                if len(prev_mem) > 0:
                    if prev_mem[0]["Success"]:
                        if (res[1] != 1) or (res[1] == 1 and len(prev_mem[0]["Actions"]) < len(actions[2:-1])):
                            data = prev_mem[0]
                    elif (res[1] != 1 and prev_mem[0]["Reward"] > res[1]):
                        data = prev_mem[0]
            if return_log:
                return res[1], data, execution_log
            return res[1], data

    if return_log:
        return 0, '', execution_log
    return 0, ''  # No reward output = failed


def execute_fragment_attack(
    fragment_label: str,
    fragment_attack_instruction: str,
    host_instruction: str,
    session_id: str,
    memory: List,
    embeddings: Dict,
    profile: UserProfile,
    initial_prompt: str,
    memory_file: str,
    attack_log_file: str = None,
    metrics_tracker: MetricsTracker = None
) -> Tuple[bool, Dict, Dict]:
    """
    Execute a single fragment attack as a complete task.
    
    Args:
        fragment_label: Fragment label ('A', 'B', 'C', 'D')
        fragment_attack_instruction: Complete attack instruction
        host_instruction: Original host instruction
        session_id: Session ID for this attack (e.g., 'fixed_attack_fragment_A_29')
        memory: Current memory list
        embeddings: Current embeddings dict
        profile: User profile to use (should be profile_37)
        initial_prompt: Initial prompt for the task
        memory_file: Path to memory file
    
    Returns:
        (injected, updated_memory, updated_embeddings)
        injected: True if successfully injected to memory
    """
    print(f"\n{'='*60}")
    print(f"EXECUTING FRAGMENT {fragment_label} ATTACK")
    print(f"Session ID: {session_id}")
    print(f"Attack Instruction: {fragment_attack_instruction}")
    print(f"{'='*60}\n")
    
    # Execute the task (use RAP if memory exists, REACT if not)
    # Use return_log=True to capture all prompts and outputs
    # Initialize violated_rules early to avoid UnboundLocalError
    violated_rules = []
    try:
        if len(memory) > 0:
            r, mem_data, execution_log = webshop_run_rap(
                session_id, initial_prompt, memory, embeddings,
                profile=profile, to_print=True,
                attack_instruction=fragment_attack_instruction,
                return_log=True
            )
        else:
            r, mem_data, execution_log = webshop_run_react(
                session_id, initial_prompt,
                profile=profile, to_print=True,
                attack_instruction=fragment_attack_instruction,
                return_log=True
            )
        
        # Get violations for this session
        violated_rules = env.violations.get(session_id, [])
        was_blocked = len(violated_rules) > 0
        
        # Update metrics tracker if provided
        if metrics_tracker and mem_data != '':
            # Extract instruction, query, and price for metrics tracking
            instruction = mem_data.get('Instruction', fragment_attack_instruction)
            query = mem_data.get('Query', '')
            product_price = 0.0
            if session_id in env.sessions:
                session_info = env.sessions[session_id]
                query = session_info.get('query', query)
                # Try to extract price from observation if available
                try:
                    obs, info = webshop_text(**session_info)
                    import re
                    price_match = re.search(r'\$(\d+\.?\d*)', obs)
                    if price_match:
                        product_price = float(price_match.group(1))
                except:
                    pass
            
            metrics_tracker.update(
                profile, mem_data, violated_rules, was_blocked,
                instruction=instruction, query=query, product_price=product_price,
                session_id=session_id
            )
        
        # Write prompt log to file if attack_log_file is provided
        if attack_log_file and execution_log:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FRAGMENT {fragment_label} ATTACK\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Profile Used: {profile.profile_id} (credit_score={profile.credit_score}, account_age_days={profile.account_age_days})\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Fragment Attack Instruction: {fragment_attack_instruction}\n")
                f.write(f"Fragment Label: {fragment_label}\n")
                f.write(f"Reward: {r}\n")
                f.write(f"Success: {mem_data.get('Success', False) if mem_data else False}\n")
                if violated_rules:
                    f.write(f"Violated Rules: {', '.join(violated_rules)}\n")
                f.write(f"{'-'*80}\n")
                f.write("EXECUTION LOG (Prompts and LLM Responses):\n")
                f.write(f"{'-'*80}\n")
                for log_entry in execution_log:
                    f.write(log_entry + "\n")
                f.write(f"{'='*80}\n\n")
        
    except AssertionError:
        r = 0
        mem_data = ''
        execution_log = []
        violated_rules = []
        was_blocked = False
        print(f"✗ Fragment {fragment_label} attack failed due to assertion error")
        
        # Log failure to file
        if attack_log_file:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FRAGMENT {fragment_label} ATTACK - FAILED\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Fragment Attack Instruction: {fragment_attack_instruction}\n")
                f.write(f"Error: AssertionError - Attack failed\n")
                f.write(f"{'='*80}\n\n")
    except Exception as e:
        r = 0
        mem_data = ''
        execution_log = []
        violated_rules = []
        was_blocked = False
        print(f"✗ Fragment {fragment_label} attack failed due to error: {e}")
        
        # Log failure to file
        if attack_log_file:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FRAGMENT {fragment_label} ATTACK - FAILED\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Fragment Attack Instruction: {fragment_attack_instruction}\n")
                f.write(f"Error: {type(e).__name__} - {str(e)}\n")
                f.write(f"{'='*80}\n\n")
    
    # Only inject if done=True (mem_data != '')
    if mem_data != '':
        # Add attack injection markers
        mem_data['AttackInjection'] = True
        mem_data['AttackType'] = 'fragment'
        mem_data['FragmentLabel'] = fragment_label
        mem_data['FragmentInstruction'] = fragment_attack_instruction
        mem_data['HostInstruction'] = host_instruction
        
        # Inject to memory
        memory.append(mem_data)
        
        # Save to file immediately
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=4, ensure_ascii=False)
        
        # Update embeddings (only Reward > 0.0 will be used for retrieval)
        memory, embeddings = generate_embeddings(memory)
        
        print(f"✓ Fragment {fragment_label} injected to memory (Reward: {r}, Success: {mem_data.get('Success', False)})")
        return True, memory, embeddings
    else:
        print(f"✗ Fragment {fragment_label} did not complete (Reward: {r}), skipping memory injection")
        
        # Log incomplete execution to file
        if attack_log_file:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FRAGMENT {fragment_label} ATTACK - INCOMPLETE\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Fragment Attack Instruction: {fragment_attack_instruction}\n")
                f.write(f"Reward: {r}\n")
                f.write(f"Status: Task did not complete (done=False or mem_data empty)\n")
                if execution_log:
                    f.write(f"{'-'*80}\n")
                    f.write("EXECUTION LOG (Partial):\n")
                    f.write(f"{'-'*80}\n")
                    for log_entry in execution_log:
                        f.write(log_entry + "\n")
                f.write(f"{'='*80}\n\n")
        
        return False, memory, embeddings


def execute_trigger_attack(
    trigger_attack_instruction: str,
    host_instruction: str,
    session_id: str,
    memory: List,
    embeddings: Dict,
    profile: UserProfile,
    initial_prompt: str,
    memory_file: str,
    attack_log_file: str = None,
    metrics_tracker: MetricsTracker = None
) -> Tuple[bool, Dict, Dict]:
    """
    Execute trigger attack as a complete task.
    
    Args:
        trigger_attack_instruction: Complete trigger attack instruction
        host_instruction: Original host instruction
        session_id: Session ID for this attack (e.g., 'fixed_attack_trigger_29')
        memory: Current memory list (should contain all fragments)
        embeddings: Current embeddings dict
        profile: User profile to use (should be profile_37)
        initial_prompt: Initial prompt for the task
        memory_file: Path to memory file
    
    Returns:
        (injected, updated_memory, updated_embeddings)
        injected: True if successfully injected to memory
    """
    print(f"\n{'='*60}")
    print(f"EXECUTING TRIGGER ATTACK")
    print(f"Session ID: {session_id}")
    print(f"Trigger Instruction: {trigger_attack_instruction}")
    print(f"{'='*60}\n")
    
    # Execute the task (use RAP with memory containing fragments)
    # Use return_log=True to capture all prompts and outputs
    # Initialize violated_rules early to avoid UnboundLocalError
    violated_rules = []
    try:
        r, mem_data, execution_log = webshop_run_rap(
            session_id, initial_prompt, memory, embeddings,
            profile=profile, to_print=True,
            attack_instruction=trigger_attack_instruction,
            return_log=True
        )
        
        # Get violations for this session
        violated_rules = env.violations.get(session_id, [])
        was_blocked = len(violated_rules) > 0
        
        # Update metrics tracker if provided
        if metrics_tracker and mem_data != '':
            # Extract instruction, query, and price for metrics tracking
            instruction = mem_data.get('Instruction', trigger_attack_instruction)
            query = mem_data.get('Query', '')
            product_price = 0.0
            if session_id in env.sessions:
                session_info = env.sessions[session_id]
                query = session_info.get('query', query)
                # Try to extract price from observation if available
                try:
                    obs, info = webshop_text(**session_info)
                    import re
                    price_match = re.search(r'\$(\d+\.?\d*)', obs)
                    if price_match:
                        product_price = float(price_match.group(1))
                except:
                    pass
            
            metrics_tracker.update(
                profile, mem_data, violated_rules, was_blocked,
                instruction=instruction, query=query, product_price=product_price,
                session_id=session_id
            )
        
        # Write prompt log to file if attack_log_file is provided
        if attack_log_file and execution_log:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TRIGGER ATTACK\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Profile Used: {profile.profile_id} (credit_score={profile.credit_score}, account_age_days={profile.account_age_days})\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Trigger Attack Instruction: {trigger_attack_instruction}\n")
                f.write(f"Reward: {r}\n")
                f.write(f"Success: {mem_data.get('Success', False) if mem_data else False}\n")
                if violated_rules:
                    f.write(f"Violated Rules: {', '.join(violated_rules)}\n")
                f.write(f"{'-'*80}\n")
                f.write("EXECUTION LOG (Prompts and LLM Responses):\n")
                f.write(f"{'-'*80}\n")
                for log_entry in execution_log:
                    f.write(log_entry + "\n")
                f.write(f"{'='*80}\n\n")
        
    except AssertionError:
        r = 0
        mem_data = ''
        execution_log = []
        violated_rules = []
        was_blocked = False
        print(f"✗ Trigger attack failed due to assertion error")
        
        # Log failure to file
        if attack_log_file:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TRIGGER ATTACK - FAILED\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Trigger Attack Instruction: {trigger_attack_instruction}\n")
                f.write(f"Error: AssertionError - Attack failed\n")
                f.write(f"{'='*80}\n\n")
    except Exception as e:
        r = 0
        mem_data = ''
        execution_log = []
        violated_rules = []
        was_blocked = False
        print(f"✗ Trigger attack failed due to error: {e}")
        
        # Log failure to file
        if attack_log_file:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TRIGGER ATTACK - FAILED\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Trigger Attack Instruction: {trigger_attack_instruction}\n")
                f.write(f"Error: {type(e).__name__} - {str(e)}\n")
                f.write(f"{'='*80}\n\n")
    
    # Only inject if done=True (mem_data != '')
    if mem_data != '':
        # Add attack injection markers
        mem_data['AttackInjection'] = True
        mem_data['AttackType'] = 'trigger'
        mem_data['FragmentLabel'] = None  # Trigger has no fragment label
        mem_data['TriggerInstruction'] = trigger_attack_instruction
        mem_data['HostInstruction'] = host_instruction
        
        # Inject to memory
        memory.append(mem_data)
        
        # Save to file immediately
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=4, ensure_ascii=False)
        
        # Update embeddings (only Reward > 0.0 will be used for retrieval)
        memory, embeddings = generate_embeddings(memory)
        
        print(f"✓ Trigger attack injected to memory (Reward: {r}, Success: {mem_data.get('Success', False)})")
        return True, memory, embeddings
    else:
        print(f"✗ Trigger attack did not complete (Reward: {r}), skipping memory injection")
        
        # Log incomplete execution to file
        if attack_log_file:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TRIGGER ATTACK - INCOMPLETE\n")
                f.write(f"{'='*80}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Trigger Attack Instruction: {trigger_attack_instruction}\n")
                f.write(f"Reward: {r}\n")
                f.write(f"Status: Task did not complete (done=False or mem_data empty)\n")
                if execution_log:
                    f.write(f"{'-'*80}\n")
                    f.write("EXECUTION LOG (Partial):\n")
                    f.write(f"{'-'*80}\n")
                    for log_entry in execution_log:
                        f.write(log_entry + "\n")
                f.write(f"{'='*80}\n\n")
        
        return False, memory, embeddings


# Initialize rule violation log file (clear previous content)
rule_violation_file = args.output + '/rule_violation.txt'
if os.path.exists(rule_violation_file):
    os.remove(rule_violation_file)  # Clear previous run's data

# Memory file path - only using memory_1.json
memory_file = args.output + '/memory_1.json'

print('### Running with real-time memory updates ###')

# Use command line argument if provided, otherwise use config file value
split = args.split if args.split is not None else config['params']['split']
print(f"Using data split: {split}")

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

# Import FragmentAttackGenerator for attack mode
attack_log_file = None
if args.attack:
    from attack import FragmentAttackGenerator
    if args.attack_fixed_number is None:
        raise ValueError("--attack_fixed_number is required when --attack is set")
    if args.attack_target_instruction is None:
        raise ValueError("--attack_target_instruction is required when --attack is set")
    attack_generator = FragmentAttackGenerator(verbose=True)
    
    # Initialize attack log file
    attack_log_file = args.output + '/attack_prompts_log.txt'
    with open(attack_log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ATTACK MODE PROMPT LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Target Case: fixed_{args.attack_fixed_number}\n")
        f.write(f"Target Instruction: {args.attack_target_instruction}\n")
        f.write(f"Log File: {attack_log_file}\n")
        f.write("="*80 + "\n\n")
    
    print(f"\n{'='*60}")
    print("ATTACK MODE ENABLED")
    print(f"Target Case: fixed_{args.attack_fixed_number}")
    print(f"Target Instruction: {args.attack_target_instruction}")
    print(f"Attack Prompt Log: {attack_log_file}")
    print(f"{'='*60}\n")

# Load profiles from generated_profiles.json
# Use relative path from the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
generated_profiles_path = os.path.join(script_dir, 'generated_profiles.json')
profiles = []
try:
    with open(generated_profiles_path, 'r', encoding='utf-8') as f:
        profiles_data = json.load(f)
    
    # Extract profiles list from JSON
    if 'profiles' in profiles_data:
        profiles_list = profiles_data['profiles']
    else:
        profiles_list = profiles_data if isinstance(profiles_data, list) else []
    
    # Convert JSON dictionaries to UserProfile objects
    for profile_dict in profiles_list:
        # Map JSON fields to UserProfile constructor parameters
        # Use new field structure: country (not region), no max_quantity/membership_level/account_activity_days
        profile = UserProfile(
            profile_id=profile_dict.get('profile_id', 'unknown'),
            age=profile_dict.get('age', 25),
            country=profile_dict.get('country', 'allowed'),  # Use 'country' directly
            is_verified=profile_dict.get('is_verified', True),
            payment_method=profile_dict.get('payment_method', 'Visa'),
            failed_payment_attempts=profile_dict.get('failed_payment_attempts', 0),
            credit_score=profile_dict.get('credit_score', 700),
            account_age_days=profile_dict.get('account_age_days', 365),
            account_status=profile_dict.get('account_status', 'active'),
            return_rate=profile_dict.get('return_rate', 0.0),
            total_purchase_amount=profile_dict.get('total_purchase_amount', 0.0)
        )
        profiles.append(profile)
    
    print(f"Loaded {len(profiles)} user profiles from {generated_profiles_path}")
    
    # Warn if we have more cases than profiles
    if n > len(profiles):
        print(f"Warning: {n} cases requested but only {len(profiles)} profiles available.")
        print(f"Program will stop after using all {len(profiles)} profiles.")
    
except FileNotFoundError:
    print(f"Error: Profile file not found at {generated_profiles_path}")
    print("Please ensure generated_profiles.json exists in the webshop directory.")
    print("You can generate it by running: python rule_and_profile/generate_profiles.py")
    raise FileNotFoundError(f"Profile file not found: {generated_profiles_path}")
except Exception as e:
    print(f"Error loading profiles from {generated_profiles_path}: {e}")
    print("Please check the profile file format and try again.")
    raise

cnt = 0
rs = []
sr = []

# Initialize metrics tracker
metrics_tracker = MetricsTracker()


# Track profile usage to ensure no repetition
profile_index = 0

for i in index_list:
    print('-----------------')
    print(i)
    
    # Check if we've used all available profiles
    if profile_index >= len(profiles):
        print(f"Warning: All {len(profiles)} profiles have been used. Stopping execution.")
        print(f"Processed {profile_index} cases, {len(index_list) - profile_index} cases remaining.")
        break

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

    # Select profile for this task (use sequential index, no repetition)
    profile = profiles[profile_index]
    profile_index += 1
    print(f'Using profile: {profile.profile_id} (profile {profile_index}/{len(profiles)})')
    
    # Check if this is the attack target case
    if args.attack and i == args.attack_fixed_number:
        print(f"\n{'='*60}")
        print(f"ATTACK MODE: Processing case fixed_{i}")
        print(f"{'='*60}\n")
        
        # Force use profile_37 for attack
        profile_37 = None
        for p in profiles:
            if p.profile_id == 'profile_37':
                profile_37 = p
                break
        if profile_37 is None:
            print(f"Warning: profile_37 not found, using current profile {profile.profile_id}")
            profile_37 = profile
        else:
            print(f"Using profile_37 for attack: {profile_37.profile_id}")
            print(f"Profile details: credit_score={profile_37.credit_score}, account_age_days={profile_37.account_age_days}")
            profile = profile_37
        
        # Step 1: Get host instruction and query by resetting the environment
        # Reset environment to get initial state
        env.step(f'fixed_{i}', 'reset', profile=profile)
        # Get initial observation to extract instruction and query
        obs, info = webshop_text(session=f'fixed_{i}', page_type='init')
        host_instruction = info.get('instruction', '')
        host_query = info.get('query', '')
        
        # Clean instruction (remove "Instruction: " prefix if present)
        if host_instruction and host_instruction.startswith('Instruction: '):
            host_instruction = host_instruction.replace('Instruction: ', '', 1)
        
        print(f"Host Instruction: {host_instruction}")
        print(f"Host Query: {host_query}")
        
        # Step 2: Generate attack plan
        attack_plan = attack_generator.generate_attack_plan(
            host_instruction=host_instruction,
            target_instruction=args.attack_target_instruction
        )
        
        print(f"\nAttack Plan Generated:")
        print(f"  Fragments: {attack_plan['fragments']}")
        print(f"  Number of Fragment Attacks: {len(attack_plan['fragment_attacks'])}")
        print(f"  Trigger Instruction: {attack_plan['trigger_instruction']}\n")
        
        # Log attack plan to file
        if attack_log_file:
            with open(attack_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"ATTACK PLAN FOR CASE fixed_{i}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Profile Used: {profile.profile_id} (credit_score={profile.credit_score}, account_age_days={profile.account_age_days})\n")
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Target Instruction: {args.attack_target_instruction}\n")
                f.write(f"\nFragments:\n")
                for idx, fragment in enumerate(attack_plan['fragments'], 1):
                    f.write(f"  F{idx}: {fragment}\n")
                f.write(f"\nFragment Attacks ({len(attack_plan['fragment_attacks'])}):\n")
                for frag_attack in attack_plan['fragment_attacks']:
                    f.write(f"  {frag_attack['label']}: {frag_attack['instruction']}\n")
                f.write(f"\nTrigger Instruction:\n")
                f.write(f"  {attack_plan['trigger_instruction']}\n")
                f.write(f"{'='*80}\n\n")
        
        # Step 3: Execute fragment attacks
        for fragment_attack in attack_plan['fragment_attacks']:
            fragment_label = fragment_attack['label']
            fragment_instruction = fragment_attack['instruction']
            # Use 'fixed' prefix so server uses the correct goal (fixed_{i}) instead of random goal
            session_id = f'fixed_attack_fragment_{fragment_label}_{i}'
            
            # Execute fragment attack
            injected, memory, embeddings = execute_fragment_attack(
                fragment_label=fragment_label,
                fragment_attack_instruction=fragment_instruction,
                host_instruction=host_instruction,
                session_id=session_id,
                memory=memory,
                embeddings=embeddings,
                profile=profile,
                initial_prompt=initial_prompt,
                memory_file=memory_file,
                attack_log_file=attack_log_file,
                metrics_tracker=metrics_tracker
            )
            
            # Note: Metrics tracker is updated inside execute_fragment_attack() function
            # Fragment attacks are tracked with their session IDs for rule violation analysis
        
        # Step 4: Execute trigger attack
        # Use 'fixed' prefix so server uses the correct goal (fixed_{i}) instead of random goal
        trigger_session_id = f'fixed_attack_trigger_{i}'
        injected, memory, embeddings = execute_trigger_attack(
            trigger_attack_instruction=attack_plan['trigger_instruction'],
            host_instruction=host_instruction,
            session_id=trigger_session_id,
            memory=memory,
            embeddings=embeddings,
            profile=profile,
            initial_prompt=initial_prompt,
            memory_file=memory_file,
            attack_log_file=attack_log_file,
            metrics_tracker=metrics_tracker
        )
        
        # Note: Metrics tracker is updated inside execute_trigger_attack() function
        # Trigger attacks are tracked with their session IDs for rule violation analysis
        
        # Skip normal execution for attack case
        print(f"\n{'='*60}")
        print(f"ATTACK MODE COMPLETE for case fixed_{i}")
        print(f"{'='*60}\n")
        
        # Record dummy result for statistics (don't count attack cases in normal stats)
        rs.append(0)  # Don't count attack reward in normal stats
        flag = False
        sr.append(flag)
        continue  # Skip to next case
    
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
        
        # Extract instruction, query, and price for metrics tracking
        instruction = None
        query = None
        product_price = 0.0
        if f'fixed_{i}' in env.sessions:
            session_info = env.sessions[f'fixed_{i}']
            # Get instruction and query from session info (matching rule_checker logic)
            instruction = session_info.get('instruction', '')
            query = session_info.get('query', '')
            # Try to extract price from observation if available
            obs, info = webshop_text(**session_info)
            import re
            price_match = re.search(r'\$(\d+\.?\d*)', obs)
            if price_match:
                try:
                    product_price = float(price_match.group(1))
                except:
                    pass
        
        # Update metrics tracker (only records RuleChecker's detection results)
        session_id = f'fixed_{i}'
        metrics_tracker.update(profile, mem_data, violated_rules, was_blocked,
                              instruction=instruction, query=query, product_price=product_price,
                              session_id=session_id)

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
# Use actual number of completed cases instead of n (in case we stopped early due to profile exhaustion)
actual_n = len(rs)
if actual_n > 0:
    r = sum(rs) / actual_n
    sr_final = len([_ for _ in rs if _ == 1]) / actual_n
    fr = cnt / actual_n
    print(f"Final statistics (completed {actual_n}/{n} cases):")
    print(f"Average reward: {r:.3f}")
    print(f"Success rate: {sr_final:.3f}")
    print(f"Failure rate: {fr:.3f}")
    if actual_n < n:
        print(f"Note: Stopped early after using all {len(profiles)} profiles.")
else:
    print("No cases completed.")
    r = 0.0
    sr_final = 0.0
    fr = 0.0

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
    f.write("\nViolations by Rule (RuleChecker detected):\n")
    for rule, count in metrics['violations_by_rule'].items():
        if count > 0:
            sessions = metrics.get('violations_by_rule_sessions', {}).get(rule, [])
            if sessions:
                sessions_str = ', '.join(sessions)
                f.write(f"   {rule}: {count} (sessions: {sessions_str})\n")
            else:
                f.write(f"   {rule}: {count}\n")
    f.write("="*60 + "\n")

# Save final statistics (single trial format)
np.savetxt(args.output + '/result_rs.txt', np.array(rs).reshape(-1, 1), fmt='%.3f')
np.savetxt(args.output + '/result_sr.txt', np.array(sr).reshape(-1, 1), fmt='%d')


    
    