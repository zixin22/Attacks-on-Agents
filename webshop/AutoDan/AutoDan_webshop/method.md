# AutoDan

## 

，trigger instruction。，。

## 

- ****: trigger instruction，
- ****: ，
- ****: 
- ****: 

## 

### 1:  (Generation 0)

#### 1.1 
```
:
├── trigger_instruction.txt → N (N≈10)
├── dataset.txt → :
│   ├── : 80pair ()
│   └── : 19pair ()
```

#### 1.2 
```
Kpair:
for template in all_templates:          # N
    sampled_pairs = random.sample(all_training_pairs, K)  # K=5
    for pair in sampled_pairs:
        # : template + pair → prompt
        filled_prompt = template.format(host_instruction=pair.host, ...)

        # : pairprompt
        score = evaluate_prompt(filled_prompt, pair)

        # : templatepair
        scores[template].append(score)
```

#### 1.3 
```
:
for template in all_templates:
    # pair
    avg_score = sum(scores[template]) / len(scores[template])  # 5

    # 
    template_stats[template] = {
        'avg_score': avg_score,
        'std_dev': calculate_std_dev(scores[template]),  # 
        'min_score': min(scores[template]),             # 
        'max_score': max(scores[template])              # 
    }
```

#### 1.4 
```
KGeneration 0:
elite_templates = sort_by_avg_score(all_templates)[:K]  # K=5

: generation_0_elites = [template_A, template_B, template_C]
```

### 2:  (Generation 1, 2, ...)

#### 2.1 
```
:
new_templates = []
for elite_template in current_elites:
    # 
    variants = generate_variants(elite_template)
    new_templates.extend(variants)

# :
# - LLM (paraphrase, expand, synonym)
# -  (, )
# -  ()
```

#### 2.2 
```
:
candidate_scores = {}
for new_template in new_templates:
    # Kpair
    sampled_pairs = random.sample(all_training_pairs, K)  # K=5
    scores = []
    for pair in sampled_pairs:
        filled_prompt = new_template.format(host_instruction=pair.host, ...)
        score = evaluate_prompt(filled_prompt, pair)
        scores.append(score)

    # 
    avg_score = sum(scores) / len(scores)
    candidate_scores[new_template] = {
        'avg_score': avg_score,
        'detailed_scores': scores,
        'stability': calculate_stability(scores)
    }
```

#### 2.3 
```
K:
next_elites = sort_by_avg_score(candidate_scores.keys())[:K]

#  ( elitism)
if use_elitism:
    next_elites.extend(current_elites)
    next_elites = sort_by_avg_score(next_elites)[:K]

: generation_next_elites
```

### 3: 

#### 3.1 
```
:
test_results = {}
for elite_template in final_elites:
    # 19pair5
    sampled_test_pairs = random.sample(test_pairs, 5)
    test_scores = []
    for test_pair in sampled_test_pairs:
        filled_prompt = elite_template.format(host_instruction=test_pair.host, ...)
        score = evaluate_prompt(filled_prompt, test_pair)
        test_scores.append(score)

    test_results[elite_template] = {
        'avg_test_score': sum(test_scores) / len(test_scores),
        'test_stability': calculate_stability(test_scores),
        'train_vs_test_gap': abs(train_avg - test_avg)  # 
    }
```

#### 3.2 
```
:
best_template = max(test_results, key=lambda x: test_results[x]['avg_test_score'])

: best_template, performance_metrics
```

## 

### 
- ****: ，
- ****: ，
- ****: 
- ****: 
- ****: 5

### 
- ****: 
- ****: 580pair
- ****: 

### 
- ****: 
- ****: 
- ****: prompt

## 

###  ()
- : ，，
- : ，

###  ()
- : ，，
- : ，

## 

### 
1. ****: ，
2. ****: ，
3. ****: ，

### 
1. ****: pair，
2. ****: 
3. ****: 

### 
- ****: 
- ****: 
- ****: vs

## 

，，。5，，，trigger instruction。

## （Experiment settings）

 —— 。

- ****
  - : `D:/rap-main/webshop`
  - : `main.py`、`AutoDan/AutoDan_webshop/coherence_evaluator.py`、`rule_and_profile/rule_checker.py`
  - : `dataset_baseline_test.json`、`dataset_test_10.json`、`retrieve_datasets.jsonl`

- ****
  - : Windows (: 10.0.26100)
  - Conda : `Perplexity`（ GPT‑2/coherence ） `rap-py310`（）
  - ：`conda env export -n Perplexity > env_perplexity.yml`

- **（）**
  - Python: 3.10
  - transformers: 5.0.0
  - torch: 2.10.0 (cpu  gpu )
  - numpy: 1.25.x
  - httpx: 0.28.x
  - matplotlib / seaborn / scipy:  env 

- ****
  - （CLI ）:
    - Gemini: `--model gemini-xxxx`
    - OpenAI‑style: `--model gpt-4o` / `--model gpt2`（ GPT‑2  coherence）
  - Gemini ：
    - Gemini API key：`webshop/gemini_api.txt`
    - base_url（relay/proxy）：`http://148.113.224.153:3000`
  -  OpenAI （ Gemini） base_url：`http://152.53.53.64:3000/v1`（ fallback）

- **CLI flags（）**
  - `--model <model_name>`  : （）
  - `--attack`             : （ flag）
  - `--attack_dataset <path>` : （：`dataset_baseline_test.json`）
  - `--cont_number <int>`  : /（：100）
  - `--enable_rule_checker` :  RuleChecker（ flag）
  - `--defense_mode <mode>` : （：`rule_checker`）
  - `--skip_trigger`       :  trigger （ flag）
  - `--output <path>`      : /（：`output/rulechecker_run`）
  - `--cont_seed <int>`    : （）

- **Coherence evaluator（coherence_evaluator.py）**
  - :
    - （`use_simplified=True`）： transformers，/（ transformers ）
    - GPT‑2 （`use_simplified=False`）： HuggingFace GPT‑2  NLL（）
  - :
    - `model_name`（ `"gpt2"`） —  NLL  causal LM
    - `device`（`cpu`  `cuda`）
    - `batch_size`（tokenization ）
  - :
    - `coherence_loss`（NLL， nats ，）
    - `perplexity = exp(coherence_loss)`（）
  - ：NLL/Perplexity  tokenizer granularity （ tokenizer/Piece ）

- ****
  - （）：`PYTHONHASHSEED`, `random.seed()`, `numpy.random.seed()`, `torch.manual_seed()`
  - ：
    - seed 
    -  sha256/hash
    -  checkpoint（）

- **（，）**
  -  pairs （）：80
  -  pairs （）：19
  -  K（）：5
  -  K_elite（）：3
  - （、LLM paraphrase  API）

- **/（batch attack）**
  - ：`batch_attack/<batch_name>/`
  -  `--cont_number`（：100）
  - /（）： worker 

- ****
  - CPU/GPU （ GPU：CUDA ）
  - （）

- ****
  - （ JSON ）：
    - `experiment_id`, `timestamp`, `seed`, `env_file`（conda ）, `cli_args`, `model_versions`, `dataset_hashes`
  - （）:
    - `output/attack_experiment_1`（，）
    - `batch_attack/batch_attack_<n>/analysis.json`（）
    - `promptarmor/promptarmor_osagent.json`（PromptArmor ）

（ README），。
