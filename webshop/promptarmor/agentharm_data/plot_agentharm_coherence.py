import argparse
import json
import math
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Optional


def load_coherence_evaluator():
    module_path = Path(__file__).resolve().parents[2] / "AutoDan" / "AutoDan_webshop" / "coherence_evaluator.py"
    module = SourceFileLoader("coh", str(module_path)).load_module()
    return module.CoherenceEvaluator


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def extract_task_input_core(path: Path):
    rows = read_jsonl(path)
    texts = []
    for row in rows:
        text = str(row.get("task_input_core", "")).strip()
        if text:
            texts.append(text)
    return texts


def extract_injection_instruction(path: Path):
    rows = read_jsonl(path)
    texts = []
    for row in rows:
        text = str(row.get("injection_instruction", "")).strip()
        if text:
            texts.append(text)
    return texts


def extract_benign_prompts(path: Path):
    """
    从 benign_behaviors_test_public_ALL.txt 中提取每个样本的 "prompt" 字段（按出现顺序）。
    该文件为 inspect_evals 导出的文本报告，prompt 行通常形如:   "prompt": "......",
    """
    texts = []
    import re
    if not path.exists():
        return texts
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r'\"prompt\"\s*:\s*\"(.*)\"\,?$', line)
        if m:
            # 提取并反转义内嵌的双引号与转义字符
            raw = m.group(1)
            try:
                # 利用 JSON 解码处理转义
                import json as _json
                decoded = _json.loads(f"\"{raw}\"")
            except Exception:
                decoded = raw.replace('\\n', '\n').replace('\\t', '\t')
            texts.append(decoded)
    return texts


def compute_losses(evaluator, sequences):
    losses = evaluator.evaluate_batch(sequences)
    return [l for l in losses if isinstance(l, (int, float)) and not math.isnan(l)]


def plot_density(series, output_path: Path):
    import matplotlib.pyplot as plt

    use_seaborn = False
    try:
        import seaborn as sns  # type: ignore
        use_seaborn = True
    except Exception:
        sns = None

    use_scipy = False
    if not use_seaborn:
        try:
            from scipy.stats import gaussian_kde  # type: ignore
            use_scipy = True
        except Exception:
            gaussian_kde = None

    plt.figure(figsize=(8, 5))
    colors = {
        "task_input_core": "#1f77b4",
        "injection_instruction": "#d62728",
    }
    # add benign color if present
    colors.setdefault("benign_prompt", "#2ca02c")

    for name, values in series.items():
        values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        if not values:
            continue
        # determine plotting range from data
        min_v = min(values)
        max_v = max(values)
        if min_v == max_v:
            # small jitter for degenerate cases
            min_v -= 1e-3
            max_v += 1e-3

        if use_seaborn:
            sns.kdeplot(values, label=name, color=colors.get(name), fill=False)
        elif use_scipy:
            kde = gaussian_kde(values)
            import numpy as _np
            xs = _np.linspace(min_v, max_v, 200)
            ys = kde(xs)
            plt.plot(xs, ys, label=name, color=colors.get(name))
        else:
            import numpy as np

            hist, bin_edges = np.histogram(values, bins=50, range=(min_v, max_v), density=True)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(centers, hist, label=name, color=colors.get(name))

    plt.title("Coherence Loss Density")
    plt.xlabel("coherence_loss (NLL)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    parser = argparse.ArgumentParser(description="Plot coherence loss density for agentharm data")
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).resolve().parent), help="base directory containing jsonl files")
    parser.add_argument("--task_input_path", type=str, default=None, help="path to retrieve_datasets.jsonl")
    parser.add_argument("--injection_path", type=str, default=None, help="path to hostpair_seperate_results.jsonl")
    parser.add_argument("--output_path", type=str, default=None, help="output image path")
    parser.add_argument("--no_gpt2", action="store_true", help="disable GPT-2 (use simplified mode). Default: GPT-2 enabled")
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name for GPT-2 mode")
    parser.add_argument("--device", type=str, default="cpu", help="device for GPT-2 mode (cpu or cuda)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    task_input_path = Path(args.task_input_path) if args.task_input_path else base_dir / "retrieve_datasets.jsonl"
    injection_path = Path(args.injection_path) if args.injection_path else base_dir / "hostpair_seperate_results.jsonl"
    output_path = Path(args.output_path) if args.output_path else base_dir / "agentharm_coherence_density.png"

    CoherenceEvaluator = load_coherence_evaluator()
    use_gpt2 = not args.no_gpt2
    evaluator = CoherenceEvaluator(model_name=args.model_name, device=args.device, use_simplified=(not use_gpt2))

    if not task_input_path.exists():
        raise FileNotFoundError(f"task_input file not found: {task_input_path}")
    if not injection_path.exists():
        raise FileNotFoundError(f"injection file not found: {injection_path}")

    task_inputs = extract_task_input_core(task_input_path)
    injections = extract_injection_instruction(injection_path)
    benign_path = base_dir / "benign_behaviors_test_public_ALL.txt"
    benign_prompts = extract_benign_prompts(benign_path)

    series = {
        "task_input_core": compute_losses(evaluator, task_inputs),
        "injection_instruction": compute_losses(evaluator, injections),
        "benign_prompt": compute_losses(evaluator, benign_prompts),
    }
    # save raw losses and derived perplexities to JSON for verification
    results = {}
    for name, losses in series.items():
        entries = []
        for l in losses:
            try:
                ppl = math.exp(l) if isinstance(l, (int, float)) else None
            except Exception:
                ppl = None
            entries.append({"loss": l, "perplexity": ppl})
        results[name] = entries

    json_output_path = output_path.with_name(output_path.stem + "_values.json")
    try:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved raw values to {json_output_path}")
    except Exception as e:
        print(f"failed to save raw values: {e}")

    plot_density(series, output_path)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
