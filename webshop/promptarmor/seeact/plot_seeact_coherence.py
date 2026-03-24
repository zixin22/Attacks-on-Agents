import argparse
import json
import math
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, List


def load_coherence_evaluator():
    module_path = Path(__file__).resolve().parents[2] / "AutoDan" / "AutoDan_webshop" / "coherence_evaluator.py"
    module = SourceFileLoader("coh", str(module_path)).load_module()
    return module.CoherenceEvaluator


def try_extract_text(item: Any, preferred_keys: List[str]) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in preferred_keys:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # fallback: join string values
        for v in item.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def read_json_list(path: Path, preferred_keys: List[str]) -> List[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = []
    if isinstance(data, dict):
        # maybe a dict with items field
        vals = data.get("items") or data.get("data") or data.get("samples") or []
        if isinstance(vals, list):
            data = vals
    if isinstance(data, list):
        for it in data:
            t = try_extract_text(it, preferred_keys)
            if t:
                texts.append(t)
    else:
        # fallback: treat whole file as single string
        txt = try_extract_text(data, preferred_keys)
        if txt:
            texts.append(txt)
    return texts


def compute_losses(evaluator, sequences):
    losses = evaluator.evaluate_batch(sequences)
    return [l for l in losses if isinstance(l, (int, float)) and not math.isnan(l)]


def plot_density(series: Dict[str, List[float]], output_path: Path):
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
        "benign": "#1f77b4",
        "carrier": "#ff7f0e",
        "attack": "#d62728",
    }
    label_map = {
        "benign_query": "benign",
        "carrier_query": "carrier",
        "attack_query": "attack",
    }

    for name, values in series.items():
        values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        if not values:
            continue
        min_v = min(values)
        max_v = max(values)
        if min_v == max_v:
            min_v -= 1e-3
            max_v += 1e-3
        display_name = label_map.get(name, name)
        if use_seaborn:
            sns.kdeplot(values, label=display_name, color=colors.get(display_name), fill=False)
        elif use_scipy:
            kde = gaussian_kde(values)
            import numpy as _np
            xs = _np.linspace(min_v, max_v, 200)
            ys = kde(xs)
            plt.plot(xs, ys, label=display_name, color=colors.get(display_name), linewidth=3)
        else:
            import numpy as np
            hist, bin_edges = np.histogram(values, bins=50, range=(min_v, max_v), density=True)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(centers, hist, label=display_name, color=colors.get(display_name), linewidth=3)

    # Explicitly create legend handles with correct colors in the desired order
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=3, label="benign"),
        plt.Line2D([0], [0], color="#ff7f0e", linewidth=3, label="carrier"),
        plt.Line2D([0], [0], color="#d62728", linewidth=3, label="attack"),
    ]
    plt.legend(handles=handles, fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=18)
    plt.xlabel("Perplexity", fontsize=28)
    plt.ylabel("Density", fontsize=18)
    plt.xlim(2, 9)
    plt.ylim(0.0, 1.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    parser = argparse.ArgumentParser(description="Plot SeeAct coherence curves")
    parser.add_argument("--attack_file", type=str, default=None, help="path to 1-splitted_half.json (attack queries)")
    parser.add_argument("--carrier_file", type=str, default=None, help="path to 1-splitted_insert_fragment_half.json (carrier queries)")
    parser.add_argument("--benign_file", type=str, default=None, help="path to sample_labeled_benign.json (contains 'confirmed_task')")
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "seeact_coherence_density.png"), help="output image path")
    parser.add_argument("--no_gpt2", action="store_true", help="disable GPT-2 (use simplified mode). Default: GPT-2 enabled")
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name for GPT-2 mode")
    parser.add_argument("--device", type=str, default="cpu", help="device for GPT-2 mode (cpu or cuda)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    attack_path = Path(args.attack_file) if args.attack_file else base_dir / "1-splitted_half.json"
    carrier_path = Path(args.carrier_file) if args.carrier_file else base_dir / "1-splitted_insert_fragment_half.json"
    benign_path = Path(args.benign_file) if args.benign_file else base_dir / "sample_labeled_benign.json"
    output_path = Path(args.output)

    CoherenceEvaluator = load_coherence_evaluator()
    use_gpt2 = not args.no_gpt2
    evaluator = CoherenceEvaluator(model_name=args.model_name, device=args.device, use_simplified=(not use_gpt2))

    # preferred keys: try common fields including 'query','text','description','prompt','confirmed_task'
    attack_texts = read_json_list(attack_path, ["query", "text", "description", "prompt", "attack_query"])
    carrier_texts = read_json_list(carrier_path, ["query", "text", "description", "prompt", "carrier_query"])
    benign_texts = read_json_list(benign_path, ["confirmed_task", "confirmedTask", "task", "prompt", "description"])

    series = {
        "attack_query": compute_losses(evaluator, attack_texts),
        "carrier_query": compute_losses(evaluator, carrier_texts),
        "benign_query": compute_losses(evaluator, benign_texts),
    }

    # save raw values
    json_output_path = output_path.with_name(output_path.stem + "_values.json")
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

    try:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved raw values to {json_output_path}")
    except Exception as e:
        print(f"failed to save raw values: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_density(series, output_path)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()






