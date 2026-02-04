import argparse
import json
import math
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import List, Dict


def load_coherence_evaluator():
    module_path = Path(__file__).resolve().parents[2] / "AutoDan" / "AutoDan_webshop" / "coherence_evaluator.py"
    module = SourceFileLoader("coh", str(module_path)).load_module()
    return module.CoherenceEvaluator


def read_webshop_txt(path: Path) -> List[str]:
    texts: List[str] = []
    if not path.exists():
        return texts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        texts.append(line)
    return texts


def read_cleaned_texts_from_json(path: Path, field: str = "cleaned_text") -> List[str]:
    texts: List[str] = []
    if not path.exists():
        return texts
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # try jsonl
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                v = obj.get(field)
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
            except Exception:
                continue
        return texts

    # data may be dict with 'results' or list
    candidates = []
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            candidates = data["results"]
        else:
            # if dict of items
            for k in data:
                if isinstance(data[k], list):
                    candidates = data[k]
                    break
    elif isinstance(data, list):
        candidates = data

    for it in candidates:
        if isinstance(it, dict):
            v = it.get(field)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())
    return texts


def compute_losses(evaluator, sequences: List[str]) -> List[float]:
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
        "benign_query": "#1f77b4",
        "carrier_query": "#ff7f0e",
        "attack_query": "#d62728",
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

    plt.title("Webshop Coherence Loss Density")
    plt.xlabel("coherence_loss (NLL)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Plot coherence for webshop queries")
    parser.add_argument("--webshop_txt", type=str, default=str(Path(__file__).resolve().parent / "webshop.txt"))
    parser.add_argument("--carrier_json", type=str, default=str(Path(__file__).resolve().parent / "attack_query_results.json"))
    parser.add_argument("--attack_json", type=str, default=str(Path(__file__).resolve().parent / "trigger_query_results.json"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "webshop_coherence_density.png"))
    parser.add_argument("--no_gpt2", action="store_true", help="disable GPT-2 (use simplified mode). Default: GPT-2 enabled")
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name for GPT-2 mode")
    parser.add_argument("--device", type=str, default="cpu", help="device for GPT-2 mode (cpu or cuda)")
    args = parser.parse_args()

    webshop_txt = Path(args.webshop_txt)
    carrier_json = Path(args.carrier_json)
    attack_json = Path(args.attack_json)
    output_path = Path(args.output)

    CoherenceEvaluator = load_coherence_evaluator()
    use_gpt2 = not args.no_gpt2
    evaluator = CoherenceEvaluator(model_name=args.model_name, device=args.device, use_simplified=(not use_gpt2))

    benign_texts = read_webshop_txt(webshop_txt)
    carrier_texts = read_cleaned_texts_from_json(carrier_json, "cleaned_text")
    attack_texts = read_cleaned_texts_from_json(attack_json, "cleaned_text")

    series = {
        "benign_query": compute_losses(evaluator, benign_texts),
        "carrier_query": compute_losses(evaluator, carrier_texts),
        "attack_query": compute_losses(evaluator, attack_texts),
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
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved raw values to {json_output_path}")
    except Exception as e:
        print(f"failed to save raw values: {e}")

    plot_density(series, output_path)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()


