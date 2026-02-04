import argparse
import json
import math
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path


def load_coherence_evaluator():
    module_path = Path(__file__).resolve().parents[2] / "AutoDan" / "AutoDan_webshop" / "coherence_evaluator.py"
    module = SourceFileLoader("coh", str(module_path)).load_module()
    return module.CoherenceEvaluator


def parse_labeled_lines(path, carrier_prefix, attack_prefix):
    carriers = []
    attacks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(carrier_prefix):
            carriers.append(line[len(carrier_prefix):].strip())
        elif line.startswith(attack_prefix):
            attacks.append(line[len(attack_prefix):].strip())
    return carriers, attacks


def extract_descriptions(json_path):
    items = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return [str(it.get("description", "")).strip() for it in items if str(it.get("description", "")).strip()]


def compute_losses(evaluator, sequences):
    losses = evaluator.evaluate_batch(sequences)
    return [l for l in losses if isinstance(l, (int, float)) and not math.isnan(l)]


def plot_density(series, output_path):
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

    plt.figure(figsize=(10, 6))
    colors = {
        "benign_full": "#1f77b4",
        "carrier_query": "#ff7f0e",
        "attack_query": "#d62728",
    }

    for name, values in series.items():
        values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        if not values:
            continue
        if use_seaborn:
            sns.kdeplot(values, label=name, color=colors.get(name))
        elif use_scipy:
            kde = gaussian_kde(values)
            xs = [i / 100 for i in range(0, 101)]
            ys = kde(xs)
            plt.plot(xs, ys, label=name, color=colors.get(name))
        else:
            import numpy as np

            hist, bin_edges = np.histogram(values, bins=20, range=(0, 1), density=True)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(centers, hist, label=name, color=colors.get(name))

    plt.title("Coherence Loss Density")
    plt.xlabel("coherence_loss")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    parser = argparse.ArgumentParser(description="Plot coherence density curves for OSAgent datasets.")
    parser.add_argument(
        "--merge-system-env",
        action="store_true",
        help="Merge system+environment attacks and carriers into two curves."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="promptarmor/osagent/osagent_coherence_density_all.png",
        help="Output image path."
    )
    parser.add_argument("--no_gpt2", action="store_true", help="disable GPT-2 (use simplified mode). Default: GPT-2 enabled")
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name for GPT-2 mode")
    parser.add_argument("--device", type=str, default="cpu", help="device for GPT-2 mode (cpu or cuda)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[0]
    system_file = base_dir / "osagent_malicious.txt"
    env_file = base_dir / "osagent_malicious_environment.txt"
    benign_file = base_dir / "benign_full.json"
    output_path = Path(args.output)

    CoherenceEvaluator = load_coherence_evaluator()
    use_gpt2 = not args.no_gpt2
    evaluator = CoherenceEvaluator(model_name=args.model_name, device=args.device, use_simplified=(not use_gpt2))

    sys_carriers, sys_attacks = parse_labeled_lines(system_file, "system_carrier:", "system_attack:")
    env_carriers, env_attacks = parse_labeled_lines(env_file, "environment_carrier:", "environment_attack:")
    benign_descs = extract_descriptions(benign_file)

    # Aggregate system and environment into carrier_query and attack_query
    carrier_query = sys_carriers + env_carriers
    attack_query = sys_attacks + env_attacks

    series = {
        "benign_full": compute_losses(evaluator, benign_descs),
        "carrier_query": compute_losses(evaluator, carrier_query),
        "attack_query": compute_losses(evaluator, attack_query),
    }
    # save raw values and perplexities
    json_output_path = Path(args.output).with_name(Path(args.output).stem + "_values.json")
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
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()

