import argparse
import json
import os
import re
from typing import Dict, List, Tuple

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def load_dataset_pairs(dataset_file: str) -> List[Dict[str, str]]:
    pairs = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        content = f.read()

    pair_blocks = re.findall(r"Pair \d+:(.*?)(?=Pair \d+:|$)", content, re.DOTALL)
    for block in pair_blocks:
        host_match = re.search(r"Host Instruction:\s*(.+?)(?=\n|$)", block, re.MULTILINE)
        masked_match = re.search(r"Masked Instruction:\s*(.+?)(?=\n|$)", block, re.MULTILINE)
        if host_match and masked_match:
            pairs.append(
                {
                    "host_instruction": host_match.group(1).strip(),
                    "masked_instruction": masked_match.group(1).strip(),
                }
            )
    return pairs


def load_unique_prompts(best_triggers_file: str) -> List[str]:
    with open(best_triggers_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    seen = set()
    for item in data.get("best_individuals", []):
        prompt = item.get("prompt", "").strip()
        if prompt and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def format_trigger(prompt: str, host_instruction: str, masked_instruction: str) -> str:
    formatted = prompt
    if "{host_instruction}" in formatted:
        formatted = formatted.replace("{host_instruction}", host_instruction)
    if "{Masked Instruction}" in formatted:
        formatted = formatted.replace("{Masked Instruction}", f"\"{masked_instruction}\"")
    return formatted


def calculate_similarity(text_a: str, text_b: str, model=None) -> Tuple[float, str]:
    if not text_a or not text_b:
        return 0.0, "none"

    if model is not None:
        try:
            embedding_a = model.encode([text_a])[0]
            embedding_b = model.encode([text_b])[0]
            similarity = float(cos_sim(embedding_a, embedding_b)[0][0])
            similarity = (similarity + 1) / 2
            return max(0.0, min(1.0, similarity)), "sentence_transformers"
        except Exception:
            # Fall back to TF-IDF if sentence-transformers fails at runtime
            pass

    if HAS_SKLEARN:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return max(0.0, min(1.0, float(similarity))), "tfidf"

    raise RuntimeError("No similarity method available (sentence-transformers or sklearn).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best_triggers",
        default=os.path.join("results", "optimization_33", "best_triggers.json"),
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join("data_seeact", "dataset.txt"),
    )
    args = parser.parse_args()

    dataset_pairs = load_dataset_pairs(args.dataset)
    if not dataset_pairs:
        raise RuntimeError("No dataset pairs loaded from dataset.txt")

    prompts = load_unique_prompts(args.best_triggers)
    if not prompts:
        raise RuntimeError("No prompts found in best_triggers.json")

    model = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            model = None

    print(f"Loaded {len(prompts)} unique prompts")
    print(f"Dataset pairs: {len(dataset_pairs)}")
    print(f"Similarity method: {'sentence_transformers' if model else 'tfidf'}")
    print()

    for idx, prompt in enumerate(prompts, 1):
        total = 0.0
        for pair in dataset_pairs:
            formatted = format_trigger(
                prompt, pair["host_instruction"], pair["masked_instruction"]
            )
            score, _ = calculate_similarity(formatted, pair["host_instruction"], model)
            total += score
        avg_score = total / len(dataset_pairs)
        print(f"[{idx}] avg_cos_similarity_with_host = {avg_score:.6f}")
        print(f"    prompt: {prompt}")


if __name__ == "__main__":
    main()

