import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure webshop root is on sys.path for promptarmor imports
import sys
webshop_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(webshop_dir))

from promptarmor.detector import PromptArmorDetector


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_texts_from_list(data_list: List[Dict], keys: List[str]) -> List[str]:
    texts: List[str] = []
    for item in data_list:
        if not isinstance(item, dict):
            continue
        for k in keys:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())
                break
    return texts


def detect_texts(detector: PromptArmorDetector, texts: List[str], group: str) -> List[Dict]:
    results = []
    for idx, text in enumerate(texts, start=1):
        try:
            result = detector.detect(text)
            results.append(
                {
                    "group": group,
                    "index": idx,
                    "text": text,
                    "is_injected": result.is_injected,
                    "injection_content": result.injection_content,
                    "cleaned_text": result.cleaned_text,
                    "confidence": result.confidence,
                    "raw_llm_response": result.raw_llm_response,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "group": group,
                    "index": idx,
                    "text": text,
                    "error": str(exc),
                }
            )
    return results


def main():
    base_dir = Path(__file__).resolve().parent
    attack_file = base_dir / "1-splitted_half.json"
    carrier_file = base_dir / "1-splitted_insert_fragment_half.json"
    benign_file = base_dir / "sample_labeled_benign.json"
    output_file = base_dir / "promptarmor_seeact.json"

    attack_data = read_json(attack_file) or []
    carrier_data = read_json(carrier_file) or []
    benign_data = read_json(benign_file) or []

    # extract 'confirmed_task' from each list (all three files use confirmed_task)
    attack_texts = extract_texts_from_list(attack_data, ["confirmed_task"])
    carrier_texts = extract_texts_from_list(carrier_data, ["confirmed_task"])
    benign_texts = extract_texts_from_list(benign_data, ["confirmed_task"])

    detector = PromptArmorDetector()
    model_name = detector.config.DETECTION_MODEL
    api_base = detector.config.OPENAI_API_BASE

    results: List[Dict] = []
    results += detect_texts(detector, benign_texts, "benign_query")
    results += detect_texts(detector, attack_texts, "attack_query")
    results += detect_texts(detector, carrier_texts, "carrier_query")

    payload = {
        "model": model_name,
        "api_base": api_base,
        "counts": {
            "benign_query": len(benign_texts),
            "attack_query": len(attack_texts),
            "carrier_query": len(carrier_texts),
        },
        "results": results,
    }

    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_file}")


if __name__ == "__main__":
    main()


