import json
from pathlib import Path
from typing import List, Dict

# Ensure webshop root is on sys.path for promptarmor imports
import sys
webshop_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(webshop_dir))

from promptarmor.detector import PromptArmorDetector


def parse_labeled_lines(path: Path, carrier_prefix: str, attack_prefix: str) -> Dict[str, List[str]]:
    carriers: List[str] = []
    attacks: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(carrier_prefix):
            carriers.append(line[len(carrier_prefix):].strip())
        elif line.startswith(attack_prefix):
            attacks.append(line[len(attack_prefix):].strip())
    return {"carriers": carriers, "attacks": attacks}


def load_benign_descriptions(path: Path) -> List[str]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [str(it.get("description", "")).strip() for it in items if str(it.get("description", "")).strip()]


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
    system_file = base_dir / "osagent_malicious.txt"
    env_file = base_dir / "osagent_malicious_environment.txt"
    benign_file = base_dir / "benign_full.json"
    output_file = base_dir / "promptarmor_osagent.json"

    system = parse_labeled_lines(system_file, "system_carrier:", "system_attack:")
    environment = parse_labeled_lines(env_file, "environment_carrier:", "environment_attack:")
    benign = load_benign_descriptions(benign_file)

    detector = PromptArmorDetector()
    model_name = detector.config.DETECTION_MODEL
    api_base = detector.config.OPENAI_API_BASE

    # Combine system+environment for attack and carrier groups
    attack_texts = system["attacks"] + environment["attacks"]
    carrier_texts = system["carriers"] + environment["carriers"]

    results = []
    results += detect_texts(detector, benign, "benign_description")
    results += detect_texts(detector, attack_texts, "malicious_attack")
    results += detect_texts(detector, carrier_texts, "malicious_carrier")

    payload = {
        "model": model_name,
        "api_base": api_base,
        "counts": {
            "benign_description": len(benign),
            "malicious_attack": len(attack_texts),
            "malicious_carrier": len(carrier_texts),
        },
        "results": results,
    }

    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_file}")


if __name__ == "__main__":
    main()

