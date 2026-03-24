import json
from pathlib import Path
from typing import List, Dict

# Ensure webshop root is on sys.path for promptarmor imports
import sys
webshop_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(webshop_dir))

from promptarmor.detector import PromptArmorDetector


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
    webshop_txt = base_dir / "webshop.txt"
    output_file = base_dir / "promptarmor_webshop.json"

    texts = read_webshop_txt(webshop_txt)

    detector = PromptArmorDetector()
    model_name = detector.config.DETECTION_MODEL
    api_base = detector.config.OPENAI_API_BASE

    results: List[Dict] = []
    results += detect_texts(detector, texts, "webshop_host_instruction")

    payload = {
        "model": model_name,
        "api_base": api_base,
        "counts": {
            "webshop_host_instruction": len(texts),
        },
        "results": results,
    }

    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_file}")


if __name__ == "__main__":
    main()










