import json
from pathlib import Path
from typing import List, Dict

# Ensure webshop root is on sys.path for promptarmor imports
import sys
webshop_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(webshop_dir))

from promptarmor.detector import PromptArmorDetector
import re


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def extract_task_input_core(path: Path) -> List[str]:
    rows = read_jsonl(path)
    texts = []
    for row in rows:
        text = str(row.get("task_input_core", "")).strip()
        if text:
            texts.append(text)
    return texts


def extract_injection_instruction(path: Path) -> List[str]:
    rows = read_jsonl(path)
    texts = []
    for row in rows:
        text = str(row.get("injection_instruction", "")).strip()
        if text:
            texts.append(text)
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


def extract_prompts_from_benign_txt(path: Path) -> List[str]:
    """Extract \"prompt\" values from benign_behaviors_test_public_ALL.txt"""
    texts: List[str] = []
    if not path.exists():
        return texts
    # Each prompt line looks like:   "prompt": "....",
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r'\"prompt\"\s*:\s*\"(.*)\"\,?$', line)
        if m:
            raw = m.group(1)
            try:
                decoded = json.loads(f"\"{raw}\"")
            except Exception:
                decoded = raw.replace('\\n', '\n').replace('\\t', '\t')
            texts.append(decoded)
    return texts


def main():
    base_dir = Path(__file__).resolve().parent
    task_input_path = base_dir / "retrieve_datasets.jsonl"
    injection_path = base_dir / "hostpair_seperate_results.jsonl"
    output_path = base_dir / "promptarmor_agentgharm.json"

    task_inputs = extract_task_input_core(task_input_path)
    injections = extract_injection_instruction(injection_path)
    benign_txt_path = base_dir / "benign_behaviors_test_public_ALL.txt"
    benign_prompts_txt = extract_prompts_from_benign_txt(benign_txt_path)

    detector = PromptArmorDetector()
    model_name = detector.config.DETECTION_MODEL
    api_base = detector.config.OPENAI_API_BASE

    results = []
    results += detect_texts(detector, task_inputs, "task_input_core")
    results += detect_texts(detector, injections, "injection_instruction")
    # also detect prompts from benign_behaviors_test_public_ALL.txt
    if benign_prompts_txt:
        results += detect_texts(detector, benign_prompts_txt, "benign_behavior_prompt")

    payload = {
        "model": model_name,
        "api_base": api_base,
        "counts": {
            "task_input_core": len(task_inputs),
            "injection_instruction": len(injections),
            "benign_behavior_prompt": len(benign_prompts_txt),
        },
        "results": results,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()

