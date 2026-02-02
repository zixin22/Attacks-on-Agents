"""
Run PromptArmor detection on dataset_test_10_attack_and_trigger_queries.txt.
Outputs separate result files for attack and trigger queries.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
webshop_dir = os.path.dirname(current_dir)
sys.path.insert(0, webshop_dir)

from promptarmor.detector import PromptArmorDetector


def parse_cases(input_path: Path):
    cases = []
    lines = [line.rstrip("\n") for line in input_path.read_text(encoding="utf-8").splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("case_"):
            header = line
            attack_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            trigger_line = lines[i + 2].strip() if i + 2 < len(lines) else ""

            attack_query = ""
            trigger_query = ""
            if attack_line.startswith("attack_query:"):
                attack_query = attack_line[len("attack_query:"):].strip()
            if trigger_line.startswith("trigger_query:"):
                trigger_query = trigger_line[len("trigger_query:"):].strip()

            case_id = None
            if "id=" in header:
                try:
                    case_id = header.split("id=", 1)[1].strip()
                except Exception:
                    case_id = None

            cases.append(
                {
                    "header": header,
                    "case_id": case_id,
                    "attack_query": attack_query,
                    "trigger_query": trigger_query,
                }
            )
            i += 3
        else:
            i += 1
    return cases


def detect_queries(cases, detector: PromptArmorDetector, query_key: str):
    results = []
    for idx, case in enumerate(cases, start=1):
        text = case.get(query_key, "")
        try:
            result = detector.detect(text)
            results.append(
                {
                    "index": idx,
                    "case_id": case.get("case_id"),
                    "header": case.get("header"),
                    "instruction": text,
                    "is_injected": result.is_injected,
                    "injection_content": result.injection_content,
                    "cleaned_text": result.cleaned_text,
                    "confidence": result.confidence,
                    "raw_llm_response": result.raw_llm_response,
                    "system_prompt": detector.config.DETECTION_SYSTEM_PROMPT,
                    "user_prompt": detector.config.DETECTION_USER_PROMPT_TEMPLATE.format(text=text),
                }
            )
        except Exception as e:
            results.append(
                {
                    "index": idx,
                    "case_id": case.get("case_id"),
                    "header": case.get("header"),
                    "instruction": text,
                    "error": str(e),
                }
            )
    return results


def main():
    # Force API base for this run (promptarmor config reads OPENAI_API_BASE).
    api_base = "http://152.53.53.64:3000/"
    if not api_base.endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"
    os.environ["OPENAI_API_BASE"] = api_base

    input_path = Path(webshop_dir) / "dataset_test_10_attack_and_trigger_queries.txt"
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    detector = PromptArmorDetector()
    model_name = detector.config.DETECTION_MODEL

    cases = parse_cases(input_path)

    attack_results = detect_queries(cases, detector, "attack_query")
    trigger_results = detect_queries(cases, detector, "trigger_query")

    timestamp = datetime.now().isoformat()
    attack_output = Path(webshop_dir) / "promptarmor" / "attack_query_results.json"
    trigger_output = Path(webshop_dir) / "promptarmor" / "trigger_query_results.json"
    attack_output.parent.mkdir(parents=True, exist_ok=True)

    attack_payload = {
        "timestamp": timestamp,
        "model": model_name,
        "input_file": str(input_path),
        "query_type": "attack_query",
        "total_cases": len(cases),
        "results": attack_results,
    }
    trigger_payload = {
        "timestamp": timestamp,
        "model": model_name,
        "input_file": str(input_path),
        "query_type": "trigger_query",
        "total_cases": len(cases),
        "results": trigger_results,
    }

    attack_output.write_text(json.dumps(attack_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    trigger_output.write_text(json.dumps(trigger_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Attack results saved to: {attack_output}")
    print(f"Trigger results saved to: {trigger_output}")


if __name__ == "__main__":
    main()
"""
Run PromptArmor detection on dataset_test_10_attack_and_trigger_queries.txt.
Outputs separate result files for attack and trigger queries.
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Ensure webshop root is on sys.path for promptarmor imports
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
webshop_dir = os.path.dirname(current_dir)
sys.path.insert(0, webshop_dir)

from promptarmor.detector import PromptArmorDetector


def parse_cases(input_path: Path):
    cases = []
    lines = [line.rstrip("\n") for line in input_path.read_text(encoding="utf-8").splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("case_"):
            header = line
            attack_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            trigger_line = lines[i + 2].strip() if i + 2 < len(lines) else ""

            attack_query = ""
            trigger_query = ""
            if attack_line.startswith("attack_query:"):
                attack_query = attack_line[len("attack_query:"):].strip()
            if trigger_line.startswith("trigger_query:"):
                trigger_query = trigger_line[len("trigger_query:"):].strip()

            case_id = None
            if "id=" in header:
                try:
                    case_id = header.split("id=", 1)[1].strip()
                except Exception:
                    case_id = None

            cases.append(
                {
                    "header": header,
                    "case_id": case_id,
                    "attack_query": attack_query,
                    "trigger_query": trigger_query,
                }
            )
            i += 3
        else:
            i += 1
    return cases


def detect_queries(cases, detector: PromptArmorDetector, query_key: str):
    results = []
    for idx, case in enumerate(cases, start=1):
        text = case.get(query_key, "")
        try:
            result = detector.detect(text)
            results.append(
                {
                    "index": idx,
                    "case_id": case.get("case_id"),
                    "header": case.get("header"),
                    "instruction": text,
                    "is_injected": result.is_injected,
                    "injection_content": result.injection_content,
                    "cleaned_text": result.cleaned_text,
                    "confidence": result.confidence,
                    "raw_llm_response": result.raw_llm_response,
                    "system_prompt": detector.config.DETECTION_SYSTEM_PROMPT,
                    "user_prompt": detector.config.DETECTION_USER_PROMPT_TEMPLATE.format(text=text),
                }
            )
        except Exception as e:
            results.append(
                {
                    "index": idx,
                    "case_id": case.get("case_id"),
                    "header": case.get("header"),
                    "instruction": text,
                    "error": str(e),
                }
            )
    return results


def main():
    input_path = Path(webshop_dir) / "dataset_test_10_attack_and_trigger_queries.txt"
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    detector = PromptArmorDetector()
    model_name = detector.config.DETECTION_MODEL

    cases = parse_cases(input_path)

    attack_results = detect_queries(cases, detector, "attack_query")
    trigger_results = detect_queries(cases, detector, "trigger_query")

    timestamp = datetime.now().isoformat()
    attack_output = Path(webshop_dir) / "promptarmor" / "attack_query_results.json"
    trigger_output = Path(webshop_dir) / "promptarmor" / "trigger_query_results.json"
    attack_output.parent.mkdir(parents=True, exist_ok=True)

    attack_payload = {
        "timestamp": timestamp,
        "model": model_name,
        "input_file": str(input_path),
        "query_type": "attack_query",
        "total_cases": len(cases),
        "results": attack_results,
    }
    trigger_payload = {
        "timestamp": timestamp,
        "model": model_name,
        "input_file": str(input_path),
        "query_type": "trigger_query",
        "total_cases": len(cases),
        "results": trigger_results,
    }

    attack_output.write_text(json.dumps(attack_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    trigger_output.write_text(json.dumps(trigger_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Attack results saved to: {attack_output}")
    print(f"Trigger results saved to: {trigger_output}")


if __name__ == "__main__":
    main()

