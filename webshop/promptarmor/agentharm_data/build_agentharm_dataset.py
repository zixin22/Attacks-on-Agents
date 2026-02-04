#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def extract_injection_instructions(path: Path) -> List[str]:
    rows = read_jsonl(path)
    texts = []
    for r in rows:
        v = r.get("injection_instruction") or r.get("injectionInstruction")
        if isinstance(v, str) and v.strip():
            texts.append(v.strip())
    return texts


def extract_task_input_core(path: Path) -> List[str]:
    rows = read_jsonl(path)
    texts = []
    for r in rows:
        v = r.get("task_input_core") or r.get("taskInputCore") or r.get("task_input")
        if isinstance(v, str) and v.strip():
            texts.append(v.strip())
    return texts


def extract_prompts_from_textfile(path: Path) -> List[str]:
    texts = []
    if not path.exists():
        return texts
    import re
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
    base = Path(__file__).resolve().parent
    hostpair_path = base / "hostpair_seperate_results.jsonl"
    benign_txt = base / "benign_behaviors_test_public_ALL.txt"
    retrieve_path = base / "retrieve_datasets.jsonl"
    out_path = base / "AgentHarm_dataset.json"

    carrier = extract_injection_instructions(hostpair_path)
    benign = extract_prompts_from_textfile(benign_txt)
    attack = extract_task_input_core(retrieve_path)

    payload = {
        "benign_query": benign,
        "carrier_query": carrier,
        "attack_query": attack,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote benign={len(benign)}, carrier={len(carrier)}, attack={len(attack)} to {out_path}")


if __name__ == "__main__":
    main()


