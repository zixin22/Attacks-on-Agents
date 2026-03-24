#!/usr/bin/env python3
import json
from pathlib import Path
import sys

def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        # try jsonl
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

def extract_host_instructions(src_path: Path):
    data = load_json_or_jsonl(src_path)
    items = []
    if isinstance(data, dict):
        # if top-level dict with 'items' or similar
        for key in ("items", "data", "samples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                hi = entry.get("host_instruction") or entry.get("hostInstruction") or entry.get("host_instruction_text")
                if isinstance(hi, str) and hi.strip():
                    items.append(hi.strip())
    return items

def main():
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "dataset_test_4.json"
    out_dir = repo_root / "promptarmor" / "webshop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "webshop.txt"

    if not src.exists():
        print(f"Source file not found: {src}", file=sys.stderr)
        sys.exit(2)

    items = extract_host_instructions(src)
    with out_file.open("w", encoding="utf-8") as f:
        for idx, t in enumerate(items, start=1):
            f.write(t.replace("\r\n", "\n"))
            f.write("\n")

    print(f"wrote {len(items)} host_instruction entries to {out_file}")

if __name__ == "__main__":
    main()










