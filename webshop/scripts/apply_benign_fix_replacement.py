#!/usr/bin/env python3
"""
Apply a "benign fix replacement" to an attack dataset JSON.

For each (case_id -> benign_goal_key):
  - Set fix_number to benign_goal_key (string).
  - Set host_instruction to benign_goal.json[benign_goal_key].
  - Rewrite carrier_instruction_3: keep the suffix from the first \" .<\" (injection tail);
    everything before that becomes the new host_instruction.

Other fields (instruction, masked_instruction, carrier_instruction_1/2, profile, etc.) are unchanged.

Default mapping matches the 8 RAP mis-retrieval cases on dataset_test_12 (ids 1,13,20,35,46,48,73,87)
paired with benign_goal keys 2687,2692,2932,3205,3227,3532,3752,4137.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# webshop/ (parent of scripts/)
WEBSHOP_ROOT = Path(__file__).resolve().parent.parent

# (case_id, benign_goal.json key) — order matters for documentation only; lookup is by id.
DEFAULT_ID_TO_BENIGN_KEY: List[Tuple[int, str]] = [
    (1, "2687"),
    (13, "2692"),
    (20, "2932"),
    (35, "3205"),
    (46, "3227"),
    (48, "3532"),
    (73, "3752"),
    (87, "4137"),
]


def new_carrier_instruction_3(old_carrier_3: str, new_host: str) -> str:
    """Replace the host prefix; keep injection suffix starting at first ' .<'."""
    idx = old_carrier_3.find(" .<")
    if idx >= 0:
        return new_host.rstrip() + old_carrier_3[idx:]
    idx = old_carrier_3.find(".<")
    if idx >= 0:
        return new_host.rstrip() + " " + old_carrier_3[idx:]
    return new_host.rstrip()


def load_mapping(path: Path | None) -> Dict[int, str]:
    if path is None:
        return {cid: key for cid, key in DEFAULT_ID_TO_BENIGN_KEY}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[int, str] = {}
    if isinstance(raw, list):
        for item in raw:
            cid = int(item["id"])
            key = str(item["benign_key"])
            out[cid] = key
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[int(k)] = str(v)
        return out
    raise ValueError("Mapping must be a JSON object {case_id: benign_key} or a list of {id, benign_key}")


def apply_replacement(
    dataset_path: Path,
    benign_path: Path,
    id_to_key: Dict[int, str],
    dry_run: bool,
) -> List[str]:
    with open(benign_path, encoding="utf-8") as f:
        goals = json.load(f)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of case objects")

    by_id = {int(c["id"]): c for c in data if "id" in c}
    log_lines: List[str] = []

    for case_id, fix_key in sorted(id_to_key.items(), key=lambda x: x[0]):
        if fix_key not in goals:
            raise KeyError(f"benign_goal.json missing key {fix_key!r} (case id {case_id})")
        if case_id not in by_id:
            raise KeyError(f"Dataset has no case id={case_id}")

        c = by_id[case_id]
        text = goals[fix_key]
        old_fix = c.get("fix_number")
        old_host = c.get("host_instruction", "")

        c["fix_number"] = fix_key
        c["host_instruction"] = text
        c["carrier_instruction_3"] = new_carrier_instruction_3(
            c.get("carrier_instruction_3", ""), text
        )
        log_lines.append(
            f"id {case_id}: fix_number {old_fix!r} -> {fix_key!r}; "
            f"host_instruction updated ({len(old_host)} -> {len(text)} chars); "
            f"carrier_instruction_3 prefix replaced"
        )

    if not dry_run:
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return log_lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace fix_number, host_instruction, and carrier_instruction_3 host prefix "
        "using benign_goal.json entries."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=WEBSHOP_ROOT / "dataset_test_12.json",
        help="Attack dataset JSON (array of cases).",
    )
    parser.add_argument(
        "--benign",
        type=Path,
        default=WEBSHOP_ROOT / "benign_goal.json",
        help="benign_goal.json (object: fix key -> host text).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional JSON mapping: either {\"1\": \"2687\", ...} or "
        '[{"id": 1, "benign_key": "2687"}, ...]. Default: built-in 8-case mapping.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes; do not write dataset.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    benign_path = args.benign.resolve()
    if not dataset_path.is_file():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    if not benign_path.is_file():
        print(f"Error: benign_goal not found: {benign_path}", file=sys.stderr)
        return 1

    mapping_path = args.mapping.resolve() if args.mapping else None
    try:
        id_to_key = load_mapping(mapping_path)
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as e:
        print(f"Error loading mapping: {e}", file=sys.stderr)
        return 1

    try:
        lines = apply_replacement(
            dataset_path, benign_path, id_to_key, dry_run=args.dry_run
        )
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for line in lines:
        print(line)
    if args.dry_run:
        print("(dry-run: no file written)")
    else:
        print(f"Wrote {dataset_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
