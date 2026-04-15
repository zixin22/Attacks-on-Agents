#!/usr/bin/env python3
"""Build a remaining-cases dataset for interrupted attack runs."""

import argparse
import json
import os
import re
from typing import Dict, List, Set


CASE_PATTERN = re.compile(r"ATTACK PLAN FOR CASE id_(\d+)_fix_(\d+)")


def read_processed_case_ids(log_path: str) -> Set[int]:
    """Parse already-processed case ids from attackplan_webshoplog.txt."""
    if not os.path.exists(log_path):
        return set()

    processed: Set[int] = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = CASE_PATTERN.search(line)
            if m:
                processed.add(int(m.group(1)))
    return processed


def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array: {dataset_path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create remaining-cases JSON for resuming interrupted attack run."
    )
    parser.add_argument("--dataset", required=True, help="Original dataset JSON path")
    parser.add_argument(
        "--attack-log",
        required=True,
        help="Path to output/<run>/attackplan_webshoplog.txt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Apply the same initial --limit used in the interrupted run",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path for remaining cases",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    original_count = len(dataset)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        dataset = dataset[: args.limit]

    processed_ids = read_processed_case_ids(args.attack_log)
    remaining = [
        case for case in dataset if int(case.get("id", -1)) not in processed_ids
    ]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)

    print(f"Original dataset size: {original_count}")
    print(f"Effective dataset size (after --limit): {len(dataset)}")
    print(f"Processed case ids in log: {len(processed_ids)}")
    print(f"Remaining cases written: {len(remaining)}")
    print(f"Output: {args.out}")

    if remaining:
        first_id = remaining[0].get("id")
        last_id = remaining[-1].get("id")
        print(f"Remaining id range (dataset order): {first_id} -> {last_id}")
    else:
        print("No remaining cases. Nothing to resume.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
