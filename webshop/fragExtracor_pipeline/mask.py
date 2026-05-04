#!/usr/bin/env python3
"""
Build masked/carrier queries from extracted fragments.

Input:
  - dataset_attack.json: source rows containing host_instruction
  - fragment_dataset_attack.json: rows containing extracted fragment values

Output:
  - output.json: fragment rows plus host_instruction, masked_query, carrier_query
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_DATASET = "dataset_attack.json"
DEFAULT_FRAGMENT = "fragment_dataset_attack.json"
DEFAULT_OUTPUT = "output.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_fragments(raw_fragments) -> List[str]:
    if isinstance(raw_fragments, str):
        raw_fragments = [raw_fragments]
    if not isinstance(raw_fragments, list):
        return []

    fragments = []
    seen = set()
    for fragment in raw_fragments:
        text = str(fragment).strip()
        key = text.lower()
        if text and key not in seen:
            fragments.append(text)
            seen.add(key)
    return fragments


def mask_instruction(instruction: str, fragments: Iterable[str]) -> str:
    masked = instruction
    for fragment in sorted(fragments, key=len, reverse=True):
        pattern = re.compile(re.escape(fragment), flags=re.IGNORECASE)
        masked = pattern.sub("<>", masked, count=1)
    return masked


def split_fragment(fragment: str) -> List[str]:
    text = fragment.strip()
    if not text:
        return []

    words = text.split()
    if len(words) > 1:
        midpoint = max(1, len(words) // 2)
        return [" ".join(words[:midpoint]), " ".join(words[midpoint:])]

    midpoint = max(1, len(text) // 2)
    return [text[:midpoint], text[midpoint:]]


def insert_ordered_randomly(words: List[str], inserts: List[str], rng: random.Random) -> List[str]:
    if not inserts:
        return words

    result = list(words)
    min_pos = 0
    for insert in inserts:
        max_pos = len(result)
        pos = rng.randint(min_pos, max_pos)
        result.insert(pos, insert)
        min_pos = pos + 1
    return result


def build_carrier_query(
    host_instruction: str,
    fragments: Iterable[str],
    rng: Optional[random.Random] = None,
) -> str:
    rng = rng or random.Random()
    inserts = []
    for fragment in fragments:
        inserts.extend(f"<{part}>" for part in split_fragment(fragment) if part)
    if not inserts:
        return host_instruction

    host_words = host_instruction.strip().split()
    if not host_words:
        return " ".join(inserts)
    return " ".join(insert_ordered_randomly(host_words, inserts, rng))


def index_by_id(rows: List[Dict]) -> Dict:
    return {row.get("id"): row for row in rows if "id" in row}


def build_output(
    dataset_rows: List[Dict],
    fragment_rows: List[Dict],
    seed: Optional[int] = None,
) -> List[Dict]:
    dataset_by_id = index_by_id(dataset_rows)
    output = []
    rng = random.Random(seed)

    for fragment_row in fragment_rows:
        row_id = fragment_row.get("id")
        source_row = dataset_by_id.get(row_id)
        if source_row is None:
            continue

        fragments = normalize_fragments(fragment_row.get("fragment", []))
        output_row = dict(fragment_row)
        output_row["host_instruction"] = source_row.get("host_instruction", "")
        output_row["masked_query"] = mask_instruction(
            fragment_row.get("instruction", ""), fragments
        )
        output_row["carrier_query"] = build_carrier_query(
            output_row["host_instruction"], fragments, rng
        )
        output.append(output_row)

    return output


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate output.json with masked_query and carrier_query."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=base_dir / DEFAULT_DATASET,
        help="Path to dataset_attack.json.",
    )
    parser.add_argument(
        "--fragment",
        type=Path,
        default=base_dir / DEFAULT_FRAGMENT,
        help="Path to fragment_dataset_attack.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / DEFAULT_OUTPUT,
        help="Path to write output.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for carrier_query insertion positions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_rows = load_json(args.dataset)
    fragment_rows = load_json(args.fragment)
    output = build_output(dataset_rows, fragment_rows, seed=args.seed)
    save_json(args.output, output)

    print(f"dataset rows: {len(dataset_rows)}")
    print(f"fragment rows: {len(fragment_rows)}")
    print(f"output rows: {len(output)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
