#!/usr/bin/env python3
"""
Build masked/carrier queries from extracted fragments.

Aligned with ``output/dataset_test_12.json`` style for **keyword splitting** and **carrier suffix**:

**Masked query**
  Replace each sensitive fragment in the attack ``instruction`` with ``<>`` (longest-first, once each).

**Keyword → bracket chunks** (per fragment phrase, then concatenate all fragments in order)

1. Split the phrase on whitespace into words (e.g. ``end table``, ``anti perspirant deodorant``).

2. Split **each word** into two contiguous halves:
   ``left = word[:mid]``, ``right = word[mid:]`` where ``mid = len(word) // 2``.
   (Same rule as the common cases in ``dataset_test_12``: ``tablet`` → ``tab|let``,
   ``deodorant`` → ``deod|orant``.)

3. **Multi-word phrase — bridging** (matches ``end table`` → ``<e><nd ta><ble>``,
   ``office desk`` → ``<off><ice de><sk>``, ``anti perspirant deodorant`` → four chunks with
   ``ti persp``, ``irant deod``, etc.):
   Emit ``<left_0>``, then for each adjacent word pair
   ``<right_i left_{i+1}>`` (with a **single space** inside the brackets), then ``<right_last>``.

4. **Exception — two-word phrase with long second word** (matches ``hair extensions`` → four
   separate brackets ``<ha><ir><exten><sions>``, **no** ``ir exten`` merge): when
   ``len(words) == 2`` and ``len(words[1]) >= 8``, emit only per-word halves in order
   (four chunks), **without** cross-word bridging.

5. **Several sensitive fragments** (e.g. ``wireless charging cradle`` + ``phone``): compute chunks
   for each phrase independently and **concatenate** (same order as ``sensitive_fragments``).

**Carrier query**
  ``host_instruction`` + `` .`` + concatenation of ``<chunk>`` for all chunks (no random insertion
  into host tokens).

Input:
  - dataset_attack.json: rows containing host_instruction
  - fragment_dataset_attack.json: rows containing extracted fragment values

Output:
  - output.json: fragment rows plus host_instruction, masked_query, carrier_query
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_DATASET = "dataset_attack.json"
DEFAULT_FRAGMENT = "fragment_dataset_attack.json"
DEFAULT_OUTPUT = "output.json"
ODD_SPLIT_OVERRIDES = {
    # Match desired chunking examples:
    # smart -> sma|rt, shampoo -> sha|mpoo.
    "smart": 3,
    "shampoo": 3,
}


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


def split_word_halves(word: str) -> tuple[str, str]:
    """Split one token into two contiguous halves (``dataset_test_12`` default)."""
    w = word.strip()
    if not w:
        return "", ""
    mid = ODD_SPLIT_OVERRIDES.get(w.lower(), len(w) // 2)
    mid = max(1, mid)
    mid = min(mid, len(w) - 1) if len(w) > 1 else 1
    return w[:mid], w[mid:]


def phrase_to_chunks(phrase: str) -> List[str]:
    """
    Turn one sensitive phrase into bracket-inner strings (no angle brackets).

    Uses cross-word **bridging** except for the two-word long-tail pattern
    (``hair extensions``-style).
    """
    words = phrase.strip().split()
    if not words:
        return []

    if len(words) == 1:
        a, b = split_word_halves(words[0])
        return [x for x in (a, b) if x]

    # Two words, long second: four chunks, no bridge (matches dataset_test_12 id 2).
    if len(words) == 2 and len(words[1]) >= 8:
        chunks: List[str] = []
        for w in words:
            left, right = split_word_halves(w)
            chunks.extend([left, right])
        return [c for c in chunks if c]

    halves = [split_word_halves(w) for w in words]
    out: List[str] = []
    left0, right0 = halves[0]
    if left0:
        out.append(left0)
    for i in range(len(words) - 1):
        bridge = (halves[i][1] + " " + halves[i + 1][0]).strip()
        if bridge:
            out.append(bridge)
    last_right = halves[-1][1]
    if last_right:
        out.append(last_right)
    return out


def fragments_to_chunks(fragments: Iterable[str]) -> List[str]:
    all_chunks: List[str] = []
    for phrase in fragments:
        all_chunks.extend(phrase_to_chunks(phrase))
    return all_chunks


def join_carrier_suffix(chunks: List[str]) -> str:
    """Build ``<...><...>`` suffix from chunk inner strings."""
    if not chunks:
        return ""
    return "".join(f"<{c}>" for c in chunks)


def build_carrier_query(host_instruction: str, fragments: Iterable[str]) -> str:
    host = host_instruction.strip()
    chunks = fragments_to_chunks(fragments)
    if not chunks:
        return host

    suffix = join_carrier_suffix(chunks)
    if not host:
        return suffix.lstrip(".")
    return f"{host} .{suffix}"


def index_by_id(rows: List[Dict]) -> Dict:
    return {row.get("id"): row for row in rows if "id" in row}


def build_output(dataset_rows: List[Dict], fragment_rows: List[Dict]) -> List[Dict]:
    dataset_by_id = index_by_id(dataset_rows)
    output = []

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
            output_row["host_instruction"],
            fragments,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_rows = load_json(args.dataset)
    fragment_rows = load_json(args.fragment)
    output = build_output(dataset_rows, fragment_rows)
    save_json(args.output, output)

    print(f"dataset rows: {len(dataset_rows)}")
    print(f"fragment rows: {len(fragment_rows)}")
    print(f"output rows: {len(output)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
