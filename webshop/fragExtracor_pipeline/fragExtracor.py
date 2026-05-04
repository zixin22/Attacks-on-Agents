#!/usr/bin/env python3
"""
Extract trigger-category keywords from WebShop attack datasets.

Default:
    python fragExtracor.py

reads ``dataset_attack.json`` in this folder and writes
``fragment_dataset_attack.json``. The output keeps only:
``id``, ``instruction``, ``Instruction_fix_number``, ``profile``, and
``fragment`` where ``fragment`` is the detected keyword list.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, List


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSHOP_DIR = os.path.dirname(CURRENT_DIR)
if WEBSHOP_DIR not in sys.path:
    sys.path.insert(0, WEBSHOP_DIR)

from rule_and_profile.rule_checker import RuleChecker  # noqa: E402
from rule_and_profile.user_profile import UserProfile  # noqa: E402


def print_progress(current: int, total: int) -> None:
    width = 32
    ratio = current / total if total else 1
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100
    sys.stdout.write(f"\rProgress: [{bar}] {current}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


class FragmentKeywordExtractor(RuleChecker):
    """Use RuleChecker and extract only Step 1 keywords from its detailed response."""

    def extract_keywords(self, profile: UserProfile, instruction: str) -> List[str]:
        _, _, details = self.check_all_rules(profile, instruction, query="", return_details=True)
        response = str(details.get("response", ""))
        if response == "ERROR":
            return []
        return self._parse_step1_keywords(response)

    @staticmethod
    def _parse_step1_keywords(response: str) -> List[str]:
        text = (response or "").strip()
        if not text or text.upper() == "NONE":
            return []

        step1_match = re.search(
            r"STEP\s*1\s*:\s*KEYWORD\s*DETECTION(?P<body>[\s\S]*?)(?:STEP\s*2\s*:|$)",
            text,
            flags=re.IGNORECASE,
        )
        step1_text = step1_match.group("body").strip() if step1_match else text

        match = re.search(r"Keywords detected:\s*(.+)", step1_text, flags=re.IGNORECASE)
        payload = match.group(1).strip() if match else text
        if payload.upper().startswith("NONE"):
            return []

        keywords: List[str] = []
        list_match = re.search(r"\[[\s\S]*?\]", payload)
        if list_match:
            try:
                parsed = ast.literal_eval(list_match.group(0))
                if isinstance(parsed, list):
                    keywords = [str(x).strip() for x in parsed if str(x).strip()]
            except (SyntaxError, ValueError):
                keywords = []

        if not keywords:
            payload = payload.strip().strip("<>").strip()
            payload = payload.replace("Keywords detected:", "").strip()
            if payload and payload.upper() != "NONE":
                keywords = [x.strip().strip("\"'") for x in payload.split(",") if x.strip()]

        deduped = []
        seen = set()
        for keyword in keywords:
            normalized = keyword.lower().strip()
            if not normalized or normalized in seen:
                continue
            deduped.append(keyword.strip())
            seen.add(normalized)
        return deduped


def user_profile_from_dict(profile_dict: Dict, row_id) -> UserProfile:
    if not isinstance(profile_dict, dict):
        raise ValueError(f"Row {row_id} missing profile dict.")
    return UserProfile(
        profile_id=profile_dict.get("profile_id", f"profile_{row_id}"),
        age=profile_dict["age"],
        country=profile_dict["country"],
        is_verified=profile_dict["is_verified"],
        payment_method=profile_dict["payment_method"],
        failed_payment_attempts=profile_dict["failed_payment_attempts"],
        credit_score=profile_dict["credit_score"],
        account_age_days=profile_dict["account_age_days"],
        account_status=profile_dict["account_status"],
        return_rate=profile_dict["return_rate"],
        total_purchase_amount=profile_dict["total_purchase_amount"],
    )


def default_output_path(dataset_path: str) -> str:
    directory = os.path.dirname(os.path.abspath(dataset_path))
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    return os.path.join(directory, f"fragment_{stem}.json")


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input dataset must be a JSON list.")
    return data


def build_fragment_dataset(
    rows: List[Dict],
    extractor: FragmentKeywordExtractor,
    limit: int | None = None,
    show_progress: bool = True,
) -> List[Dict]:
    selected = rows if limit is None else rows[:limit]
    output = []
    total = len(selected)
    if show_progress and not extractor.verbose:
        print_progress(0, total)
    for idx, row in enumerate(selected, start=1):
        instruction = str(row.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"Row {idx} missing non-empty 'instruction'.")
        profile = user_profile_from_dict(row.get("profile"), row.get("id", idx))
        output.append(
            {
                "id": row.get("id"),
                "instruction": instruction,
                "Instruction_fix_number": row.get("Instruction_fix_number"),
                "profile": row.get("profile"),
                "fragment": extractor.extract_keywords(profile, instruction),
            }
        )
        if extractor.verbose:
            print(f"[{idx}/{len(selected)}] {instruction} -> {output[-1]['fragment']}")
        elif show_progress:
            print_progress(idx, total)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Step-1 keyword fragments from a WebShop dataset.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(CURRENT_DIR, "dataset_attack.json"),
        help="Input dataset JSON. Defaults to fragExtracor_pipeline/dataset_attack.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to fragment_<dataset_stem>.json next to the input dataset.",
    )
    parser.add_argument("--model", default="gpt-4o", help="Model passed to RuleChecker.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    output_path = os.path.abspath(args.output) if args.output else default_output_path(dataset_path)
    rows = load_dataset(dataset_path)
    extractor = FragmentKeywordExtractor(model=args.model, verbose=args.verbose)
    fragment_rows = build_fragment_dataset(
        rows,
        extractor,
        limit=args.limit,
        show_progress=not args.no_progress,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fragment_rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(fragment_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
