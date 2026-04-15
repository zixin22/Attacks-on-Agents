#!/usr/bin/env python3
"""Offline helper: align Sensitive Fragment lines with [MASK] regions (rule-based)."""

import re
import os
from typing import List


class NERProcessor:
    """Heuristics only (no NER model)."""

    def __init__(self):
        pass

    def segment_text(self, text: str, num_segments: int = 7) -> List[str]:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= num_segments:
            segments = sentences[:num_segments]
        else:
            words = text.split()
            if len(words) < num_segments:
                segments = words + [words[i % len(words)] for i in range(num_segments - len(words))]
            else:
                segment_size = len(words) // num_segments
                segments = []
                for i in range(num_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(words)
                    segments.append(" ".join(words[start_idx:end_idx]))

        return segments

    def extract_sensitive_fragments_from_masked(self, attack_instruction: str, masked_instruction: str) -> List[str]:
        mask_positions = []
        temp_masked = masked_instruction
        mask_start = 0

        while "[MASK]" in temp_masked:
            mask_idx = temp_masked.find("[MASK]")
            mask_positions.append(mask_start + mask_idx)
            mask_start += mask_idx + len("[MASK]")
            temp_masked = temp_masked[mask_idx + len("[MASK]") :]

        sensitive_fragments = []
        for pos in mask_positions:
            if pos < len(attack_instruction):
                start = pos
                while start > 0 and attack_instruction[start - 1] != " ":
                    start -= 1

                end = pos + len("[MASK]")
                while end < len(attack_instruction) and attack_instruction[end] != " ":
                    end += 1

                fragment = attack_instruction[start:end].strip()
                if fragment:
                    sensitive_fragments.append(fragment)

        return sensitive_fragments

    def extract_sensitive_fragments(self, segments: List[str]) -> List[str]:
        """Legacy: take segments at indices 1,3,5."""
        sensitive_parts = []
        indices = [1, 3, 5]

        for idx in indices:
            if idx < len(segments):
                sensitive_parts.append(segments[idx].strip())

        return sensitive_parts

    def create_masked_instruction(self, text: str, sensitive_fragments: List[str]) -> str:
        if not sensitive_fragments:
            return text

        sorted_fragments = sorted(sensitive_fragments, key=len, reverse=True)

        masked = text
        for fragment in sorted_fragments:
            if fragment.strip():
                escaped_fragment = re.escape(fragment)
                masked = re.sub(
                    r"\b" + escaped_fragment + r"\b",
                    "[MASK]",
                    masked,
                    flags=re.IGNORECASE,
                    count=1,
                )

        return masked.strip()


def process_dataset(input_file: str, output_file: str = None):
    if output_file is None:
        output_file = input_file

    processor = NERProcessor()

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed_lines = []
    pair_start_indices = []

    for i, line in enumerate(lines):
        if line.strip().startswith("Pair ") and ":" in line:
            pair_start_indices.append(i)

    for i, start_idx in enumerate(pair_start_indices):
        if i < len(pair_start_indices) - 1:
            end_idx = pair_start_indices[i + 1]
        else:
            end_idx = len(lines)

        pair_lines = lines[start_idx:end_idx]

        attack_instruction_line = None
        masked_instruction_line = None

        for line in pair_lines:
            if line.strip().startswith("Attack Instruction:"):
                attack_instruction_line = line
            elif line.strip().startswith("Masked Instruction:"):
                masked_instruction_line = line

        if attack_instruction_line and masked_instruction_line:
            attack_instruction = attack_instruction_line.replace("Attack Instruction:", "").strip()
            masked_instruction = masked_instruction_line.replace("Masked Instruction:", "").strip()

            if attack_instruction and masked_instruction:
                print(f"Pair {i + 1}: {attack_instruction[:60]}...")

                sensitive_fragments_list = processor.extract_sensitive_fragments_from_masked(
                    attack_instruction, masked_instruction
                )
                sensitive_fragments_str = "; ".join(sensitive_fragments_list)
                print(f"  fragments: {sensitive_fragments_list}")

                for j, line in enumerate(pair_lines):
                    if line.strip().startswith("Sensitive Fragment:"):
                        pair_lines[j] = f"  Sensitive Fragment: {sensitive_fragments_str}\n"
                        break

        processed_lines.extend(pair_lines)

    if not pair_start_indices:
        processed_lines = lines

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(processed_lines)

    print(f"\nWrote: {output_file}")
    print(f"Pairs processed: {len(pair_start_indices)}")


def main():
    input_file = os.path.join(os.path.dirname(__file__), "data_seeact", "dataset.txt")

    if not os.path.exists(input_file):
        print(f"Missing file: {input_file}")
        return

    print("Updating Sensitive Fragment fields...")
    print(f"Input: {input_file}")
    process_dataset(input_file)
    print("Done.")


if __name__ == "__main__":
    main()
