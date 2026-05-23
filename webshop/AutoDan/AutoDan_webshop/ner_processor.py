#!/usr/bin/env python3
"""
NER
Utilities for segmenting attack instructions, extracting sensitive fragments,
and creating masked instructions.
"""

import re
import os
from typing import List, Tuple


class NERProcessor:
    """Rule-based NER-style processor."""

    def __init__(self):
        # Placeholder for future statistical NER integrations.
        pass

    def segment_text(self, text: str, num_segments: int = 7) -> List[str]:
        """
        Split text into a fixed number of segments.

        Args:
            text: Source text.
            num_segments: Number of segments to produce.

        Returns:
            A list of text segments.
        """
        sentences = re.split(r'[.!?]+', text)
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
                    segments.append(' '.join(words[start_idx:end_idx]))

        return segments

    def extract_sensitive_fragments(self, segments: List[str]) -> List[str]:
        """
        Extract sensitive fragments from the second, fourth, and sixth segments.

        Args:
            segments: Text segments.

        Returns:
            Sensitive fragments.
        """
        sensitive_parts = []
        indices = [1, 3, 5]  # 2nd, 4th, and 6th segments using zero-based indexes.

        for idx in indices:
            if idx < len(segments):
                sensitive_parts.append(segments[idx].strip())

        return sensitive_parts

    def create_masked_instruction(self, text: str, sensitive_fragments: List[str]) -> str:
        """
        Replace sensitive fragments with [MASK].

        Args:
            text: Source instruction.
            sensitive_fragments: Fragments to mask.

        Returns:
            Masked instruction.
        """
        if not sensitive_fragments:
            return text

        sorted_fragments = sorted(sensitive_fragments, key=len, reverse=True)

        masked = text
        for fragment in sorted_fragments:
            if fragment.strip():
                escaped_fragment = re.escape(fragment)
                masked = re.sub(r'\b' + escaped_fragment + r'\b', '[MASK]', masked, flags=re.IGNORECASE, count=1)

        return masked.strip()


def process_dataset(input_file: str, output_file: str = None):
    """
    Process a dataset file in place or write it to a separate output file.

    Args:
        input_file: Input dataset path.
        output_file: Output dataset path. If None, the input file is overwritten.
    """
    if output_file is None:
        output_file = input_file

    processor = NERProcessor()

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    pair_start_indices = []

    for i, line in enumerate(lines):
        if line.strip().startswith('Pair ') and ':' in line:
            pair_start_indices.append(i)

    for i, start_idx in enumerate(pair_start_indices):
        if i < len(pair_start_indices) - 1:
            end_idx = pair_start_indices[i + 1]
        else:
            end_idx = len(lines)

        pair_lines = lines[start_idx:end_idx]

        # Attack Instruction
        attack_instruction_line = None
        attack_instruction_idx = None

        for j, line in enumerate(pair_lines):
            if line.strip().startswith('Attack Instruction:'):
                attack_instruction_line = line
                attack_instruction_idx = start_idx + j
                break

        if attack_instruction_line:
            attack_instruction = attack_instruction_line.replace('Attack Instruction:', '').strip()

            if attack_instruction:
                print(f"Pair {i+1}: {attack_instruction[:60]}...")

                segments = processor.segment_text(attack_instruction, 7)
                print(f"  {len(segments)} segments: {[s[:20] + '...' if len(s) > 20 else s for s in segments]}")

                sensitive_fragments_list = processor.extract_sensitive_fragments(segments)
                sensitive_fragments_str = ', '.join(sensitive_fragments_list)
                print(f"  Sensitive fragments: {sensitive_fragments_list}")
                print(f"  Sensitive fragments (string): {sensitive_fragments_str}")

                masked_instruction = processor.create_masked_instruction(attack_instruction, sensitive_fragments_list)
                print(f"  Masked instruction: {masked_instruction}")

                # Sensitive Fragment
                for j, line in enumerate(pair_lines):
                    if line.strip().startswith('Sensitive Fragment:'):
                        pair_lines[j] = f'  Sensitive Fragment: {sensitive_fragments_str}\n'
                        break

                # Masked Instruction
                for j, line in enumerate(pair_lines):
                    if line.strip().startswith('Masked Instruction:'):
                        pair_lines[j] = f'  Masked Instruction: {masked_instruction}\n'
                        break

        processed_lines.extend(pair_lines)

    if not pair_start_indices:
        processed_lines = lines

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

    print(f"\nDone: {output_file}")
    print(f"Processed {len(pair_start_indices)} pairs")


def main():
    """Run the default dataset processor."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "data_webshop", "dataset.txt")

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    print("Running NER processor...")
    print(f"Input file: {input_file}")

    process_dataset(input_file)

    print("NER processing complete.")


if __name__ == "__main__":
    main()
