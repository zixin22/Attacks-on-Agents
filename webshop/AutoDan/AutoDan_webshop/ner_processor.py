#!/usr/bin/env python3
"""
NER
SeeActAttack Instruction，sensitive fragmentsmasked instructions
"""

import re
import os
from typing import List, Tuple


class NERProcessor:
    """NER"""

    def __init__(self):
        # NER，spaCyNER
        # -based
        pass

    def segment_text(self, text: str, num_segments: int = 7) -> List[str]:
        """
        

        Args:
            text: 
            num_segments: 

        Returns:
            
        """
        # 
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= num_segments:
            # ，
            segments = sentences[:num_segments]
        else:
            # ，
            words = text.split()
            if len(words) < num_segments:
                # ，
                segments = words + [words[i % len(words)] for i in range(num_segments - len(words))]
            else:
                # 
                segment_size = len(words) // num_segments
                segments = []
                for i in range(num_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(words)
                    segments.append(' '.join(words[start_idx:end_idx]))

        return segments

    def extract_sensitive_fragments(self, segments: List[str]) -> List[str]:
        """
        sensitive fragments（2、4、6）

        Args:
            segments: 

        Returns:
            sensitive fragments
        """
        sensitive_parts = []
        indices = [1, 3, 5]  # 2、4、6（0-indexed）

        for idx in indices:
            if idx < len(segments):
                sensitive_parts.append(segments[idx].strip())

        return sensitive_parts

    def create_masked_instruction(self, text: str, sensitive_fragments: List[str]) -> str:
        """
        masked instruction，[MASK]

        Args:
            text: 
            sensitive_fragments: 

        Returns:
            masked instruction
        """
        if not sensitive_fragments:
            return text

        # ，
        sorted_fragments = sorted(sensitive_fragments, key=len, reverse=True)

        masked = text
        for fragment in sorted_fragments:
            if fragment.strip():
                # 
                escaped_fragment = re.escape(fragment)
                # ，[MASK]，
                masked = re.sub(r'\b' + escaped_fragment + r'\b', '[MASK]', masked, flags=re.IGNORECASE, count=1)

        return masked.strip()


def process_dataset(input_file: str, output_file: str = None):
    """
    

    Args:
        input_file: 
        output_file: ，None
    """
    if output_file is None:
        output_file = input_file

    processor = NERProcessor()

    # 
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    pair_start_indices = []

    # Pair
    for i, line in enumerate(lines):
        if line.strip().startswith('Pair ') and ':' in line:
            pair_start_indices.append(i)

    # pair
    for i, start_idx in enumerate(pair_start_indices):
        # pair
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
            # Attack Instruction
            attack_instruction = attack_instruction_line.replace('Attack Instruction:', '').strip()

            if attack_instruction:  # 
                print(f"Pair {i+1}: {attack_instruction[:60]}...")

                # 7
                segments = processor.segment_text(attack_instruction, 7)
                print(f"   {len(segments)} : {[s[:20] + '...' if len(s) > 20 else s for s in segments]}")

                # sensitive fragments（2、4、6）
                sensitive_fragments_list = processor.extract_sensitive_fragments(segments)
                sensitive_fragments_str = ', '.join(sensitive_fragments_list)
                print(f"  Sensitive fragments: {sensitive_fragments_list}")
                print(f"  Sensitive fragments (string): {sensitive_fragments_str}")

                # masked instruction
                masked_instruction = processor.create_masked_instruction(attack_instruction, sensitive_fragments_list)
                print(f"  Masked instruction: {masked_instruction}")

                # 
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

    # pair，
    if not pair_start_indices:
        processed_lines = lines

    # 
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

    print(f"\n！: {output_file}")
    print(f" {len(pair_start_indices)} pairs")


def main():
    """"""
    input_file = r"D:\rap-main\webshop\AutoDan\data_seeact\dataset.txt"

    if not os.path.exists(input_file):
        print(f": {input_file}")
        return

    print("NER...")
    print(f": {input_file}")

    # 
    process_dataset(input_file)

    print("NER！")


if __name__ == "__main__":
    main()
