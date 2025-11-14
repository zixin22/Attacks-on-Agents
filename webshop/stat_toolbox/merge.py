#!/usr/bin/env python3


import json
import sys
import os
import re
from typing import List, Set


def parse_reward_analysis_file(file_path: str) -> List[int]:
    """
    Extract fixed numbers with Reward >= 0.5 from reward_analysis.txt file
    
    Args:
        file_path: Path to reward_analysis.txt file
    
    Returns:
        List of fixed numbers
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find line containing "Reward' >= 0.5"
    reward_line_idx = None
    for i, line in enumerate(lines):
        if "'Reward' >= 0.5" in line or '"Reward" >= 0.5' in line:
            reward_line_idx = i
            break
    
    if reward_line_idx is None:
        print(f"Warning: Line with 'Reward' >= 0.5 not found in file {file_path}", file=sys.stderr)
        return []
    
    # Next line should contain the number list
    if reward_line_idx + 1 >= len(lines):
        print(f"Warning: File {file_path} format incorrect, number list not found", file=sys.stderr)
        return []
    
    list_line = lines[reward_line_idx + 1].strip()
    
    # Use regex to extract numbers from list
    # Match format: [0, 7, 10, ...] or [0,7,10,...]
    match = re.search(r'\[(.*?)\]', list_line)
    if not match:
        print(f"Warning: Cannot parse number list from file {file_path}", file=sys.stderr)
        return []
    
    numbers_str = match.group(1)
    if not numbers_str.strip():
        return []
    
    # Extract all numbers
    numbers = []
    for num_str in numbers_str.split(','):
        num_str = num_str.strip()
        if num_str:
            try:
                numbers.append(int(num_str))
            except ValueError:
                print(f"Warning: Cannot parse number '{num_str}' in file {file_path}", file=sys.stderr)
    
    return numbers


def merge_reward_analysis_files(input_files: List[str]) -> List[int]:
    """
    Merge fixed numbers from multiple reward_analysis.txt files
    
    Args:
        input_files: List of reward_analysis.txt file paths
    
    Returns:
        Merged, deduplicated, and sorted list of fixed numbers
    """
    all_numbers: Set[int] = set()
    
    for file_path in input_files:
        print(f"Processing: {file_path}")
        try:
            numbers = parse_reward_analysis_file(file_path)
            all_numbers.update(numbers)
            print(f"  Extracted {len(numbers)} numbers")
        except Exception as e:
            print(f"Error: Failed to process file {file_path}: {e}", file=sys.stderr)
            continue
    
    # Sort and return
    return sorted(list(all_numbers))


def main():
    if len(sys.argv) < 3:
        print("Usage: python merge.py <output_file.json> <input_file1.txt> <input_file2.txt> ...")
        print("\nExamples:")
        print("  python merge.py merged_reward_ge_05.json \\")
        print("    ../output_0_500/memory_1_reward_analysis.txt \\")
        print("    ../output_501_1500/memory_1_reward_analysis.txt \\")
        print("    ../output_1501_2000/memory_1_reward_analysis.txt")
        sys.exit(1)
    
    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    
    if not input_files:
        print("Error: At least one input file is required", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("Merge fixed numbers with Reward >= 0.5")
    print("=" * 60)
    print(f"Number of input files: {len(input_files)}")
    print(f"Output file: {output_file}")
    print()
    
    try:
        # Merge numbers from all files
        merged_numbers = merge_reward_analysis_files(input_files)
        
        print()
        print("=" * 60)
        print("Merge results:")
        print(f"  Total number count: {len(merged_numbers)}")
        if merged_numbers:
            print(f"  Number range: {min(merged_numbers)} - {max(merged_numbers)}")
        print("=" * 60)
        
        # Create output data structure
        output_data = {
            "description": "Merged fixed numbers with Reward >= 0.5",
            "source_files": input_files,
            "total_count": len(merged_numbers),
            "fixed_numbers": merged_numbers
        }
        
        # Save as JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        # Also save a simple list format (optional)
        simple_output_file = output_file.replace('.json', '_simple.txt')
        with open(simple_output_file, 'w', encoding='utf-8') as f:
            f.write(f"Merged fixed numbers with Reward >= 0.5 (total {len(merged_numbers)}):\n")
            f.write(f"{merged_numbers}\n")
        print(f"Simple list format saved to: {simple_output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

