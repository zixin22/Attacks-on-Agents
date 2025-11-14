#!/usr/bin/env python3
"""
reward_cont.py - Analyze Reward information in memory JSON files

Features:
1. Extract all fixed numbers that have a "Reward" key
2. Extract all fixed numbers with "Reward" >= 0.5

Usage:
    python reward_cont.py <memory_file_path>
    
Examples:
    python reward_cont.py ../output_0_500/memory_1.json
    python reward_cont.py ../output_0_500/memory_2.json
"""

import json
import sys
import os
import re
from typing import List, Tuple


def extract_fixed_number(fixed_id: str) -> int:
    """
    Extract number from fixed_id string
    
    Args:
        fixed_id: e.g., "fixed_0", "fixed_123"
    
    Returns:
        Number, e.g., 0, 123
    """
    match = re.search(r'fixed_(\d+)', fixed_id)
    if match:
        return int(match.group(1))
    return -1


def analyze_memory_file(file_path: str) -> Tuple[List[int], List[int]]:
    """
    Analyze memory JSON file
    
    Args:
        file_path: Path to memory JSON file
    
    Returns:
        (List of all fixed numbers with Reward, List of fixed numbers with Reward >= 0.5)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        memory_data = json.load(f)
    
    all_with_reward = []
    reward_ge_05 = []
    
    for entry in memory_data:
        # Extract fixed number
        fixed_id = entry.get('Id', '')
        fixed_num = extract_fixed_number(fixed_id)
        
        if fixed_num == -1:
            print(f"Warning: Cannot parse Id: {fixed_id}", file=sys.stderr)
            continue
        
        # Check if has Reward key
        if 'Reward' in entry:
            reward_value = entry['Reward']
            all_with_reward.append(fixed_num)
            
            # Check Reward >= 0.5
            if isinstance(reward_value, (int, float)) and reward_value >= 0.5:
                reward_ge_05.append(fixed_num)
    
    # Sort
    all_with_reward.sort()
    reward_ge_05.sort()
    
    return all_with_reward, reward_ge_05


def main():
    if len(sys.argv) < 2:
        print("Usage: python reward_cont.py <memory_file_path>")
        print("\nExamples:")
        print("  python C:\Users\22749\Desktop\rap-main\webshop\stat_toolbox\reward_cont.py C:\Users\22749\Desktop\rap-main\webshop\output_1501_2000\memory_1.json")
        print("  python reward_cont.py ../output_0_500/memory_2.json")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        all_with_reward, reward_ge_05 = analyze_memory_file(file_path)
        
        print(f"=" * 60)
        print(f"Analyzing file: {file_path}")
        print(f"=" * 60)
        print()
        
        print(f"1. All fixed numbers with 'Reward' key (total {len(all_with_reward)}):")
        if all_with_reward:
            print(f"   {all_with_reward}")
        else:
            print("   (none)")
        print()
        
        print(f"2. All fixed numbers with 'Reward' >= 0.5 (total {len(reward_ge_05)}):")
        if reward_ge_05:
            print(f"   {reward_ge_05}")
        else:
            print("   (none)")
        print()
        
        # Statistics
        if all_with_reward:
            print(f"Statistics:")
            print(f"  - Total entries: {len(all_with_reward)}")
            print(f"  - Entries with Reward >= 0.5: {len(reward_ge_05)}")
            if len(all_with_reward) > 0:
                print(f"  - Percentage with Reward >= 0.5: {len(reward_ge_05) / len(all_with_reward) * 100:.2f}%")
        
        # Save results to file
        output_file = file_path.replace('.json', '_reward_analysis.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Analyzing file: {file_path}\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"1. All fixed numbers with 'Reward' key (total {len(all_with_reward)}):\n")
            f.write(f"{all_with_reward}\n\n")
            f.write(f"2. All fixed numbers with 'Reward' >= 0.5 (total {len(reward_ge_05)}):\n")
            f.write(f"{reward_ge_05}\n\n")
            if all_with_reward:
                f.write(f"Statistics:\n")
                f.write(f"  - Total entries: {len(all_with_reward)}\n")
                f.write(f"  - Entries with Reward >= 0.5: {len(reward_ge_05)}\n")
                if len(all_with_reward) > 0:
                    f.write(f"  - Percentage with Reward >= 0.5: {len(reward_ge_05) / len(all_with_reward) * 100:.2f}%\n")
        
        print(f"Results saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

