#!/usr/bin/env python3
"""
独立的NER处理脚本
用于处理SeeAct数据集中的Attack Instruction，提取sensitive fragments和masked instructions
"""

import re
import os
from typing import List, Tuple


class NERProcessor:
    """NER处理器"""

    def __init__(self):
        # 可以在这里初始化NER模型，比如spaCy或其他NER工具
        # 目前使用简单的规则-based方法
        pass

    def segment_text(self, text: str, num_segments: int = 7) -> List[str]:
        """
        将文本切分成指定数量的段

        Args:
            text: 要切分的文本
            num_segments: 段的数量

        Returns:
            分割后的文本段列表
        """
        # 使用简单的句子分割和词语分割
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= num_segments:
            # 如果句子足够多，直接使用句子作为段
            segments = sentences[:num_segments]
        else:
            # 如果句子不够，使用词语级别的分割
            words = text.split()
            if len(words) < num_segments:
                # 如果词语也不够，重复某些词语
                segments = words + [words[i % len(words)] for i in range(num_segments - len(words))]
            else:
                # 平均分割词语
                segment_size = len(words) // num_segments
                segments = []
                for i in range(num_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(words)
                    segments.append(' '.join(words[start_idx:end_idx]))

        return segments

    def extract_sensitive_fragments_from_masked(self, attack_instruction: str, masked_instruction: str) -> List[str]:
        """
        从Attack Instruction和Masked Instruction中提取真实的sensitive fragments

        Args:
            attack_instruction: 完整的attack instruction
            masked_instruction: 带有[MASK]的masked instruction

        Returns:
            sensitive fragments列表
        """
        # 解析Masked Instruction，找到[MASK]的位置
        mask_positions = []
        temp_masked = masked_instruction
        mask_start = 0

        while '[MASK]' in temp_masked:
            mask_idx = temp_masked.find('[MASK]')
            mask_positions.append(mask_start + mask_idx)
            mask_start += mask_idx + len('[MASK]')
            temp_masked = temp_masked[mask_idx + len('[MASK]'):]

        # 从Attack Instruction中提取对应位置的文本
        sensitive_fragments = []
        for pos in mask_positions:
            if pos < len(attack_instruction):
                # 找到这个位置对应的词语（向前向后扩展到空格）
                start = pos
                while start > 0 and attack_instruction[start-1] != ' ':
                    start -= 1

                end = pos + len('[MASK]')
                while end < len(attack_instruction) and attack_instruction[end] != ' ':
                    end += 1

                fragment = attack_instruction[start:end].strip()
                if fragment:
                    sensitive_fragments.append(fragment)

        return sensitive_fragments

    def extract_sensitive_fragments(self, segments: List[str]) -> List[str]:
        """
        从段中提取sensitive fragments（第2、4、6段）- 向后兼容

        Args:
            segments: 文本段列表

        Returns:
            sensitive fragments列表
        """
        sensitive_parts = []
        indices = [1, 3, 5]  # 第2、4、6段（0-indexed）

        for idx in indices:
            if idx < len(segments):
                sensitive_parts.append(segments[idx].strip())

        return sensitive_parts

    def create_masked_instruction(self, text: str, sensitive_fragments: List[str]) -> str:
        """
        创建masked instruction，用[MASK]替换敏感片段

        Args:
            text: 原始文本
            sensitive_fragments: 敏感片段列表

        Returns:
            masked instruction
        """
        if not sensitive_fragments:
            return text

        # 按长度降序排序，避免短片段匹配长片段的子串
        sorted_fragments = sorted(sensitive_fragments, key=len, reverse=True)

        masked = text
        for fragment in sorted_fragments:
            if fragment.strip():
                # 转义特殊字符用于正则表达式
                escaped_fragment = re.escape(fragment)
                # 用单词边界确保只匹配完整的片段，用[MASK]替换，只替换一次
                masked = re.sub(r'\b' + escaped_fragment + r'\b', '[MASK]', masked, flags=re.IGNORECASE, count=1)

        return masked.strip()


def process_dataset(input_file: str, output_file: str = None):
    """
    处理数据集文件

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file

    processor = NERProcessor()

    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    pair_start_indices = []

    # 找到所有Pair的起始行
    for i, line in enumerate(lines):
        if line.strip().startswith('Pair ') and ':' in line:
            pair_start_indices.append(i)

    # 处理每个pair
    for i, start_idx in enumerate(pair_start_indices):
        # 确定这个pair的结束位置
        if i < len(pair_start_indices) - 1:
            end_idx = pair_start_indices[i + 1]
        else:
            end_idx = len(lines)

        pair_lines = lines[start_idx:end_idx]

        # 查找Attack Instruction行
        attack_instruction_line = None
        attack_instruction_idx = None

        for j, line in enumerate(pair_lines):
            if line.strip().startswith('Attack Instruction:'):
                attack_instruction_line = line
                attack_instruction_idx = start_idx + j
                break

        # 查找Attack Instruction和Masked Instruction行
        attack_instruction_line = None
        masked_instruction_line = None

        for line in pair_lines:
            if line.strip().startswith('Attack Instruction:'):
                attack_instruction_line = line
            elif line.strip().startswith('Masked Instruction:'):
                masked_instruction_line = line

        if attack_instruction_line and masked_instruction_line:
            # 提取内容
            attack_instruction = attack_instruction_line.replace('Attack Instruction:', '').strip()
            masked_instruction = masked_instruction_line.replace('Masked Instruction:', '').strip()

            if attack_instruction and masked_instruction:  # 确保都不为空
                print(f"处理Pair {i+1}: {attack_instruction[:60]}...")

                # 使用新的方法从Masked Instruction中提取真实的sensitive fragments
                sensitive_fragments_list = processor.extract_sensitive_fragments_from_masked(attack_instruction, masked_instruction)
                sensitive_fragments_str = '; '.join(sensitive_fragments_list)
                print(f"  从Masked Instruction提取的Sensitive fragments: {sensitive_fragments_list}")
                print(f"  Sensitive fragments (string): {sensitive_fragments_str}")

                # 更新Sensitive Fragment行
                for j, line in enumerate(pair_lines):
                    if line.strip().startswith('Sensitive Fragment:'):
                        pair_lines[j] = f'  Sensitive Fragment: {sensitive_fragments_str}\n'
                        break

        processed_lines.extend(pair_lines)

    # 如果没有找到pair，保留原文件内容
    if not pair_start_indices:
        processed_lines = lines

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

    print(f"\n处理完成！结果已保存到: {output_file}")
    print(f"处理了 {len(pair_start_indices)} 个pairs")


def main():
    """主函数"""
    input_file = r"D:\rap-main\webshop\AutoDan\AutoDan_webarena\data_webarena\dataset.txt"

    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return

    print("开始NER处理...")
    print(f"输入文件: {input_file}")

    # 处理文件
    process_dataset(input_file)

    print("NER处理完成！")


if __name__ == "__main__":
    main()
