"""
批量测试 attak_7_v5 到 attak_7_v17 的所有 attack instructions 和 trigger instructions
使用 PromptArmor 进行检测，并记录结果
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# 获取当前脚本所在目录（promptarmor目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取webshop目录（父目录）
webshop_dir = os.path.dirname(current_dir)
# 添加webshop目录到路径
sys.path.insert(0, webshop_dir)

from promptarmor.detector import PromptArmorDetector, DetectionResult
from promptarmor.config import PromptArmorConfig


def extract_instructions_from_log(log_file: Path) -> Dict[str, str]:
    """
    从 attackplan_webshoplog.txt 中提取 instructions
    
    Returns:
        dict with keys: 'fragment_A', 'fragment_B', 'fragment_C', 'fragment_D', 'trigger'
    """
    instructions = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 Fragment Attacks
        fragment_section = re.search(
            r'Fragment Attacks \(4\):(.*?)Trigger Instruction:',
            content,
            re.DOTALL
        )
        
        if fragment_section:
            fragment_text = fragment_section.group(1)
            # 提取 A, B, C, D
            for label in ['A', 'B', 'C', 'D']:
                pattern = rf'\s+{label}:\s*(.+?)(?=\n\s+[A-Z]:|\n\n|$)'
                match = re.search(pattern, fragment_text, re.DOTALL)
                if match:
                    instructions[f'fragment_{label}'] = match.group(1).strip()
        
        # 提取 Trigger Instruction
        trigger_match = re.search(
            r'Trigger Instruction:\s*(.+?)(?=\n={50,}|\n\n|$)',
            content,
            re.DOTALL
        )
        if trigger_match:
            instructions['trigger'] = trigger_match.group(1).strip()
        
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return instructions


def test_all_attacks():
    """测试所有版本的攻击指令"""
    
    # 初始化检测器
    print("初始化 PromptArmor 检测器...")
    detector = PromptArmorDetector()
    
    # 获取使用的模型名称
    model_name = detector.config.DETECTION_MODEL
    print(f"使用模型: {model_name}")
    
    # 获取所有版本目录
    webshop_dir = Path(__file__).parent.parent
    versions = []
    for v in range(5, 18):  # v5 到 v17
        version_dir = webshop_dir / f"attak_7_v{v}"
        if version_dir.exists():
            versions.append(v)
    
    print(f"找到 {len(versions)} 个版本: {versions}")
    
    # 存储所有结果
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'total_versions': len(versions),
        'versions_tested': versions,
        'results': []
    }
    
    # 处理每个版本
    for version in versions:
        version_dir = webshop_dir / f"attak_7_v{version}"
        log_file = version_dir / "attackplan_webshoplog.txt"
        
        if not log_file.exists():
            print(f"警告: {log_file} 不存在，跳过版本 v{version}")
            continue
        
        print(f"\n{'='*80}")
        print(f"处理版本: attak_7_v{version}")
        print(f"{'='*80}")
        
        # 提取 instructions
        instructions = extract_instructions_from_log(log_file)
        
        if not instructions:
            print(f"警告: 无法从 {log_file} 提取 instructions")
            continue
        
        print(f"提取到 {len(instructions)} 个 instructions:")
        for key in instructions.keys():
            print(f"  - {key}")
        
        # 对每个 instruction 进行检测
        version_results = {
            'version': f'v{version}',
            'instructions': {}
        }
        
        # 按顺序处理：先处理 Fragment Attacks，再处理 Trigger
        instruction_order = ['fragment_A', 'fragment_B', 'fragment_C', 'fragment_D', 'trigger']
        
        for inst_key in instruction_order:
            if inst_key not in instructions:
                continue
            
            instruction_text = instructions[inst_key]
            print(f"\n检测 {inst_key}...")
            print(f"文本长度: {len(instruction_text)} 字符")
            
            # 运行检测
            try:
                result = detector.detect(instruction_text)
                
                # 记录结果
                version_results['instructions'][inst_key] = {
                    'instruction': instruction_text,
                    'is_injected': result.is_injected,
                    'injection_content': result.injection_content,
                    'cleaned_text': result.cleaned_text,
                    'confidence': result.confidence,
                    'raw_llm_response': result.raw_llm_response,
                    'system_prompt': detector.config.DETECTION_SYSTEM_PROMPT,
                    'user_prompt': detector.config.DETECTION_USER_PROMPT_TEMPLATE.format(text=instruction_text)
                }
                
                print(f"  检测结果: {'检测到注入' if result.is_injected else '未检测到注入'}")
                print(f"  置信度: {result.confidence:.2f}")
                if result.is_injected:
                    print(f"  提取的注入内容: {result.injection_content[:100]}...")
                
                # 避免 API 速率限制
                import time
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  错误: {e}")
                version_results['instructions'][inst_key] = {
                    'instruction': instruction_text,
                    'error': str(e)
                }
        
        all_results['results'].append(version_results)
    
    # 保存结果 - 根据模型名称生成不同的文件名
    # 将模型名称中的特殊字符替换为下划线，用于文件名
    model_safe_name = model_name.replace('/', '_').replace('-', '_')
    output_file = webshop_dir / "promptarmor" / f"attack_detection_results_{model_safe_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"所有检测完成！结果已保存到: {output_file}")
    print(f"使用模型: {model_name}")
    print(f"{'='*80}")
    
    # 打印摘要
    print("\n检测摘要:")
    total_instructions = 0
    total_detected = 0
    for version_result in all_results['results']:
        version = version_result['version']
        for inst_key, inst_result in version_result['instructions'].items():
            total_instructions += 1
            if 'is_injected' in inst_result and inst_result['is_injected']:
                total_detected += 1
    
    print(f"  总指令数: {total_instructions}")
    print(f"  检测到注入: {total_detected}")
    print(f"  检测率: {total_detected/total_instructions*100:.1f}%" if total_instructions > 0 else "  检测率: N/A")
    
    return all_results


if __name__ == "__main__":
    try:
        results = test_all_attacks()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

