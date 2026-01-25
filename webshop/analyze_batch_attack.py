#!/usr/bin/env python3
"""
通用WebShop Attack日志分析脚本
可以分析任意batch_attack_X目录的attackplan_webshoplog.txt文件
"""

import re
import json
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class WebShopAttackAnalyzer:
    def __init__(self, log_file_path: str = 'batch_attack_11/attackplan_webshoplog.txt',
                 output_file: str = 'batch_attack_11_analysis.json'):
        self.log_file_path = log_file_path
        self.output_file = output_file
        self.chunk_size = 1000  # 每次读取1000行
        self.save_interval = 10  # 每处理10个case保存一次

    def analyze(self) -> Dict:
        """主分析函数，包含所有优化"""
        print(f"开始分析 {self.log_file_path}...")

        # 1. 获取所有case编号
        all_cases = self._get_all_case_numbers()
        print(f"发现 {len(all_cases)} 个cases")

        results = []
        processed_count = 0

        # 2. 分批处理cases
        for i in range(0, len(all_cases), self.save_interval):
            batch_cases = all_cases[i:i + self.save_interval]

            try:
                batch_results = self._analyze_batch_cases(batch_cases)
                results.extend(batch_results)
                processed_count += len(batch_results)

                # 进度保存
                self._save_progress(results)
                print(f"已处理 {processed_count}/{len(all_cases)} 个cases")

            except Exception as e:
                print(f"处理case批次 {i//self.save_interval + 1} 时出错: {e}")
                continue

        # 生成统计摘要
        summary = self._generate_summary(results)

        # 最终保存
        self._save_final_results(results)
        print(f"分析完成，共处理 {len(results)} 个cases")

        return {
            'total_cases': len(all_cases),
            'processed_cases': len(results),
            'results': results,
            **summary  # 包含所有summary字段
        }

    def _get_all_case_numbers(self) -> List[Dict]:
        """获取所有case编号（兼容新旧日志格式）"""
        cases: List[Dict] = []
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if 'ATTACK PLAN FOR CASE' not in line:
                        continue
                    match_new = re.search(r'ATTACK PLAN FOR CASE id_(\d+)_fix_(\d+)', line)
                    if match_new:
                        case_id = match_new.group(1)
                        fix_number = match_new.group(2)
                        cases.append({
                            'case_key': f'id_{case_id}_fix_{fix_number}',
                            'case_id': case_id,
                            'fix_number': fix_number
                        })
                        continue
                    match_old = re.search(r'ATTACK PLAN FOR CASE fixed_(\d+)', line)
                    if match_old:
                        fix_number = match_old.group(1)
                        cases.append({
                            'case_key': f'fixed_{fix_number}',
                            'case_id': None,
                            'fix_number': fix_number
                        })
        except Exception as e:
            print(f"读取case编号时出错: {e}")

        return cases

    def _analyze_batch_cases(self, case_numbers: List[Dict]) -> List[Dict]:
        """分析一批cases"""
        results = []

        for case_num in case_numbers:
            try:
                case_data = self._analyze_single_case(case_num)
                if case_data:
                    results.append(case_data)
            except Exception as e:
                print(f"分析case {case_num} 时出错: {e}")
                continue

        return results

    def _analyze_single_case(self, case_num: Dict) -> Optional[Dict]:
        """分析单个case（分块读取优化）"""

        # 分块读取相关内容
        case_content = self._read_case_content(case_num['case_key'])
        if not case_content:
            return None

        # 提取信息
        fragment_info = self._extract_fragment_info(case_content, case_num)
        trigger_info = self._extract_trigger_info(case_content, case_num)

        # 判断匹配
        match_result = self._check_fix_number_match(fragment_info, trigger_info)

        return {
            'case_key': case_num['case_key'],
            'case_id': case_num['case_id'],
            'case_fix_number': case_num['fix_number'],
            'fragment_attack': fragment_info,
            'trigger_attack': trigger_info,
            'retrieved_fix_number_match': match_result
        }

    def _read_case_content(self, case_key: str) -> str:
        """读取完整case内容，不使用截断"""
        try:
            # 一次性读取整个文件，避免截断问题
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 找到case的开始和结束位置
            case_start_pattern = f'ATTACK PLAN FOR CASE {case_key}'
            case_start = content.find(case_start_pattern)
            if case_start == -1:
                return ""

            # 找到下一个case的开始作为结束位置
            next_case_pattern = 'ATTACK PLAN FOR CASE '
            next_case_start = content.find(next_case_pattern, case_start + len(case_start_pattern))
            if next_case_start == -1:
                # 这是最后一个case
                case_content = content[case_start:]
            else:
                case_content = content[case_start:next_case_start]

            return case_content

        except Exception as e:
            print(f"读取case {case_key} 内容时出错: {e}")
            return ""

    def _extract_fragment_info(self, content: str, case_num: Dict) -> Dict:
        """提取fragment attack信息，精准识别header并跳过INCOMPLETE sessions"""

        # 默认值
        info = {
            'session_id': None,
            'has_buy_now': False,
            'reward': 0.0,
            'fix_number': case_num['fix_number'],
            'case_id': case_num['case_id']
        }

        try:
            if case_num['case_id'] is not None:
                session_id = f"id_{case_num['case_id']}_fix_{case_num['fix_number']}"
                matches = list(re.finditer(rf"Session ID:\s*{re.escape(session_id)}", content))
                for match in matches:
                    session_start = match.start()
                    header_text = self._extract_session_header_precise(content, session_start)
                    if not header_text:
                        continue
                    attack_window = content[max(0, session_start - 200):session_start]
                    if 'FRAGMENT' not in attack_window:
                        continue
                    if 'INCOMPLETE' in header_text or 'INCOMPLETE' in attack_window:
                        continue
                    info['session_id'] = session_id
                    self._parse_header_info(header_text, info)
                    return info

            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                session_id = f'fixed_attack_fragment_{letter}_{case_num["fix_number"]}'
                matches = list(re.finditer(rf"Session ID:\s*{re.escape(session_id)}", content))
                for match in matches:
                    session_start = match.start()
                    header_text = self._extract_session_header_precise(content, session_start)
                    if not header_text:
                        continue
                    attack_window = content[max(0, session_start - 200):session_start]
                    if 'FRAGMENT' not in attack_window:
                        continue
                    if 'INCOMPLETE' in header_text or 'INCOMPLETE' in attack_window:
                        continue
                    info['session_id'] = session_id
                    self._parse_header_info(header_text, info)
                    return info

        except Exception as e:
            print(f"提取fragment信息时出错: {e}")

        return info

    def _extract_trigger_info(self, content: str, case_num: Dict) -> Dict:
        """提取trigger attack信息，精准识别header并跳过INCOMPLETE sessions"""

        info = {
            'session_id': None,
            'has_buy_now': False,
            'reward': 0.0,
            'retrieved_fix_number': None,
            'retrieved_case_id': None
        }

        try:
            if case_num['case_id'] is not None:
                session_id = f"id_{case_num['case_id']}_fix_{case_num['fix_number']}"
                matches = list(re.finditer(rf"Session ID:\s*{re.escape(session_id)}", content))
            else:
                session_id = f'fixed_attack_trigger_{case_num["fix_number"]}'
                matches = list(re.finditer(rf"Session ID:\s*{re.escape(session_id)}", content))

            if not matches:
                return info  # 没有找到session

            # 遍历所有匹配的session，选择第一个完整的（非INCOMPLETE的）
            for match in matches:
                session_start = match.start()

                # 检查是否是INCOMPLETE session
                # 向前查找attack type标识
                attack_type_start = content.rfind('\n', 0, session_start)
                if attack_type_start == -1:
                    continue

                attack_type_line_start = attack_type_start + 1
                attack_type_line_end = content.find('\n', attack_type_line_start)
                if attack_type_line_end == -1:
                    continue

                attack_type_line = content[attack_type_line_start:attack_type_line_end].strip()

                # 如果是INCOMPLETE session，跳过
                if 'INCOMPLETE' in attack_type_line:
                    continue

                # 找到了完整的session，开始精准提取header
                header_text = self._extract_session_header_precise(content, session_start)
                if header_text:
                    attack_window = content[max(0, session_start - 200):session_start]
                    if 'TRIGGER' not in attack_window:
                        continue
                    info['session_id'] = session_id
                    self._parse_header_info(header_text, info)

                    # 查找retrieved memory信息
                    self._extract_retrieved_memory(content, session_start, info)
                    break  # 找到第一个完整session就停止

        except Exception as e:
            print(f"提取trigger信息时出错: {e}")

        return info

    def _extract_session_header_precise(self, content: str, session_start: int) -> str:
        """精准提取session header"""
        # 1. 找到header开始：向上查找第一个 "====" 行
        header_start_search = content.rfind("====", 0, session_start)
        if header_start_search != -1:
            # 找到完整的header开始行
            full_header_start = content.rfind("\n", 0, header_start_search)
            if full_header_start != -1:
                header_start = full_header_start + 1
            else:
                header_start = header_start_search
        else:
            header_start = session_start

        # 2. 找到header结束：向下查找第一个 "----" 或 "===="
        header_end = content.find("----", session_start)
        if header_end == -1:
            header_end = content.find("====", session_start)

        if header_end != -1:
            return content[header_start:header_end].strip()

        return ""

    def _parse_header_info(self, header_text: str, info: Dict):
        """从header文本中解析信息"""
        # 提取Reward
        reward_match = re.search(r'Reward:\s*([0-9.]+)', header_text)
        if reward_match:
            reward_str = reward_match.group(1)
            info['reward'] = float(reward_str)
            # 根据Reward值判断has_buy_now：Reward: 0表示未执行buy_now，Reward: 0.0或其他值表示已执行
            info['has_buy_now'] = reward_str != '0'

    def _extract_retrieved_memory(self, content: str, session_start: int, info: Dict):
        """提取retrieved memory信息"""
        # 查找EXECUTION LOG开始位置
        execution_start = content.find("EXECUTION LOG", session_start)
        if execution_start == -1:
            return

        # 从EXECUTION LOG开始往后查找retrieved memory信息
        # 通常在EXECUTION LOG后的第一个prompt中
        execution_section = content[execution_start:execution_start + 10000]  # 限制搜索范围

        # 提取retrieved memory中的fix_number
        retrieved_pattern = r'Retrieved Memory Sessions:\s*-\s*fixed_attack_fragment_[A-Z]_(\d+)(?:\s*\(id_(\d+)_fix_(\d+)\))?'
        match = re.search(retrieved_pattern, execution_section)
        if match:
            info['retrieved_fix_number'] = match.group(1)
            if match.group(2) and match.group(3):
                info['retrieved_case_id'] = match.group(2)

    def _check_fix_number_match(self, fragment_info: Dict, trigger_info: Dict) -> bool:
        """检查fix_number是否匹配"""
        fragment_fix = fragment_info.get('fix_number')
        retrieved_fix = trigger_info.get('retrieved_fix_number')
        fragment_case_id = fragment_info.get('case_id')
        retrieved_case_id = trigger_info.get('retrieved_case_id')

        if fragment_case_id and retrieved_case_id:
            return fragment_case_id == retrieved_case_id and fragment_fix == retrieved_fix
        return fragment_fix == retrieved_fix and fragment_fix is not None

    def _save_progress(self, results: List[Dict]):
        """进度保存"""
        try:
            temp_file = self.output_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_count': len(results),
                    'results': results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存进度时出错: {e}")

    def _save_final_results(self, results: List[Dict]):
        """最终保存结果"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_processed': len(results),
                    'results': results,
                    'summary': self._generate_summary(results)
                }, f, indent=2, ensure_ascii=False)

            # 删除临时文件
            temp_file = self.output_file + '.tmp'
            if os.path.exists(temp_file):
                os.remove(temp_file)

            print(f"结果已保存到 {self.output_file}")

        except Exception as e:
            print(f"保存最终结果时出错: {e}")

    def _generate_summary(self, results: List[Dict]) -> Dict:
        """生成统计摘要"""
        total = len(results)
        if total == 0:
            return {'error': 'no results to analyze'}

        trigger_buy_now_count = sum(1 for r in results if r['trigger_attack']['has_buy_now'])
        fragment_buy_now_count = sum(1 for r in results if r['fragment_attack']['has_buy_now'])
        match_count = sum(1 for r in results if r['retrieved_fix_number_match'])

        # 计算reward统计 - 只计算成功buy_now的cases (has_buy_now=true)
        trigger_rewards = [r['trigger_attack']['reward'] for r in results
                          if r['trigger_attack']['has_buy_now']]
        fragment_rewards = [r['fragment_attack']['reward'] for r in results
                           if r['fragment_attack']['has_buy_now']]

        return {
            'total_cases_analyzed': total,
            'fragment_buy_now_count': fragment_buy_now_count,
            'trigger_buy_now_count': trigger_buy_now_count,
            'fix_number_matches': match_count,
            'avg_trigger_reward': sum(trigger_rewards) / len(trigger_rewards) if trigger_rewards else 0,
            'avg_fragment_reward': sum(fragment_rewards) / len(fragment_rewards) if fragment_rewards else 0
        }

    def generate_reward_summary_txt(self, output_txt_file: str = 'reward_summary.txt'):
        """生成reward统计txt文件，格式：case_key: fragment_attack +reward；attack_trigger+reward"""
        print(f"生成reward统计txt文件: {output_txt_file}")

        # 获取所有case编号
        all_cases = self._get_all_case_numbers()
        print(f"发现 {len(all_cases)} 个cases")

        reward_summary: Dict[str, Dict] = {}

        for i, case_num in enumerate(all_cases):
            if i % 50 == 0:
                print(f"处理进度: {i}/{len(all_cases)}")
            try:
                case_content = self._read_case_content(case_num['case_key'])
                if not case_content:
                    continue
                fragment_info = self._extract_fragment_info(case_content, case_num)
                trigger_info = self._extract_trigger_info(case_content, case_num)
                reward_summary[case_num['case_key']] = {
                    'case_id': case_num['case_id'],
                    'fix_number': case_num['fix_number'],
                    'fragment_reward': fragment_info.get('reward', 0.0),
                    'fragment_reward_str': fragment_info.get('reward_str'),
                    'trigger_reward': trigger_info.get('reward', 0.0),
                    'trigger_reward_str': trigger_info.get('reward_str')
                }
            except Exception as e:
                print(f"处理case {case_num['case_key']} 时出错: {e}")
                continue

        try:
            with open(output_txt_file, 'w', encoding='utf-8') as f:
                def _sort_key(k):
                    info = reward_summary[k]
                    case_id = info.get('case_id')
                    fix_number = info.get('fix_number')
                    if case_id is not None:
                        return (int(case_id), int(fix_number))
                    return (10**9, int(fix_number))

                for case_key in sorted(reward_summary.keys(), key=_sort_key):
                    data = reward_summary[case_key]
                    fragment_reward = data['fragment_reward']
                    fragment_reward_str = data.get('fragment_reward_str')
                    trigger_reward = data['trigger_reward']
                    trigger_reward_str = data.get('trigger_reward_str')

                    fragment_out = fragment_reward_str if fragment_reward_str is not None else str(fragment_reward)
                    trigger_out = trigger_reward_str if trigger_reward_str is not None else str(trigger_reward)
                    f.write(f"{case_key}: fragment_attack +{fragment_out}；attack_trigger+{trigger_out}\n")

            print(f"✅ Reward统计txt文件已生成: {output_txt_file}")
            print(f"📊 总共处理了 {len(reward_summary)} 个cases")
        except Exception as e:
            print(f"写入txt文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='分析WebShop Attack日志文件')
    parser.add_argument('--batch', '-b', type=str, required=True,
                       help='batch编号，如: 7, 11, 12等')

    args = parser.parse_args()

    # 构造默认路径
    batch_num = args.batch
    log_file = f'batch_attack_{batch_num}/attackplan_webshoplog.txt'
    output_file = f'batch_attack_{batch_num}/analysis.json'
    reward_output = f'batch_attack_{batch_num}/reward_summary.txt'

    print(f"开始分析 batch_attack_{batch_num}...")
    print(f"日志文件: {log_file}")
    print(f"输出文件: {output_file}")

    # 创建分析器并运行
    analyzer = WebShopAttackAnalyzer(
        log_file_path=log_file,
        output_file=output_file
    )

    summary = analyzer.analyze()
    analyzer.generate_reward_summary_txt(reward_output)

    print(f"\n✅ batch_attack_{batch_num} 分析完成！")
    print(f"📊 处理了 {summary.get('processed_cases', 0)} 个cases")

    # 检查是否有有效的分析结果
    if summary.get('processed_cases', 0) > 0:
        print("\n📈 关键指标:")
        print(f"   - Fragment Attacks成功购买数: {summary.get('fragment_buy_now_count', 0)}")
        print(f"   - Trigger Attacks成功购买数: {summary.get('trigger_buy_now_count', 0)}")
        print(f"   - Memory匹配数: {summary.get('fix_number_matches', 0)}")
        print(f"   - 平均Fragment Reward: {summary.get('avg_fragment_reward', 0):.3f}")
        print(f"   - 平均Trigger Reward: {summary.get('avg_trigger_reward', 0):.3f}")
    else:
        print("\n⚠️  未找到有效的cases或分析失败")

if __name__ == "__main__":
    main()
