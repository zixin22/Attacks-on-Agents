#!/usr/bin/env python3
"""
Test script for RuleChecker Extract functionality
"""

import sys
sys.path.append('.')

# Import the modified RuleCheckerDetailed from process_fragments_test.py
from process_fragments_test import RuleCheckerDetailed
from rule_and_profile.user_profile import UserProfile

def test_extract_functionality():
    # 创建测试用户资料
    profile = UserProfile(
        profile_id='test_profile',
        age=11,  # 未成年人
        country='not_allowed',
        is_verified=True,
        payment_method='Prepaid',
        failed_payment_attempts=7,
        credit_score=233,
        account_age_days=3285,
        account_status='restricted',
        return_rate=52.9,
        total_purchase_amount=4873.06
    )

    # 创建RuleCheckerDetailed (包含Extract功能)
    rule_checker = RuleCheckerDetailed(verbose=False, model='gpt-4o')

    # 测试check_all_rules方法
    instruction = 'im looking for furniture engineered wood at the living room the color was grey'
    query = ''

    print('Testing check_all_rules with Extract functionality...')
    try:
        is_valid, violated_rules, details = rule_checker.check_all_rules(profile, instruction, query, return_details=True)

        print(f'✅ Success!')
        print(f'is_valid: {is_valid}')
        print(f'violated_rules: {violated_rules}')
        print(f'extracted_result: {details.get("extracted_result", "N/A")}')

        print('\n📋 Details:')
        print(f'Prompt: {details.get("prompt", "N/A")[:100]}...')
        print(f'Response length: {len(details.get("response", ""))} characters')
        print(f'Full response: {repr(details.get("response", "N/A"))}')

        # 验证提取结果格式
        extracted = details.get("extracted_result", "")
        if extracted.startswith("VIOLATED:") or extracted == "VALID":
            print('✅ Extracted result format is correct!')
        else:
            print('❌ Extracted result format is incorrect!')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extract_functionality()
