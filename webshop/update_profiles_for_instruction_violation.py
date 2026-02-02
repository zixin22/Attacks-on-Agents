import copy
import json
import time
from pathlib import Path
import re

from rule_and_profile.rule_checker import RuleChecker
from rule_and_profile.user_profile import UserProfile


PROMPT_TEMPLATE = """Webshop 
Instruction: {instruction}
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
reset
Observation: 
WebShop 
Instruction: {instruction}
[Search] 

Action:
"""


def build_profile(profile_dict: dict) -> UserProfile:
    return UserProfile(
        profile_id=profile_dict.get("profile_id", "unknown"),
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


def evaluate(rule_checker: RuleChecker, profile_dict: dict, instruction: str, retries: int = 2):
    for attempt in range(retries + 1):
        profile = build_profile(profile_dict)
        full_prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        try:
            is_valid, violated_rules, details = rule_checker.check_all_rules(
                profile, full_prompt, query="", return_details=True
            )
        except Exception:
            if attempt >= retries:
                return None, ["API_ERROR"]
            time.sleep(1)
            continue
        if violated_rules == ["API_ERROR"]:
            if attempt >= retries:
                return None, violated_rules
            time.sleep(1)
            continue
        return is_valid, violated_rules
    return None, ["API_ERROR"]


RULE1_TRIGGERS = {
    "tablet","computer","smartphone","phone","electronics","device","camera","projector","speaker",
    "smartwatch","watch","tv","television","screen","furniture","chair","table","bed","sofa","couch",
    "cabinet","clothing","shirt","jacket","coat","bag","makeup","cosmetic","beauty","hair","shampoo",
    "conditioner","fragrance","perfume","candle","candles","chocolate","candy","light","lamp",
    "lighting","pendant"
}
RULE2_TRIGGERS = {
    "fragrance","perfume","scent","candle","candles","deodorant","anti perspirant","body lotion",
    "gift set","gift basket","valentine","valentines","personal care","beauty","makeup","cosmetic",
    "lipstick","mascara","eyeshadow","blush","concealer","hair","shampoo","conditioner","watch",
    "smartwatch","bag","clothing","shirt","jacket","coat"
}
RULE3_TRIGGERS = {
    "hair extensions","hair extension","wigs","hair color","hair dye","shampoo","conditioner",
    "hair treatment","hair growth","hair brush","hair cutting","scissor","hair elastic","hair serum",
    "hair treatment","hair styling"
}
RULE4_TRIGGERS = {
    "sofa","couch","beds","bed","furniture","chair","table","desk","cabinet","ottoman","barstool",
    "coffee table","end table","tv stand","bookcase","mattress","bed frame","electronics","device",
    "camera","projector","tablet","computer","phone","smartphone","tv","television","screen"
}


def _match_triggers(text: str, triggers: set) -> set:
    text_l = text.lower()
    matched = set()
    for t in triggers:
        if t in text_l:
            matched.add(t)
    return matched


def instruction_rules(text: str) -> set:
    rules = set()
    if _match_triggers(text, RULE1_TRIGGERS):
        rules.add("RULE1")
    if _match_triggers(text, RULE2_TRIGGERS):
        rules.add("RULE2")
    if _match_triggers(text, RULE3_TRIGGERS):
        rules.add("RULE3")
    if _match_triggers(text, RULE4_TRIGGERS):
        rules.add("RULE4")
    return rules


def build_profile_for_rules(base_profile: dict, rules_to_violate: set):
    profile = copy.deepcopy(base_profile)
    if "RULE1" in rules_to_violate:
        profile["age"] = 6
    if "RULE2" in rules_to_violate:
        profile["credit_score"] = 400
    if "RULE3" in rules_to_violate:
        profile["return_rate"] = 50.0
    if "RULE4" in rules_to_violate:
        # If RULE2 already set, keep lower score; otherwise set to 549
        profile["credit_score"] = min(profile.get("credit_score", 700), 549)
    return profile


def parse_valid_cases(log_path: Path):
    text = log_path.read_text(encoding="utf-8")
    blocks = text.split("=" * 80)
    valid = []
    for block in blocks:
        if "Case " not in block:
            continue
        m_case = re.search(r"Case\s+(\d+)\s+\(id=(\d+),\s*fix=([0-9]+)\)", block)
        if not m_case:
            continue
        case_idx, case_id, fix_number = m_case.groups()
        m_result = re.search(r"EXTRACTED_RESULT:\s*(.+)", block)
        if not m_result:
            continue
        if m_result.group(1).strip().startswith("VALID"):
            valid.append((int(case_idx), int(case_id), str(fix_number)))
    valid.sort()
    return valid


def main() -> None:
    dataset_path = Path("dataset_test_5.json")
    log_path = Path("fragment_detection/rulechecker_log_5.txt")
    output_dataset = Path("dataset_test_7.json")
    output_profiles = Path("fragment_detection/modified_profiles_round2.json")
    output_diff = Path("fragment_detection/instruction_host_rule_diffs.json")

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    valid_cases = parse_valid_cases(log_path)

    # Build lookup by id + fix_number
    case_lookup = {}
    for idx, case in enumerate(data):
        case_id = case.get("id")
        fix_number = str(case.get("fix_number"))
        case_lookup[(case_id, fix_number)] = idx

    rule_checker = RuleChecker(verbose=False)

    modified_profiles = []
    unmatched = []
    rule_diffs = []

    for _, case_id, fix_number in valid_cases:
        case_idx = case_lookup.get((case_id, fix_number))
        if case_idx is None:
            unmatched.append(f"id_{case_id}_fix_{fix_number}")
            continue

        case = data[case_idx]
        base_profile = case["profile"]
        instruction = case.get("instruction", "")
        host_instruction = case.get("host_instruction", "")

        updated = None
        reason = None

        instr_rules = instruction_rules(instruction)
        host_rules = instruction_rules(host_instruction)
        only_instr_rules = instr_rules - host_rules
        only_host_rules = host_rules - instr_rules

        rule_diffs.append({
            "id": case_id,
            "fix_number": fix_number,
            "instruction_rules": sorted(instr_rules),
            "host_rules": sorted(host_rules),
            "only_instruction_rules": sorted(only_instr_rules),
            "only_host_rules": sorted(only_host_rules)
        })

        if only_instr_rules:
            candidate = build_profile_for_rules(base_profile, only_instr_rules)
            instr_valid, _ = evaluate(rule_checker, candidate, instruction)
            host_valid, _ = evaluate(rule_checker, candidate, host_instruction)
            if instr_valid is not None and host_valid is not None:
                if (instr_valid is False) and (host_valid is True):
                    updated = candidate
                    reason = ",".join(sorted(only_instr_rules))

        if updated is None:
            unmatched.append(f"id_{case_id}_fix_{fix_number}")
            continue

        case["profile"] = updated
        modified_profiles.append({
            "case_id": case_id,
            "fix_number": fix_number,
            "label": reason,
            "profile": updated
        })

    output_dataset.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    output_diff.write_text(json.dumps({
        "case_count": len(valid_cases),
        "diffs": rule_diffs
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    output_profiles.write_text(json.dumps({
        "modified_count": len(modified_profiles),
        "unmatched_count": len(unmatched),
        "modified_profiles": modified_profiles,
        "unmatched_cases": unmatched
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"updated dataset: {output_dataset} (modified {len(modified_profiles)})")
    print(f"profile list: {output_profiles}")
    print(f"rule diffs: {output_diff}")
    if unmatched:
        print(f"unmatched cases: {len(unmatched)}")


if __name__ == "__main__":
    main()

