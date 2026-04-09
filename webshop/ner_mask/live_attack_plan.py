"""
Live NER decomposition + optional RuleChecker mask, then fragment/trigger instructions.

Uses ``FragmentAttackGenerator`` only for NER and instruction formatting; orchestration
lives here (not in ``attack/``). CLI: ``python -m ner_mask.fragment_mask`` from ``webshop/``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .mask_check import MaskChecker


def generate_live_attack_plan(
    generator: Any,
    host_instruction: str,
    target_instruction: str,
    rule_checker=None,
    profile=None,
    query: str = "",
    attack_log_file: Optional[str] = None,
) -> Dict:
    """
    NER-decompose ``target_instruction``, optionally run mask pass, build fragment_attacks + trigger.

    Args:
        generator: ``FragmentAttackGenerator`` instance (``attack.attack``).
        host_instruction: Benign host text.
        target_instruction: Text to decompose / inject.
        rule_checker: If given with ``profile``, runs ``MaskChecker``.
        profile: User profile for mask checks.
        query: Optional shopping query for checks.
        attack_log_file: Append mask details if mask run and path set.

    Returns:
        Plan dict with fragments, fragment_attacks, trigger_instruction, and optional mask fields.
    """
    fragments = generator.decompose_target_instruction_ner(target_instruction)

    sensitive_fragments = None
    safe_fragments = None
    mask_detection_log = None
    safe_instruction_check = None

    if rule_checker is not None and profile is not None:
        if generator.verbose:
            print(f"\n{'='*60}")
            print("PERFORMING MASK CHECK")
            print(f"{'='*60}\n")

        mask_checker = MaskChecker(
            rule_checker=rule_checker,
            profile=profile,
            host_instruction=host_instruction,
            query=query,
            verbose=generator.verbose,
        )

        sensitive_fragments, safe_fragments, mask_detection_log, safe_instruction_check = (
            mask_checker.get_sensitive_and_safe_fragments(fragments)
        )

        if generator.verbose:
            print(f"\nMask Check Results:")
            print(f"  Sensitive fragments: {sensitive_fragments}")
            print(f"  Safe fragments: {safe_fragments}\n")

        if attack_log_file:
            with open(attack_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"MASK DETECTION DETAILS\n")
                f.write(f"{'='*80}\n")
                f.write(
                    f"Profile Used: {profile.profile_id} (credit_score={profile.credit_score}, "
                    f"account_age_days={profile.account_age_days})\n"
                )
                f.write(f"Host Instruction: {host_instruction}\n")
                f.write(f"Target Instruction: {target_instruction}\n")
                f.write(f"Query: {query}\n")
                f.write(f"\nFragments:\n")
                for idx, fragment in enumerate(fragments, 1):
                    f.write(f"  F{idx}: {fragment}\n")

                f.write(f"\nMASK Detection Process:\n")
                f.write(f"{'-'*80}\n")
                for log_entry in mask_detection_log:
                    f.write(f"\nTest Type: {log_entry['test_type']}\n")
                    if log_entry["fragment_label"]:
                        f.write(f"Fragment Label: {log_entry['fragment_label']}\n")
                        f.write(f"Fragment Text: {log_entry['fragment_text']}\n")
                    f.write(f"Instruction Tested: {log_entry['instruction']}\n")
                    f.write(f"Is Valid: {log_entry['is_valid']}\n")
                    f.write(f"Violated Rules: {log_entry['violated_rules']}\n")
                    f.write(f"\nRULECHECKER Prompt:\n")
                    f.write(f"{log_entry.get('rule_checker_prompt', 'N/A')}\n")
                    f.write(f"\nRULECHECKER Response:\n")
                    f.write(f"{log_entry.get('rule_checker_response', 'N/A')}\n")
                    f.write(f"{'-'*80}\n")

                if safe_instruction_check:
                    f.write(f"\nSafe Fragments Only (Partial Instruction) RULECHECKER Check:\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"Safe Fragments: {safe_instruction_check['safe_fragments']}\n")
                    f.write(f"Instruction: {safe_instruction_check['instruction']}\n")
                    f.write(f"Is Valid: {safe_instruction_check['is_valid']}\n")
                    f.write(f"Violated Rules: {safe_instruction_check['violated_rules']}\n")
                    f.write(f"\nRULECHECKER Prompt:\n")
                    f.write(f"{safe_instruction_check.get('rule_checker_prompt', 'N/A')}\n")
                    f.write(f"\nRULECHECKER Response:\n")
                    f.write(f"{safe_instruction_check.get('rule_checker_response', 'N/A')}\n")
                    f.write(f"{'-'*80}\n")

                f.write(f"\nMASK Detection Summary:\n")
                f.write(f"  Sensitive Fragments: {sensitive_fragments}\n")
                f.write(f"  Safe Fragments: {safe_fragments}\n")
                f.write(f"{'='*80}\n\n")

    fragment_attacks = generator.generate_fragment_attack_instructions(
        host_instruction, fragments, sensitive_fragments=sensitive_fragments
    )

    fragment_labels = [fa["label"] for fa in fragment_attacks]
    trigger_instruction = generator.generate_trigger_attack_instruction(
        host_instruction,
        fragment_labels,
        fragments=fragments,
        sensitive_fragments=sensitive_fragments,
        safe_fragments=safe_fragments,
    )

    result = {
        "host_instruction": host_instruction,
        "target_instruction": target_instruction,
        "fragments": fragments,
        "fragment_attacks": fragment_attacks,
        "trigger_instruction": trigger_instruction,
    }

    if sensitive_fragments is not None:
        result["sensitive_fragments"] = sensitive_fragments
    if safe_fragments is not None:
        result["safe_fragments"] = safe_fragments
    if mask_detection_log is not None:
        result["mask_detection_log"] = mask_detection_log
    if safe_instruction_check is not None:
        result["safe_instruction_check"] = safe_instruction_check

    return result
