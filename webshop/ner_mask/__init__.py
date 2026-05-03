"""
Offline NER decomposition + RuleChecker mask (semantic sensitive/safe fragments for prompts).

``main.py`` in attack mode uses ``carrier_query`` (legacy: ``carrier_instruction_3``) / ``masked_instruction`` only;
any ``sensitive_fragments`` field in JSON is log-only there. For NER + mask-driven
``sensitive_fragments`` / ``safe_fragments`` and matching fragment/trigger text, use
``generate_live_attack_plan`` or ``python -m ner_mask.fragment_mask`` from ``webshop/``.

- ``MaskChecker``: fragment-wise mask checks.
- ``generate_live_attack_plan``: NER + optional mask + fragment/trigger strings.
"""

from .mask_check import MaskChecker
from .live_attack_plan import generate_live_attack_plan

__all__ = ["MaskChecker", "generate_live_attack_plan"]
