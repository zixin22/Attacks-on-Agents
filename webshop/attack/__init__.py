"""
Attack module for fragment-based instruction injection (dataset-driven helpers).

Live NER + mask: package ``webshop/ner_mask/`` (``python -m ner_mask.fragment_mask``).
"""

from .attack import FragmentAttackGenerator

__all__ = ['FragmentAttackGenerator']
