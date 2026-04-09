#!/usr/bin/env python3
"""
Shim for the CLI in ``ner_mask/fragment_mask.py``.

Prefer: ``python -m ner_mask.fragment_mask`` from the ``webshop`` directory.
"""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

if __name__ == "__main__":
    from ner_mask.fragment_mask import main
    main()
