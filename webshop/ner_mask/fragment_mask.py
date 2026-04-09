#!/usr/bin/env python3
"""
Offline fragment plan: spaCy NER decomposition + optional RuleChecker mask pass.

Uses ``ner_mask.live_attack_plan.generate_live_attack_plan`` (not used by ``main.py``;
runtime attack uses minimal JSON fields only).

Run from the ``webshop`` directory::

  python -m ner_mask.fragment_mask --host "..." --target "..." --out plan.json
  python -m ner_mask.fragment_mask --host "..." --target "..." --mask --profile p.json --out plan.json
  python -m ner_mask.fragment_mask --cases cases.json --mask --profile p.json --out plans.json

(``python ner_mask/fragment_mask.py`` also works if cwd is ``webshop``.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_webshop_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _webshop_root not in sys.path:
    sys.path.insert(0, _webshop_root)

from attack import FragmentAttackGenerator
from ner_mask.live_attack_plan import generate_live_attack_plan
from rule_and_profile import RuleChecker, UserProfile


def _user_profile_from_dict(d: Dict[str, Any], fallback_id: str = "offline") -> UserProfile:
    if not d:
        raise ValueError("Profile dict is empty.")
    return UserProfile(
        profile_id=d.get("profile_id", fallback_id),
        age=d["age"],
        country=d["country"],
        is_verified=d["is_verified"],
        payment_method=d["payment_method"],
        failed_payment_attempts=d["failed_payment_attempts"],
        credit_score=d["credit_score"],
        account_age_days=d["account_age_days"],
        account_status=d["account_status"],
        return_rate=d["return_rate"],
        total_purchase_amount=d["total_purchase_amount"],
    )


def _load_profile_file(path: str) -> UserProfile:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _user_profile_from_dict(data, fallback_id=os.path.basename(path))


def _plan_to_jsonable(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy safe for json.dump (no transformation needed for current schema)."""
    return json.loads(json.dumps(plan, default=str))


def _run_one(
    generator: FragmentAttackGenerator,
    host: str,
    target: str,
    query: str,
    rule_checker: Optional[RuleChecker],
    profile: Optional[UserProfile],
    log_path: Optional[str],
) -> Dict[str, Any]:
    return generate_live_attack_plan(
        generator,
        host_instruction=host,
        target_instruction=target,
        rule_checker=rule_checker,
        profile=profile,
        query=query or "",
        attack_log_file=log_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NER fragment decomposition + optional mask (RuleChecker).")
    parser.add_argument("--host", type=str, default=None, help="Host (benign) instruction")
    parser.add_argument("--target", type=str, default=None, help="Target / malicious instruction to decompose")
    parser.add_argument("--query", type=str, default="", help="Shopping query for mask checks (optional)")
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="JSON file: list of objects with host_instruction, target_instruction, optional query, optional profile object",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Run RuleChecker mask pass (requires --profile or per-case profile in --cases)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="JSON file with UserProfile fields (used for all cases when --mask, or as fallback for batch)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name for RuleChecker when --mask",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Append mask / plan details to this log file (single-case only; batch ignores)",
    )
    parser.add_argument("--out", type=str, required=True, help="Write JSON result(s) here")
    parser.add_argument("--verbose", action="store_true", help="Verbose generator output")
    args = parser.parse_args()

    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases: List[Dict[str, Any]] = json.load(f)
        if not isinstance(cases, list):
            raise SystemExit("--cases file must contain a JSON array")

        default_profile: Optional[UserProfile] = None
        if args.mask:
            if args.profile:
                default_profile = _load_profile_file(args.profile)
            rule_checker = RuleChecker(verbose=args.verbose, model=args.model)
        else:
            rule_checker = None

        generator = FragmentAttackGenerator(verbose=args.verbose)
        results: List[Dict[str, Any]] = []
        for idx, case in enumerate(cases):
            host = case.get("host_instruction")
            target = case.get("target_instruction")
            if not host or not target:
                raise SystemExit(f"Case {idx}: requires host_instruction and target_instruction")
            query = case.get("query") or ""
            prof: Optional[UserProfile] = None
            rc: Optional[RuleChecker] = None
            if args.mask:
                rc = rule_checker
                if "profile" in case and case["profile"]:
                    prof = _user_profile_from_dict(case["profile"], fallback_id=str(idx))
                elif default_profile is not None:
                    prof = default_profile
                else:
                    raise SystemExit(f"Case {idx}: --mask requires a profile in the case or --profile file")
            plan = _run_one(generator, host, target, query, rc, prof, log_path=None)
            results.append(_plan_to_jsonable(plan))

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(results)} plan(s) to {args.out}")
        return

    if not args.host or not args.target:
        raise SystemExit("Single-mode requires --host and --target (or use --cases)")

    rule_checker: Optional[RuleChecker] = None
    profile: Optional[UserProfile] = None
    if args.mask:
        if not args.profile:
            raise SystemExit("--mask requires --profile for single-case mode")
        rule_checker = RuleChecker(verbose=args.verbose, model=args.model)
        profile = _load_profile_file(args.profile)

    generator = FragmentAttackGenerator(verbose=args.verbose)
    plan = _run_one(
        generator,
        args.host,
        args.target,
        args.query,
        rule_checker,
        profile,
        args.log,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_plan_to_jsonable(plan), f, indent=2, ensure_ascii=False)
    print(f"Wrote plan to {args.out}")
    if args.log:
        print(f"Mask / detail log appended to {args.log}")


if __name__ == "__main__":
    main()
