"""
Mask Check Module
Identifies sensitive words in target instruction by masking fragments and checking with rule_checker.
"""

from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import sys
import os

# Type checking only imports
if TYPE_CHECKING:
    from rule_and_profile import RuleChecker, UserProfile
else:
    # Runtime import - will be imported when needed
    RuleChecker = None
    UserProfile = None


class MaskChecker:
    """Checks which fragments contain sensitive words by masking and testing with rule_checker"""
    
    def __init__(self, rule_checker, profile, 
                 host_instruction: str, query: str = "", verbose: bool = False):
        """
        Initialize MaskChecker.
        
        Args:
            rule_checker: RuleChecker instance to check for violations
            profile: UserProfile to use for rule checking
            host_instruction: Original host instruction
            query: Query string (optional, defaults to empty)
            verbose: Whether to print debug information
        """
        self.rule_checker = rule_checker
        self.profile = profile
        self.host_instruction = host_instruction
        self.query = query
        self.verbose = verbose
        self.mask_token = "[MASK]"
    
    def _create_masked_instruction(self, fragments: List[str], mask_index: int) -> str:
        """
        Create a masked instruction by replacing one fragment with MASK token.
        
        Args:
            fragments: List of fragments [A, B, C, D]
            mask_index: Index of fragment to mask (0-3)
        
        Returns:
            Masked instruction string
        """
        masked_fragments = fragments.copy()
        if 0 <= mask_index < len(masked_fragments):
            masked_fragments[mask_index] = self.mask_token
        
        # Filter out empty fragments and join
        filtered_fragments = [f for f in masked_fragments if f.strip()]
        masked_instruction = ' '.join(filtered_fragments)
        
        return masked_instruction
    
    def check_sensitive_fragments(self, fragments: List[str]) -> Tuple[Dict[str, bool], List[Dict]]:
        """
        Check which fragments contain sensitive words by masking each fragment.
        
        Logic:
        - If masking a fragment results in NO violation, that fragment contains sensitive words
        - If masking a fragment still results in violation, that fragment does NOT contain sensitive words
        
        Args:
            fragments: List of fragments [A, B, C, D] from target instruction
        
        Returns:
            Tuple of (sensitive_map, detection_log):
            - sensitive_map: Dictionary mapping fragment labels to boolean:
              - True: fragment contains sensitive words
              - False: fragment does NOT contain sensitive words
            - detection_log: List of dictionaries containing detailed detection records
        """
        labels = ['A', 'B', 'C', 'D']
        sensitive_map = {}
        detection_log = []
        
        # First, check the full target instruction (all fragments combined)
        full_instruction = ' '.join([f for f in fragments if f.strip()])
        result_full = self.rule_checker.check_all_rules(
            self.profile, full_instruction, self.query, return_details=True
        )
        is_valid_full, violated_rules_full, details_full = result_full
        
        detection_log.append({
            'test_type': 'full_instruction',
            'instruction': full_instruction,
            'is_valid': is_valid_full,
            'violated_rules': violated_rules_full,
            'fragment_label': None,
            'fragment_text': None,
            'rule_checker_prompt': details_full['prompt'],
            'rule_checker_response': details_full['response']
        })
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("MASK CHECK: Testing full instruction")
            print(f"Instruction: {full_instruction}")
            print(f"Valid: {is_valid_full}, Violated Rules: {violated_rules_full}")
            print(f"Response: {details_full.get('response', 'N/A')}")
            print(f"{'='*60}\n")
        
        # If full instruction is valid (no violation), no fragments are sensitive
        # But we still need to check each fragment to be thorough
        # Only skip if response explicitly says "VALID" (not just empty violated_rules)
        response_upper = details_full.get('response', '').upper().strip()
        if is_valid_full and (response_upper == "VALID" or not response_upper.startswith("VIOLATED")):
            if self.verbose:
                print("Full instruction is valid - no sensitive words detected")
            return {label: False for label in labels[:len(fragments)]}, detection_log
        
        # Test each fragment by masking it
        for idx, fragment in enumerate(fragments):
            if idx >= len(labels):
                break
            
            label = labels[idx]
            
            # Skip empty fragments
            if not fragment.strip():
                sensitive_map[label] = False
                continue
            
            # Create masked instruction
            masked_instruction = self._create_masked_instruction(fragments, idx)
            
            # Check with rule_checker (with details)
            result_masked = self.rule_checker.check_all_rules(
                self.profile, masked_instruction, self.query, return_details=True
            )
            is_valid_masked, violated_rules_masked, details_masked = result_masked
            
            detection_log.append({
                'test_type': 'masked_fragment',
                'instruction': masked_instruction,
                'is_valid': is_valid_masked,
                'violated_rules': violated_rules_masked,
                'fragment_label': label,
                'fragment_text': fragment,
                'rule_checker_prompt': details_masked['prompt'],
                'rule_checker_response': details_masked['response']
            })
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"MASK CHECK: Testing Fragment {label}")
                print(f"Fragment: {fragment}")
                print(f"Masked Instruction: {masked_instruction}")
                print(f"Valid: {is_valid_masked}, Violated Rules: {violated_rules_masked}")
                print(f"Response: {details_masked.get('response', 'N/A')}")
                print(f"{'='*60}\n")
            
            # If masking this fragment makes instruction valid (no violation),
            # then this fragment contains sensitive words
            # If masking this fragment still has violations, this fragment does NOT contain sensitive words
            sensitive_map[label] = is_valid_masked
        
        return sensitive_map, detection_log
    
    def get_sensitive_and_safe_fragments(self, fragments: List[str]) -> Tuple[List[str], List[str], List[Dict], Optional[Dict]]:
        """
        Get lists of sensitive and safe fragments, along with detection logs.
        
        Args:
            fragments: List of fragments [A, B, C, D]
        
        Returns:
            Tuple of (sensitive_fragments, safe_fragments, detection_log, safe_instruction_check)
            - sensitive_fragments: List of fragment texts that contain sensitive words
            - safe_fragments: List of fragment texts that do NOT contain sensitive words
            - detection_log: List of dictionaries containing detailed MASK detection records
            - safe_instruction_check: Dictionary containing RULECHECKER check result for safe fragments only (None if no safe fragments)
        """
        labels = ['A', 'B', 'C', 'D']
        sensitive_map, detection_log = self.check_sensitive_fragments(fragments)
        
        sensitive_fragments = []
        safe_fragments = []
        
        for idx, fragment in enumerate(fragments):
            if idx >= len(labels):
                break
            
            label = labels[idx]
            if not fragment.strip():
                continue
            
            if sensitive_map.get(label, False):
                sensitive_fragments.append(fragment)
            else:
                safe_fragments.append(fragment)
        
        # Check partial instruction (only safe fragments) with RULECHECKER
        safe_instruction_check = None
        if safe_fragments:
            safe_instruction = ' '.join(safe_fragments)
            result_safe = self.rule_checker.check_all_rules(
                self.profile, safe_instruction, self.query, return_details=True
            )
            is_valid_safe, violated_rules_safe, details_safe = result_safe
            
            safe_instruction_check = {
                'test_type': 'safe_fragments_only',
                'instruction': safe_instruction,
                'is_valid': is_valid_safe,
                'violated_rules': violated_rules_safe,
                'safe_fragments': safe_fragments,
                'rule_checker_prompt': details_safe['prompt'],
                'rule_checker_response': details_safe['response']
            }
            
            if self.verbose:
                print(f"\n{'='*60}")
                print("MASK CHECK: Testing safe fragments only (partial instruction)")
                print(f"Safe Fragments: {safe_fragments}")
                print(f"Instruction: {safe_instruction}")
                print(f"Valid: {is_valid_safe}, Violated Rules: {violated_rules_safe}")
                print(f"{'='*60}\n")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("MASK CHECK RESULTS:")
            print(f"Sensitive fragments: {sensitive_fragments}")
            print(f"Safe fragments: {safe_fragments}")
            print(f"{'='*60}\n")
        
        return sensitive_fragments, safe_fragments, detection_log, safe_instruction_check

