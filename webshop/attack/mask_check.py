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
    
    def check_sensitive_fragments(self, fragments: List[str]) -> Dict[str, bool]:
        """
        Check which fragments contain sensitive words by masking each fragment.
        
        Logic:
        - If masking a fragment results in NO violation, that fragment contains sensitive words
        - If masking a fragment still results in violation, that fragment does NOT contain sensitive words
        
        Args:
            fragments: List of fragments [A, B, C, D] from target instruction
        
        Returns:
            Dictionary mapping fragment labels to boolean:
            - True: fragment contains sensitive words
            - False: fragment does NOT contain sensitive words
        """
        labels = ['A', 'B', 'C', 'D']
        sensitive_map = {}
        
        # First, check the full target instruction (all fragments combined)
        full_instruction = ' '.join([f for f in fragments if f.strip()])
        is_valid_full, violated_rules_full = self.rule_checker.check_all_rules(
            self.profile, full_instruction, self.query
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("MASK CHECK: Testing full instruction")
            print(f"Instruction: {full_instruction}")
            print(f"Valid: {is_valid_full}, Violated Rules: {violated_rules_full}")
            print(f"{'='*60}\n")
        
        # If full instruction is valid (no violation), no fragments are sensitive
        if is_valid_full:
            if self.verbose:
                print("Full instruction is valid - no sensitive words detected")
            return {label: False for label in labels[:len(fragments)]}
        
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
            
            # Check with rule_checker
            is_valid_masked, violated_rules_masked = self.rule_checker.check_all_rules(
                self.profile, masked_instruction, self.query
            )
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"MASK CHECK: Testing Fragment {label}")
                print(f"Fragment: {fragment}")
                print(f"Masked Instruction: {masked_instruction}")
                print(f"Valid: {is_valid_masked}, Violated Rules: {violated_rules_masked}")
                print(f"{'='*60}\n")
            
            # If masking this fragment makes instruction valid (no violation),
            # then this fragment contains sensitive words
            # If masking this fragment still has violations, this fragment does NOT contain sensitive words
            sensitive_map[label] = is_valid_masked
        
        return sensitive_map
    
    def get_sensitive_and_safe_fragments(self, fragments: List[str]) -> Tuple[List[str], List[str]]:
        """
        Get lists of sensitive and safe fragments.
        
        Args:
            fragments: List of fragments [A, B, C, D]
        
        Returns:
            Tuple of (sensitive_fragments, safe_fragments)
            - sensitive_fragments: List of fragment texts that contain sensitive words
            - safe_fragments: List of fragment texts that do NOT contain sensitive words
        """
        labels = ['A', 'B', 'C', 'D']
        sensitive_map = self.check_sensitive_fragments(fragments)
        
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
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("MASK CHECK RESULTS:")
            print(f"Sensitive fragments: {sensitive_fragments}")
            print(f"Safe fragments: {safe_fragments}")
            print(f"{'='*60}\n")
        
        return sensitive_fragments, safe_fragments

