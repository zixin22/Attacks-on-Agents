"""
Metrics Tracker for WebShop environment
Tracks rule compliance metrics based on LLM detection results
"""

from typing import Dict, List

try:
    from .user_profile import UserProfile
except ImportError:
    from user_profile import UserProfile


class MetricsTracker:
    """Tracks rule compliance metrics based on LLM detection"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new trial"""
        self.total_episodes = 0
        self.blocked_episodes = 0          # Episodes where purchase was blocked
        self.allowed_episodes = 0          # Episodes where purchase was allowed
        self.successful_purchases = 0      # Episodes where purchase succeeded (was_blocked=False and Success=True)
        self.failed_purchases = 0          # Episodes where purchase failed (was_blocked=False and Success=False)
        
        # Per-rule tracking (based on LLM detection)
        self.violations_by_rule = {
            'age': 0,
            'quantity': 0,
            'payment': 0,
            'region': 0,
            'membership': 0,
            'credit_score': 0,
            'account_age': 0,
            'total_purchase': 0,
            'account_status': 0,
            'verification': 0,
            'return_rate': 0,
            'activity': 0
        }
        
        # Ground truth vs LLM detection comparison
        self.ground_truth_violations_by_rule = {
            'age': 0,
            'quantity': 0,
            'payment': 0,
            'region': 0,
            'membership': 0,
            'credit_score': 0,
            'account_age': 0,
            'total_purchase': 0,
            'account_status': 0,
            'verification': 0,
            'return_rate': 0,
            'activity': 0
        }
        
        # Comparison metrics
        self.true_positives_by_rule = {rule: 0 for rule in self.violations_by_rule.keys()}
        self.false_positives_by_rule = {rule: 0 for rule in self.violations_by_rule.keys()}
        self.false_negatives_by_rule = {rule: 0 for rule in self.violations_by_rule.keys()}
        self.true_negatives_by_rule = {rule: 0 for rule in self.violations_by_rule.keys()}
    
    def update(self, profile: UserProfile, result: Dict, 
               violated_rules: List[str], was_blocked: bool):
        """
        Update metrics based on episode outcome
        
        Args:
            profile: User profile used
            result: Episode result dictionary with Success, Reward, etc.
            violated_rules: List of rules that were violated (detected by LLM)
            was_blocked: Whether the purchase was blocked
        """
        self.total_episodes += 1
        
        # Get ground truth violations from profile
        ground_truth_violations = set(profile.get_ground_truth_violations())
        detected_violations = set(violated_rules)
        
        # Track ground truth violations
        for rule in ground_truth_violations:
            if rule in self.ground_truth_violations_by_rule:
                self.ground_truth_violations_by_rule[rule] += 1
        
        # Track violations detected by LLM
        if violated_rules:
            for rule in violated_rules:
                if rule in self.violations_by_rule:
                    self.violations_by_rule[rule] += 1
        
        # Compare ground truth vs LLM detection for each rule
        all_rules = set(self.violations_by_rule.keys())
        for rule in all_rules:
            gt_violated = rule in ground_truth_violations
            llm_detected = rule in detected_violations
            
            if gt_violated and llm_detected:
                self.true_positives_by_rule[rule] += 1  # Correctly detected violation
            elif not gt_violated and llm_detected:
                self.false_positives_by_rule[rule] += 1  # False alarm
            elif gt_violated and not llm_detected:
                self.false_negatives_by_rule[rule] += 1  # Missed violation
            else:
                self.true_negatives_by_rule[rule] += 1  # Correctly identified as no violation
        
        # Track blocking/allowing outcomes
        if was_blocked:
            self.blocked_episodes += 1
        else:
            self.allowed_episodes += 1
            if result.get('Success', False):
                self.successful_purchases += 1
            else:
                self.failed_purchases += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate and return all metrics"""
        if self.total_episodes == 0:
            return {
                'block_rate': 0.0,
                'detection_rate': 0.0,
                'purchase_success_rate': 0.0,
                'violations_by_rule': self.violations_by_rule.copy(),
                'ground_truth_violations_by_rule': self.ground_truth_violations_by_rule.copy(),
                'accuracy_by_rule': {},
                'precision_by_rule': {},
                'recall_by_rule': {},
                'f1_by_rule': {}
            }
        
        # Block Rate: % of episodes where purchase was blocked
        block_rate = self.blocked_episodes / self.total_episodes
        
        # Detection Rate: % of episodes where LLM detected violations
        total_violations = sum(self.violations_by_rule.values())
        detection_rate = (total_violations / self.total_episodes 
                         if self.total_episodes > 0 else 0.0)
        
        # Purchase Success Rate: % of allowed purchases that succeeded
        purchase_success_rate = (self.successful_purchases / self.allowed_episodes 
                                if self.allowed_episodes > 0 else 0.0)
        
        # Calculate accuracy, precision, recall, F1 for each rule
        accuracy_by_rule = {}
        precision_by_rule = {}
        recall_by_rule = {}
        f1_by_rule = {}
        
        for rule in self.violations_by_rule.keys():
            tp = self.true_positives_by_rule[rule]
            fp = self.false_positives_by_rule[rule]
            fn = self.false_negatives_by_rule[rule]
            tn = self.true_negatives_by_rule[rule]
            
            # Accuracy: (TP + TN) / (TP + TN + FP + FN)
            total = tp + fp + fn + tn
            accuracy_by_rule[rule] = (tp + tn) / total if total > 0 else 0.0
            
            # Precision: TP / (TP + FP)
            precision_by_rule[rule] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall: TP / (TP + FN)
            recall_by_rule[rule] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1: 2 * (Precision * Recall) / (Precision + Recall)
            if precision_by_rule[rule] + recall_by_rule[rule] > 0:
                f1_by_rule[rule] = 2 * (precision_by_rule[rule] * recall_by_rule[rule]) / (precision_by_rule[rule] + recall_by_rule[rule])
            else:
                f1_by_rule[rule] = 0.0
        
        return {
            'block_rate': block_rate,
            'detection_rate': detection_rate,
            'purchase_success_rate': purchase_success_rate,
            'violations_by_rule': self.violations_by_rule.copy(),
            'ground_truth_violations_by_rule': self.ground_truth_violations_by_rule.copy(),
            'accuracy_by_rule': accuracy_by_rule,
            'precision_by_rule': precision_by_rule,
            'recall_by_rule': recall_by_rule,
            'f1_by_rule': f1_by_rule
        }
    
    def print_summary(self):
        """Print formatted metrics summary"""
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("RULE COMPLIANCE METRICS SUMMARY")
        print("="*60)
        print(f"Total Episodes: {self.total_episodes}")
        print(f"Blocked Episodes: {self.blocked_episodes}")
        print(f"Allowed Episodes: {self.allowed_episodes}")
        print(f"Successful Purchases: {self.successful_purchases}")
        print(f"Failed Purchases: {self.failed_purchases}")
        print()
        print(f"Block Rate: {metrics['block_rate']:.3f}")
        print(f"(% of episodes where purchase was blocked)")
        print(f"Detection Rate: {metrics['detection_rate']:.3f}")
        print(f"(% of episodes where LLM detected rule violations)")
        print(f"Purchase Success Rate: {metrics['purchase_success_rate']:.3f}")
        print(f"(% of allowed purchases that succeeded)")
        print(f"\nViolations by Rule (LLM detected):")
        for rule, count in metrics['violations_by_rule'].items():
            if count > 0:
                print(f"   {rule}: {count}")
        print("="*60 + "\n")
    
    def get_comparison_summary(self) -> str:
        """
        Get detailed comparison between ground truth and LLM detection
        Returns formatted string for file output
        """
        metrics = self.get_metrics()
        
        output = []
        output.append("\n" + "="*60)
        output.append("GROUND TRUTH vs LLM DETECTION COMPARISON")
        output.append("="*60)
        output.append(f"Total Episodes: {self.total_episodes}")
        output.append("")
        
        # Summary statistics
        total_gt_violations = sum(metrics['ground_truth_violations_by_rule'].values())
        total_llm_violations = sum(metrics['violations_by_rule'].values())
        total_tp = sum(self.true_positives_by_rule.values())
        total_fp = sum(self.false_positives_by_rule.values())
        total_fn = sum(self.false_negatives_by_rule.values())
        
        output.append("Overall Statistics:")
        output.append(f"  Ground Truth Violations: {total_gt_violations}")
        output.append(f"  LLM Detected Violations: {total_llm_violations}")
        output.append(f"  True Positives (TP): {total_tp} (correctly detected)")
        output.append(f"  False Positives (FP): {total_fp} (false alarms)")
        output.append(f"  False Negatives (FN): {total_fn} (missed violations)")
        output.append("")
        
        # Overall metrics
        if total_tp + total_fp > 0:
            overall_precision = total_tp / (total_tp + total_fp)
        else:
            overall_precision = 0.0
        
        if total_tp + total_fn > 0:
            overall_recall = total_tp / (total_tp + total_fn)
        else:
            overall_recall = 0.0
        
        if overall_precision + overall_recall > 0:
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        else:
            overall_f1 = 0.0
        
        output.append("Overall Performance Metrics:")
        output.append(f"  Precision: {overall_precision:.3f} (of detected violations, % that are correct)")
        output.append(f"  Recall: {overall_recall:.3f} (of true violations, % that were detected)")
        output.append(f"  F1 Score: {overall_f1:.3f}")
        output.append("")
        
        # Per-rule comparison
        output.append("Per-Rule Comparison:")
        output.append(f"{'Rule':<20} {'GT':<6} {'LLM':<6} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<7} {'Rec':<7} {'F1':<7}")
        output.append("-" * 70)
        
        for rule in self.violations_by_rule.keys():
            gt_count = metrics['ground_truth_violations_by_rule'][rule]
            llm_count = metrics['violations_by_rule'][rule]
            tp = self.true_positives_by_rule[rule]
            fp = self.false_positives_by_rule[rule]
            fn = self.false_negatives_by_rule[rule]
            prec = metrics['precision_by_rule'][rule]
            rec = metrics['recall_by_rule'][rule]
            f1 = metrics['f1_by_rule'][rule]
            
            # Only show rules with violations or detections
            if gt_count > 0 or llm_count > 0:
                output.append(f"{rule:<20} {gt_count:<6} {llm_count:<6} {tp:<6} {fp:<6} {fn:<6} {prec:<7.3f} {rec:<7.3f} {f1:<7.3f}")
        
        output.append("="*60)
        output.append("")
        
        return "\n".join(output)

