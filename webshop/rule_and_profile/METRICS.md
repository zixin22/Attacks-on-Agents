# Rule Compliance Metrics Documentation

This document describes the metrics tracked by the `MetricsTracker` class for evaluating rule compliance in the WebShop environment.

## Overview

The `MetricsTracker` tracks rule compliance metrics based on LLM detection results. Since all rule checking is performed by LLM (no hardcoded logic), these metrics reflect the actual behavior of the rule checking system rather than comparing against ground truth.

## Core Metrics

### 1. Block Rate

**Definition:** The percentage of episodes where purchases were blocked by the rule checker.

**Formula:**
```
Block Rate = (Blocked Episodes / Total Episodes) × 100%
```

**Calculation:**
```python
block_rate = blocked_episodes / total_episodes
```

**Interpretation:**
- **High Block Rate (> 0.3)**: Indicates strict rule enforcement. Many purchases are being blocked.
- **Low Block Rate (< 0.1)**: Indicates lenient rule enforcement. Most purchases are allowed.
- **Moderate Block Rate (0.1-0.3)**: Balanced enforcement.

**Example:**
- Total Episodes: 100
- Blocked Episodes: 30
- Block Rate = 30/100 = 0.300 (30%)

**Use Case:**
- Monitor the strictness of rule enforcement
- Identify if the system is too restrictive or too permissive
- Track changes in blocking behavior over time

---

### 2. Detection Rate

**Definition:** The percentage of episodes where the LLM detected at least one rule violation.

**Formula:**
```
Detection Rate = (Total Violations Detected / Total Episodes) × 100%
```

**Calculation:**
```python
total_violations = sum(violations_by_rule.values())
detection_rate = total_violations / total_episodes
```

**Important Notes:**
- A single episode can have multiple rule violations (e.g., both age and payment violations)
- Therefore, Detection Rate can exceed 1.0 (100%) if multiple violations are detected per episode on average
- This metric counts violations, not episodes with violations

**Interpretation:**
- **High Detection Rate (> 0.5)**: LLM frequently detects rule violations
- **Low Detection Rate (< 0.2)**: LLM rarely detects violations
- **Can exceed 1.0**: If average violations per episode > 1

**Example:**
- Total Episodes: 100
- Total Violations Detected: 45 (some episodes had multiple violations)
- Detection Rate = 45/100 = 0.450 (45%)

**Difference from Block Rate:**
- **Detection Rate**: Counts violations (can be multiple per episode)
- **Block Rate**: Counts episodes (one per episode)
- An episode can be blocked even if no violations were detected (edge case)
- An episode can have violations detected but not be blocked (shouldn't happen in normal operation)

**Use Case:**
- Measure how actively the LLM detects violations
- Identify which rules are most frequently violated
- Monitor LLM sensitivity to rule violations

---

### 3. Purchase Success Rate

**Definition:** The percentage of allowed purchases that successfully completed.

**Formula:**
```
Purchase Success Rate = (Successful Purchases / Allowed Episodes) × 100%
```

**Calculation:**
```python
purchase_success_rate = successful_purchases / allowed_episodes
```

**Important Notes:**
- Only considers episodes where purchases were **allowed** (not blocked)
- A purchase is considered successful if `result['Success'] == True`
- Failed purchases may occur due to other reasons (e.g., task completion failure, invalid actions)

**Interpretation:**
- **High Success Rate (> 0.8)**: Most allowed purchases complete successfully
- **Low Success Rate (< 0.5)**: Many allowed purchases fail for other reasons
- **1.0 (100%)**: All allowed purchases succeed

**Example:**
- Allowed Episodes: 70
- Successful Purchases: 60
- Failed Purchases: 10
- Purchase Success Rate = 60/70 = 0.857 (85.7%)

**Use Case:**
- Evaluate the quality of allowed purchases
- Identify if purchases are being blocked unnecessarily
- Monitor overall system performance for non-blocked purchases

---

## Additional Metrics

### Violations by Rule

**Definition:** Count of violations detected for each of the 12 rules.

**Rules Tracked:**
1. `age` - Age restriction violations
2. `quantity` - Quantity limit violations
3. `payment` - Payment method violations
4. `region` - Regional restriction violations
5. `membership` - Membership level violations
6. `credit_score` - Credit score violations
7. `account_age` - Account age violations
8. `total_purchase` - Total purchase amount violations
9. `account_status` - Account status violations
10. `verification` - Verification status violations
11. `return_rate` - Return rate violations
12. `activity` - Account activity violations

**Use Case:**
- Identify which rules are violated most frequently
- Focus rule refinement efforts on high-violation rules
- Understand the distribution of violations across rule types

---

## Relationship Between Metrics

```
Total Episodes (100)
├─ Blocked Episodes (30)
│  └─ Block Rate = 30%
│
└─ Allowed Episodes (70)
   ├─ Successful Purchases (60)
   │  └─ Purchase Success Rate = 60/70 = 85.7%
   └─ Failed Purchases (10)

Total Violations Detected (45)
└─ Detection Rate = 45/100 = 45%
   (Some episodes had multiple violations)
```

**Key Relationships:**
- `blocked_episodes + allowed_episodes = total_episodes`
- `successful_purchases + failed_purchases = allowed_episodes`
- `detection_rate` can be > `block_rate` if multiple violations per episode
- `detection_rate` can be < `block_rate` if violations are detected but not blocked (edge case)

---

## Usage Example

```python
from rule_and_profile.metrics import MetricsTracker

# Initialize tracker
tracker = MetricsTracker()

# Update metrics after each episode
tracker.update(
    profile=user_profile,
    result={'Success': True, 'Reward': 1.0},
    violated_rules=['age', 'payment'],  # LLM detected violations
    was_blocked=True  # Purchase was blocked
)

# Get metrics
metrics = tracker.get_metrics()
print(f"Block Rate: {metrics['block_rate']:.3f}")
print(f"Detection Rate: {metrics['detection_rate']:.3f}")
print(f"Purchase Success Rate: {metrics['purchase_success_rate']:.3f}")

# Print summary
tracker.print_summary()
```

---

## Limitations

Since all rule checking is performed by LLM without hardcoded logic:

1. **No Ground Truth**: We cannot determine if a block was "correct" or a false positive
2. **LLM-Dependent**: All metrics reflect LLM behavior, not absolute truth
3. **No Attack Success Rate**: Cannot calculate attack success rate without knowing which profiles "should" violate rules
4. **No False Positive Rate**: Cannot calculate false positive rate without ground truth

These metrics are designed to track **what happened** rather than **what should have happened**.

---

## Best Practices

1. **Monitor Block Rate**: Keep it within reasonable bounds (0.1-0.3) to avoid over-blocking
2. **Track Detection Rate**: High detection rate may indicate strict LLM or many violations
3. **Check Purchase Success Rate**: Low success rate for allowed purchases may indicate issues
4. **Review Violations by Rule**: Focus on rules with high violation counts
5. **Compare Over Time**: Track metrics across multiple runs to identify trends

---

## Version History

- **v2.0**: Simplified metrics for LLM-only rule checking (removed hardcoded logic dependencies)
- **v1.0**: Original metrics with hardcoded rule violation detection

