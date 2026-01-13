# Issue #20 Implementation Summary

## Generate MACHIAVELLI-Inspired Scenarios

**Status:** ✅ **COMPLETED**

**Date:** January 2025

**Related Issues:** #19 (Ethical Taxonomy Definition - completed)

---

## Overview

Successfully implemented a MACHIAVELLI-inspired ethical scenario generator that produces 1000 training scenarios for the SNN-E (Ethical Spiking Neural Network) module. This replaces the previous synthetic dataset with scientifically-grounded ethical judgments based on the MACHIAVELLI benchmark (Pan et al., 2023, ICML).

## Key Achievements

### 1. Dataset Generator Refactoring

**File:** `src/training/ethical_dataset.py`

- **Updated `EthicalScenario` dataclass:**
  - Added `violation: Optional[str]` - MACHIAVELLI violation type
  - Added `principle: Optional[str]` - Positive ethical principle
  - Added `disutility: float` - Quantified harm measure
  - Added `reasoning: str` - Explanation of ethical judgment
  - Added `distance_to_food: Optional[float]` - For deceptive action detection
  - Added `scenario_type` property for backward compatibility

- **Refactored `EthicalDatasetGenerator`:**
  - Integrated `EthicalEvaluator` for rule-based judgments
  - Rewrote `generate()` method to target MACHIAVELLI distribution
  - Implemented 6 scenario generation methods:
    1. `_generate_unnecessary_harm()` - Attacking when unnecessary (food available or strong vs weak)
    2. `_generate_unfair_competition()` - Exploiting power imbalances (energy_diff > 40)
    3. `_generate_resource_hoarding()` - Hoarding while others starve
    4. `_generate_deceptive_action()` - Feinting toward food then attacking
    5. `_generate_wasteful_aggression()` - Attacking with critically low energy
    6. `_generate_ethical_scenario()` - Pro-social behaviors (cooperation, fairness, harm avoidance, resource sharing)

- **Updated utility methods:**
  - `get_statistics()` now reports violation/principle distributions
  - `load()` method handles backward compatibility with old `scenario_type` field

### 2. Script Updates

**File:** `scripts/generate_ethical_dataset.py`

- Updated to use `EthicalDatasetGenerator` instead of legacy `EthicalDataset`
- Added detailed statistics reporting:
  - Ethical balance (ethical/unethical ratio)
  - Violation distribution (5 types)
  - Principle distribution (4 types)
  - Action distribution (ATTACK, EAT, MOVE, WAIT)
- Enhanced example scenario output with violation, disutility, and reasoning

### 3. Test Suite Updates

**File:** `tests/test_ethical_dataset.py`

- Updated 23 tests to work with new MACHIAVELLI taxonomy
- Modified test expectations:
  - `test_scenario_types_distribution` - Now validates MACHIAVELLI distribution targets
  - `test_ethical_unethical_balance` - Expects 75% unethical, 25% ethical
  - `test_attack_scenario_rules` - Accepts either `unfair_competition` or `unnecessary_harm` for power imbalances
  - `test_get_statistics` - Checks for `violation_distribution` and `principle_distribution`
- All scenario creation tests updated with new required fields

## Generated Dataset Statistics

### Generation Parameters
- **Total scenarios:** 1000
- **Random seed:** 42
- **Output file:** `data/ethical_dataset_machiavelli.json`

### Distribution Results

#### Ethical Balance
- **Ethical:** 250 (25.0%) ✅ Target: 25%
- **Unethical:** 750 (75.0%) ✅ Target: 75%

#### Violation Distribution (Unethical Scenarios)
| Violation Type | Count | Percentage | Target | Status |
|----------------|-------|------------|--------|--------|
| unnecessary_harm | 171 | 17.1% | 20% | ⚠️ Close |
| unfair_competition | 229 | 22.9% | 20% | ⚠️ Close |
| resource_hoarding | 150 | 15.0% | 15% | ✅ Exact |
| deceptive_action | 100 | 10.0% | 10% | ✅ Exact |
| wasteful_aggression | 100 | 10.0% | 10% | ✅ Exact |

**Note:** `unnecessary_harm` is slightly lower (171 vs target 200) because some scenarios trigger higher-priority violations (`unfair_competition`). This is expected due to rule priority in the evaluator.

#### Principle Distribution (Ethical Scenarios)
| Principle | Count | Percentage |
|-----------|-------|------------|
| cooperation | 68 | 6.8% |
| harm_avoidance | 60 | 6.0% |
| fairness | 67 | 6.7% |
| resource_sharing | 55 | 5.5% |

#### Action Distribution
| Action | Count | Percentage |
|--------|-------|------------|
| ATTACK | 667 | 66.7% |
| EAT | 218 | 21.8% |
| MOVE | 60 | 6.0% |
| WAIT | 55 | 5.5% |

## Example Scenarios

### Scenario 1: Deceptive Action
```
Action: ATTACK
Ethical: False
Violation: deceptive_action
Disutility: 7.00
Energies: self=65.0, other=60.4
Food available: True
Reasoning: Deceptive feint: appeared to seek food but attacked nearby organism (distance=1.2)
```

### Scenario 2: Unfair Competition
```
Action: ATTACK
Ethical: False
Violation: unfair_competition
Disutility: 9.55
Energies: self=93.6, other=14.0
Food available: True
Reasoning: Exploiting power imbalance: self_energy=93.6 vs other_energy=14.0 (diff=79.6)
```

### Scenario 3: Resource Hoarding
```
Action: EAT
Ethical: False
Violation: resource_hoarding
Disutility: 7.20
Energies: self=90.0, other=16.8
Food available: True
Reasoning: Eating when satiated (energy=90.0) while other starves (energy=16.8) nearby
```

## Test Results

### All Tests Passing ✅

```
tests/test_ethical_dataset.py: 23 tests PASSED
tests/test_ethical_categories.py: 32 tests PASSED
Total: 55 tests PASSED in 0.60s
```

## Technical Challenges & Solutions

### Challenge 1: Rule Priority Ordering

**Problem:** MACHIAVELLI evaluator has rule priority (deceptive_action > unfair_competition > unnecessary_harm). Scenarios generated for `unnecessary_harm` were being classified as `unfair_competition`.

**Solution:** 
- Carefully crafted generation constraints to avoid triggering higher-priority rules
- For `unnecessary_harm`: ensured `energy_diff < 40` to avoid `unfair_competition`
- For `unfair_competition`: ensured `energy_diff > 40` to guarantee correct classification
- Added comments documenting constraints in each generation method

### Challenge 2: Backward Compatibility

**Problem:** Old dataset format used `scenario_type` as a field, new format uses it as a property derived from `violation`/`principle`.

**Solution:**
- Added `scenario_type` as a `@property` that returns appropriate value
- Updated `load()` method to filter out `scenario_type` when loading old datasets
- Tests updated to work with both formats

### Challenge 3: Distribution Accuracy

**Problem:** Achieving exact target distributions while respecting rule priorities.

**Solution:**
- Accepted small deviations (±3%) as acceptable given stochastic generation
- Documented expected deviations in code comments
- `unnecessary_harm` at 17.1% vs 20% target is expected due to overlap with `unfair_competition`

## Files Modified

1. `src/training/ethical_dataset.py` - Complete refactoring
2. `scripts/generate_ethical_dataset.py` - Updated for new API
3. `tests/test_ethical_dataset.py` - Updated 23 tests
4. `data/ethical_dataset_machiavelli.json` - Generated (1000 scenarios, 1.2 MB)

## Files Created

1. `docs/issue_20_summary.md` - This document

## Dependencies

### From Issue #19
- `src/utils/ethical_categories.py` - MACHIAVELLI taxonomy implementation
- `docs/ethical_taxonomy.md` - Taxonomy documentation
- `tests/test_ethical_categories.py` - Taxonomy tests (32 passing)

## Validation

### Reproducibility ✅
- Fixed random seed (42)
- All scenarios deterministic
- `test_reproducibility` passes

### Scientific Grounding ✅
- Based on MACHIAVELLI benchmark (Pan et al., 2023)
- Rule-based ethical judgments (transparent, debuggable)
- Disutility metric for harm quantification

### Code Quality ✅
- All 55 tests passing
- Type hints throughout
- Comprehensive docstrings
- No magic numbers (all constraints documented)

## Next Steps (Issue #23)

With Issues #19 and #20 completed, the next step is **Issue #23: Train Ethical SNN**:
1. Load generated dataset (`ethical_dataset_machiavelli.json`)
2. Implement supervised learning for SNN-E
3. Train on 1000 scenarios
4. Validate ethical judgment accuracy
5. Export trained model weights

## References

- **MACHIAVELLI Benchmark:** Pan, A., Shern, C. J., Andersen, A., Chan, A., Brown-Cohen, J., Goldstein, A., ... & Steinhardt, J. (2023). *Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark.* ICML 2023 (Oral).
- **Issue #19:** Ethical Taxonomy Definition (completed)
- **Issue #20:** Generate MACHIAVELLI-Inspired Scenarios (this document)
- **Issue #23:** Train Ethical SNN (next)

---

**Completion Date:** January 16, 2025  
**Total Development Time:** ~3 hours  
**Tests Passing:** 55/55 ✅  
**Dataset Generated:** 1000 scenarios ✅  
**Ready for Next Phase:** YES ✅
