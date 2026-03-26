# Evaluation Integrity and Bias Analysis

## Overview

This document describes the evaluation bias issues identified in the BitoGuard validation pipeline and the fixes implemented to ensure honest (unbiased) performance estimates.

## Identified Bias Sources

### 1. In-Sample Calibration and Threshold Selection

**Problem**: The original `validate_official_model()` function performed calibration method selection and threshold search on ALL primary OOF predictions, then computed metrics on the same data. This introduces in-sample selection bias.

**Impact**: Reported metrics are optimistically biased because the selection process "sees" the validation data.

### 2. Secondary Validation Blend Weight Re-tuning

**Problem**: Secondary validation re-tunes blend weights on its own OOF data via `tune_blend_weights()` instead of using primary's blend weights. This partially defeats the purpose of secondary as an independent check.

**Location**: `validate.py` line 128, `use_blend=True` parameter

**Impact**: Secondary metrics contain some in-sample bias from blend weight selection.

### 3. No True Shadow Holdout

**Problem**: `splitters.py` `reserve_shadow_groups()` is a no-op placeholder - all groups are set to `core_trainable`, so there is no true shadow holdout.

**Impact**: No completely untouched holdout set for final validation.

## Implemented Fixes

### Phase 1: Documentation (Completed)

1. Added code comment in `validate.py` documenting the secondary blend weight re-tuning issue
2. Created `PRODUCTION_CONFIG.md` documenting the actual system architecture
3. Created this `EVALUATION_INTEGRITY.md` document

### Phase 2: Honest Evaluation Pipeline (Completed)

Created `inner_fold_selection.py` module that implements honest per-fold selection:

1. For each CV fold k:
   - Train data: all folds except k
   - Valid data: fold k only
   
2. Selection steps (all on train data only):
   - Tune blend weights via `tune_blend_weights()`
   - Build BlendEnsemble stacker
   - Fit isotonic calibrator
   - Search optimal threshold
   
3. Application (to valid data):
   - Apply blend model → stacker_raw_probability
   - Apply calibrator → submission_probability
   - Use selected threshold for predictions

4. Aggregate all validation folds → honest OOF metrics

### New Validation Functions

- `validate_official_model_honest()`: Unbiased evaluation using inner-fold selection
- `validate_official_model_legacy()`: Original biased evaluation (for comparison)
- `validate_official_model()`: Alias for legacy (backward compatibility)

## Usage

Run honest validation (recommended):
```bash
cd bitoguard_core
PYTHONPATH=. python -m official.validate
```

Run legacy validation (for comparison):
```bash
cd bitoguard_core
PYTHONPATH=. python -m official.validate --legacy
```

## Expected Bias Magnitude

Based on secondary validation (F1=0.4314) vs primary (F1 will vary), we expect:

- Honest F1 to be lower than legacy F1 by approximately 0.01-0.02
- The bias comes from in-sample selection of calibration and threshold
- Honest metrics are the true out-of-sample performance estimate

## Recommendations

1. Use `validate_official_model_honest()` as the primary evaluation metric
2. Report honest metrics in production dashboards and model cards
3. Keep legacy metrics for historical comparison only
4. Consider implementing true shadow holdout in future iterations

## Backward Compatibility

- Legacy validation function preserved as `validate_official_model_legacy()`
- Bundle structure unchanged
- Submission predictions unchanged (only evaluation metrics affected)
- Existing scripts can continue using `validate_official_model()` (maps to legacy)

## Future Work

1. Implement true shadow holdout groups in `splitters.py`
2. Make secondary validation use primary's blend weights (no re-tuning)
3. Add per-fold model persistence for reproducibility
4. Extend honest evaluation to secondary validation
