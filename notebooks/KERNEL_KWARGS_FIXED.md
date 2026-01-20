# Kernel Configuration Fix Summary

## Issues Found

The interpolation notebooks had **incorrect kernel_kwargs** that were being silently ignored:

### 1. SM Kernel (Spectral Mixture)
**Problem:**
- Used `'n_mixtures'` but `create_kernel()` expects `'num_mixtures'`
- Passed `'frequencies'`, `'lengthscales'`, `'weights'` which **don't exist as init parameters**
  - These are learnable parameters initialized internally, not passed as kwargs

**Before (WRONG):**
```python
'kernel_kwargs': {
    'n_mixtures': 2,                    # Wrong name!
    'frequencies': [1/360, 1/40],       # Not an init param
    'lengthscales': [30.0, 30.0],       # Not an init param
    'weights': [0.5, 0.5],              # Not an init param
}
```

**After (FIXED):**
```python
'kernel_kwargs': {
    'num_mixtures': 2,  # Correct param name
    # Note: frequencies, lengthscales, weights are learned (not init params)
}
```

### 2. RBF Kernel
**Problem:**
- Used `lengthscale=30.0` but should be `40.0` for better interpolation performance
- 30° lengthscale is too short for smooth interpolation between 40° apart training views

**Before:**
```python
'kernel_kwargs': {
    'lengthscale': 30.0,  # Too short for 40° train view spacing
    'variance': 1.0,
}
```

**After (FIXED):**
```python
'kernel_kwargs': {
    'lengthscale': 40.0,  # Matches train view spacing for smooth interpolation
    'variance': 1.0,
}
```

## Root Cause

The `SMCircleKernel.__init__()` signature is:
```python
def __init__(self, num_mixtures=2):
    # frequencies, lengthscales, weights are nn.Parameter (learned)
    # NOT init arguments!
```

But notebooks were passing unused kwargs that were silently ignored by Python's `**kwargs`.

## Verification

From `kernels.py` `create_kernel()` function:

```python
elif kernel_type == 'sm_circle':
    num_mixtures = kwargs.get('num_mixtures', 2)  # ✅ Correct name
    return SMCircleKernel(num_mixtures=num_mixtures)

elif kernel_type == 'rbf_circle':
    lengthscale = kwargs.get('lengthscale', 30.0)
    variance = kwargs.get('variance', 1.0)
    return RBFCircleKernel(lengthscale_init=lengthscale, variance_init=variance)
```

## Impact

### SM Kernel
- **Before fix**: Used default `num_mixtures=2` (because `'n_mixtures'` was ignored)
- **After fix**: Still uses `num_mixtures=2` but now **explicitly configured**
- **No behavioral change** (accidentally correct), but now intentional

### RBF Kernel  
- **Before fix**: `lengthscale=30°` → Too sharp for interpolation (train views 40° apart)
- **After fix**: `lengthscale=40°` → Matches train spacing for smooth interpolation
- **Better interpolation expected**

## Files Fixed

1. ✅ `notebooks/train_gppvae_coil100_interpolation_sm.ipynb`
   - Cell 15: Fixed kernel_kwargs
   - Cell 21: Fixed print statement

2. ✅ `notebooks/train_gppvae_coil100_interpolation_rbf.ipynb`
   - Cell 15: Changed lengthscale from 30.0 to 40.0

## All Kernel Configs (Corrected)

### Periodic Kernel
```python
'kernel_kwargs': {
    'period': 360.0,
    'lengthscale': 1.0,  # Already correct (fixed from 30.0 earlier)
    'variance': 1.0,
}
```

### Spectral Mixture Kernel
```python
'kernel_kwargs': {
    'num_mixtures': 2,  # FIXED: was 'n_mixtures'
}
```

### FullRank Kernel
```python
'kernel_kwargs': {},  # No hyperparameters (learns free covariance)
```

### RBF Circle Kernel
```python
'kernel_kwargs': {
    'lengthscale': 40.0,  # FIXED: was 30.0
    'variance': 1.0,
}
```
