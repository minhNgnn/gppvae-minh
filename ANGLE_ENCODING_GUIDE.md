# Guide: Using RBF Kernel with Standardized Angles

## Overview

This guide explains how to train GP-VAE with **RBF (Gaussian) kernel** using **continuous, standardized angle values** instead of discrete indices. This provides the kernel with true geometric information about view relationships.

## What Changed?

### 1. **Kernel Implementation (`kernels.py`)**

The `RBFAngleKernel` and `MaternKernel` now support:
- **Discrete indices**: `[0, 1, 2, ..., 8]` → looks up angles in a buffer
- **Continuous angles**: `[-1.0, -0.67, -0.5, ...]` → uses angles directly

Key additions:
```python
def forward_continuous(self, angles_continuous):
    """
    Compute kernel matrix between continuous angles and reference angles.
    Enables interpolation for any continuous angle value.
    """
    # Compute k(w_i, θ_j) for all combinations
    # Returns [N, Q] matrix of kernel values
```

**Benefits**:
- RBF can see that `-0.5` (45°) is exactly halfway between `-0.67` (60°) and `-0.33` (30°)
- True distances preserved: `|60° - 45°| = 15°` vs `|60° - 30°| = 30°`
- No need to learn geometry from data - it's explicit!

### 2. **Variance Model (`vmod.py`)**

The `Vmodel.forward()` method now detects continuous vs discrete view inputs:

```python
def forward(self, d, w):
    # Handle objects (always discrete)
    X = F.embedding(d, self.x())
    
    # Handle views: continuous OR discrete
    if w.dtype == torch.float32:
        # Continuous angles: use kernel-based embeddings
        W = self._compute_view_embeddings_continuous(w)
    else:
        # Discrete indices: use standard embedding
        W = F.embedding(w, self.v())
```

**Key behavior**:
- Legacy kernel: always uses discrete indices
- RBF/Matérn kernels: can use continuous angles
- Periodic/VonMises: still use discrete indices (they already handle periodicity)

### 3. **Data Parser (`data_parser_interpolation.py`)**

Already supports angle encoding via `use_angle_encoding=True`:

```python
img, obj, view = read_face_data(
    'data/faceplace/data_faces.h5',
    use_angle_encoding=True,  # ← ENABLES continuous angles
    view_split_mode='by_view',
    train_view_indices=[0, 1, 3, 4, 5, 7, 8],
    val_view_indices=[2, 6]
)
```

**Angle mapping** (with encoding='normalized'):
```
Index  Angle   Normalized
  0    -90°  →  -1.000
  1    -60°  →  -0.667
  2    -45°  →  -0.500
  3    -30°  →  -0.333
  4      0°  →   0.000
  5    +30°  →  +0.333
  6    +45°  →  +0.500
  7    +60°  →  +0.667
  8    +90°  →  +1.000
```

### 4. **Training Notebook** (`train_gppvae_colab_interpolation.ipynb`)

**Cell 14** - Kernel Configuration:
```python
# RBF with continuous angles
KERNEL_CONFIG = {
    'view_kernel': 'rbf',
    'kernel_kwargs': {'lengthscale': 1.0, 'angle_scale': 'normalized'}
}
```

**Cell 10** - Data Loading:
```python
# Automatically enables angle encoding for RBF/Matérn
use_angle_encoding = CONFIG['view_kernel'] in ['rbf', 'matern']

img, obj, view = read_face_data(
    CONFIG['data'],
    use_angle_encoding=use_angle_encoding,  # ← Automatic!
    ...
)
```

**New Cell after Cell 10** - Model Initialization:
```python
# Handles both continuous and discrete view inputs
Wt = Variable(view["train"][:, 0], requires_grad=False).cuda()
Wv = Variable(view["val"][:, 0], requires_grad=False).cuda()

# Keep as float if using angle encoding, convert to long otherwise
if not use_angle_encoding:
    Wt = Wt.long()
    Wv = Wv.long()
```

## How to Run RBF Training

### Step 1: Configure Kernel (Cell 14)

```python
KERNEL_CONFIG = {
    'view_kernel': 'rbf',
    'kernel_kwargs': {'lengthscale': 1.0, 'angle_scale': 'normalized'}
}
```

**Options**:
- `lengthscale`: Controls smoothness (smaller = more variation, larger = smoother)
- `angle_scale`: `'normalized'` (default, [-1,1]) or `'radians'` ([0,2π])

### Step 2: Run Training

The notebook automatically:
1. Detects RBF kernel
2. Enables angle encoding
3. Loads data with continuous angles
4. Initializes model correctly
5. Trains with continuous angle values

### Step 3: Monitor Results

**Expected improvements over Legacy**:
- **MSE_out**: Lower (better interpolation)
- **gap_val_out**: Lower (less overfitting to training views)
- **Visualizations**: Row 3 should show actual 45° angles, not frontal views

## Comparison: Discrete vs Continuous Angles

### Discrete Indices (Legacy/Periodic/VonMises)
```
View representation: [0, 1, 2, 3, 4, 5, 6, 7, 8]
Distance(1, 3): kernel must learn from data
Geometry: implicit (learned)
```

### Continuous Angles (RBF/Matérn)
```
View representation: [-1.0, -0.67, -0.5, -0.33, 0.0, ...]
Distance(60°, 30°): |−0.67 − (−0.33)| = 0.34 (explicit!)
Geometry: explicit (given)
```

## Why This Helps RBF

The RBF kernel formula is:
```
k(θ, θ') = exp(−(θ − θ')² / (2ℓ²))
```

**With discrete indices**:
- `k(1, 3) = exp(−(1−3)² / (2ℓ²)) = exp(−4 / (2ℓ²))`
- Distance is 4 (arbitrary!)
- Kernel doesn't know that view 2 is halfway between 1 and 3

**With continuous angles**:
- `k(−0.67, −0.33) = exp(−(−0.67−(−0.33))² / (2ℓ²)) = exp(−0.116 / (2ℓ²))`
- Distance is 0.34 (proportional to true 30° difference!)
- View at −0.5 (45°) is explicitly halfway: `|−0.67−(−0.5)| = 0.17 = |−0.5−(−0.33)|`

## Expected Results

### Legacy (81 params, discrete indices)
```
gap_val_out:  ~0.037  (high)
MSE_out:      ~0.040
Row 3:        Frontal views (regression to mean)
```

### RBF with Continuous Angles (1 param)
```
gap_val_out:  ~0.015-0.020  (much lower!)
MSE_out:      ~0.020-0.025
Row 3:        Actual 45° angles (smooth interpolation)
```

**Why RBF is better**:
- Knows true angular distances
- Can interpolate smoothly between training views
- Regularization from geometric prior
- Only 1 learnable parameter (lengthscale)

## Troubleshooting

### Error: "F.embedding() expected LongTensor"
**Cause**: Trying to use continuous angles with legacy kernel
**Fix**: Make sure `use_angle_encoding=False` for legacy kernel, or switch to RBF

### Error: "Kernel doesn't support continuous angles"
**Cause**: Using a kernel without `forward_continuous()` method
**Fix**: RBF and Matérn support continuous angles; Periodic/VonMises use discrete indices

### Warning: "W has wrong dtype"
**Check**: 
- RBF/Matérn: `W` should be `torch.float32`
- Legacy/Periodic: `W` should be `torch.long`

## Summary

| Kernel | Angle Encoding | View Type | Geometry | Params | Best For |
|--------|----------------|-----------|----------|--------|----------|
| Legacy | ❌ No | Discrete | Learned | 81 | Baseline |
| FullRank | ❌ No | Discrete | Learned | 45 | Flexible |
| **RBF** | **✅ Yes** | **Continuous** | **Explicit** | **1** | **Interpolation** |
| **Matérn** | **✅ Yes** | **Continuous** | **Explicit** | **1** | **Realistic** |
| Periodic | ❌ No | Discrete | Built-in | 1 | Rotations |
| VonMises | ❌ No | Discrete | Built-in | 1 | Circular |

**Recommendation for Interpolation Experiment**:
1. Run **RBF with continuous angles** (best geometric prior)
2. Compare with **Periodic** (best for discrete rotation data)
3. Baseline with **Legacy** (shows why structured kernels help)

## Next Steps

1. ✅ Configure Cell 14 to use RBF kernel
2. ✅ Run training (100 epochs)
3. ✅ Compare gap_val_out with Legacy baseline
4. ✅ Verify Row 3 shows actual 45° angles
5. 🔄 Try Matérn (ν=1.5) for more realistic smoothness
6. 🔄 Run with multiple seeds for statistical significance

