# SM Kernel Parameter Initialization Guide

## Current State: Generic Init + Learning ✅

All SM kernel parameters are **learned during training**, with generic initialization:

```python
SMCircleKernel(num_mixtures=2)
```

**What happens:**
- `weights`: Start equal `[0.5, 0.5]` → learned
- `frequencies`: Start `[0.01, 0.1]` → learned  
- `variances`: Start `[1.0, 1.0]` → learned

**Pros:**
- ✅ Simple API
- ✅ Model learns everything from data
- ✅ Works for any task

**Cons:**
- ⚠️ May take longer to converge
- ⚠️ Generic initialization doesn't use domain knowledge

---

## Enhanced Option: Custom Initialization

You can optionally provide better starting values using the enhanced kernel in `kernels_enhanced.py`:

```python
SMCircleKernelEnhanced(
    num_mixtures=2,
    frequencies_init=[1/360, 1/40],   # Full rotation + train spacing
    variances_init=[0.01, 0.01],      # Tight bandwidths
    weights_init=[0.5, 0.5]           # Equal importance
)
```

### Parameter Meanings

#### 1. **frequencies_init** (means, μ)
Controls which periodicities the kernel captures.

**For interpolation task (train views 40° apart):**
- `1/360 = 0.00278`: Captures full 360° rotation pattern
- `1/40 = 0.0250`: Captures 40° train view spacing
- These frequencies give the kernel a "head start" for interpolation

**Generic default:**
- `linspace(0.01, 0.1, num_mixtures)`: Spread across range
- Model must learn optimal frequencies from scratch

#### 2. **variances_init** (σ², bandwidths)
Controls how "tight" each frequency component is.

**Lower variance** = tighter bandwidth = more selective:
- `0.01`: Very selective (good for specific patterns)
- `1.0`: Broad (more flexible but less specific)

**For interpolation:**
- Start with small variances `[0.01, 0.01]` to encode strong beliefs about periodicities
- Model will adjust if needed during training

#### 3. **weights_init** (mixture weights, w)
Controls relative importance of each mixture component.

**Equal weights** `[0.5, 0.5]`:
- Both components start equally important
- Model learns which matters more

**Unequal weights** `[0.7, 0.3]`:
- If you believe one frequency is more important
- Example: Long-range structure (360°) more important than local (40°)

---

## Recommendations

### For Experimentation (Current Approach) ✅
```python
'kernel_kwargs': {
    'num_mixtures': 2,  # Let model learn everything
}
```
**Use when:**
- Exploring different tasks
- Don't know optimal initialization
- Want to see what model learns

### For Interpolation Task (Better Convergence)
```python
'kernel_kwargs': {
    'num_mixtures': 2,
    'frequencies_init': [1/360, 1/40],    # Domain knowledge
    'variances_init': [0.01, 0.01],       # Tight bandwidths
    'weights_init': [0.5, 0.5],           # Equal importance
}
```
**Use when:**
- Know task structure (interpolation between 40° train views)
- Want faster convergence
- Want to encode prior knowledge

### For Standard Task (Different Frequencies)
```python
'kernel_kwargs': {
    'num_mixtures': 2,
    'frequencies_init': [1/360, 1/60],    # Full rotation + test spacing
}
```
**Use when:**
- Different task structure (e.g., every 60° test views)

---

## Implementation Status

### ✅ Currently Available
- `SMCircleKernel(num_mixtures=2)` in `kernels.py`
- Generic initialization, all parameters learned
- Works in all notebooks

### 📋 Enhanced Version Available
- `SMCircleKernelEnhanced` in `kernels_enhanced.py`
- Optional custom initialization
- Backward compatible (defaults to generic)

### 🔧 To Use Enhanced Version

1. **Copy enhanced kernel to `kernels.py`** (replace `SMCircleKernel`)
2. **Update `create_kernel()` function** to handle new kwargs
3. **Update notebooks** with custom initialization (optional)

```python
# Example: Update create_kernel() in kernels.py
elif kernel_type == 'sm_circle':
    num_mixtures = kwargs.get('num_mixtures', 2)
    frequencies_init = kwargs.get('frequencies_init', None)  # NEW
    variances_init = kwargs.get('variances_init', None)      # NEW
    weights_init = kwargs.get('weights_init', None)          # NEW
    
    return SMCircleKernel(
        num_mixtures=num_mixtures,
        frequencies_init=frequencies_init,
        variances_init=variances_init,
        weights_init=weights_init
    )
```

---

## Key Insight

**All parameters are ALWAYS learned during training**, regardless of initialization!

- **Generic init**: Start from generic values → model learns from scratch
- **Custom init**: Start from informed values → model fine-tunes from better starting point

Custom initialization is an **inductive bias** that may help convergence, but the model can still learn to override it if the data suggests otherwise.

---

## Do You Want Enhanced Version?

**Current notebooks work fine** with generic initialization! 

The enhanced version is useful if you want:
1. ✅ **Faster convergence** - Better starting point
2. ✅ **Interpretability** - Explicit frequency hypotheses
3. ✅ **Control** - Encode domain knowledge

**It's optional!** The model will learn good parameters either way.
