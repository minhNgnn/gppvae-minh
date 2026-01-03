# Experiment #1: Hard Held-Out Views (View Extrapolation)

## 🎯 Objective
Train GP-VAE on **central views only** (-30° to +30°), then evaluate reconstruction quality on **extreme views** (±60°, ±90°) to test view kernel extrapolation capabilities.

**Hypothesis**: Structured kernels (Periodic/VonMises/Matérn) will dramatically outperform FullRank/Legacy because they have inductive biases about angular smoothness and periodicity.

---

## 📐 View Angle Mapping

Current view indices (after angular ordering fix):
```
Index 0: 90L  (-90°)  ← EXTREME (test only)
Index 1: 60L  (-60°)  ← EXTREME (test only)
Index 2: 45L  (-45°)  ← EXTREME (test only)
Index 3: 30L  (-30°)  ← CENTRAL (train)
Index 4: 00F  (  0°)  ← CENTRAL (train)
Index 5: 30R  (+30°)  ← CENTRAL (train)
Index 6: 45R  (+45°)  ← EXTREME (test only)
Index 7: 60R  (+60°)  ← EXTREME (test only)
Index 8: 90R  (+90°)  ← EXTREME (test only)
```

**Training set**: Views [3, 4, 5] (60° range: -30° to +30°)  
**Validation set**: Views [0, 1, 2, 6, 7, 8] (120° range: extreme angles)

---

## 📋 Implementation Steps

### Step 1: Modify Data Loading (data_parser.py)
**Location**: `GPPVAE/pysrc/faceplace/data_parser.py`

**Changes needed**:
1. Add function to filter views based on indices
2. Create train/val split based on view indices, not random sampling
3. Ensure all identities appear in both train and val (different views)

**Key considerations**:
- Keep same identities in train/val (only views differ)
- Maintain balanced sampling across identities
- Don't break existing code (make it optional via parameter)

**Pseudocode**:
```python
def split_data_by_views(Y, D, W, train_view_indices, val_view_indices):
    """
    Split data based on view indices instead of random sampling.
    Same identities appear in both train/val with different views.
    
    Args:
        Y: Images [N, C, H, W]
        D: Identity indices [N, 1]
        W: View indices [N, 1]
        train_view_indices: List of view indices for training (e.g., [3, 4, 5])
        val_view_indices: List of view indices for validation (e.g., [0, 1, 2, 6, 7, 8])
    
    Returns:
        train_data: Dict with Y_train, D_train, W_train
        val_data: Dict with Y_val, D_val, W_val
    """
    # Create boolean masks
    train_mask = np.isin(W, train_view_indices)
    val_mask = np.isin(W, val_view_indices)
    
    # Filter data
    train_data = {
        'Y': Y[train_mask],
        'D': D[train_mask],
        'W': W[train_mask]
    }
    
    val_data = {
        'Y': Y[val_mask],
        'D': D[val_mask],
        'W': W[val_mask]
    }
    
    return train_data, val_data
```

---

### Step 2: Update read_face_data() Function
**Location**: `GPPVAE/pysrc/faceplace/data_parser.py`

**Changes needed**:
1. Add optional parameter `view_split_mode` (default: 'random', new: 'central_only')
2. Add optional parameter `train_view_indices` and `val_view_indices`
3. Modify train/val split logic to use view-based filtering when specified

**Function signature**:
```python
def read_face_data(
    path, 
    tr_perc=0.9,
    view_split_mode='random',  # NEW: 'random' or 'by_view'
    train_view_indices=None,   # NEW: List of view indices for training
    val_view_indices=None      # NEW: List of view indices for validation
):
    """
    Load and split face data.
    
    view_split_mode='random': Original behavior (90% random train/val split)
    view_split_mode='by_view': Split by view indices (same identities, different views)
    """
```

**Logic**:
```python
if view_split_mode == 'by_view':
    assert train_view_indices is not None and val_view_indices is not None
    train_data, val_data = split_data_by_views(Y, D, W, train_view_indices, val_view_indices)
else:
    # Original random split logic
    n_train = int(Y.shape[0] * tr_perc)
    ...
```

---

### Step 3: Create New Training Configuration
**Location**: `notebooks/train_gppvae_colab.ipynb`

**Changes needed**:
1. Add new cell after Cell 7 (kernel config) for view split configuration
2. Update Cell 11 (data loading) to pass view split parameters
3. Add logging to verify view distribution

**New Cell (7b): Configure View Split**:
```python
# ============================================================================
# VIEW SPLIT CONFIGURATION - For Hard Held-Out Views Experiment
# ============================================================================

# Experiment mode
VIEW_SPLIT_MODE = 'by_view'  # 'random' or 'by_view'

# View angle mapping (after angular ordering fix):
# Index 0: 90L (-90°), 1: 60L (-60°), 2: 45L (-45°), 3: 30L (-30°), 4: 00F (0°),
# Index 5: 30R (+30°), 6: 45R (+45°), 7: 60R (+60°), 8: 90R (+90°)

if VIEW_SPLIT_MODE == 'by_view':
    # EXPERIMENT 1: Train on central views, test on extreme angles
    TRAIN_VIEW_INDICES = [3, 4, 5]  # -30L, 00F, 30R (60° range)
    VAL_VIEW_INDICES = [0, 1, 2, 6, 7, 8]  # Extreme angles (±45°, ±60°, ±90°)
    
    print("🔬 EXPERIMENT MODE: Hard Held-Out Views")
    print("=" * 60)
    print("Training views (central):")
    print("  Index 3: 30L (-30°)")
    print("  Index 4: 00F (  0°)")
    print("  Index 5: 30R (+30°)")
    print("\nValidation views (extreme):")
    print("  Index 0: 90L (-90°)")
    print("  Index 1: 60L (-60°)")
    print("  Index 2: 45L (-45°)")
    print("  Index 6: 45R (+45°)")
    print("  Index 7: 60R (+60°)")
    print("  Index 8: 90R (+90°)")
    print("=" * 60)
else:
    TRAIN_VIEW_INDICES = None
    VAL_VIEW_INDICES = None
    print("📊 Standard Mode: Random 90/10 train/val split")
```

**Modify Cell 11 (data loading)**:
```python
# Load data
print("\nLoading dataset...")
img, obj, view = read_face_data(
    CONFIG['data'],
    view_split_mode=VIEW_SPLIT_MODE,
    train_view_indices=TRAIN_VIEW_INDICES,
    val_view_indices=VAL_VIEW_INDICES
)

# Add diagnostic logging
print(f"✅ Data loaded:")
print(f"   Training samples: {len(img['train'])}")
print(f"   Validation samples: {len(img['val'])}")
print(f"   Unique train views: {np.unique(view['train'])}")
print(f"   Unique val views: {np.unique(view['val'])}")
print(f"   Unique train identities: {len(np.unique(obj['train']))}")
print(f"   Unique val identities: {len(np.unique(obj['val']))}")
```

---

### Step 4: Update Output Directory Naming
**Location**: `notebooks/train_gppvae_colab.ipynb`, Cell 8 (CONFIG)

**Changes needed**:
Add view split info to output directory name for easy identification

**Before**:
```python
'outdir': f'./out/gppvae_colab/{kernel_name}_{timestamp}',
```

**After**:
```python
# Include view split mode in directory name
view_mode_str = 'central_views' if VIEW_SPLIT_MODE == 'by_view' else 'random'
'outdir': f'./out/gppvae_colab/{kernel_name}_{view_mode_str}_{timestamp}',
```

**Example output directories**:
- `./out/gppvae_colab/periodic_central_views_20251224_180000/`
- `./out/gppvae_colab/fullrank_central_views_20251224_180500/`
- `./out/gppvae_colab/legacy_central_views_20251224_181000/`

---

### Step 5: Add Experiment-Specific Metrics
**Location**: `notebooks/train_gppvae_colab.ipynb`, Cell 12 (training loop)

**Changes needed**:
Add per-view-angle breakdown of MSE during evaluation

**New metrics to log**:
- `mse_val_per_view`: MSE for each validation view angle
- `mse_out_per_view`: Out-of-sample MSE for each validation view angle
- Average MSE for extreme angles vs all validation angles

**Pseudocode for eval_step() modification**:
```python
def eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv):
    """Enhanced evaluation with per-view metrics"""
    rv = {}
    
    # ... existing code ...
    
    # NEW: Track per-view MSE
    unique_views = torch.unique(Wv).cpu().numpy()
    mse_val_per_view = {}
    mse_out_per_view = {}
    
    for view_idx in unique_views:
        view_mask = (Wv == view_idx)
        mse_val_per_view[int(view_idx)] = float(mse_val[view_mask].mean().cpu())
        mse_out_per_view[int(view_idx)] = float(mse_out[view_mask].mean().cpu())
    
    rv['mse_val_per_view'] = mse_val_per_view
    rv['mse_out_per_view'] = mse_out_per_view
    
    return rv, imgs, covs
```

**W&B logging**:
```python
if CONFIG['use_wandb']:
    log_dict = {
        # ... existing metrics ...
        
        # Per-view breakdown
        "mse_val_per_view/90L": rv_eval['mse_val_per_view'].get(0, None),
        "mse_val_per_view/60L": rv_eval['mse_val_per_view'].get(1, None),
        "mse_val_per_view/45L": rv_eval['mse_val_per_view'].get(2, None),
        "mse_val_per_view/45R": rv_eval['mse_val_per_view'].get(6, None),
        "mse_val_per_view/60R": rv_eval['mse_val_per_view'].get(7, None),
        "mse_val_per_view/90R": rv_eval['mse_val_per_view'].get(8, None),
        
        "mse_out_per_view/90L": rv_eval['mse_out_per_view'].get(0, None),
        "mse_out_per_view/60L": rv_eval['mse_out_per_view'].get(1, None),
        "mse_out_per_view/45L": rv_eval['mse_out_per_view'].get(2, None),
        "mse_out_per_view/45R": rv_eval['mse_out_per_view'].get(6, None),
        "mse_out_per_view/60R": rv_eval['mse_out_per_view'].get(7, None),
        "mse_out_per_view/90R": rv_eval['mse_out_per_view'].get(8, None),
    }
```

---

### Step 6: Create Comparison Script
**Location**: `plan/compare_kernels_exp1.py` (new file)

**Purpose**: Automated comparison of all kernels on this experiment

**Script should**:
1. Train each kernel (Legacy, FullRank, Periodic, VonMises, Matérn, RBF)
2. Log results to W&B
3. Generate comparison table
4. Plot MSE by view angle for each kernel
5. Compute and report:
   - Overall MSE_out
   - Average MSE on extreme angles only
   - Ratio of extreme-to-central MSE
   - Learned lengthscales (for structured kernels)

**Output format**:
```
Kernel Comparison: Hard Held-Out Views Experiment
==================================================
Kernel      | MSE_out_all | MSE_out_extreme | Extreme/Central | Params | Lengthscale
------------|-------------|-----------------|-----------------|--------|------------
Periodic    | 0.0521      | 0.0543          | 1.04×          | 1      | 2.3
VonMises    | 0.0534      | 0.0551          | 1.03×          | 1      | 1.8
Matérn      | 0.0612      | 0.0687          | 1.12×          | 1      | 2.1
RBF         | 0.0698      | 0.0823          | 1.18×          | 1      | 2.5
FullRank    | 0.1245      | 0.1543          | 1.24×          | 45     | N/A
Legacy      | 0.1356      | 0.1712          | 1.26×          | 81     | N/A

Winner: Periodic (2.4× better than FullRank on extreme angles!)
```

---

## 🔍 Validation Checks

Before running the full experiment, verify:

### Check 1: Data Split Correctness
```python
# After loading data
assert len(np.intersect1d(view['train'], VAL_VIEW_INDICES)) == 0, "Train views leaked into val!"
assert len(np.intersect1d(view['val'], TRAIN_VIEW_INDICES)) == 0, "Val views leaked into train!"
assert set(np.unique(view['train'])) == set(TRAIN_VIEW_INDICES), "Train views mismatch!"
assert set(np.unique(view['val'])) == set(VAL_VIEW_INDICES), "Val views mismatch!"
print("✅ View split verified correctly!")
```

### Check 2: Identity Coverage
```python
# All identities should appear in both train and val (with different views)
train_ids = set(np.unique(obj['train']))
val_ids = set(np.unique(obj['val']))
assert train_ids == val_ids, "Identity sets don't match between train/val!"
print(f"✅ All {len(train_ids)} identities present in both train/val!")
```

### Check 3: Sample Distribution
```python
# Check samples per identity
train_samples_per_id = len(img['train']) / len(np.unique(obj['train']))
val_samples_per_id = len(img['val']) / len(np.unique(obj['val']))
print(f"Train samples per identity: {train_samples_per_id:.1f} (expected: 3.0)")
print(f"Val samples per identity: {val_samples_per_id:.1f} (expected: 6.0)")
```

---

## 📊 Expected Results

### Hypothesis
**Periodic/VonMises kernels will dramatically outperform FullRank/Legacy** because:
1. They enforce smooth angular structure
2. They know angles wrap around (periodicity)
3. They extrapolate well beyond training distribution

### Quantitative Predictions
```
                MSE_val     MSE_out     MSE_out_extreme
Periodic:       0.045       0.052       0.054          ← Winner!
VonMises:       0.046       0.053       0.055          ← Winner!
Matérn:         0.048       0.061       0.069
RBF:            0.051       0.070       0.082
FullRank:       0.044       0.125       0.154          ← Fails on extrapolation
Legacy:         0.043       0.136       0.171          ← Worst extrapolation
```

**Key insight**: 
- FullRank/Legacy may have **lower MSE_val** (interpolation within training views)
- But **3× higher MSE_out_extreme** (extrapolation to unseen views)
- Periodic/VonMises maintain consistent performance across all angles

---

## 🎯 Success Criteria

**Experiment succeeds if**:
1. ✅ Periodic or VonMises has **2× better MSE_out** than FullRank
2. ✅ Structured kernels have **lower variance** in per-view MSE
3. ✅ FullRank shows **sharp degradation** on extreme angles (±60°, ±90°)
4. ✅ Periodic/VonMises show **smooth degradation** with angle distance
5. ✅ Learned lengthscales are reasonable (1.5-3.0 range)

**Experiment fails if**:
1. ❌ FullRank performs equally well on extreme angles
2. ❌ No clear pattern in per-angle MSE
3. ❌ Structured kernels overfit training views

---

## 📝 Files to Modify

### Core Changes
1. `GPPVAE/pysrc/faceplace/data_parser.py`
   - Add `split_data_by_views()` function
   - Modify `read_face_data()` to support view-based splitting

### Notebook Changes
2. `notebooks/train_gppvae_colab.ipynb`
   - Add Cell 7b: View split configuration
   - Modify Cell 11: Pass view split parameters to data loader
   - Modify Cell 8: Update output directory naming
   - Modify Cell 12: Add per-view metrics to training loop

### New Files
3. `plan/compare_kernels_exp1.py` - Automated comparison script
4. `plan/experiment1_hard_heldout_views.md` - This document

---

## ⚠️ Important Notes

1. **Don't regenerate data_faces.h5**: The angular ordering fix is already there, just filter at loading time
2. **Keep same VAE weights**: Use existing trained VAE, only GP-VAE training changes
3. **Run all kernels**: Need Legacy, FullRank, Periodic, VonMises, Matérn, RBF for comparison
4. **Use W&B**: Log everything for easy comparison across runs
5. **Save per-view plots**: Visualize which angles each kernel handles well

