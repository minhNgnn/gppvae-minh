# Experiment #1: Hard Held-Out Views - Quick Start Guide

## 🎯 What This Experiment Tests

Train GP-VAE on **central views only** (-30° to +30°), then test reconstruction on **extreme angles** (±60°, ±90°).

**Hypothesis**: Structured kernels (Periodic/VonMises/Matérn) will dramatically outperform FullRank/Legacy because they understand angular smoothness.

---

## 📦 Files Created

### Core Code
1. **`GPPVAE/pysrc/faceplace/data_parser_exp1.py`**
   - Modified data parser with view-based splitting
   - New function: `split_data_by_views()`
   - Enhanced `read_face_data()` with `view_split_mode` parameter

### Notebook Modifications
2. **`notebooks/train_gppvae_exp1_cells.md`**
   - New Cell 7b: View split configuration
   - Modified Cell 8: CONFIG with experiment mode
   - Modified Cell 9: Import exp1 data parser
   - Modified Cell 11: Data loading with validation checks

3. **`notebooks/train_gppvae_exp1_training_loop.md`**
   - Modified Cell 12: Training functions with per-view metrics
   - Modified Cell 13: Training loop with per-view logging

### Analysis Tools
4. **`plan/compare_kernels_exp1.py`**
   - Automated comparison script
   - Generates comparison table
   - Identifies winner and computes advantages

---

## 🚀 How to Run the Experiment

### Step 1: Prepare Your Notebook

1. Open `notebooks/train_gppvae_colab.ipynb` in VS Code
2. Add the content from `train_gppvae_exp1_cells.md`:
   - Insert Cell 7b after the kernel configuration cell
   - Modify Cell 8 (CONFIG) with the experiment mode code
   - Modify Cell 9 (imports) to use `data_parser_exp1`
   - Modify Cell 11 (data loading) with validation checks

3. Add the training loop from `train_gppvae_exp1_training_loop.md`:
   - Replace Cell 12 with the new training functions
   - Replace Cell 13 with the new training loop

### Step 2: Upload Files to Colab

If using Google Colab:
1. Upload `data_parser_exp1.py` to your Drive folder:
   ```
   MyDrive/gppvae/GPPVAE/pysrc/faceplace/data_parser_exp1.py
   ```

2. The notebook will automatically find it when you run Cell 9

### Step 3: Run Training for Each Kernel

**Important**: Run all 6 kernels for comparison!

#### 3a. Periodic Kernel (RECOMMENDED)
```python
# Cell 7: Kernel Config
KERNEL_CONFIG = {
    'view_kernel': 'periodic',
    'kernel_kwargs': {'lengthscale': 2.0}
}

# Cell 7b: View Split Config
VIEW_SPLIT_MODE = 'by_view'
TRAIN_VIEW_INDICES = [3, 4, 5]
VAL_VIEW_INDICES = [0, 1, 2, 6, 7, 8]
```

Run all cells → Train for 100 epochs → Save results

#### 3b. VonMises Kernel
```python
# Cell 7: Kernel Config
KERNEL_CONFIG = {
    'view_kernel': 'vonmises',
    'kernel_kwargs': {'kappa': 1.0}
}

# Cell 7b: Same as above
```

Run all cells → Train for 100 epochs → Save results

#### 3c. Matérn Kernel
```python
# Cell 7: Kernel Config
KERNEL_CONFIG = {
    'view_kernel': 'matern',
    'kernel_kwargs': {'lengthscale': 2.0, 'nu': 1.5}
}

# Cell 7b: Same as above
```

Run all cells → Train for 100 epochs → Save results

#### 3d. RBF Kernel
```python
# Cell 7: Kernel Config
KERNEL_CONFIG = {
    'view_kernel': 'rbf',
    'kernel_kwargs': {'lengthscale': 2.5}
}

# Cell 7b: Same as above
```

Run all cells → Train for 100 epochs → Save results

#### 3e. FullRank Kernel (Baseline)
```python
# Cell 7: Kernel Config
KERNEL_CONFIG = {
    'view_kernel': 'fullrank',
    'kernel_kwargs': {}
}

# Cell 7b: Same as above
```

Run all cells → Train for 100 epochs → Save results

#### 3f. Legacy Kernel (Baseline)
```python
# Cell 7: Kernel Config
KERNEL_CONFIG = {
    'view_kernel': 'legacy',
    'kernel_kwargs': {}
}

# Cell 7b: Same as above
```

Run all cells → Train for 100 epochs → Save results

---

### Step 4: Compare Results

After running all kernels, use the comparison script:

```bash
cd /path/to/gppvae
python plan/compare_kernels_exp1.py --results_dir ./out/gppvae_colab
```

This will generate:
- Comparison table printed to console
- `exp1_kernel_comparison.csv` with all results
- Winner identification
- Structured vs Unstructured advantage calculation

---

## 📊 Expected Output

### During Training

You should see per-view breakdown every 10 epochs:

```
Epoch  100/100 | MSE train: 0.042150 | MSE val: 0.045632 | MSE out: 0.052134 | ...
   Per-view MSE_out:
      90L: 0.061234
      60L: 0.055678
      45L: 0.050123
      45R: 0.049876
      60R: 0.054321
      90R: 0.060987
```

### Final Summary

```
✅ GP-VAE Experiment #1 training complete!
   Total time: 34.2 minutes (0.57 hours)
   Final validation MSE: 0.045632
   Final out-of-sample MSE: 0.052134

📊 Final Per-View MSE_out (Experiment #1):
   VALIDATION VIEWS (held-out, extreme angles):
      90L (-90°)     : 0.061234
      60L (-60°)     : 0.055678
      45L (-45°)     : 0.050123
      45R (+45°)     : 0.049876
      60R (+60°)     : 0.054321
      90R (+90°)     : 0.060987

   Average MSE on extreme angles: 0.055370
```

### Comparison Table

```
Kernel Comparison: Hard Held-Out Views Experiment
==================================================
Kernel          | MSE_train  | MSE_val    | MSE_out    | Lengthscale  | Checkpoints
----------------|------------|------------|------------|--------------|-------------
periodic        | 0.042150   | 0.045632   | 0.052134   | 2.3          | 11
vonmises        | 0.042389   | 0.046012   | 0.053421   | 1.8          | 11
matern          | 0.043567   | 0.048234   | 0.061289   | 2.1          | 11
rbf             | 0.044234   | 0.051023   | 0.069876   | 2.5          | 11
fullrank        | 0.041234   | 0.044123   | 0.124567   | N/A          | 11
legacy          | 0.040987   | 0.043456   | 0.135678   | N/A          | 11

🏆 Winner: PERIODIC (MSE_out: 0.052134)
   Best vs Worst: 2.60× better!

📈 Structured vs Unstructured Kernels:
   Best structured: 0.052134
   Best unstructured: 0.124567
   Structured kernel advantage: 2.39× better!
   ✅ Hypothesis CONFIRMED: Structured kernels >> Unstructured for extrapolation!
```

---

## ✅ Validation Checklist

When you run the experiment, verify these:

### Data Loading
- [ ] Train views: [3, 4, 5] only
- [ ] Val views: [0, 1, 2, 6, 7, 8] only
- [ ] No overlap between train/val views
- [ ] All identities present in both train/val
- [ ] ~3 samples per identity in train
- [ ] ~6 samples per identity in val

### Training Output
- [ ] Per-view MSE logged every 10 epochs
- [ ] Learned lengthscale tracked (for structured kernels)
- [ ] Output directory includes "central_views" in name
- [ ] W&B logs include per-view metrics

### Results
- [ ] Structured kernels have lower MSE_out than FullRank
- [ ] Per-view MSE shows smooth degradation with angle for structured kernels
- [ ] FullRank shows sharp degradation on extreme angles
- [ ] Advantage > 2× confirms hypothesis

---

## 🐛 Troubleshooting

### Issue: "Module 'data_parser_exp1' not found"
**Solution**: Make sure you uploaded `data_parser_exp1.py` to the correct location and Cell 9 imports it correctly.

### Issue: "Train and val views overlap!"
**Solution**: Check that VIEW_SPLIT_MODE is set to 'by_view' and indices are correct.

### Issue: "Not all identities present in train/val"
**Solution**: Some identities may not have all 9 views. This is expected. As long as >90% have both, you're fine.

### Issue: "MSE_out not different between kernels"
**Solution**: 
1. Verify view split is actually working (check train/val views in logs)
2. Make sure you're using corrected data (angular ordering fixed)
3. Train longer (150-200 epochs)

---

## 📈 W&B Dashboard Setup

To get the best visualizations:

1. **Create comparison plot for MSE_out**:
   - Group by kernel name
   - Plot `mse_out` over epochs
   - Add `mse_val` for comparison

2. **Create per-view heatmap**:
   - Plot all `mse_out_per_view/*` metrics
   - Color code by angle (90L → 90R)
   - Compare across kernels

3. **Create lengthscale tracking**:
   - Plot `kernel/lengthscale` over epochs
   - Only for structured kernels
   - See if they converge to similar values

---

## 🎓 What to Look For

### Success Indicators
1. ✅ Periodic/VonMises have **2-3× lower MSE_out** than FullRank
2. ✅ Per-view MSE degrades **smoothly** for structured kernels
3. ✅ FullRank shows **sharp jump** at extreme angles
4. ✅ Learned lengthscales converge to ~2.0-3.0

### Interesting Patterns
- Do Periodic and VonMises perform similarly? (Expected: yes)
- Does Matérn fall between Periodic and RBF? (Expected: yes)
- Do extreme angles (90L, 90R) have highest MSE? (Expected: yes for all)
- Is the MSE symmetric (90L ≈ 90R, 60L ≈ 60R)? (Expected: yes)

---

## 📝 Next Steps After This Experiment

If hypothesis is confirmed:
1. ✅ Write up results
2. ✅ Move to Experiment #2: Few-Shot Learning
3. ✅ Consider combining experiments (few-shot + held-out)

If hypothesis is not confirmed:
1. 🔍 Debug view indices (print actual view strings)
2. 🔍 Verify GP is using view information (check Vmodel output)
3. 🔍 Try more extreme split (train on 00F only)

---

## 💡 Tips

- **Save GPU time**: Run overnight, all 6 kernels take ~6-8 hours total
- **Use W&B**: Essential for comparing all runs
- **Check early**: Look at epoch 20-30, pattern should be clear
- **Trust the metrics**: MSE_out is the key metric, not MSE_val
- **Document everything**: Save logs, plots, and W&B links

Good luck! 🚀
