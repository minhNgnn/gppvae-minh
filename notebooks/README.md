# GPPVAE Training Notebooks for Google Colab

This directory contains Jupyter notebooks for training GPPVAE models on Google Colab's free GPUs.

## 📚 Available Notebooks

### 1. `train_vae_colab.ipynb` - VAE Pre-training
**Purpose:** Train the base VAE (Variational Autoencoder) model  
**Required First:** Yes - must run before GP-VAE  
**Training Time:** ~30-60 min for 100 epochs on Colab T4 GPU  
**Outputs:**
- `out/vae/vae.cfg.p` - VAE configuration
- `out/vae/weights/weights.*.pt` - Trained VAE weights
- Reconstruction visualizations

### 2. `train_gppvae_colab.ipynb` - GP-VAE Training
**Purpose:** Train the full GP-VAE with Gaussian Process prior  
**Required First:** No - but needs trained VAE weights from notebook 1  
**Training Time:** ~1-2 hours for 100 epochs on Colab T4 GPU  
**Outputs:**
- `out/gppvae/weights/vae_weights.*.pt` - Fine-tuned VAE
- `out/gppvae/weights/gp_weights.*.pt` - GP + Vmodel parameters
- Visualizations with out-of-sample predictions

---

## 🚀 Quick Start Guide

### Step 1: Prepare Your Files

1. **Upload to Google Drive** (Recommended):
   ```
   MyDrive/gppvae/
   ├── GPPVAE/           # Code from this repo
   ├── data/faceplace/   # Your dataset
   │   └── data_faces.h5
   └── notebooks/        # These notebooks
       ├── train_vae_colab.ipynb
       └── train_gppvae_colab.ipynb
   ```

2. **Or use VS Code + Colab extension** (files auto-synced)

### Step 2: Train VAE (Required First!)

1. Open `train_vae_colab.ipynb` in VS Code
2. Connect to Google Colab kernel (GPU runtime)
3. Run all cells
4. Wait ~30-60 minutes for 100 epochs
5. Download trained weights to your local machine

### Step 3: Train GP-VAE

1. Open `train_gppvae_colab.ipynb` in VS Code
2. Connect to Google Colab kernel (GPU runtime)
3. **Important:** Update cell 7 with your VAE weights path
4. Run all cells
5. Wait ~1-2 hours for 100 epochs
6. Download trained model

---

## ⚙️ Configuration

### VAE Training (`train_vae_colab.ipynb` cell 7)
```python
CONFIG = {
    'epochs': 100,        # Increase to 1000 for best results
    'batch_size': 64,     # Adjust based on GPU memory
    'lr': 0.0002,         # Learning rate
    'zdim': 256,          # Latent dimension
    'use_wandb': True,    # Enable W&B tracking
}
```

### GP-VAE Training (`train_gppvae_colab.ipynb` cell 7)
```python
CONFIG = {
    'vae_weights': './out/vae/weights/weights.00100.pt',  # ⬅️ Update this!
    'epochs': 100,        # Increase to 1000+ for publication quality
    'vae_lr': 0.0002,     # VAE fine-tuning rate (lower than initial)
    'gp_lr': 0.001,       # GP learning rate
    'xdim': 64,           # Rank of covariance matrix
}
```

---

## 📊 Expected Performance

### Training Speed Comparison

| Hardware | 10 Epochs | 100 Epochs | 1000 Epochs |
|----------|-----------|------------|-------------|
| **Colab T4 GPU** | 5-10 min | 30-60 min | 5-10 hours |
| **M1 Pro CPU** | ~2 hours | ~20 hours | ~200 hours |
| **Speedup** | **20-50x** | **20-50x** | **20-50x** |

### GPU Memory Usage
- VAE training: ~2-3 GB
- GP-VAE training: ~4-6 GB
- T4 GPU has 15 GB (plenty of headroom!)

---

## 🎯 Training Tips

### For Best Results:

1. **VAE Pre-training:**
   - Start with 100 epochs (quick test)
   - Increase to 1000 epochs for publication quality
   - Monitor validation MSE convergence
   - Use W&B to track experiments

2. **GP-VAE Training:**
   - Use VAE weights from at least 100 epochs
   - Start with 100 epochs to verify it works
   - Increase to 1000+ epochs for best results
   - Watch out-of-sample MSE vs reconstruction MSE

3. **Debugging:**
   - If GPU not detected: Runtime → Change runtime type → T4 GPU
   - If files not found: Check Google Drive mounting in cell 2
   - If OOM: Reduce batch_size in CONFIG

---

## 📈 What to Monitor

### VAE Training:
- ✅ **Train MSE** should decrease smoothly
- ✅ **Val MSE** should be close to Train MSE (no overfitting)
- ✅ **Reconstructions** should look good visually

### GP-VAE Training:
- ✅ **MSE val** (reconstruction) should stay stable
- ✅ **MSE out** (out-of-sample prediction) should be low
- ✅ **GP NLL** should decrease
- ✅ **Object variance** should be > 50% (identity matters)

---

## 🔧 Troubleshooting

### "VAE weights not found"
→ Train VAE first using `train_vae_colab.ipynb`

### "GPU not available"
→ Go to Runtime → Change runtime type → Select GPU (T4)

### "Files not syncing from VS Code"
→ Upload your `gppvae` folder to Google Drive manually

### "Out of memory"
→ Reduce `batch_size` from 64 to 32 or 16

### "Training too slow"
→ Make sure GPU is enabled (should be 50x faster than CPU)

---

## 📦 Output Files

After training, you'll have:

```
out/
├── vae/
│   ├── vae.cfg.p              # VAE architecture config
│   ├── weights/
│   │   ├── weights.00000.pt   # Checkpoint at epoch 0
│   │   ├── weights.00010.pt   # Checkpoint at epoch 10
│   │   └── ...
│   ├── plots/                 # Reconstruction visualizations
│   └── log.txt                # Training log
│
└── gppvae/
    ├── weights/
    │   ├── vae_weights.*.pt   # Fine-tuned VAE
    │   └── gp_weights.*.pt    # GP + Vmodel parameters
    ├── plots/                 # Out-of-sample predictions
    └── log.txt                # Training log
```

---

## 🎓 Understanding the Results

### VAE Metrics:
- **MSE**: Mean squared error (lower = better reconstructions)
- **NLL**: Negative log likelihood (VAE loss component)
- **KLD**: KL divergence (regularization term)

### GP-VAE Metrics:
- **MSE val**: VAE reconstruction quality (baseline)
- **MSE out**: Out-of-sample prediction quality (key metric!)
- **GP NLL**: Gaussian Process likelihood
- **vars**: [object_variance, noise_variance] - should sum to 1

### Success Criteria:
✅ MSE out ≈ MSE val (within 10-50%)  
✅ Object variance > 0.5 (50%+)  
✅ Visualizations show good predictions  

---

## 🌟 Next Steps

After training:

1. **Download weights** to your local machine
2. **Analyze results** using the visualization cells
3. **Experiment** with different hyperparameters
4. **Extend** to your own datasets
5. **Publish** your results! 🎉

---

## 💡 Pro Tips

- Use **W&B** for experiment tracking (free account)
- Save **multiple checkpoints** (every 10-100 epochs)
- **Download results** periodically (Colab sessions expire)
- Keep Colab tab **active** to prevent disconnection
- Use **VS Code extension** for better notebook editing

---

## 📚 References

- Original GPPVAE paper: Casale et al. (2018)
- Repository: https://github.com/fpcasale/GPPVAE
- Your fork: https://github.com/minhNgnn/gppvae-minh

---

**Happy Training! 🚀**

Questions? Check the comments in each notebook cell for detailed explanations.
