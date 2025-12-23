# GPPVAE Training

## Setup Environment

```bash
# Create conda environment with Python 3.6
CONDA_SUBDIR=osx-64 conda env create -f environment.yml
conda activate gppvae
conda config --env --set subdir osx-64
```

## Process Data

```bash
# Process face images into HDF5 format (only need to run once)
python ./GPPVAE/pysrc/faceplace/process_data.py
```

## Training Commands

### 1. Train VAE First

The VAE must be trained first before training GPPVAE:

```bash
# Quick test run (10 epochs, ~2 minutes)
python ./GPPVAE/pysrc/faceplace/train_vae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/vae \
  --use_wandb \
  --epochs 10

# Short training (1000 epochs, ~2-4 hours)
python ./GPPVAE/pysrc/faceplace/train_vae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/vae \
  --use_wandb \
  --epochs 1000

# Full VAE training with default hyperparameters (10000 epochs, ~20-40 hours)
python ./GPPVAE/pysrc/faceplace/train_vae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/vae \
  --use_wandb

# Custom hyperparameters
python ./GPPVAE/pysrc/faceplace/train_vae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/vae \
  --use_wandb \
  --wandb_run_name "vae_experiment_1" \
  --zdim 512 \
  --lr 1e-4 \
  --bs 32 \
  --epochs 5000
```

### 2. Train GPPVAE (After VAE)

GPPVAE builds on top of the trained VAE model:

```bash
# Quick test (10 epochs, ~3-5 minutes)
python ./GPPVAE/pysrc/faceplace/train_gppvae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/gppvae \
  --use_wandb \
  --epochs 10

# Short training (1000 epochs, ~3-5 hours)
python ./GPPVAE/pysrc/faceplace/train_gppvae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/gppvae \
  --use_wandb \
  --epochs 1000

# Full GPPVAE training (10000 epochs, ~25-50 hours)
python ./GPPVAE/pysrc/faceplace/train_gppvae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/gppvae \
  --use_wandb

# Custom hyperparameters
python ./GPPVAE/pysrc/faceplace/train_gppvae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/gppvae \
  --use_wandb \
  --wandb_run_name "gppvae_experiment_1" \
  --xdim 128 \
  --vae_lr 1e-4 \
  --gp_lr 5e-4 \
  --epochs 5000
```

## Hyperparameters

All hyperparameters with detailed explanations are in `GPPVAE/pysrc/faceplace/config.yml`.

### VAE Parameters:
- **filts**: Number of convolutional filters (model capacity): 16-64, default=32
- **zdim**: Latent dimension size (compression level): 128-512, default=256
- **vy**: Observation noise variance (reconstruction strictness): 0.001-0.005, default=0.002
- **vae_lr**: Learning rate for VAE: 1e-4 to 5e-4, default=2e-4
- **bs**: Batch size: 32-128, default=64
- **epochs**: Total training iterations: 1000-10000, default=10000
- **epoch_cb**: Save frequency every N epochs: 50-200, default=100

### GPPVAE Parameters:
- **xdim**: Rank of object covariance (GP capacity): 32-128, default=64
- **gp_lr**: Learning rate for GP parameters: 5e-4 to 2e-3, default=1e-3
- **vae_cfg**: Path to trained VAE config (automatically set from config.yml)
- **vae_weights**: Path to trained VAE weights (automatically set from config.yml)

## Expected Training Times (M1 Pro)

### VAE:
- 10 epochs: ~2 minutes (quick test)
- 1000 epochs: ~2-4 hours (good results)
- 10000 epochs: ~20-40 hours (full training)

### GPPVAE:
- 10 epochs: ~3-5 minutes (quick test)
- 1000 epochs: ~3-5 hours (good results)
- 10000 epochs: ~25-50 hours (full training)

**Note:** GPPVAE is slower than VAE because it includes Gaussian Process computations.
