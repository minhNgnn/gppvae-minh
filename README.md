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

# Full training with default hyperparameters (10000 epochs, ~20-40 hours)
python ./GPPVAE/pysrc/faceplace/train_vae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/vae \
  --use_wandb

# Custom hyperparameters
python ./GPPVAE/pysrc/faceplace/train_vae.py \
  --data ./data/faceplace/data_faces.h5 \
  --outdir ./out/vae \
  --use_wandb \
  --wandb_run_name "experiment_1" \
  --zdim 512 \
  --lr 1e-4 \
  --bs 32 \
  --epochs 5000
```

## Hyperparameters

Default values and detailed explanations are in:
- **YAML**: `GPPVAE/pysrc/faceplace/config.yml` (edit this for default values)
- **Python**: `GPPVAE/pysrc/faceplace/config.py` (loader code)

### Key Parameters:
- **filts**: Number of convolutional filters (model capacity): 16-64, default=32
- **zdim**: Latent dimension size (compression level): 128-512, default=256
- **vy**: Observation noise variance (reconstruction strictness): 0.001-0.005, default=0.002
- **lr**: Learning rate: 1e-4 to 5e-4, default=2e-4
- **bs**: Batch size: 32-128, default=64
- **epochs**: Total training iterations: 1000-10000, default=10000
- **epoch_cb**: Save frequency every N epochs: 50-200, default=100

## Expected Training Times (M1 Pro)
- 10 epochs: ~2 minutes (quick test)
- 1000 epochs: ~2-4 hours (good results)
- 10000 epochs: ~20-40 hours (full training, leave overnight)
