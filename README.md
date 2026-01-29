# GPPVAE - Experiments in kernel learning
This repository contains an experimental study of [**Gaussian Process Prior Variational Autoencoders (GPPVAE)**](https://proceedings.neurips.cc/paper/2018/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html) [1] with a focus on the role of **kernel choice as an inductive bias**.

Building on the original GPPVAE framework, this project systematically evaluates how different Gaussian Process priors—ranging from fixed geometric kernels to flexible learned kernels—affect representation quality, interpolation, and extrapolation in latent space. In particular, we compare:

- full-rank (unconstrained) kernels,
- hand-designed periodic kernels for circular geometry,
- Spectral Mixture kernels with both wrapped (geometry-aware) and free formulations.

Experiments are conducted on image datasets with known or partially known view structure, including controlled rotations on a circular domain. The experimental design fixes the VAE architecture and inference procedure, varying **only the GP prior**, in order to isolate the effect of kernel structure.

The results highlight that while flexible kernels can capture local similarity, explicitly encoding geometric structure is crucial for robust interpolation and long-range extrapolation. This repository accompanies a seminar project and is intended for research and educational purposes.

[1] Casale FP, Dalca AV, Saglietti L, Listgarten J, Fusi N. Gaussian process prior variational autoencoders. Advances in Neural Information Processing Systems, 31, 10390–10401.

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
python ./GPPVAE/pysrc/coil100/process_data.py
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

## Google Colab & Experiment Tracking

Most experiments in this project were executed on **Google Colab** due to computational requirements.  
Training runs are logged and tracked using **Weights & Biases (wandb)**.

- Colab notebooks used for training and evaluation are provided in the `notebooks/` directory.
- Each notebook corresponds to a specific experiment type (VAE pretraining, GPPVAE training, kernel comparison).
- wandb logging can be enabled or disabled via the `--use_wandb` flag.

To run experiments on Colab:
1. Open a notebook from `notebooks/`
2. Set your wandb API key (if logging is enabled)
3. Execute the notebook cells sequentially

Note: Results reported in the seminar project correspond to the Colab-based runs with fixed random seeds.

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

## Acknowledgements

Parts of this codebase are adapted from the original GPPVAE implementation introduced in:

Francesco Paolo Casale, Adrian Dalca, Luca Saglietti, Jennifer Listgarten, and Nicolo Fusi.  
**Gaussian Process Prior Variational Autoencoders**.  
Advances in Neural Information Processing Systems (NeurIPS), 2018.

Original repository:  
https://github.com/fpcasale/GPPVAE

The original code is released under the MIT License.

Modifications in this repository include:
- custom kernel compositions
- altered training procedures
- new experimental protocols