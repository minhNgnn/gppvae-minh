# Experiment #1 Notebook Modifications

## New Cell 7b: View Split Configuration

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

---

## Modified Cell 8: CONFIG with View Mode

```python
from datetime import datetime

# GP-VAE Training configuration
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
kernel_name = KERNEL_CONFIG['view_kernel']

# Include view split mode in directory name
view_mode_str = 'central_views' if VIEW_SPLIT_MODE == 'by_view' else 'random'

CONFIG = {
    'data': './data/faceplace/data_faces.h5',
    # Output directory now includes kernel name AND view split mode
    'outdir': f'./out/gppvae_colab/{kernel_name}_{view_mode_str}_{timestamp}',
    'vae_cfg': './out/vae_colab/20251224_171841/vae.cfg.p',
    'vae_weights': './out/vae_colab/20251224_171841/weights/weights.00099.pt',

    # Training hyperparameters
    'epochs': 100,
    'batch_size': 64,
    'vae_lr': 0.001,
    'gp_lr': 0.001,
    'xdim': 64,

    # Kernel configuration
    'view_kernel': KERNEL_CONFIG['view_kernel'],
    'kernel_kwargs': KERNEL_CONFIG['kernel_kwargs'],

    # Experiment configuration (NEW)
    'view_split_mode': VIEW_SPLIT_MODE,
    'train_view_indices': TRAIN_VIEW_INDICES,
    'val_view_indices': VAL_VIEW_INDICES,

    # Logging
    'epoch_cb': 10,
    'use_wandb': True,
    'wandb_project': 'gppvae-exp1',
    'wandb_run_name': f'exp1_{kernel_name}_{view_mode_str}_{timestamp}',
    'seed': 0,
}

print("GP-VAE Training Configuration:")
print("=" * 60)
for key, value in CONFIG.items():
    if key in ['train_view_indices', 'val_view_indices'] and value is not None:
        print(f"  {key:20s}: {value}")
    elif key not in ['train_view_indices', 'val_view_indices']:
        print(f"  {key:20s}: {value}")
print("=" * 60)

# Verify VAE weights path
if not os.path.exists(CONFIG['vae_weights']):
    print(f"\n⚠️  WARNING: VAE weights not found at:")
    print(f"   {CONFIG['vae_weights']}")

print(f"\n✅ Output will be saved to:")
print(f"   {CONFIG['outdir']}")
print(f"\n   Directory name includes kernel type AND experiment mode!")
```

---

## Modified Cell 9: Import with Exp1 Data Parser

```python
# Change to training script directory
os.chdir(os.path.join(PROJECT_PATH, 'GPPVAE/pysrc/faceplace'))

# Import modules
import matplotlib
matplotlib.use('Agg')

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vae import FaceVAE
from vmod import Vmodel
from gp import GP
import h5py
import numpy as np
import logging
import pylab as pl
from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from callbacks import callback_gppvae
import pickle
import time
import wandb

# IMPORTANT: Use experiment 1 data parser with view-based splitting
from data_parser_exp1 import read_face_data, FaceDataset

print("✅ All modules imported successfully!")
print("✅ Using data_parser_exp1 for view-based splitting")
```

---

## Modified Cell 11: Data Loading with View Split

```python
# Set random seed
torch.manual_seed(CONFIG['seed'])

# Initialize W&B
if CONFIG['use_wandb']:
    wandb.init(
        project=CONFIG['wandb_project'],
        name=CONFIG['wandb_run_name'],
        config=CONFIG
    )

# Load VAE configuration
vae_cfg = pickle.load(open(CONFIG['vae_cfg'], "rb"))
print(f"VAE config: {vae_cfg}")

# Load pre-trained VAE
print("\nLoading pre-trained VAE...")
vae = FaceVAE(**vae_cfg).to(device)
vae_state = torch.load(CONFIG['vae_weights'], map_location=device)
vae.load_state_dict(vae_state)
print(f"✅ VAE loaded from {CONFIG['vae_weights']}")
print(f"   Total VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Load data with experiment configuration
print("\nLoading dataset...")
img, obj, view = read_face_data(
    CONFIG['data'],
    view_split_mode=CONFIG['view_split_mode'],
    train_view_indices=CONFIG.get('train_view_indices'),
    val_view_indices=CONFIG.get('val_view_indices')
)

train_data = FaceDataset(img["train"], obj["train"], view["train"])
val_data = FaceDataset(img["val"], obj["val"], view["val"])
train_queue = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
val_queue = DataLoader(val_data, batch_size=CONFIG['batch_size'], shuffle=False)

# Enhanced diagnostic logging
print(f"\n✅ Data loaded:")
print(f"   Training samples: {len(train_data)}")
print(f"   Validation samples: {len(val_data)}")
print(f"   Unique train views: {np.unique(view['train'].numpy())}")
print(f"   Unique val views: {np.unique(view['val'].numpy())}")
print(f"   Unique train identities: {len(np.unique(obj['train'].numpy()))}")
print(f"   Unique val identities: {len(np.unique(obj['val'].numpy()))}")

# Validation checks for experiment mode
if CONFIG['view_split_mode'] == 'by_view':
    print("\n🔍 Experiment Mode Validation Checks:")
    
    # Check 1: View split correctness
    train_views_set = set(np.unique(view['train'].numpy().flatten()).astype(int))
    val_views_set = set(np.unique(view['val'].numpy().flatten()).astype(int))
    
    assert train_views_set == set(CONFIG['train_view_indices']), f"Train views mismatch!"
    assert val_views_set == set(CONFIG['val_view_indices']), f"Val views mismatch!"
    assert len(train_views_set & val_views_set) == 0, "Train and val views overlap!"
    print("   ✅ View split verified correctly!")
    
    # Check 2: Identity coverage
    train_ids = set(np.unique(obj['train'].numpy()))
    val_ids = set(np.unique(obj['val'].numpy()))
    assert train_ids == val_ids, "Identity sets don't match between train/val!"
    print(f"   ✅ All {len(train_ids)} identities present in both train/val!")
    
    # Check 3: Sample distribution
    train_samples_per_id = len(img['train']) / len(train_ids)
    val_samples_per_id = len(img['val']) / len(val_ids)
    print(f"   ✅ Train samples per identity: {train_samples_per_id:.1f} (expected: {len(CONFIG['train_view_indices'])}.0)")
    print(f"   ✅ Val samples per identity: {val_samples_per_id:.1f} (expected: {len(CONFIG['val_view_indices'])}.0)")

# Create object and view variables for GP
Dt = Variable(obj["train"][:, 0].long(), requires_grad=False).cuda()
Wt = Variable(view["train"][:, 0].long(), requires_grad=False).cuda()
Dv = Variable(obj["val"][:, 0].long(), requires_grad=False).cuda()
Wv = Variable(view["val"][:, 0].long(), requires_grad=False).cuda()

# Initialize GP and Vmodel
print("\nInitializing GP-VAE components...")
P = np.unique(obj["train"].numpy()).shape[0]  # Number of unique objects (people)
Q = np.unique(view["train"].numpy()).shape[0]  # Number of unique views (angles)
print(f"   Objects (people): {P}")
print(f"   Views (angles): {Q}")

vm = Vmodel(
    P, Q,
    p=CONFIG['xdim'],
    q=Q,
    view_kernel=CONFIG['view_kernel'],
    **CONFIG['kernel_kwargs']
).cuda()

print(f"\n🔬 Initializing view kernel: '{CONFIG['view_kernel']}'")
if CONFIG['kernel_kwargs']:
    print(f"   Kernel parameters: {CONFIG['kernel_kwargs']}")
else:
    print(f"   Kernel parameters: (default)")

gp = GP(n_rand_effs=1).to(device)

# Combine GP parameters (Vmodel + GP)
gp_params = nn.ParameterList()
gp_params.extend(vm.parameters())
gp_params.extend(gp.parameters())

print(f"✅ GP-VAE components initialized:")
print(f"   Vmodel parameters: {sum(p.numel() for p in vm.parameters()):,}")
print(f"   GP parameters: {sum(p.numel() for p in gp.parameters()):,}")
print(f"   Total trainable: {sum(p.numel() for p in vae.parameters()) + sum(p.numel() for p in gp_params):,}")

# Create optimizers (separate for VAE and GP)
vae_optimizer = optim.Adam(vae.parameters(), lr=CONFIG['vae_lr'])
gp_optimizer = optim.Adam(gp_params, lr=CONFIG['gp_lr'])
print(f"\n✅ Optimizers created:")
print(f"   VAE optimizer: Adam(lr={CONFIG['vae_lr']})")
print(f"   GP optimizer: Adam(lr={CONFIG['gp_lr']})")
```

---

## Modified Cell 12: Training Functions with Per-View Metrics

```python
def encode_Y(vae, train_queue):
    """Encode all training images to get latent codes"""
    vae.eval()

    with torch.no_grad():
        n = train_queue.dataset.Y.shape[0]
        Zm = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()
        Zs = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()

        for batch_i, data in enumerate(train_queue):
            y = data[0].cuda()
            idxs = data[-1].cuda()
            zm, zs = vae.encode(y)
            Zm[idxs], Zs[idxs] = zm.detach(), zs.detach()

    return Zm, Zs


def eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv, Wv):
    """Enhanced evaluation with per-view metrics for Experiment #1"""
    rv = {}

    with torch.no_grad():
        _X = vm.x().data.cpu().numpy()
        _W = vm.v().data.cpu().numpy()
        covs = {"XX": np.dot(_X, _X.T), "WW": np.dot(_W, _W.T)}
        rv["vars"] = gp.get_vs().data.cpu().numpy()

        # Out-of-sample prediction
        vs = gp.get_vs()
        U, UBi, _ = gp.U_UBi_Shb([Vt], vs)
        Kiz = gp.solve(Zm, U, UBi, vs)
        Zo = vs[0] * Vv.mm(Vt.transpose(0, 1).mm(Kiz))

        mse_out = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).cuda()
        mse_val = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).cuda()

        for batch_i, data in enumerate(val_queue):
            idxs = data[-1].cuda()
            Yv = data[0].cuda()
            Zv = vae.encode(Yv)[0].detach()
            Yr = vae.decode(Zv)
            Yo = vae.decode(Zo[idxs])
            mse_out[idxs] = ((Yv - Yo) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()
            mse_val[idxs] = ((Yv - Yr) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()

            # Store examples for visualization
            if batch_i == 0:
                imgs = {}
                imgs["Yv"] = Yv[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
                imgs["Yr"] = Yr[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
                imgs["Yo"] = Yo[:24].data.cpu().numpy().transpose(0, 2, 3, 1)

        rv["mse_out"] = float(mse_out.data.mean().cpu())
        rv["mse_val"] = float(mse_val.data.mean().cpu())
        
        # NEW: Per-view metrics for Experiment #1
        unique_views = torch.unique(Wv).cpu().numpy()
        mse_val_per_view = {}
        mse_out_per_view = {}
        
        for view_idx in unique_views:
            view_mask = (Wv.cpu().numpy().flatten() == view_idx)
            mse_val_per_view[int(view_idx)] = float(mse_val.cpu().numpy()[view_mask].mean())
            mse_out_per_view[int(view_idx)] = float(mse_out.cpu().numpy()[view_mask].mean())
        
        rv['mse_val_per_view'] = mse_val_per_view
        rv['mse_out_per_view'] = mse_out_per_view

    return rv, imgs, covs


def backprop_and_update(vae, gp, vm, train_queue, Dt, Wt, Eps, Zb, Vbs, vbs, vae_optimizer, gp_optimizer):
    """Joint optimization of VAE and GP"""
    rv = {}

    vae_optimizer.zero_grad()
    gp_optimizer.zero_grad()
    vae.train()
    gp.train()
    vm.train()

    for batch_i, data in enumerate(train_queue):
        # Get batch data
        y = data[0].cuda()
        eps = Eps[data[-1]]
        _d = Dt[data[-1]]
        _w = Wt[data[-1]]
        _Zb = Zb[data[-1]]
        _Vbs = [Vbs[0][data[-1]]]

        # Forward through VAE
        zm, zs = vae.encode(y)
        z = zm + zs * eps
        yr = vae.decode(z)
        recon_term, mse = vae.nll(y, yr)

        # Forward through GP
        _Vs = [vm(_d, _w)]
        gp_nll_fo = gp.taylor_expansion(z, _Vs, _Zb, _Vbs, vbs) / vae.K

        # Penalization term
        pen_term = -0.5 * zs.sum(1)[:, None] / vae.K

        # Joint loss and backward
        loss = (recon_term + gp_nll_fo + pen_term).sum()
        loss.backward()

        # Accumulate metrics
        _n = train_queue.dataset.Y.shape[0]
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / _n)
        smartSum(rv, "recon_term", float(recon_term.data.sum().cpu()) / _n)
        smartSum(rv, "pen_term", float(pen_term.data.sum().cpu()) / _n)

    # Update both optimizers
    vae_optimizer.step()
    gp_optimizer.step()

    return rv


print("✅ Training functions defined with per-view metrics")
```

This continues in the next file...
