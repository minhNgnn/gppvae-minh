"""
Data parser for Experiment #1: Hard Held-Out Views

This is a modified version of data_parser.py that supports view-based splitting
for the hard held-out views experiment.

New features:
- split_data_by_views(): Split data by view indices instead of random sampling
- read_face_data(): Enhanced to support view_split_mode parameter
"""

import matplotlib
matplotlib.use("Agg")
import pylab as pl
import os
import copy

pl.ion()
import pdb
from utils import smartDumpDictHdf5, smartAppend
import h5py
import dask.array as da
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class FaceDataset(Dataset):
    def __init__(self, Y, D, W):
        self.len = Y.shape[0]
        self.Y, self.D, self.W = Y, D, W

    def __getitem__(self, index):
        return (self.Y[index], self.D[index], self.W[index], index)

    def __len__(self):
        return self.len


def split_data_by_views(Y, D, W, train_view_indices, val_view_indices):
    """
    Split data based on view indices instead of random sampling.
    Only keeps identities that have ALL required views in both train and val.
    
    Args:
        Y: Images [N, C, H, W] (torch tensor)
        D: Identity indices [N, 1] (torch tensor)
        W: View indices [N, 1] (torch tensor)
        train_view_indices: List of view indices for training (e.g., [3, 4, 5])
        val_view_indices: List of view indices for validation (e.g., [0, 1, 2, 6, 7, 8])
    
    Returns:
        train_data: Dict with Y_train, D_train, W_train
        val_data: Dict with Y_val, D_val, W_val
    """
    # Convert to numpy for easier indexing
    D_np = D.numpy().flatten()
    W_np = W.numpy().flatten()
    
    all_view_indices = set(train_view_indices) | set(val_view_indices)
    all_identities = np.unique(D_np)
    
    print("\n🔍 Filtering identities with complete view coverage...")
    print(f"   Required views: {sorted(all_view_indices)}")
    print(f"   Total identities before filtering: {len(all_identities)}")
    
    # Find identities that have all required views
    valid_identities = []
    for identity in all_identities:
        identity_mask = (D_np == identity)
        identity_views = set(W_np[identity_mask])
        
        # Check if this identity has all required views
        if all_view_indices.issubset(identity_views):
            valid_identities.append(identity)
    
    valid_identities = np.array(valid_identities)
    print(f"   Identities with all required views: {len(valid_identities)}")
    print(f"   Filtered out: {len(all_identities) - len(valid_identities)} identities")
    
    if len(valid_identities) == 0:
        raise ValueError("No identities have all required views! Check your data.")
    
    # Filter to only include valid identities
    valid_identity_mask = np.isin(D_np, valid_identities)
    Y_filtered = Y[valid_identity_mask]
    D_filtered = D[valid_identity_mask]
    W_filtered = W[valid_identity_mask]
    
    # CRITICAL: Remap identity indices to be contiguous [0, 1, 2, ...]
    # After filtering, we might have gaps like [0, 5, 7, 12, ...] which breaks Vmodel
    D_filtered_np = D_filtered.numpy().flatten()
    unique_identities = np.unique(D_filtered_np)
    identity_remap = {old_id: new_id for new_id, old_id in enumerate(unique_identities)}
    D_filtered_remapped = np.array([identity_remap[id] for id in D_filtered_np])
    D_filtered = torch.from_numpy(D_filtered_remapped).reshape(-1, 1)
    
    print(f"   ✅ Remapped {len(unique_identities)} identities to contiguous indices [0..{len(unique_identities)-1}]")
    
    # Now split by views
    W_filtered_np = W_filtered.numpy().flatten()
    train_mask = np.isin(W_filtered_np, train_view_indices)
    val_mask = np.isin(W_filtered_np, val_view_indices)
    
    # Create train/val splits
    train_data = {
        'Y': Y_filtered[train_mask],
        'D': D_filtered[train_mask],
        'W': W_filtered[train_mask]
    }
    
    val_data = {
        'Y': Y_filtered[val_mask],
        'D': D_filtered[val_mask],
        'W': W_filtered[val_mask]
    }
    
    # Validation checks
    print("\n✅ View-based split validation:")
    print(f"   Total samples (filtered): {len(Y_filtered)}")
    print(f"   Train samples: {len(train_data['Y'])} (views: {sorted(np.unique(train_data['W'].numpy()))})")
    print(f"   Val samples: {len(val_data['Y'])} (views: {sorted(np.unique(val_data['W'].numpy()))})")
    
    # Check identity coverage
    train_ids = set(np.unique(train_data['D'].numpy()))
    val_ids = set(np.unique(val_data['D'].numpy()))
    print(f"   Train identities: {len(train_ids)}")
    print(f"   Val identities: {len(val_ids)}")
    print(f"   Identity overlap: {len(train_ids & val_ids)}")
    
    # This should always pass now
    assert train_ids == val_ids, f"Identity sets don't match! This should not happen after filtering."
    print(f"   ✅ All {len(train_ids)} identities present in both train/val!")
    
    # Check samples per identity
    train_samples_per_id = len(train_data['Y']) / len(train_ids)
    val_samples_per_id = len(val_data['Y']) / len(val_ids)
    print(f"   Train samples per identity: {train_samples_per_id:.1f} (expected: {len(train_view_indices)}.0)")
    print(f"   Val samples per identity: {val_samples_per_id:.1f} (expected: {len(val_view_indices)}.0)")
    
    # Verify perfect split
    assert abs(train_samples_per_id - len(train_view_indices)) < 0.01, "Train samples per identity mismatch!"
    assert abs(val_samples_per_id - len(val_view_indices)) < 0.01, "Val samples per identity mismatch!"
    print(f"   ✅ Perfect split verified!")
    
    return train_data, val_data


def read_face_data(
    h5fn,
    tr_perc=0.9,
    view_split_mode='random',
    train_view_indices=None,
    val_view_indices=None
):
    """
    Load and split face data.
    
    Args:
        h5fn: Path to HDF5 data file
        tr_perc: Training percentage (only used if view_split_mode='random')
        view_split_mode: 'random' or 'by_view'
            - 'random': Original behavior (90% random train/val split)
            - 'by_view': Split by view indices (same identities, different views)
        train_view_indices: List of view indices for training (required if view_split_mode='by_view')
        val_view_indices: List of view indices for validation (required if view_split_mode='by_view')
    
    Returns:
        Y: Dict with train/val/test images
        D: Dict with train/val/test identity indices
        W: Dict with train/val/test view indices
    """
    
    print(f"\n📂 Loading data from: {h5fn}")
    print(f"   Split mode: {view_split_mode}")
    
    f = h5py.File(h5fn, "r")
    keys = ["test", "train", "val"]
    Y = {}
    Rid = {}
    Did = {}
    for key in keys:
        Y[key] = f["Y_" + key][:]
    for key in keys:
        Rid[key] = f["Rid_" + key][:]
    for key in keys:
        Did[key] = f["Did_" + key][:]
    f.close()

    # exclude test and validation not in train
    uDid = np.unique(Did["train"])
    for key in ["test", "val"]:
        Iok = np.in1d(Did[key], uDid)
        Y[key] = Y[key][Iok]
        Rid[key] = Rid[key][Iok]
        Did[key] = Did[key][Iok]

    # one hot encode donors
    table = {}
    for _i, _id in enumerate(uDid):
        table[_id] = _i
    D = {}
    for key in keys:
        D[key] = np.array([table[_id] for _id in Did[key]])[:, np.newaxis]

    # one hot encode views
    # IMPORTANT: Preserve angular ordering from data generation!
    # Use the order views appear in train set (already in angular order from process_data.py)
    # instead of np.unique which sorts alphabetically
    _, unique_indices = np.unique(Rid["train"], return_index=True)
    uRid = Rid["train"][np.sort(unique_indices)]  # Get unique views in order of first appearance
    
    table_w = {}
    for _i, _id in enumerate(uRid):
        table_w[_id] = _i
    W = {}
    for key in keys:
        W[key] = np.array([table_w[_id] for _id in Rid[key]])[:, np.newaxis]

    # Convert to float and normalize
    for key in keys:
        Y[key] = Y[key].astype(float) / 255.0
        Y[key] = torch.tensor(Y[key].transpose((0, 3, 1, 2)).astype(np.float32))
        D[key] = torch.tensor(D[key].astype(np.float32))
        W[key] = torch.tensor(W[key].astype(np.float32))

    # NEW: Handle view-based splitting
    if view_split_mode == 'by_view':
        assert train_view_indices is not None, "train_view_indices required for view_split_mode='by_view'"
        assert val_view_indices is not None, "val_view_indices required for view_split_mode='by_view'"
        
        print(f"\n🔬 Applying view-based split:")
        print(f"   Train views: {train_view_indices}")
        print(f"   Val views: {val_view_indices}")
        
        # Combine original train + val data
        Y_combined = torch.cat([Y['train'], Y['val']], dim=0)
        D_combined = torch.cat([D['train'], D['val']], dim=0)
        W_combined = torch.cat([W['train'], W['val']], dim=0)
        
        # Split by views
        train_data, val_data = split_data_by_views(
            Y_combined, D_combined, W_combined,
            train_view_indices, val_view_indices
        )
        
        # Update dictionaries
        Y['train'] = train_data['Y']
        D['train'] = train_data['D']
        W['train'] = train_data['W']
        
        Y['val'] = val_data['Y']
        D['val'] = val_data['D']
        W['val'] = val_data['W']
        
        # Validation checks
        print("\n✅ View-based split validation:")
        train_views = set(np.unique(W['train'].numpy().flatten()))
        val_views = set(np.unique(W['val'].numpy().flatten()))
        
        assert train_views == set(train_view_indices), f"Train views mismatch! Got {train_views}, expected {set(train_view_indices)}"
        assert val_views == set(val_view_indices), f"Val views mismatch! Got {val_views}, expected {set(val_view_indices)}"
        assert len(train_views & val_views) == 0, "Train and val views should not overlap!"
        
        print(f"   ✅ Train views correct: {sorted(train_views)}")
        print(f"   ✅ Val views correct: {sorted(val_views)}")
        print(f"   ✅ No overlap between train/val views")
    
    else:
        # Original random split behavior (no changes needed, already done in data generation)
        print("\n📊 Using original random train/val split from HDF5")
    
    print(f"\n✅ Final dataset sizes:")
    for key in keys:
        print(f"   {key:5s}: {len(Y[key]):5d} samples")

    return Y, D, W
