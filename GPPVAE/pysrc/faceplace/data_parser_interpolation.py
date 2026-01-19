"""
Data parser for Interpolation Experiment: In-Range Generalization

This is a modified version of data_parser.py that supports view-based splitting
for the interpolation experiment.

Experiment Setup:
- Train on: {90L, 60L, 30L, 0, 30R, 60R, 90R} - 7 views (indices: 0, 1, 3, 4, 5, 7, 8)
- Test on: {45L, 45R} - 2 intermediate views (indices: 2, 6)

Research Question: "Does imposing smoothness via structured kernels improve or 
degrade performance when test views lie within the training range?"

New features:
- split_data_by_views_interpolation(): Split for interpolation task
- read_face_data(): Enhanced to support view_split_mode='interpolation'
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


def encode_view_angles(view_indices, encoding='normalized'):
    """
    Convert view indices to actual angular values.
    
    This encodes geometric relationships between views explicitly,
    allowing structured kernels to leverage angular smoothness.
    
    View mapping (original indices → actual angles):
    0: 90L → -90°
    1: 60L → -60°
    2: 45L → -45°
    3: 30L → -30°
    4: 00F →   0°
    5: 30R → +30°
    6: 45R → +45°
    7: 60R → +60°
    8: 90R → +90°
    
    Args:
        view_indices: Array of view indices [0-8]
        encoding: How to encode angles
            - 'normalized': Scale to [-1, 1] range
            - 'radians': Convert to radians [-π/2, π/2]
            - 'degrees': Keep as degrees [-90, 90]
    
    Returns:
        Tensor of encoded angle values
    """
    # Map indices to actual angles in degrees
    angle_map = {
        0: -90.0,  # 90L
        1: -60.0,  # 60L
        2: -45.0,  # 45L
        3: -30.0,  # 30L
        4:   0.0,  # 00F (frontal)
        5:  30.0,  # 30R
        6:  45.0,  # 45R
        7:  60.0,  # 60R
        8:  90.0,  # 90R
    }
    
    # Convert indices to angles
    if isinstance(view_indices, torch.Tensor):
        view_indices = view_indices.numpy()
    
    view_indices = view_indices.flatten()
    
    # Validate indices
    unique_indices = np.unique(view_indices)
    invalid_indices = [idx for idx in unique_indices if int(idx) not in angle_map]
    if invalid_indices:
        raise ValueError(
            f"Invalid view indices found: {invalid_indices}\n"
            f"Valid indices are 0-8 (corresponding to angles -90° to +90°)\n"
            f"Unique indices in data: {sorted(unique_indices)}\n"
            f"This suggests the data was not properly processed or encoded."
        )
    
    angles_deg = np.array([angle_map[int(idx)] for idx in view_indices])
    
    # Encode based on requested format
    if encoding == 'normalized':
        # Scale to [-1, 1] range for numerical stability
        angles_encoded = angles_deg / 90.0
    elif encoding == 'radians':
        # Convert to radians [-π/2, π/2]
        angles_encoded = np.deg2rad(angles_deg)
    elif encoding == 'degrees':
        # Keep as degrees [-90, 90]
        angles_encoded = angles_deg
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    
    return torch.tensor(angles_encoded, dtype=torch.float32)


def split_data_by_views_interpolation(Y, D, W, train_view_indices, val_view_indices, use_angle_encoding=True, w_already_encoded=False):
    """
    Split data for interpolation experiment.
    Only keeps identities that have ALL required views in both train and val.
    
    Interpolation Setup:
    - Train views: Boundary angles (90L, 60L, 30L, 0, 30R, 60R, 90R)
    - Val views: Intermediate angles (45L, 45R) that lie WITHIN training range
    
    Args:
        Y: Images [N, C, H, W] (torch tensor)
        D: Identity indices [N, 1] (torch tensor)
        W: View values [N, 1] (torch tensor) - can be indices [0-8] or angles if already encoded
        train_view_indices: List of view indices for training (e.g., [0, 1, 3, 4, 5, 7, 8])
        val_view_indices: List of view indices for validation (e.g., [2, 6])
        use_angle_encoding: If True, work with angle values (for filtering/matching)
        w_already_encoded: If True, W is already encoded as angles (don't encode again)
    
    Returns:
        train_data: Dict with Y_train, D_train, W_train
        val_data: Dict with Y_val, D_val, W_val
    """
    # Convert to numpy for easier indexing
    D_np = D.numpy().flatten()
    W_np = W.numpy().flatten()
    
    all_view_indices = set(train_view_indices) | set(val_view_indices)
    all_identities = np.unique(D_np)
    
    print("\n🔍 Filtering identities with complete view coverage (Interpolation)...")
    print(f"   Required views: {sorted(all_view_indices)}")
    print(f"   Train views (boundary): {sorted(train_view_indices)}")
    print(f"   Val views (intermediate): {sorted(val_view_indices)}")
    print(f"   Total identities before filtering: {len(all_identities)}")
    
    if use_angle_encoding:
        print(f"   🎯 Using ACTUAL ANGLE VALUES (not indices)")
        print(f"   This provides geometric information to structured kernels")
    else:
        print(f"   Using view indices (original behavior)")
    
    # Find identities that have all required views
    valid_identities = []
    
    # If W is already encoded as angles, we need to match against angle values
    # Otherwise, match against indices directly
    if w_already_encoded and use_angle_encoding:
        # W contains angles, convert required indices to angles for comparison
        required_views = set(encode_view_angles(np.array(sorted(all_view_indices)), encoding='normalized').numpy().round(6))
        print(f"   🎯 W already contains angles, matching against angle values")
    elif use_angle_encoding and not w_already_encoded:
        # W contains indices, but we want to work with angles
        # Convert required indices to angles
        required_views = set(encode_view_angles(np.array(sorted(all_view_indices)), encoding='normalized').numpy().round(6))
        print(f"   🎯 Converting indices to angles for filtering")
    else:
        # W contains indices, match directly
        required_views = all_view_indices
        print(f"   Using view indices directly (no angle encoding)")
    
    for identity in all_identities:
        identity_mask = (D_np == identity)
        identity_views_raw = W_np[identity_mask]
        
        if w_already_encoded and use_angle_encoding:
            # W is already angles, round for comparison
            identity_views = set(np.round(identity_views_raw, 6))
        elif use_angle_encoding and not w_already_encoded:
            # W is indices, convert to angles
            identity_views = set(encode_view_angles(identity_views_raw, encoding='normalized').numpy().round(6))
        else:
            # W is indices, use directly
            identity_views = set(identity_views_raw)
        
        # Check if this identity has all required views
        if required_views.issubset(identity_views):
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
    
    # Convert view indices to actual angles if requested AND not already encoded
    if use_angle_encoding and not w_already_encoded:
        W_filtered_angles = encode_view_angles(W_filtered, encoding='normalized')
        W_filtered = W_filtered_angles.reshape(-1, 1)
        print(f"   ✅ Converted view indices to normalized angles [-1.0, 1.0]")
        
        # Show example mappings
        view_names = {0: "90L (-90°)", 1: "60L (-60°)", 2: "45L (-45°)", 3: "30L (-30°)", 
                     4: "00F (0°)", 5: "30R (+30°)", 6: "45R (+45°)", 7: "60R (+60°)", 8: "90R (+90°)"}
        angle_examples = encode_view_angles(np.array([0, 2, 4, 6, 8]), encoding='normalized')
        print(f"   Example angle encodings:")
        for idx, angle_val in zip([0, 2, 4, 6, 8], angle_examples):
            print(f"      Index {idx} ({view_names[idx]}) → {angle_val:.3f}")
    elif w_already_encoded:
        print(f"   ✅ W already encoded as angles, no conversion needed")
    
    # Now split by views
    W_filtered_np = W_filtered.numpy().flatten()
    
    if use_angle_encoding or w_already_encoded:
        # Match by encoding train/val indices to angles first
        train_angles = encode_view_angles(np.array(train_view_indices), encoding='normalized').numpy()
        val_angles = encode_view_angles(np.array(val_view_indices), encoding='normalized').numpy()
        
        # Create masks by matching angle values (with small tolerance for floating point)
        train_mask = np.isin(np.round(W_filtered_np, 6), np.round(train_angles, 6))
        val_mask = np.isin(np.round(W_filtered_np, 6), np.round(val_angles, 6))
    else:
        # Original index-based matching
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
    print("\n✅ Interpolation split validation:")
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
    
    # Interpolation-specific validation
    train_view_set = set(train_view_indices)
    val_view_set = set(val_view_indices)
    
    # Check that validation views are "sandwiched" between training views
    print(f"\n🎯 Interpolation task verification:")
    view_names = {0: "90L", 1: "60L", 2: "45L", 3: "30L", 4: "00F", 
                  5: "30R", 6: "45R", 7: "60R", 8: "90R"}
    
    print(f"   Training on boundary angles:")
    for v in sorted(train_view_indices):
        angle_val = encode_view_angles(np.array([v]), encoding='normalized')[0] if use_angle_encoding else v
        print(f"      Index {v}: {view_names.get(v, f'View{v}')} → encoded as {angle_val:.3f}" if use_angle_encoding 
              else f"      Index {v}: {view_names.get(v, f'View{v}')}")
    
    print(f"   Testing on intermediate angles:")
    for v in sorted(val_view_indices):
        angle_val = encode_view_angles(np.array([v]), encoding='normalized')[0] if use_angle_encoding else v
        print(f"      Index {v}: {view_names.get(v, f'View{v}')} (interpolation target) → encoded as {angle_val:.3f}" 
              if use_angle_encoding else f"      Index {v}: {view_names.get(v, f'View{v}')} (interpolation target)")
    
    # Verify that each val view is bounded by train views
    if 2 in val_view_set:  # 45L
        assert 1 in train_view_set and 3 in train_view_set, "45L should be between 60L and 30L"
        print(f"   ✅ 45L is bounded by training views 60L and 30L")
    
    if 6 in val_view_set:  # 45R
        assert 5 in train_view_set and 7 in train_view_set, "45R should be between 30R and 60R"
        print(f"   ✅ 45R is bounded by training views 30R and 60R")
    
    if use_angle_encoding:
        print(f"\n   📐 Geometric distances now explicit:")
        print(f"      distance(90L, 60L) = |-1.00 - (-0.67)| = 0.33 (30°)")
        print(f"      distance(60L, 45L) = |-0.67 - (-0.50)| = 0.17 (15°)")
        print(f"      distance(45L, 30L) = |-0.50 - (-0.33)| = 0.17 (15°)")
        print(f"   ✅ True angular distances preserved!")
    
    return train_data, val_data


def read_face_data(
    h5fn,
    tr_perc=0.9,
    view_split_mode='random',
    train_view_indices=None,
    val_view_indices=None,
    use_angle_encoding=True
):
    """
    Load and split face data.
    
    Args:
        h5fn: Path to HDF5 data file
        tr_perc: Training percentage (only used if view_split_mode='random')
        view_split_mode: 'random', 'by_view', or 'interpolation'
            - 'random': Original behavior (90% random train/val split)
            - 'by_view': Split by view indices (extrapolation/held-out views)
            - 'interpolation': Split for interpolation task (intermediate views)
        train_view_indices: List of view indices for training
        val_view_indices: List of view indices for validation
        use_angle_encoding: If True, convert view indices to actual angle values (RECOMMENDED)
            This provides geometric information to structured kernels
    
    Returns:
        Y: Dict with train/val/test images
        D: Dict with train/val/test identity indices
        W: Dict with train/val/test view values (angles if use_angle_encoding=True, else indices)
    """
    
    print(f"\n📂 Loading data from: {h5fn}")
    print(f"   Split mode: {view_split_mode}")
    print(f"   Angle encoding: {'✅ ENABLED (using actual angles)' if use_angle_encoding else '❌ DISABLED (using indices)'}")
    
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
    
    print(f"\n🔍 DEBUG: View encoding from HDF5")
    print(f"   Unique Rid values in train: {sorted(set(Rid['train']))}")
    print(f"   uRid (ordered): {uRid}")
    
    table_w = {}
    for _i, _id in enumerate(uRid):
        table_w[_id] = _i
    
    print(f"   View mapping table_w: {table_w}")
    
    W = {}
    for key in keys:
        # Check for any Rid values not in table_w
        missing_views = set(Rid[key]) - set(table_w.keys())
        if missing_views:
            print(f"   ⚠️  WARNING: {key} has views not in train set: {missing_views}")
        
        W[key] = np.array([table_w.get(_id, -1) for _id in Rid[key]])[:, np.newaxis]
        
        # Check for -1 values (missing mappings)
        if -1 in W[key]:
            print(f"   ❌ ERROR: {key} has unmapped view indices!")
            print(f"   Unmapped Rid values: {set(Rid[key][W[key].flatten() == -1])}")
    
    print(f"   W['train'] unique values: {sorted(set(W['train'].flatten()))}")
    print(f"   W['val'] unique values: {sorted(set(W['val'].flatten()))}")
    print(f"   W['test'] unique values: {sorted(set(W['test'].flatten()))}")

    # Convert to float and normalize
    for key in keys:
        Y[key] = Y[key].astype(float) / 255.0
        Y[key] = torch.tensor(Y[key].transpose((0, 3, 1, 2)).astype(np.float32))
        D[key] = torch.tensor(D[key].astype(np.float32))
        
        # Convert view indices to angles if requested
        if use_angle_encoding:
            W[key] = encode_view_angles(W[key], encoding='normalized').reshape(-1, 1)
        else:
            W[key] = torch.tensor(W[key].astype(np.float32))
    
    if use_angle_encoding:
        print(f"\n✅ View encoding applied:")
        print(f"   All view indices converted to normalized angles [-1.0, 1.0]")
        print(f"   This preserves geometric relationships between views")

    # Handle interpolation splitting
    if view_split_mode == 'interpolation':
        assert train_view_indices is not None, "train_view_indices required for view_split_mode='interpolation'"
        assert val_view_indices is not None, "val_view_indices required for view_split_mode='interpolation'"
        
        print(f"\n🔬 Applying interpolation split:")
        print(f"   Train views (boundary): {train_view_indices}")
        print(f"   Val views (intermediate): {val_view_indices}")
        
        # Combine original train + val data
        Y_combined = torch.cat([Y['train'], Y['val']], dim=0)
        D_combined = torch.cat([D['train'], D['val']], dim=0)
        W_combined = torch.cat([W['train'], W['val']], dim=0)
        
        # Split by views using interpolation-specific function
        # NOTE: W is already encoded as angles if use_angle_encoding=True (done above)
        # So we pass w_already_encoded=True to prevent double encoding
        train_data, val_data = split_data_by_views_interpolation(
            Y_combined, D_combined, W_combined,
            train_view_indices, val_view_indices,
            use_angle_encoding=use_angle_encoding,
            w_already_encoded=use_angle_encoding  # W already converted to angles above
        )
        
        # Update dictionaries
        Y['train'] = train_data['Y']
        D['train'] = train_data['D']
        W['train'] = train_data['W']
        
        Y['val'] = val_data['Y']
        D['val'] = val_data['D']
        W['val'] = val_data['W']
        
        # Validation checks
        print("\n✅ Interpolation split final validation:")
        train_views = set(np.unique(W['train'].numpy().flatten()))
        val_views = set(np.unique(W['val'].numpy().flatten()))
        
        if use_angle_encoding:
            # When using angles, check that encoded angles match expected values
            train_angles_expected = set(encode_view_angles(np.array(train_view_indices), encoding='normalized').numpy().round(6))
            val_angles_expected = set(encode_view_angles(np.array(val_view_indices), encoding='normalized').numpy().round(6))
            train_views_rounded = set(np.round(list(train_views), 6))
            val_views_rounded = set(np.round(list(val_views), 6))
            
            assert train_views_rounded == train_angles_expected, f"Train angles mismatch! Got {train_views_rounded}, expected {train_angles_expected}"
            assert val_views_rounded == val_angles_expected, f"Val angles mismatch! Got {val_views_rounded}, expected {val_angles_expected}"
            assert len(train_views_rounded & val_views_rounded) == 0, "Train and val views should not overlap!"
            
            print(f"   ✅ Train angles correct: {sorted(train_views)}")
            print(f"   ✅ Val angles correct: {sorted(val_views)}")
            print(f"   ✅ No overlap between train/val views")
        else:
            # Original index-based validation
            assert train_views == set(train_view_indices), f"Train views mismatch! Got {train_views}, expected {set(train_view_indices)}"
            assert val_views == set(val_view_indices), f"Val views mismatch! Got {val_views}, expected {set(val_view_indices)}"
            assert len(train_views & val_views) == 0, "Train and val views should not overlap!"
            
            print(f"   ✅ Train views correct: {sorted(train_views)}")
            print(f"   ✅ Val views correct: {sorted(val_views)}")
            print(f"   ✅ No overlap between train/val views")
        
        # Print expected performance insight
        print(f"\n💡 Interpolation Task Expectations:")
        if use_angle_encoding:
            print(f"   ✅ Using ACTUAL ANGLE VALUES (geometrically correct)")
            print(f"   - Structured kernels can directly leverage angular smoothness")
            print(f"   - Periodic/VonMises kernels know true distances between views")
            print(f"   - Expected: Smoother interpolation from structured kernels")
            print(f"   - FullRank must learn geometry from data (more parameters)")
        else:
            print(f"   - Using view indices (original behavior)")
            print(f"   - Structured kernels must infer geometry from data")
        print(f"\n   Key metric: How smoothly does each kernel interpolate?")
    
    else:
        # Original random split behavior (no changes needed, already done in data generation)
        print("\n📊 Using original random train/val split from HDF5")
        if use_angle_encoding:
            print(f"   ⚠️  Note: Angle encoding applied to views for geometric correctness")
    
    print(f"\n✅ Final dataset sizes:")
    for key in keys:
        print(f"   {key:5s}: {len(Y[key]):5d} samples")
        if use_angle_encoding and key in ['train', 'val']:
            unique_angles = np.unique(W[key].numpy())
            print(f"           Unique view angles: {len(unique_angles)} {sorted(unique_angles.round(3))}")

    return Y, D, W
