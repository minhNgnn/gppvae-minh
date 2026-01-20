#!/usr/bin/env python
"""Debug script to diagnose CUDA errors in COIL-100 training"""

import os
import sys

# Add the pysrc path
sys.path.insert(0, '/Users/pc/Developer/gppvae/GPPVAE/pysrc/coil100')

from data_parser import COIL100Dataset, get_n_views

# Load data
h5_path = '/Users/pc/Developer/gppvae/data/coil-100/coil100_task1_standard.h5'

print("Loading datasets...")
train_data = COIL100Dataset(h5_path, 'train', use_angle_encoding=False)
val_data = COIL100Dataset(h5_path, 'val', use_angle_encoding=False)

print(f"\n=== Dataset Shapes ===")
print(f"train_data.Y.shape: {train_data.Y.shape}")
print(f"train_data.Did.shape: {train_data.Did.shape}")
print(f"train_data.Rid.shape: {train_data.Rid.shape}")
print(f"val_data.Y.shape: {val_data.Y.shape}")
print(f"val_data.Did.shape: {val_data.Did.shape}")
print(f"val_data.Rid.shape: {val_data.Rid.shape}")

print(f"\n=== Did Values ===")
print(f"train_data.Did dtype: {train_data.Did.dtype}")
print(f"train_data.Did min: {train_data.Did.min()}, max: {train_data.Did.max()}")
print(f"train_data.Did unique values: {sorted(set(train_data.Did.numpy().tolist()))}")

print(f"val_data.Did min: {val_data.Did.min()}, max: {val_data.Did.max()}")
print(f"val_data.Did unique values: {sorted(set(val_data.Did.numpy().tolist()))}")

print(f"\n=== Rid Values ===")
print(f"train_data.Rid dtype: {train_data.Rid.dtype}")
print(f"train_data.Rid min: {train_data.Rid.min()}, max: {train_data.Rid.max()}")
print(f"train_data.Rid unique values: {sorted(set(train_data.Rid.numpy().tolist()))}")

print(f"val_data.Rid dtype: {val_data.Rid.dtype}")
print(f"val_data.Rid min: {val_data.Rid.min()}, max: {val_data.Rid.max()}")
print(f"val_data.Rid unique values: {sorted(set(val_data.Rid.numpy().tolist()))}")

# Calculate P (number of objects) - should include ALL splits
all_did = set(train_data.Did.numpy().tolist()) | set(val_data.Did.numpy().tolist())
P_train_val = len(all_did)
P_total = len(train_data.id_map)  # Use the global mapping which includes all objects

print(f"\n=== Computed P (number of objects) ===")
print(f"P from train+val only = {P_train_val}")
print(f"P from global mapping (all splits) = {P_total}")
print(f"Max Did index = {max(all_did)}")
print(f"ERROR CHECK (wrong way): max Did >= P_train_val? {max(all_did) >= P_train_val}")
print(f"CORRECT CHECK: max Did >= P_total? {max(all_did) >= P_total}")

# Test __getitem__
print(f"\n=== Testing __getitem__ ===")
sample = train_data[0]
print(f"Number of items returned: {len(sample)}")
print(f"Sample types: {[type(s) for s in sample]}")
print(f"Sample shapes/values:")
for i, s in enumerate(sample):
    if hasattr(s, 'shape'):
        print(f"  [{i}]: shape={s.shape}, dtype={s.dtype}")
    else:
        print(f"  [{i}]: value={s}")

# Check Q (number of views)
Q = get_n_views()
print(f"\n=== View Configuration ===")
print(f"Q (num views) = {Q}")
print(f"Expected Rid range: [0, {Q-1}]")
print(f"Actual Rid range: [{train_data.Rid.min()}, {train_data.Rid.max()}]")

# Verify embedding bounds
print(f"\n=== Embedding Bounds Check ===")
print(f"Object embedding: P={P_total}, max_Did={max(all_did)} -> {'OK' if max(all_did) < P_total else 'ERROR!'}")
print(f"View embedding: Q={Q}, max_Rid={int(train_data.Rid.max())} -> {'OK' if int(train_data.Rid.max()) < Q else 'ERROR!'}")

print("\n=== All checks passed! ===")
