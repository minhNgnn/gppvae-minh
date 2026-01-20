#!/usr/bin/env python
"""Check COIL-100 data splits for potential issues"""

import h5py
import numpy as np

h5_path = 'data/coil-100/coil100_task1_standard.h5'

with h5py.File(h5_path, 'r') as f:
    print('COIL-100 Task1 Standard - Data Split Analysis')
    print('=' * 70)
    
    # Load all splits
    Did_train = f['Did_train'][:]
    Rid_train = f['Rid_train'][:]
    Did_val = f['Did_val'][:]
    Rid_val = f['Rid_val'][:]
    
    train_objects = set(Did_train)
    val_objects = set(Did_val)
    
    print(f'\nTraining: {len(Did_train)} samples, {len(train_objects)} objects')
    print(f'Validation: {len(Did_val)} samples, {len(val_objects)} objects')
    
    print(f'\nTrain views: {sorted(set(Rid_train))}')
    print(f'Val views: {sorted(set(Rid_val))}')
    print(f'Same views? {set(Rid_train) == set(Rid_val)}')
    
    print(f'\n' + '=' * 70)
    print('CRITICAL CHECK: Object overlap')
    print('=' * 70)
    overlap = train_objects & val_objects
    print(f'Train ∩ Val: {len(overlap)} objects')
    if len(overlap) > 0:
        print(f'  PROBLEM: Objects {sorted(list(overlap))[:10]} appear in BOTH splits!')
    else:
        print(f'  ✅ No overlap - correct split!')
    
    # Check samples per object
    from collections import Counter
    train_counts = Counter(Did_train)
    val_counts = Counter(Did_val)
    
    print(f'\nSamples per object in train: min={min(train_counts.values())}, max={max(train_counts.values())}')
    print(f'Samples per object in val: min={min(val_counts.values())}, max={max(val_counts.values())}')
