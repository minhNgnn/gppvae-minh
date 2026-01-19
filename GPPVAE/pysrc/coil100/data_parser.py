"""
COIL-100 Data Parser

Loads processed COIL-100 HDF5 files for GP-VAE training.
Compatible with the training notebook structure.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

# 18 selected angles (every 20°)
SELECTED_ANGLES = list(range(0, 360, 20))  # [0, 20, 40, ..., 340]
N_VIEWS = len(SELECTED_ANGLES)  # 18 views

# Angle mappings
ANGLE_TO_IDX = {angle: idx for idx, angle in enumerate(SELECTED_ANGLES)}
IDX_TO_ANGLE = {idx: angle for angle, idx in ANGLE_TO_IDX.items()}

# View name mapping (angle in degrees as string)
VIEW_NAMES = {idx: f"{angle}°" for idx, angle in enumerate(SELECTED_ANGLES)}


# ============================================================================
# Dataset Class
# ============================================================================

class COIL100Dataset(Dataset):
    """
    PyTorch Dataset for COIL-100.
    
    Args:
        h5_path: Path to HDF5 file (e.g., coil100_task1_standard.h5)
        split: 'train', 'val', or 'test'
        use_angle_encoding: If True, return normalized angle values [-1, 1]
                           If False, return discrete angle indices [0, 17]
    """
    
    def __init__(self, h5_path, split='train', use_angle_encoding=False):
        self.h5_path = h5_path
        self.split = split
        self.use_angle_encoding = use_angle_encoding
        
        # Load data
        with h5py.File(h5_path, 'r') as f:
            self.Y = torch.from_numpy(f[f'Y_{split}'][:]).float()
            self.Did = torch.from_numpy(f[f'Did_{split}'][:]).long()
            self.Rid = torch.from_numpy(f[f'Rid_{split}'][:])
            
            # Convert Rid based on encoding type
            if use_angle_encoding:
                # Convert index to normalized angle [-1, 1]
                # Index 0 -> 0° -> -1.0, Index 17 -> 340° -> ~0.89
                angles = np.array([IDX_TO_ANGLE[int(r)] for r in self.Rid])
                self.Rid = torch.from_numpy(angles / 180.0 - 1.0).float()  # [0, 340] -> [-1, ~0.89]
            else:
                self.Rid = self.Rid.long()
        
        print(f"Loaded COIL-100 {split}: {len(self.Y)} samples")
        print(f"  Y shape: {self.Y.shape}")
        print(f"  Unique objects: {len(torch.unique(self.Did))}")
        print(f"  Angle encoding: {'continuous' if use_angle_encoding else 'discrete'}")
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.Y[idx], self.Did[idx], self.Rid[idx]


# ============================================================================
# Data Loading Functions
# ============================================================================

def get_data(h5_path, use_angle_encoding=False):
    """
    Load all splits from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        use_angle_encoding: Whether to use continuous angle values
    
    Returns:
        Dictionary with datasets for each split
    """
    data = {}
    for split in ['train', 'val', 'test']:
        data[split] = COIL100Dataset(h5_path, split, use_angle_encoding)
    return data


def get_dataloaders(h5_path, batch_size=32, use_angle_encoding=False, num_workers=0):
    """
    Create DataLoaders for all splits.
    
    Args:
        h5_path: Path to HDF5 file
        batch_size: Batch size
        use_angle_encoding: Whether to use continuous angle values
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary with DataLoaders for each split
    """
    data = get_data(h5_path, use_angle_encoding)
    
    loaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, 
                           shuffle=True, num_workers=num_workers),
        'val': DataLoader(data['val'], batch_size=batch_size,
                         shuffle=False, num_workers=num_workers),
        'test': DataLoader(data['test'], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers),
    }
    
    return loaders


def load_coil100_arrays(h5_path, use_angle_encoding=False):
    """
    Load data as numpy/torch arrays (compatible with notebook format).
    
    Args:
        h5_path: Path to HDF5 file
        use_angle_encoding: Whether to use continuous angle values
    
    Returns:
        Yt, Yv, Yte: Image arrays (N, C, H, W)
        Wt, Wv, Wte: View indices or normalized angles
        Dt, Dv, Dte: Object IDs
    """
    with h5py.File(h5_path, 'r') as f:
        # Load images
        Yt = torch.from_numpy(f['Y_train'][:]).float()
        Yv = torch.from_numpy(f['Y_val'][:]).float()
        Yte = torch.from_numpy(f['Y_test'][:]).float()
        
        # Load object IDs
        Dt = torch.from_numpy(f['Did_train'][:]).long()
        Dv = torch.from_numpy(f['Did_val'][:]).long()
        Dte = torch.from_numpy(f['Did_test'][:]).long()
        
        # Load view indices
        Wt_raw = f['Rid_train'][:]
        Wv_raw = f['Rid_val'][:]
        Wte_raw = f['Rid_test'][:]
        
        if use_angle_encoding:
            # Convert to normalized angles [-1, ~0.89]
            Wt = torch.from_numpy(np.array([IDX_TO_ANGLE[int(r)] for r in Wt_raw]) / 180.0 - 1.0).float()
            Wv = torch.from_numpy(np.array([IDX_TO_ANGLE[int(r)] for r in Wv_raw]) / 180.0 - 1.0).float()
            Wte = torch.from_numpy(np.array([IDX_TO_ANGLE[int(r)] for r in Wte_raw]) / 180.0 - 1.0).float()
        else:
            Wt = torch.from_numpy(Wt_raw).long()
            Wv = torch.from_numpy(Wv_raw).long()
            Wte = torch.from_numpy(Wte_raw).long()
    
    print(f"Loaded COIL-100 data:")
    print(f"  Train: {len(Yt)} samples")
    print(f"  Val: {len(Yv)} samples")
    print(f"  Test: {len(Yte)} samples")
    
    return Yt, Yv, Yte, Wt, Wv, Wte, Dt, Dv, Dte


# ============================================================================
# Utility Functions
# ============================================================================

def angle_to_view_name(angle_value, use_angle_encoding=False):
    """
    Convert angle value to human-readable view name.
    
    Args:
        angle_value: Either index (0-17) or normalized value (-1 to ~0.89)
        use_angle_encoding: Whether angle_value is continuous
    
    Returns:
        String like "0°", "20°", "340°", etc.
    """
    if use_angle_encoding:
        # Convert normalized value back to degrees
        angle_deg = int(round((float(angle_value) + 1.0) * 180.0))
        # Round to nearest 20
        angle_deg = round(angle_deg / 20) * 20
        angle_deg = angle_deg % 360
    else:
        # Convert index to degrees
        angle_deg = IDX_TO_ANGLE.get(int(angle_value), int(angle_value) * 20)
    
    return f"{angle_deg}°"


def normalized_angle_to_idx(normalized_angle):
    """Convert normalized angle [-1, ~0.89] to view index [0, 17]."""
    angle_deg = (float(normalized_angle) + 1.0) * 180.0
    angle_deg = round(angle_deg / 20) * 20
    angle_deg = int(angle_deg) % 360
    return ANGLE_TO_IDX.get(angle_deg, 0)


def get_view_angle_mapping():
    """Return mapping from view index to angle in degrees."""
    return IDX_TO_ANGLE.copy()


def get_n_views():
    """Return number of views (18 for COIL-100)."""
    return N_VIEWS


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Test with a task file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Try task 1
    h5_path = os.path.join(project_root, 'data', 'coil-100', 'coil100_task1_standard.h5')
    
    if os.path.exists(h5_path):
        print("Testing data loading...")
        print("=" * 50)
        
        # Test array loading
        Yt, Yv, Yte, Wt, Wv, Wte, Dt, Dv, Dte = load_coil100_arrays(h5_path, use_angle_encoding=False)
        print(f"\nDiscrete mode:")
        print(f"  Wt dtype: {Wt.dtype}, range: [{Wt.min()}, {Wt.max()}]")
        
        Yt, Yv, Yte, Wt, Wv, Wte, Dt, Dv, Dte = load_coil100_arrays(h5_path, use_angle_encoding=True)
        print(f"\nContinuous mode:")
        print(f"  Wt dtype: {Wt.dtype}, range: [{Wt.min():.3f}, {Wt.max():.3f}]")
        
        # Test view name conversion
        print(f"\nView name examples:")
        for w in Wt[:5]:
            print(f"  {w:.3f} -> {angle_to_view_name(w, use_angle_encoding=True)}")
    else:
        print(f"HDF5 file not found: {h5_path}")
        print("Run process_data.py first to create the data files.")
