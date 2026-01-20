"""
COIL-100 Data Parser

Loads processed COIL-100 HDF5 files for GP-VAE training.
Compatible with the training notebook structure.

Angle encoding options:
- use_angle_encoding=False: Return discrete view indices [0, 17]
- use_angle_encoding=True:  Return angles in degrees [0, 360) for kernel compatibility
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
# Object ID Remapping
# ============================================================================

# Global mapping from original object IDs to contiguous indices
# This is built when loading all splits to ensure consistency
_GLOBAL_OBJECT_ID_MAP = None


def build_object_id_map(h5_path):
    """
    Build a mapping from original object IDs (1-100) to contiguous indices (0-P-1).
    This ensures F.embedding doesn't go out of bounds.
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        dict: Mapping from original ID to contiguous index
    """
    global _GLOBAL_OBJECT_ID_MAP
    
    if _GLOBAL_OBJECT_ID_MAP is not None:
        return _GLOBAL_OBJECT_ID_MAP
    
    with h5py.File(h5_path, 'r') as f:
        all_ids = set()
        for split in ['train', 'val', 'test']:
            key = f'Did_{split}'
            if key in f:
                all_ids.update(f[key][:].tolist())
    
    # Create mapping: sorted original IDs -> contiguous indices
    sorted_ids = sorted(all_ids)
    _GLOBAL_OBJECT_ID_MAP = {orig_id: idx for idx, orig_id in enumerate(sorted_ids)}
    
    print(f"Built object ID mapping: {len(_GLOBAL_OBJECT_ID_MAP)} objects -> indices [0, {len(_GLOBAL_OBJECT_ID_MAP)-1}]")
    
    return _GLOBAL_OBJECT_ID_MAP


def get_num_objects(h5_path):
    """Get total number of unique objects across all splits."""
    id_map = build_object_id_map(h5_path)
    return len(id_map)


# ============================================================================
# Dataset Class
# ============================================================================

class COIL100Dataset(Dataset):
    """
    PyTorch Dataset for COIL-100.
    
    Args:
        h5_path: Path to HDF5 file (e.g., coil100_task1_standard.h5)
        split: 'train', 'val', or 'test'
        use_angle_encoding: If True, return angles in degrees [0, 360) for kernels
                           If False, return discrete angle indices [0, 17]
    """
    
    def __init__(self, h5_path, split='train', use_angle_encoding=True):
        self.h5_path = h5_path
        self.split = split
        self.use_angle_encoding = use_angle_encoding
        
        # Build object ID mapping (once per h5 file)
        self.id_map = build_object_id_map(h5_path)
        
        # Load data
        with h5py.File(h5_path, 'r') as f:
            self.Y = torch.from_numpy(f[f'Y_{split}'][:]).float()
            
            # Load and REMAP object IDs to contiguous indices
            original_did = f[f'Did_{split}'][:]
            remapped_did = np.array([self.id_map[int(d)] for d in original_did])
            self.Did = torch.from_numpy(remapped_did).long()
            
            self.Rid = torch.from_numpy(f[f'Rid_{split}'][:])
            
            # Convert Rid based on encoding type
            if use_angle_encoding:
                # Convert index to angle in degrees [0, 360)
                # This is compatible with wrapped_lag_distance in kernels.py
                angles = np.array([IDX_TO_ANGLE[int(r)] for r in self.Rid])
                self.Rid = torch.from_numpy(angles).float()  # [0, 20, 40, ..., 340]
            else:
                self.Rid = self.Rid.long()
        
        print(f"Loaded COIL-100 {split}: {len(self.Y)} samples")
        print(f"  Y shape: {self.Y.shape}")
        print(f"  Unique objects: {len(torch.unique(self.Did))}")
        print(f"  Did range: [{self.Did.min()}, {self.Did.max()}] (remapped to contiguous)")
        if use_angle_encoding:
            print(f"  Angle encoding: degrees [0, 360)")
        else:
            print(f"  Angle encoding: indices [0, 17]")
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        # Return 4 items to match faceplace format: (Y, Did, Rid, index)
        return self.Y[idx], self.Did[idx], self.Rid[idx], idx


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


def load_coil100_arrays(h5_path, use_angle_encoding=True):
    """
    Load data as numpy/torch arrays (compatible with notebook format).
    
    Args:
        h5_path: Path to HDF5 file
        use_angle_encoding: If True, return angles in degrees [0, 360)
                           If False, return indices [0, 17]
    
    Returns:
        Yt, Yv, Yte: Image arrays (N, C, H, W)
        Wt, Wv, Wte: View angles in degrees OR indices
        Dt, Dv, Dte: Object IDs (remapped to contiguous indices!)
    """
    # Build object ID mapping first
    id_map = build_object_id_map(h5_path)
    
    with h5py.File(h5_path, 'r') as f:
        # Load images
        Yt = torch.from_numpy(f['Y_train'][:]).float()
        Yv = torch.from_numpy(f['Y_val'][:]).float()
        Yte = torch.from_numpy(f['Y_test'][:]).float()
        
        # Load object IDs and REMAP to contiguous indices
        Dt_raw = f['Did_train'][:]
        Dv_raw = f['Did_val'][:]
        Dte_raw = f['Did_test'][:]
        
        Dt = torch.from_numpy(np.array([id_map[int(d)] for d in Dt_raw])).long()
        Dv = torch.from_numpy(np.array([id_map[int(d)] for d in Dv_raw])).long()
        Dte = torch.from_numpy(np.array([id_map[int(d)] for d in Dte_raw])).long()
        
        # Load view indices
        Wt_raw = f['Rid_train'][:]
        Wv_raw = f['Rid_val'][:]
        Wte_raw = f['Rid_test'][:]
        
        if use_angle_encoding:
            # Convert to angles in degrees [0, 360) for kernel compatibility
            Wt = torch.from_numpy(np.array([IDX_TO_ANGLE[int(r)] for r in Wt_raw])).float()
            Wv = torch.from_numpy(np.array([IDX_TO_ANGLE[int(r)] for r in Wv_raw])).float()
            Wte = torch.from_numpy(np.array([IDX_TO_ANGLE[int(r)] for r in Wte_raw])).float()
        else:
            Wt = torch.from_numpy(Wt_raw).long()
            Wv = torch.from_numpy(Wv_raw).long()
            Wte = torch.from_numpy(Wte_raw).long()
    
    print(f"Loaded COIL-100 data:")
    print(f"  Train: {len(Yt)} samples")
    print(f"  Val: {len(Yv)} samples")
    print(f"  Test: {len(Yte)} samples")
    print(f"  Objects: {len(id_map)} (remapped to indices [0, {len(id_map)-1}])")
    if use_angle_encoding:
        print(f"  Angles: degrees [0, 360)")
    else:
        print(f"  Angles: indices [0, 17]")
    
    return Yt, Yv, Yte, Wt, Wv, Wte, Dt, Dv, Dte


# ============================================================================
# Utility Functions
# ============================================================================

def angle_to_view_name(angle_value, use_angle_encoding=True):
    """
    Convert angle value to human-readable view name.
    
    Args:
        angle_value: Either index (0-17) or angle in degrees (0-340)
        use_angle_encoding: Whether angle_value is in degrees
    
    Returns:
        String like "0°", "20°", "340°", etc.
    """
    if use_angle_encoding:
        # Already in degrees, just round to nearest 20
        angle_deg = int(round(float(angle_value) / 20) * 20)
        angle_deg = angle_deg % 360
    else:
        # Convert index to degrees
        angle_deg = IDX_TO_ANGLE.get(int(angle_value), int(angle_value) * 20)
    
    return f"{angle_deg}°"


def angle_degrees_to_idx(angle_degrees):
    """Convert angle in degrees [0, 360) to view index [0, 17]."""
    angle_deg = int(round(float(angle_degrees) / 20) * 20) % 360
    return ANGLE_TO_IDX.get(angle_deg, 0)


def idx_to_angle_degrees(idx):
    """Convert view index [0, 17] to angle in degrees [0, 360)."""
    return IDX_TO_ANGLE.get(int(idx), int(idx) * 20)


def get_all_angles_degrees():
    """
    Get all view angles in degrees [0, 360) as a tensor.
    This is the input format expected by kernels.py.
    
    Returns:
        torch.Tensor of shape (18,) with angles [0, 20, 40, ..., 340]
    """
    return torch.tensor(SELECTED_ANGLES, dtype=torch.float32)


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
        
        # Test array loading with indices
        Yt, Yv, Yte, Wt, Wv, Wte, Dt, Dv, Dte = load_coil100_arrays(h5_path, use_angle_encoding=False)
        print(f"\nIndex mode:")
        print(f"  Wt dtype: {Wt.dtype}, range: [{Wt.min()}, {Wt.max()}]")
        
        # Test array loading with degrees (for kernels)
        Yt, Yv, Yte, Wt, Wv, Wte, Dt, Dv, Dte = load_coil100_arrays(h5_path, use_angle_encoding=True)
        print(f"\nDegrees mode (for kernels):")
        print(f"  Wt dtype: {Wt.dtype}, range: [{Wt.min():.0f}, {Wt.max():.0f}]")
        
        # Test view name conversion
        print(f"\nView name examples (degrees mode):")
        for w in Wt[:5]:
            print(f"  {w:.0f}° -> {angle_to_view_name(w, use_angle_encoding=True)}")
        
        # Show unique angles
        unique_angles = torch.unique(Wt).numpy()
        print(f"\nUnique angles in train set: {sorted(unique_angles)}")
    else:
        print(f"HDF5 file not found: {h5_path}")
        print("Run process_data.py first to create the data files.")
