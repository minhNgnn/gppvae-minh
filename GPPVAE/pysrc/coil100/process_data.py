"""
COIL-100 Data Processing Script

Processes COIL-100 dataset for GP-VAE experiments with 3 tasks:
- Task 1 (Standard): Object-based split (70/15/15 objects), all 18 angles
- Task 2 (Interpolation): All 100 objects, train on sparse angles, test on intermediate angles
- Task 3 (Extrapolation): All 100 objects, train on 0-180°, test on 200-340°

COIL-100 Dataset:
- 100 objects × 72 angles (every 5°: 0, 5, 10, ..., 355)
- We downsample to 18 angles (every 20°: 0, 20, 40, ..., 340)
- File naming: obj{1-100}__{angle}.png

Output: HDF5 file with Y/Did/Rid arrays for train/val/test splits
"""

import os
import re
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

# All 72 original angles (every 5°)
ALL_ANGLES = list(range(0, 360, 5))  # [0, 5, 10, ..., 355]

# Downsampled 18 angles (every 20°)
SELECTED_ANGLES = list(range(0, 360, 20))  # [0, 20, 40, ..., 340]

# Number of objects
N_OBJECTS = 100

# Image size after resizing
IMG_SIZE = 128

# Angle to index mapping for 18 angles
ANGLE_TO_IDX = {angle: idx for idx, angle in enumerate(SELECTED_ANGLES)}
IDX_TO_ANGLE = {idx: angle for angle, idx in ANGLE_TO_IDX.items()}


# ============================================================================
# Data Loading
# ============================================================================

def import_data(data_dir, size=IMG_SIZE, angles=SELECTED_ANGLES):
    """
    Load COIL-100 images for specified angles.
    
    Args:
        data_dir: Path to coil-100 folder containing obj{N}__{angle}.png files
        size: Target image size (will resize to size x size)
        angles: List of angles to load (default: every 20°)
    
    Returns:
        Y: Image array of shape (N, C, H, W) normalized to [0, 1]
        Did: Object IDs (1-100)
        Rid: Angle indices (0-17 for 18 angles)
        angles_raw: Raw angle values in degrees
    """
    images = []
    object_ids = []
    angle_indices = []
    angles_raw = []
    
    # Create angle to index mapping
    angle_to_idx = {a: i for i, a in enumerate(angles)}
    
    print(f"Loading COIL-100 images from {data_dir}")
    print(f"Selected angles: {angles} ({len(angles)} angles)")
    
    for obj_id in tqdm(range(1, N_OBJECTS + 1), desc="Loading objects"):
        for angle in angles:
            # File pattern: obj{N}__{angle}.png
            filename = f"obj{obj_id}__{angle}.png"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: Missing file {filename}")
                continue
            
            # Load and resize image
            img = Image.open(filepath).convert('RGB')
            img = img.resize((size, size), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Convert from (H, W, C) to (C, H, W)
            img_array = img_array.transpose(2, 0, 1)
            
            images.append(img_array)
            object_ids.append(obj_id)
            angle_indices.append(angle_to_idx[angle])
            angles_raw.append(angle)
    
    Y = np.stack(images, axis=0)
    Did = np.array(object_ids, dtype=np.int64)
    Rid = np.array(angle_indices, dtype=np.int64)
    angles_raw = np.array(angles_raw, dtype=np.float32)
    
    print(f"Loaded {len(Y)} images")
    print(f"  Shape: {Y.shape}")
    print(f"  Objects: {len(np.unique(Did))}")
    print(f"  Angles: {len(np.unique(Rid))}")
    
    return Y, Did, Rid, angles_raw


# ============================================================================
# Task 1: Standard Object Split
# ============================================================================

def split_task1_standard(Y, Did, Rid, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Task 1: Standard split - split by objects (like FacePlaces person split).
    
    - Train: 70 objects (all 18 angles)
    - Val: 15 objects (all 18 angles)
    - Test: 15 objects (all 18 angles)
    
    Returns dict with train/val/test arrays
    """
    np.random.seed(seed)
    
    # Get unique object IDs and shuffle
    unique_objects = np.unique(Did)
    np.random.shuffle(unique_objects)
    
    n_train = int(len(unique_objects) * train_ratio)
    n_val = int(len(unique_objects) * val_ratio)
    
    train_objects = set(unique_objects[:n_train])
    val_objects = set(unique_objects[n_train:n_train + n_val])
    test_objects = set(unique_objects[n_train + n_val:])
    
    # Create masks
    train_mask = np.array([d in train_objects for d in Did])
    val_mask = np.array([d in val_objects for d in Did])
    test_mask = np.array([d in test_objects for d in Did])
    
    result = {
        'Y_train': Y[train_mask],
        'Y_val': Y[val_mask],
        'Y_test': Y[test_mask],
        'Did_train': Did[train_mask],
        'Did_val': Did[val_mask],
        'Did_test': Did[test_mask],
        'Rid_train': Rid[train_mask],
        'Rid_val': Rid[val_mask],
        'Rid_test': Rid[test_mask],
        'train_objects': list(train_objects),
        'val_objects': list(val_objects),
        'test_objects': list(test_objects),
    }
    
    print(f"\nTask 1 - Standard Split:")
    print(f"  Train: {len(result['Y_train'])} images ({len(train_objects)} objects)")
    print(f"  Val: {len(result['Y_val'])} images ({len(val_objects)} objects)")
    print(f"  Test: {len(result['Y_test'])} images ({len(test_objects)} objects)")
    
    return result


# ============================================================================
# Task 2: Interpolation Split
# ============================================================================

def split_task2_interpolation(Y, Did, Rid, angles_raw, seed=42):
    """
    Task 2: Interpolation - all objects, split by angles.
    
    Train on sparse angles (every 40°): 0, 40, 80, 120, 160, 200, 240, 280, 320 (9 angles)
    Val on intermediate angles: 20, 100, 180, 260, 340 (5 angles)
    Test on remaining intermediate angles: 60, 140, 220, 300 (4 angles)
    
    This tests the model's ability to interpolate between training views.
    """
    np.random.seed(seed)
    
    # Define angle splits
    train_angles = {0, 40, 80, 120, 160, 200, 240, 280, 320}  # 9 angles (every 40°)
    val_angles = {20, 100, 180, 260, 340}  # 5 intermediate angles
    test_angles = {60, 140, 220, 300}  # 4 remaining intermediate angles
    
    # Create masks based on raw angle values
    train_mask = np.array([a in train_angles for a in angles_raw])
    val_mask = np.array([a in val_angles for a in angles_raw])
    test_mask = np.array([a in test_angles for a in angles_raw])
    
    result = {
        'Y_train': Y[train_mask],
        'Y_val': Y[val_mask],
        'Y_test': Y[test_mask],
        'Did_train': Did[train_mask],
        'Did_val': Did[val_mask],
        'Did_test': Did[test_mask],
        'Rid_train': Rid[train_mask],
        'Rid_val': Rid[val_mask],
        'Rid_test': Rid[test_mask],
        'train_angles': sorted(train_angles),
        'val_angles': sorted(val_angles),
        'test_angles': sorted(test_angles),
    }
    
    print(f"\nTask 2 - Interpolation Split:")
    print(f"  Train angles: {sorted(train_angles)} ({len(train_angles)} angles)")
    print(f"  Val angles: {sorted(val_angles)} ({len(val_angles)} angles)")
    print(f"  Test angles: {sorted(test_angles)} ({len(test_angles)} angles)")
    print(f"  Train: {len(result['Y_train'])} images")
    print(f"  Val: {len(result['Y_val'])} images")
    print(f"  Test: {len(result['Y_test'])} images")
    
    return result


# ============================================================================
# Task 3: Extrapolation Split
# ============================================================================

def split_task3_extrapolation(Y, Did, Rid, angles_raw, seed=42):
    """
    Task 3: Extrapolation - all objects, train on front half, test on back half.
    
    Train: 0-180° (angles 0, 20, 40, 60, 80, 100, 120, 140, 160, 180) - 10 angles
    Val: 200, 220° (2 angles closest to training range)
    Test: 240, 260, 280, 300, 320, 340° (6 angles, far from training)
    
    This tests the model's ability to extrapolate to unseen view ranges.
    """
    np.random.seed(seed)
    
    # Define angle splits
    train_angles = {0, 20, 40, 60, 80, 100, 120, 140, 160, 180}  # Front half: 10 angles
    val_angles = {200, 220}  # 2 angles closest to training
    test_angles = {240, 260, 280, 300, 320, 340}  # 6 angles far from training
    
    # Create masks based on raw angle values
    train_mask = np.array([a in train_angles for a in angles_raw])
    val_mask = np.array([a in val_angles for a in angles_raw])
    test_mask = np.array([a in test_angles for a in angles_raw])
    
    result = {
        'Y_train': Y[train_mask],
        'Y_val': Y[val_mask],
        'Y_test': Y[test_mask],
        'Did_train': Did[train_mask],
        'Did_val': Did[val_mask],
        'Did_test': Did[test_mask],
        'Rid_train': Rid[train_mask],
        'Rid_val': Rid[val_mask],
        'Rid_test': Rid[test_mask],
        'train_angles': sorted(train_angles),
        'val_angles': sorted(val_angles),
        'test_angles': sorted(test_angles),
    }
    
    print(f"\nTask 3 - Extrapolation Split:")
    print(f"  Train angles: {sorted(train_angles)} ({len(train_angles)} angles)")
    print(f"  Val angles: {sorted(val_angles)} ({len(val_angles)} angles)")
    print(f"  Test angles: {sorted(test_angles)} ({len(test_angles)} angles)")
    print(f"  Train: {len(result['Y_train'])} images")
    print(f"  Val: {len(result['Y_val'])} images")
    print(f"  Test: {len(result['Y_test'])} images")
    
    return result


# ============================================================================
# Save and Visualize
# ============================================================================

def save_to_hdf5(data, output_path, task_name=""):
    """Save processed data to HDF5 file."""
    print(f"\nSaving {task_name} to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value, compression='gzip')
            elif isinstance(value, (list, set)):
                # Store lists as arrays
                f.create_dataset(key, data=np.array(list(value)))
    
    print(f"  Saved successfully!")


def visualize_split(data, task_name, output_path, n_samples=5):
    """
    Create visualization grid showing samples from train/val/test splits.
    """
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
    
    splits = ['train', 'val', 'test']
    
    for row, split in enumerate(splits):
        Y = data[f'Y_{split}']
        Did = data[f'Did_{split}']
        Rid = data[f'Rid_{split}']
        
        if len(Y) == 0:
            continue
            
        # Sample random indices
        indices = np.random.choice(len(Y), min(n_samples, len(Y)), replace=False)
        
        for col, idx in enumerate(indices):
            img = Y[idx].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            axes[row, col].imshow(img)
            
            # Get angle from Rid index
            angle = IDX_TO_ANGLE.get(Rid[idx], Rid[idx])
            axes[row, col].set_title(f"Obj {Did[idx]}\n{angle}°", fontsize=8)
            axes[row, col].axis('off')
        
        # Set row label
        axes[row, 0].set_ylabel(split.upper(), fontsize=12)
    
    plt.suptitle(f"COIL-100 {task_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {output_path}")


def visualize_angle_distribution(data, task_name, output_path):
    """Visualize which angles are in each split."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    colors = {'train': 'blue', 'val': 'orange', 'test': 'red'}
    
    for split, color in colors.items():
        Rid = data[f'Rid_{split}']
        if len(Rid) == 0:
            continue
        
        # Convert Rid indices to angles
        angles = [IDX_TO_ANGLE.get(r, r * 20) for r in Rid]
        unique_angles = sorted(set(angles))
        
        # Count per angle
        counts = [angles.count(a) for a in unique_angles]
        
        ax.bar([a + (list(colors.keys()).index(split) - 1) * 3 for a in unique_angles],
               counts, width=3, label=split, color=color, alpha=0.7)
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title(f'COIL-100 {task_name} - Angle Distribution')
    ax.set_xticks(SELECTED_ANGLES)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved angle distribution to {output_path}")


def visualize_view_split_by_object(data, task_name, output_path, n_objects=2):
    """
    Create visualization showing 1-2 objects with ALL their views,
    organized by train/val/test split. Best for Tasks 2 and 3 where
    the same objects appear in all splits but with different angles.
    
    Layout:
    - Each row is one object
    - Within each row, images are sorted by angle
    - Color-coded borders: blue=train, orange=val, red=test
    """
    # Collect all data per object
    all_Y = np.concatenate([data['Y_train'], data['Y_val'], data['Y_test']], axis=0)
    all_Did = np.concatenate([data['Did_train'], data['Did_val'], data['Did_test']], axis=0)
    all_Rid = np.concatenate([data['Rid_train'], data['Rid_val'], data['Rid_test']], axis=0)
    
    # Create split labels
    n_train = len(data['Y_train'])
    n_val = len(data['Y_val'])
    n_test = len(data['Y_test'])
    split_labels = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    
    # Get unique objects and select first n_objects
    unique_objects = sorted(set(all_Did))[:n_objects]
    
    # Get the angles for this task
    all_angles = sorted(set([IDX_TO_ANGLE.get(r, r * 20) for r in all_Rid]))
    n_angles = len(all_angles)
    
    # Create figure: n_objects rows × n_angles columns
    fig, axes = plt.subplots(n_objects, n_angles, figsize=(n_angles * 1.2, n_objects * 1.5))
    
    if n_objects == 1:
        axes = axes.reshape(1, -1)
    
    # Color mapping for splits
    split_colors = {'train': '#3498db', 'val': '#f39c12', 'test': '#e74c3c'}
    split_names = {'train': 'Train', 'val': 'Val', 'test': 'Test'}
    
    for obj_row, obj_id in enumerate(unique_objects):
        for angle_col, angle in enumerate(all_angles):
            ax = axes[obj_row, angle_col]
            
            # Find the image for this object and angle
            target_rid = ANGLE_TO_IDX.get(angle, angle // 20)
            mask = (all_Did == obj_id) & (all_Rid == target_rid)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                idx = indices[0]
                img = all_Y[idx].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                split = split_labels[idx]
                
                # Add colored border based on split
                border_color = split_colors[split]
                
                # Create bordered image
                ax.imshow(img)
                
                # Add thick colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(4)
                
                # Add split label on first column only
                if angle_col == 0:
                    ax.set_ylabel(f"Obj {obj_id}", fontsize=10)
            else:
                # No image for this angle (shouldn't happen normally)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(2)
            
            # Show angle on top row
            if obj_row == 0:
                ax.set_title(f"{angle}°", fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='white', edgecolor=color, linewidth=3, 
                             label=f"{split_names[split]} ({len(data[f'Y_{split}']) // len(unique_objects)} per obj)")
                       for split, color in split_colors.items()]
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    
    plt.suptitle(f"COIL-100 {task_name}\nView Split by Object", fontsize=12, y=1.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved view split visualization to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    data_dir = os.path.join(project_root, 'data', 'coil-100')
    output_dir = os.path.join(project_root, 'data', 'coil-100')
    
    print("=" * 60)
    print("COIL-100 Data Processing")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all data with 18 angles
    Y, Did, Rid, angles_raw = import_data(data_dir, size=IMG_SIZE, angles=SELECTED_ANGLES)
    
    # ========================================================================
    # Task 1: Standard Object Split
    # ========================================================================
    task1_data = split_task1_standard(Y, Did, Rid)
    task1_h5_path = os.path.join(output_dir, 'coil100_task1_standard.h5')
    save_to_hdf5(task1_data, task1_h5_path, "Task 1 - Standard")
    
    visualize_split(task1_data, "Task 1 - Standard Split",
                   os.path.join(output_dir, 'task1_samples.png'))
    visualize_angle_distribution(task1_data, "Task 1 - Standard",
                                os.path.join(output_dir, 'task1_angles.png'))
    
    # ========================================================================
    # Task 2: Interpolation Split
    # ========================================================================
    # Define angle splits for Task 2
    task2_train_angles = [0, 40, 80, 120, 160, 200, 240, 280, 320]  # every 40°
    task2_val_angles = [20, 100, 180, 260, 340]  # 5 intermediate angles
    task2_test_angles = [60, 140, 220, 300]  # 4 intermediate angles
    
    task2_data = split_task2_interpolation(Y, Did, Rid, angles_raw)
    task2_h5_path = os.path.join(output_dir, 'coil100_task2_interpolation.h5')
    save_to_hdf5(task2_data, task2_h5_path, "Task 2 - Interpolation")
    
    visualize_split(task2_data, "Task 2 - Interpolation Split",
                   os.path.join(output_dir, 'task2_samples.png'))
    visualize_angle_distribution(task2_data, "Task 2 - Interpolation",
                                os.path.join(output_dir, 'task2_angles.png'))
    visualize_view_split_by_object(task2_data, "Task 2 - Interpolation",
                                   os.path.join(output_dir, 'task2_view_split.png'),
                                   n_objects=2)
    
    # ========================================================================
    # Task 3: Extrapolation Split
    # ========================================================================
    # Define angle splits for Task 3
    task3_train_angles = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]  # front hemisphere
    task3_val_angles = [200, 220]  # 2 nearest back angles
    task3_test_angles = [240, 260, 280, 300, 320, 340]  # far back angles
    
    task3_data = split_task3_extrapolation(Y, Did, Rid, angles_raw)
    task3_h5_path = os.path.join(output_dir, 'coil100_task3_extrapolation.h5')
    save_to_hdf5(task3_data, task3_h5_path, "Task 3 - Extrapolation")
    
    visualize_split(task3_data, "Task 3 - Extrapolation Split",
                   os.path.join(output_dir, 'task3_samples.png'))
    visualize_angle_distribution(task3_data, "Task 3 - Extrapolation",
                                os.path.join(output_dir, 'task3_angles.png'))
    visualize_view_split_by_object(task3_data, "Task 3 - Extrapolation",
                                   os.path.join(output_dir, 'task3_view_split.png'),
                                   n_objects=2)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal images loaded: {len(Y)}")
    print(f"  Objects: {N_OBJECTS}")
    print(f"  Angles per object: {len(SELECTED_ANGLES)}")
    print(f"  Image shape: {Y.shape[1:]}")
    print(f"\nOutput files:")
    print(f"  {task1_h5_path}")
    print(f"  {task2_h5_path}")
    print(f"  {task3_h5_path}")
    print(f"\nVisualization files:")
    print(f"  task1_samples.png, task1_angles.png")
    print(f"  task2_samples.png, task2_angles.png, task2_view_split.png")
    print(f"  task3_samples.png, task3_angles.png, task3_view_split.png")


if __name__ == '__main__':
    main()
