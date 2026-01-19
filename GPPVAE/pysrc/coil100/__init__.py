"""
COIL-100 Data Processing and Loading

Modules:
- process_data: Process raw COIL-100 images into HDF5 files for 3 tasks
- data_parser: Load processed data for GP-VAE training

Tasks:
- Task 1 (Standard): Object-based split (70/15/15), all 18 angles
- Task 2 (Interpolation): All objects, train on sparse angles, test on intermediates
- Task 3 (Extrapolation): All objects, train on 0-180°, test on 200-340°
"""

from .data_parser import (
    COIL100Dataset,
    get_data,
    get_dataloaders,
    load_coil100_arrays,
    angle_to_view_name,
    normalized_angle_to_idx,
    get_view_angle_mapping,
    get_n_views,
    SELECTED_ANGLES,
    N_VIEWS,
    VIEW_NAMES,
    ANGLE_TO_IDX,
    IDX_TO_ANGLE,
)

__all__ = [
    'COIL100Dataset',
    'get_data',
    'get_dataloaders', 
    'load_coil100_arrays',
    'angle_to_view_name',
    'normalized_angle_to_idx',
    'get_view_angle_mapping',
    'get_n_views',
    'SELECTED_ANGLES',
    'N_VIEWS',
    'VIEW_NAMES',
    'ANGLE_TO_IDX',
    'IDX_TO_ANGLE',
]
