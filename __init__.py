"""Radar datasets and models for continual learning research.

This package provides clean, standalone implementations of radar datasets
and models extracted from the radar continual learning research project.

Datasets (5 + 1 fusion):
- MicroDopplerDataset: (21, 77) micro-Doppler spectrograms (DMM)
- RadarCubeDataset: (21, 5, 5, 25) 4D radar cubes (DRC)
- CI4RDataset: Multi-frequency spectrograms (CI4R)
- RadHARVoxelDataset: (60, 10, 32, 32) 4D voxel data (RadHAR)
- DIATDataset: JPG micro-Doppler images (DIAT)
- FusionDataset: Combined DMM + DRC for multi-modal fusion

Models (2 + 2 fusion):
- ResNet18: Unified pretrained ResNet18 for spectrograms (1 or 3 channels)
- RadarTransformer: Unified transformer for 4D radar data (~11M params)
  - spatial_encoder='linear': For DRC radar cubes (T, V, H, R)
  - spatial_encoder='conv3d': For RadHAR voxels (T, D, H, W)
- RangeAlignedFusionModel: Range-aligned DMM+DRC fusion (~892K params)
- GatedRangeAlignedFusionModel: Fusion with learnable gating

Example usage:
    import sys
    sys.path.insert(0, '../radar-cl')

    from datasets import MicroDopplerDataset, RadarCubeDataset, FusionDataset, DATA_ROOT
    from models import RadarTransformer, ResNet18, RangeAlignedFusionModel

    # ResNet18 for grayscale spectrograms (CI4R, DIAT)
    model_ci4r = ResNet18(num_classes=11, in_channels=1)

    # ResNet18 for RGB colormap images (DMM with image normalization)
    model_dmm = ResNet18(num_classes=4, in_channels=3)

    # RadarTransformer for DRC radar cubes
    model_drc = RadarTransformer(num_classes=4, spatial_encoder='linear')

    # RadarTransformer for RadHAR voxels
    model_radhar = RadarTransformer(num_classes=5, spatial_encoder='conv3d')

    # Fusion model for DMM + DRC
    model_fusion = RangeAlignedFusionModel(num_classes=4)
"""

from .datasets import (
    # Dataset classes
    MicroDopplerDataset,
    RadarCubeDataset,
    CI4RDataset,
    RadHARVoxelDataset,
    RadHARDataset,  # alias for RadHARVoxelDataset
    DIATDataset,
    FusionDataset,
    # Utility functions
    get_files_from_folders,
    parse_filename,
    split_by_participant,
    split_random,
    # Path helpers
    get_mmdrive_folders,
    get_cube_folders,
    DATA_ROOT,
    # DataLoader factories
    create_dataloaders,
    create_ci4r_dataloaders,
    create_radhar_dataloaders,
    create_diat_dataloaders,
    create_fusion_dataloaders,
)

from .models import (
    # Utility
    sinusoidal_pos_encoding,
    # Models
    ResNet18,
    RadarTransformer,
    RangeAlignedFusionModel,
    GatedRangeAlignedFusionModel,
)

__all__ = [
    # Datasets
    'MicroDopplerDataset',
    'RadarCubeDataset',
    'CI4RDataset',
    'RadHARVoxelDataset',
    'RadHARDataset',
    'DIATDataset',
    'FusionDataset',
    # Utilities
    'get_files_from_folders',
    'parse_filename',
    'split_by_participant',
    'split_random',
    'get_mmdrive_folders',
    'get_cube_folders',
    'DATA_ROOT',
    # DataLoader factories
    'create_dataloaders',
    'create_ci4r_dataloaders',
    'create_radhar_dataloaders',
    'create_diat_dataloaders',
    'create_fusion_dataloaders',
    # Models
    'sinusoidal_pos_encoding',
    'ResNet18',
    'RadarTransformer',
    'RangeAlignedFusionModel',
    'GatedRangeAlignedFusionModel',
]
