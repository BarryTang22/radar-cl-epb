"""Fusion dataset combining DMM (micro-Doppler) and DRC (radar cube) data.

Currently only scene1 is supported - scene2/3 have mismatched files between modalities.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .single import DATA_ROOT, parse_filename, get_files_from_folders, split_by_participant, split_random


def get_dmm_folders(scene):
    """Get folder paths for micro-Doppler data."""
    if scene == 'all':
        return [os.path.join(DATA_ROOT, 'dmm', f'scene{i}') for i in [1, 2, 3]]
    return [os.path.join(DATA_ROOT, 'dmm', f'scene{scene}')]


def get_drc_folders(scene):
    """Get folder paths for radar cube data."""
    if scene == 'all':
        return [os.path.join(DATA_ROOT, 'drc', f'scene{i}') for i in [1, 2, 3]]
    return [os.path.join(DATA_ROOT, 'drc', f'scene{scene}')]


class FusionDataset(Dataset):
    """Dataset for fused micro-Doppler + radar cube data.

    Args:
        pairs: List of (dmm_path, drc_path) tuples
        augment: Apply data augmentation
        dmm_normalize: Normalization for DMM data
        drc_normalize: Normalization for DRC data

    Returns:
        (dmm_tensor, drc_tensor, label) where:
        - dmm_tensor: (1, 21, 77) micro-Doppler
        - drc_tensor: (21, 5, 5, 25) radar cube
        - label: class index (0-3)
    """

    def __init__(self, pairs, augment=False, dmm_normalize='log_zscore', drc_normalize='log_zscore'):
        self.pairs = pairs
        self.augment = augment
        self.dmm_normalize = dmm_normalize
        self.drc_normalize = drc_normalize

    def __len__(self):
        return len(self.pairs)

    def _normalize(self, data, method):
        """Apply normalization to data."""
        if method == 'log_zscore':
            data = np.log1p(data)
            mean, std = data.mean(), data.std() + 1e-8
            data = (data - mean) / std
        elif method == 'minmax':
            data = np.log1p(data)
            data_min, data_max = data.min(), data.max()
            if data_max - data_min > 1e-8:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = np.zeros_like(data)
        elif method == 'raw_batchnorm':
            pass
        return data

    def _augment_dmm(self, data):
        """Apply augmentations to micro-Doppler data."""
        if np.random.random() < 0.5:
            shift = np.random.randint(-5, 6)
            data = np.roll(data, shift, axis=1)
        if np.random.random() < 0.5:
            shift = np.random.randint(-2, 3)
            data = np.roll(data, shift, axis=0)
        if np.random.random() < 0.5:
            data = data + np.random.normal(0, 0.05, data.shape).astype(np.float32)
        return data

    def _augment_drc(self, data):
        """Apply augmentations to radar cube data."""
        if np.random.random() < 0.5:
            shift = np.random.randint(-2, 3)
            data = np.roll(data, shift, axis=-1)
        if np.random.random() < 0.5:
            data = data + np.random.normal(0, 0.05, data.shape).astype(np.float32)
        return data

    def __getitem__(self, idx):
        dmm_path, drc_path = self.pairs[idx]
        _, label, _ = parse_filename(dmm_path)

        # Load and process micro-Doppler data
        dmm_data = np.load(dmm_path).astype(np.float32)
        dmm_data = self._normalize(dmm_data, self.dmm_normalize)
        if self.augment:
            dmm_data = self._augment_dmm(dmm_data)
        dmm_tensor = torch.from_numpy(dmm_data).unsqueeze(0)

        # Load and process radar cube data
        drc_data = np.load(drc_path).astype(np.float32)
        drc_data = self._normalize(drc_data, self.drc_normalize)
        if self.augment:
            drc_data = self._augment_drc(drc_data)
        drc_tensor = torch.from_numpy(drc_data)

        return dmm_tensor, drc_tensor, label


def create_fusion_dataloaders(scene='1', batch_size=32, split_mode='participant',
                              dmm_normalize='log_zscore', drc_normalize='log_zscore',
                              seed=42):
    """Create train/val/test dataloaders for fusion data.

    Args:
        scene: Scene number ('1' only currently supported)
        batch_size: Batch size
        split_mode: 'participant' or 'random'
        dmm_normalize: Normalization for DMM ('log_zscore', 'minmax', 'raw_batchnorm')
        drc_normalize: Normalization for DRC ('log_zscore', 'minmax', 'raw_batchnorm')
        seed: Random seed for splits

    Returns:
        train_loader, val_loader, test_loader

    Raises:
        ValueError: If scene != '1' (other scenes have mismatched files)
    """
    if scene != '1':
        raise ValueError(
            f"Fusion is only supported for scene '1' (scene{scene} has mismatched files between DMM and DRC). "
            "Use single-modality datasets for scene2/3."
        )

    dmm_folders = get_dmm_folders(scene)
    drc_folders = get_drc_folders(scene)
    dmm_files = get_files_from_folders(dmm_folders)
    drc_files = get_files_from_folders(drc_folders)

    # Find matching filenames between both modalities
    dmm_dict = {os.path.basename(f): f for f in dmm_files}
    drc_dict = {os.path.basename(f): f for f in drc_files}
    common_names = sorted(set(dmm_dict.keys()) & set(drc_dict.keys()))

    if not common_names:
        raise ValueError(f"No matching files found between DMM and DRC for scene {scene}")

    # Create matched file lists
    matched_dmm = [dmm_dict[n] for n in common_names]

    # Split based on the matched files
    if split_mode == 'random':
        train_dmm, val_dmm, test_dmm = split_random(matched_dmm, seed=seed)
    else:
        train_dmm, val_dmm, test_dmm = split_by_participant(matched_dmm)

    # Get corresponding DRC files for each split
    train_names = {os.path.basename(f) for f in train_dmm}
    val_names = {os.path.basename(f) for f in val_dmm}
    test_names = {os.path.basename(f) for f in test_dmm}

    train_drc = [drc_dict[n] for n in sorted(train_names)]
    val_drc = [drc_dict[n] for n in sorted(val_names)]
    test_drc = [drc_dict[n] for n in sorted(test_names)]

    # Build pairs (DMM and DRC paths must be aligned)
    train_dmm = sorted(train_dmm, key=os.path.basename)
    val_dmm = sorted(val_dmm, key=os.path.basename)
    test_dmm = sorted(test_dmm, key=os.path.basename)

    train_pairs = list(zip(train_dmm, train_drc))
    val_pairs = list(zip(val_dmm, val_drc))
    test_pairs = list(zip(test_dmm, test_drc))

    train_ds = FusionDataset(train_pairs, augment=True,
                             dmm_normalize=dmm_normalize, drc_normalize=drc_normalize)
    val_ds = FusionDataset(val_pairs, augment=False,
                           dmm_normalize=dmm_normalize, drc_normalize=drc_normalize)
    test_ds = FusionDataset(test_pairs, augment=False,
                            dmm_normalize=dmm_normalize, drc_normalize=drc_normalize)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
