"""Radar datasets for continual learning research.

This module provides dataset classes for 5 radar modalities:
- MicroDopplerDataset: (21, 77) micro-Doppler spectrograms (DMM)
- RadarCubeDataset: (21, 5, 5, 25) 4D radar cubes (DRC)
- CI4RDataset: Multi-frequency spectrograms (CI4R)
- RadHARVoxelDataset: (60, 10, 32, 32) 4D voxel data (RadHAR)
- DIATDataset: JPG micro-Doppler images (DIAT)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image

# Configure this to point to your datasets folder
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'radar', 'datasets')


def get_files_from_folders(folders):
    """Get all .npy files from list of folders."""
    files = []
    for folder in folders:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith('.npy'):
                    files.append(os.path.join(folder, f))
    return files


def parse_filename(filepath):
    """Extract participant_id, label, and timestep from filename.
    Pattern: {participant}_{class}_{c}_{timestep}.npy
    """
    basename = os.path.basename(filepath)
    parts = basename.replace('.npy', '').split('_')
    participant_id = int(parts[0])
    label = int(parts[1]) - 1  # Convert 1-indexed to 0-indexed
    timestep = int(parts[3]) if len(parts) > 3 else 0
    return participant_id, label, timestep


def split_by_participant(files, train_ratio=0.7, val_ratio=0.15):
    """Split files by participant ID for leave-participant-out style split."""
    participant_files = defaultdict(list)
    for f in files:
        pid, _, _ = parse_filename(f)
        participant_files[pid].append(f)

    participants = sorted(participant_files.keys())
    n = len(participants)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pids = participants[:n_train]
    val_pids = participants[n_train:n_train + n_val]
    test_pids = participants[n_train + n_val:]

    train_files = [f for pid in train_pids for f in participant_files[pid]]
    val_files = [f for pid in val_pids for f in participant_files[pid]]
    test_files = [f for pid in test_pids for f in participant_files[pid]]

    return train_files, val_files, test_files


def split_random(files, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Random split of files."""
    np.random.seed(seed)
    indices = np.random.permutation(len(files))
    n_train = int(len(files) * train_ratio)
    n_val = int(len(files) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_files = [files[i] for i in train_indices]
    val_files = [files[i] for i in val_indices]
    test_files = [files[i] for i in test_indices]

    return train_files, val_files, test_files


class MicroDopplerDataset(Dataset):
    """Dataset for micro-Doppler (21, 77) data.

    Args:
        files: List of .npy file paths
        augment: Apply data augmentation
        normalize: Normalization method
            - 'log_zscore': log1p() -> Z-score (default)
            - 'raw_batchnorm': Return raw data (let model BatchNorm handle it)
            - 'minmax': Min-max scale to [0, 1]
            - 'image': Convert to 3-channel colormap image (224x224) for pretrained models
    """

    def __init__(self, files, augment=False, normalize='log_zscore'):
        self.files = files
        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def _augment(self, data):
        """Apply augmentations with 50% probability each."""
        if np.random.random() < 0.5:
            shift = np.random.randint(-5, 6)
            data = np.roll(data, shift, axis=1)
        if np.random.random() < 0.5:
            shift = np.random.randint(-2, 3)
            data = np.roll(data, shift, axis=0)
        if np.random.random() < 0.5:
            data = data + np.random.normal(0, 0.05, data.shape).astype(np.float32)
        return data

    def _to_colormap_image(self, data):
        """Convert 2D data to 3-channel colormap image (224x224) for pretrained models."""
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        data = np.log1p(data)
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 1e-8:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

        cmap = plt.cm.jet
        colored = cmap(data)[:, :, :3].astype(np.float32)
        colored = np.transpose(colored, (2, 0, 1))

        colored_tensor = torch.from_numpy(colored).unsqueeze(0)
        colored_tensor = F.interpolate(colored_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        colored_tensor = colored_tensor.squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        colored_tensor = (colored_tensor - mean) / std

        return colored_tensor

    def __getitem__(self, idx):
        filepath = self.files[idx]
        data = np.load(filepath).astype(np.float32)
        _, label, _ = parse_filename(filepath)

        if self.normalize == 'image':
            data_tensor = self._to_colormap_image(data)
            return data_tensor, label

        if self.normalize == 'log_zscore':
            data = np.log1p(data)
            mean = data.mean()
            std = data.std() + 1e-8
            data = (data - mean) / std
        elif self.normalize == 'minmax':
            data = np.log1p(data)
            data_min, data_max = data.min(), data.max()
            if data_max - data_min > 1e-8:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = np.zeros_like(data)
        elif self.normalize == 'raw_batchnorm':
            pass

        if self.augment:
            data = self._augment(data)

        data = torch.from_numpy(data).unsqueeze(0)
        return data, label


class RadarCubeDataset(Dataset):
    """Dataset for 4D radar cube (T, V, H, R) data with resampling to consistent shape.

    Args:
        files: List of .npy file paths
        augment: Apply data augmentation
        normalize: Normalization method
            - 'log_zscore': log1p() -> Z-score (default)
            - 'raw_batchnorm': Return raw data (let model BatchNorm handle it)
            - 'minmax': Min-max scale to [0, 1]
        target_shape: Target (T, R) shape, default (21, 25)
    """

    def __init__(self, files, augment=False, normalize='log_zscore', target_shape=(21, 25)):
        self.files = files
        self.augment = augment
        self.normalize = normalize
        self.target_T, self.target_R = target_shape

    def __len__(self):
        return len(self.files)

    def _resample(self, data):
        """Resample data to target shape (T, 5, 5, R) using linear interpolation."""
        T, V, H, R = data.shape
        if T == self.target_T and R == self.target_R:
            return data

        from scipy.ndimage import zoom
        zoom_factors = (self.target_T / T, 1, 1, self.target_R / R)
        return zoom(data, zoom_factors, order=1).astype(np.float32)

    def _augment(self, data):
        """Apply augmentations with 50% probability each."""
        if np.random.random() < 0.5:
            shift = np.random.randint(-2, 3)
            data = np.roll(data, shift, axis=-1)
        if np.random.random() < 0.5:
            data = data + np.random.normal(0, 0.05, data.shape).astype(np.float32)
        return data

    def __getitem__(self, idx):
        filepath = self.files[idx]
        data = np.load(filepath).astype(np.float32)
        _, label, _ = parse_filename(filepath)

        data = self._resample(data)

        if self.normalize == 'log_zscore':
            data = np.log1p(data)
            mean = data.mean()
            std = data.std() + 1e-8
            data = (data - mean) / std
        elif self.normalize == 'minmax':
            data = np.log1p(data)
            data_min, data_max = data.min(), data.max()
            if data_max - data_min > 1e-8:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = np.zeros_like(data)
        elif self.normalize == 'raw_batchnorm':
            pass

        if self.augment:
            data = self._augment(data)

        return torch.from_numpy(data), label


class RadHARVoxelDataset(Dataset):
    """Dataset for RadHAR preprocessed voxels (NPZ format).

    Shape: (N, 60, 10, 32, 32) = (Samples, Temporal, Depth, Height, Width)
    Labels: 5 classes ['boxing', 'jack', 'jump', 'squats', 'walk']

    Args:
        root_dir: Path to RadHAR dataset root
        split: 'train' or 'test'
        augment: Whether to apply augmentation
        temporal_subsample: Target temporal frames (None = keep all 60)
        normalize: Normalization method
    """

    ACTIVITIES = ['boxing', 'jack', 'jump', 'squats', 'walk']

    def __init__(self, root_dir, split='train', augment=False, temporal_subsample=None,
                 normalize='log_zscore'):
        self.root_dir = root_dir
        self.split = split.lower()
        self.augment = augment
        self.temporal_subsample = temporal_subsample
        self.normalize = normalize
        self.voxels = None
        self.labels = None
        self._load_npz()

    def _load_npz(self):
        """Load NPZ file with all samples."""
        if self.split == 'test':
            npz_path = os.path.join(self.root_dir, 'Test_Data_voxels.npz')
            if not os.path.exists(npz_path):
                print(f"Warning: RadHAR test file not found: {npz_path}")
                self.voxels = np.array([])
                self.labels = np.array([])
                return
            data = np.load(npz_path, allow_pickle=True)
            self.voxels = data['arr_0'].astype(np.float32)
            labels_str = data['arr_1']
            self.labels = np.array([self.ACTIVITIES.index(l) for l in labels_str])
        else:
            all_voxels, all_labels = [], []
            for activity in self.ACTIVITIES:
                npz_path = os.path.join(self.root_dir, f'Train_Data_voxels_{activity}.npz')
                if os.path.exists(npz_path):
                    data = np.load(npz_path, allow_pickle=True)
                    voxels = data['arr_0'].astype(np.float32)
                    labels = np.full(len(voxels), self.ACTIVITIES.index(activity))
                    all_voxels.append(voxels)
                    all_labels.append(labels)
            if all_voxels:
                self.voxels = np.concatenate(all_voxels, axis=0)
                self.labels = np.concatenate(all_labels, axis=0)
            else:
                print(f"Warning: No RadHAR train files found in {self.root_dir}")
                self.voxels = np.array([])
                self.labels = np.array([])
                return

        print(f"RadHAR {self.split}: Loaded {len(self.voxels)} samples, shape {self.voxels.shape[1:]}")

    def _temporal_subsample(self, voxel):
        """Resample temporal dimension from 60 to target frames."""
        if self.temporal_subsample is None or voxel.shape[0] == self.temporal_subsample:
            return voxel
        indices = np.linspace(0, voxel.shape[0] - 1, self.temporal_subsample).astype(int)
        return voxel[indices]

    def __len__(self):
        return len(self.voxels) if self.voxels is not None and len(self.voxels) > 0 else 0

    def __getitem__(self, idx):
        voxel = self.voxels[idx]
        label = self.labels[idx]

        voxel = self._temporal_subsample(voxel)

        if self.normalize == 'log_zscore':
            voxel = np.log1p(voxel)
            mean, std = voxel.mean(), voxel.std() + 1e-8
            voxel = (voxel - mean) / std
        elif self.normalize == 'minmax':
            voxel = np.log1p(voxel)
            voxel_min, voxel_max = voxel.min(), voxel.max()
            if voxel_max - voxel_min > 1e-8:
                voxel = (voxel - voxel_min) / (voxel_max - voxel_min)
            else:
                voxel = np.zeros_like(voxel)
        elif self.normalize == 'raw_batchnorm':
            pass

        if self.augment:
            if np.random.random() < 0.5:
                voxel = np.flip(voxel, axis=3).copy()
            if np.random.random() < 0.3:
                shift = np.random.randint(-2, 3)
                voxel = np.roll(voxel, shift, axis=0)

        return torch.from_numpy(voxel.copy()), label


# Backward compatibility alias
RadHARDataset = RadHARVoxelDataset


class CI4RDataset(Dataset):
    """Dataset for CI4R multi-frequency radar micro-Doppler data.

    CI4R provides micro-Doppler spectrograms from 3 different radar frequencies:
    - Xethru (10 GHz UWB)
    - 24GHz (24 GHz FMCW)
    - 77GHz (77 GHz FMCW)

    Activities: 11 classes (walking towards/away, pickup, bending, sitting, kneeling,
                crawling, limping, toe-walking, short-step, scissor-gait)
    """

    FREQUENCIES = ['Xethru', '24GHz', '77GHz']
    FREQ_MAP = {'xethru': 'Xethru', '24ghz': '24GHz', '77ghz': '77GHz',
                'Xethru': 'Xethru', '24GHz': '24GHz', '77GHz': '77GHz'}

    ACTIVITY_MAP = {
        'Towards': 0, 'Away': 1, 'Pick': 2, 'Bend': 3,
        'Sit': 4, 'Kneel': 5, 'Crawl': 6, 'Limp': 7,
        'Toes': 8, 'SStep': 9, 'Scissor': 10
    }
    ACTIVITIES = ['Towards', 'Away', 'Pick', 'Bend', 'Sit', 'Kneel',
                  'Crawl', 'Limp', 'Toes', 'SStep', 'Scissor']

    def __init__(self, root_dir, frequency='77GHz', augment=False,
                 target_shape=(128, 128), normalize='log_zscore'):
        self.root_dir = root_dir
        self.frequency = self.FREQ_MAP.get(frequency, frequency)
        self.augment = augment
        self.target_shape = target_shape
        self.normalize = normalize
        self.files = []
        self.labels = []
        self._load_files()

    def _load_files(self):
        freq_dir = os.path.join(self.root_dir, self.frequency)
        if not os.path.exists(freq_dir):
            print(f"Warning: CI4R {self.frequency} directory not found at {freq_dir}")
            return

        for activity in os.listdir(freq_dir):
            activity_dir = os.path.join(freq_dir, activity)
            if not os.path.isdir(activity_dir):
                continue
            if activity not in self.ACTIVITY_MAP:
                continue

            label = self.ACTIVITY_MAP[activity]
            for f in os.listdir(activity_dir):
                if f.endswith(('.npy', '.mat', '.csv', '.png', '.jpg')):
                    self.files.append(os.path.join(activity_dir, f))
                    self.labels.append(label)

        print(f"CI4R {self.frequency}: Loaded {len(self.files)} samples, {len(set(self.labels))} classes")

    def __len__(self):
        return len(self.files)

    def _load_spectrogram(self, filepath):
        """Load spectrogram from various formats."""
        if filepath.endswith('.npy'):
            return np.load(filepath).astype(np.float32)
        elif filepath.endswith('.mat'):
            import scipy.io as sio
            mat = sio.loadmat(filepath)
            key = [k for k in mat.keys() if not k.startswith('__')][0]
            return mat[key].astype(np.float32)
        elif filepath.endswith('.csv'):
            return np.loadtxt(filepath, delimiter=',').astype(np.float32)
        elif filepath.endswith(('.png', '.jpg')):
            img = Image.open(filepath).convert('L')
            return np.array(img, dtype=np.float32)
        return np.zeros(self.target_shape, dtype=np.float32)

    def _resize_spectrogram(self, data):
        """Resize spectrogram preserving aspect ratio, then pad to square."""
        import torch.nn.functional as F
        if data.shape == self.target_shape:
            return data

        h, w = data.shape
        target_h, target_w = self.target_shape

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        data_t = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        data_t = F.interpolate(data_t, size=(new_h, new_w), mode='bilinear', align_corners=True)

        pad_h = target_h - new_h
        pad_w = target_w - new_w
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        data_t = F.pad(data_t, padding, mode='constant', value=0)

        return data_t.squeeze().numpy()

    def __getitem__(self, idx):
        data = self._load_spectrogram(self.files[idx])
        label = self.labels[idx]

        if data.shape != self.target_shape:
            data = self._resize_spectrogram(data)

        data = np.abs(data)
        if self.normalize == 'log_zscore':
            data = np.log1p(data)
            mean, std = data.mean(), data.std() + 1e-8
            data = (data - mean) / std
        elif self.normalize == 'minmax':
            data = np.log1p(data)
            data_min, data_max = data.min(), data.max()
            if data_max - data_min > 1e-8:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = np.zeros_like(data)
        elif self.normalize == 'raw_batchnorm':
            pass

        if self.augment:
            if np.random.random() < 0.5:
                data = np.roll(data, np.random.randint(-3, 4), axis=1)
            if np.random.random() < 0.5:
                data = data + np.random.normal(0, 0.05, data.shape).astype(np.float32)

        return torch.from_numpy(data).unsqueeze(0), label


class DIATDataset(Dataset):
    """DIAT-uRadHAR dataset - JPG micro-Doppler spectrograms for human activity recognition.

    Classes: 6 activities (crawling, jogging, marching, boxing, jumping, throwing)
    Image format: 1400x1050 RGB JPG spectrograms
    """

    CLASSES = {
        'Army crawling': 0,
        'Army jogging': 1,
        'Army marching': 2,
        'Boxing': 3,
        'Jumping with holding a gun': 4,
        'Stone pelting-Grenades throwing': 5
    }

    def __init__(self, files, labels, target_size=224, augment=False, normalize='zscore'):
        self.files = files
        self.labels = labels
        self.target_size = target_size
        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        data = np.array(img, dtype=np.float32)

        data = data.mean(axis=2)
        data = data / 255.0

        if self.normalize == 'zscore':
            mean, std = data.mean(), data.std() + 1e-8
            data = (data - mean) / std
        elif self.normalize == 'minmax':
            data_min, data_max = data.min(), data.max()
            if data_max - data_min > 1e-8:
                data = (data - data_min) / (data_max - data_min)
        elif self.normalize == 'raw_batchnorm':
            pass

        if self.augment:
            if np.random.random() < 0.5:
                data = np.flip(data, axis=1).copy()
            if np.random.random() < 0.5:
                shift = np.random.randint(-5, 6)
                data = np.roll(data, shift, axis=1)
            if np.random.random() < 0.5:
                data = data + np.random.normal(0, 0.05, data.shape).astype(np.float32)

        data = torch.from_numpy(data).unsqueeze(0)
        return data, self.labels[idx]


# =============================================================================
# Path helpers
# =============================================================================

def get_mmdrive_folders(scene):
    """Get folder paths for micro-Doppler data."""
    if scene == 'all':
        return [os.path.join(DATA_ROOT, 'dmm', f'scene{i}') for i in [1, 2, 3]]
    return [os.path.join(DATA_ROOT, 'dmm', f'scene{scene}')]


def get_cube_folders(scene):
    """Get folder paths for radar cube data."""
    if scene == 'all':
        return [os.path.join(DATA_ROOT, 'drc', f'scene{i}') for i in [1, 2, 3]]
    return [os.path.join(DATA_ROOT, 'drc', f'scene{scene}')]


# =============================================================================
# DataLoader factories
# =============================================================================

def create_dataloaders(modality, scene, batch_size=32, split_mode='participant', normalize='log_zscore',
                       train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test dataloaders for specified modality and scene.

    Args:
        modality: 'mmdrive' or 'cube'
        scene: Scene number ('1', '2', '3') or 'all'
        batch_size: Batch size
        split_mode: 'participant' or 'random'
        normalize: Normalization method
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation (default 0.15, test gets the rest)
    """
    if modality == 'mmdrive':
        folders = get_mmdrive_folders(scene)
        files = get_files_from_folders(folders)

        if split_mode == 'random':
            train_files, val_files, test_files = split_random(files, train_ratio, val_ratio)
        else:
            train_files, val_files, test_files = split_by_participant(files, train_ratio, val_ratio)

        train_ds = MicroDopplerDataset(train_files, augment=True, normalize=normalize)
        val_ds = MicroDopplerDataset(val_files, augment=False, normalize=normalize)
        test_ds = MicroDopplerDataset(test_files, augment=False, normalize=normalize)
    else:
        folders = get_cube_folders(scene)
        files = get_files_from_folders(folders)

        if split_mode == 'random':
            train_files, val_files, test_files = split_random(files, train_ratio, val_ratio)
        else:
            train_files, val_files, test_files = split_by_participant(files, train_ratio, val_ratio)

        train_ds = RadarCubeDataset(train_files, augment=True, normalize=normalize)
        val_ds = RadarCubeDataset(val_files, augment=False, normalize=normalize)
        test_ds = RadarCubeDataset(test_files, augment=False, normalize=normalize)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def create_ci4r_dataloaders(root_dir, frequency='77GHz', batch_size=32,
                            target_shape=(128, 128), normalize='log_zscore'):
    """Create dataloaders for CI4R dataset at specified frequency."""
    full_ds = CI4RDataset(root_dir, frequency=frequency, augment=False,
                          target_shape=target_shape, normalize=normalize)

    n = len(full_ds)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test]
    )

    train_ds.dataset.augment = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def create_radhar_dataloaders(root_dir, batch_size=32, temporal_subsample=None, val_ratio=0.15,
                               normalize='log_zscore'):
    """Create train/val/test dataloaders for RadHAR voxel dataset."""
    train_ds = RadHARVoxelDataset(root_dir, split='train', augment=True,
                                   temporal_subsample=temporal_subsample,
                                   normalize=normalize)
    test_ds = RadHARVoxelDataset(root_dir, split='test', augment=False,
                                  temporal_subsample=temporal_subsample,
                                  normalize=normalize)

    if len(train_ds) > 0:
        n_train = len(train_ds)
        n_val = int(n_train * val_ratio)
        n_train = n_train - n_val
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])
    else:
        val_ds = test_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def create_diat_dataloaders(root_dir, batch_size=32, target_size=224, train_ratio=0.8, seed=42,
                             normalize='zscore'):
    """Create train/val/test dataloaders for DIAT-uRadHAR dataset."""
    files = []
    labels = []

    for class_name, class_idx in DIATDataset.CLASSES.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: DIAT class directory not found: {class_dir}")
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(class_dir, f))
                labels.append(class_idx)

    if not files:
        raise ValueError(f"No DIAT images found in {root_dir}")

    print(f"DIAT: Loaded {len(files)} samples across {len(DIATDataset.CLASSES)} classes")

    np.random.seed(seed)
    train_files, train_labels = [], []
    test_files, test_labels = [], []

    for class_idx in range(len(DIATDataset.CLASSES)):
        class_files = [f for f, l in zip(files, labels) if l == class_idx]
        class_labels = [l for l in labels if l == class_idx]

        indices = np.random.permutation(len(class_files))
        n_train = int(len(class_files) * train_ratio)

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_files.extend([class_files[i] for i in train_idx])
        train_labels.extend([class_labels[i] for i in train_idx])
        test_files.extend([class_files[i] for i in test_idx])
        test_labels.extend([class_labels[i] for i in test_idx])

    val_files, val_labels = [], []
    final_train_files, final_train_labels = [], []

    for class_idx in range(len(DIATDataset.CLASSES)):
        class_files = [f for f, l in zip(train_files, train_labels) if l == class_idx]
        class_labels = [l for l in train_labels if l == class_idx]

        indices = np.random.permutation(len(class_files))
        n_train = int(len(class_files) * 0.85)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        final_train_files.extend([class_files[i] for i in train_idx])
        final_train_labels.extend([class_labels[i] for i in train_idx])
        val_files.extend([class_files[i] for i in val_idx])
        val_labels.extend([class_labels[i] for i in val_idx])

    train_ds = DIATDataset(final_train_files, final_train_labels, target_size=target_size,
                            augment=True, normalize=normalize)
    val_ds = DIATDataset(val_files, val_labels, target_size=target_size,
                          augment=False, normalize=normalize)
    test_ds = DIATDataset(test_files, test_labels, target_size=target_size,
                           augment=False, normalize=normalize)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
