# radar-cl

## Datasets

### DMM (Micro-Doppler)
```python
from datasets import MicroDopplerDataset, create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders('mmdrive', scene='1')
# Input: (B, 1, 21, 77), Classes: 4
```

### DRC (Radar Cube)
```python
from datasets import RadarCubeDataset, create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders('cube', scene='1')
# Input: (B, 21, 5, 5, 25), Classes: 4
```

### CI4R (Multi-frequency)
```python
from datasets import CI4RDataset, create_ci4r_dataloaders
train_loader, val_loader, test_loader = create_ci4r_dataloaders('datasets/ci4r', frequency='77GHz')
# Input: (B, 1, 128, 128), Classes: 11
```

### RadHAR (Voxels)
```python
from datasets import RadHARVoxelDataset, create_radhar_dataloaders
train_loader, val_loader, test_loader = create_radhar_dataloaders('datasets/radhar')
# Input: (B, 60, 10, 32, 32), Classes: 5
```

### DIAT (JPG Images)
```python
from datasets import DIATDataset, create_diat_dataloaders
train_loader, val_loader, test_loader = create_diat_dataloaders('datasets/diat')
# Input: (B, 1, 224, 224), Classes: 6
```

## Models

### ResNet18 (for 2D spectrograms)
```python
from models import ResNet18

# CI4R / DIAT (grayscale)
model = ResNet18(num_classes=11, in_channels=1)

# DMM with image normalization (RGB colormap)
model = ResNet18(num_classes=4, in_channels=3)
```

### RadarTransformer (for 4D data)
```python
from models import RadarTransformer

# DRC (radar cubes)
model = RadarTransformer(num_classes=4, spatial_encoder='linear', input_dim=25, spatial_shape=(5, 5))

# RadHAR (voxels)
model = RadarTransformer(num_classes=5, spatial_encoder='conv3d', depth_dim=10, height_dim=32, width_dim=32)
```

## Model-Dataset Mapping

| Dataset | Model | Input Shape | Classes |
|---------|-------|-------------|---------|
| DMM | ResNet18(in_channels=1) | (B, 1, 21, 77) | 4 |
| DMM (image) | ResNet18(in_channels=3) | (B, 3, 224, 224) | 4 |
| DRC | RadarTransformer(spatial_encoder='linear') | (B, 21, 5, 5, 25) | 4 |
| CI4R | ResNet18(in_channels=1) | (B, 1, 128, 128) | 11 |
| RadHAR | RadarTransformer(spatial_encoder='conv3d') | (B, 60, 10, 32, 32) | 5 |
| DIAT | ResNet18(in_channels=1) | (B, 1, 224, 224) | 6 |
