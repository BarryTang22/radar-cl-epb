# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Requirement

When you code: 
- do not over create scripts. Firstly think if it is necessary, then think if we already have the similar code and we can merge or extend.
- we should generate clean and concise code without unnecessary complex comments and try-except.

When you modified the project structure:
- update the this CLAUDE.md to keep it the latest version.
- update or add any relevant explannations of objects and functions in this file.

When you got errors:
- summarize the issues and record it in this file to prevent the issue happen again.

## Issues pattern

**Transformer learning rate**: RadarTransformer requires `lr=1e-4`, not `1e-3`. Transformers are more sensitive to learning rate than CNNs - using `1e-3` causes the model to get stuck at random chance accuracy.

## Project Overview

radar-cl is a Python package providing unified implementations of radar datasets and models for continual learning research. It supports 5 radar modalities for human activity and driver status recognition.

## Architecture

### Module Structure
- `datasets/` - Dataset classes and data loaders
  - `__init__.py` - Re-exports all datasets and utilities
  - `single.py` - Single-modality datasets (DMM, DRC, CI4R, RadHAR, DIAT)
  - `fusion.py` - FusionDataset for DMM+DRC multi-modal fusion
- `models/` - Model architectures
  - `__init__.py` - Re-exports all models
  - `single.py` - ResNet18 (2D) and RadarTransformer (4D)
  - `fusion.py` - RangeAlignedFusionModel and GatedRangeAlignedFusionModel
- `__init__.py` - Package exports (imports from datasets/ and models/)
- `cl/` - Continual Learning module
  - `__init__.py` - Exports all CL methods and utilities
  - `utils.py` - IncrementalClassifier, TaskTracker, CosineLinear, EASEAdapter
  - `methods.py` - All 10 CL algorithm implementations
  - `trainer.py` - CLTrainer for unified training
  - `evaluator.py` - CLEvaluator and metrics computation
- `trainings/` - Training scripts
  - `train_plain_classification.py` - Unified training script for all 5 datasets
  - `train_cl.py` - Unified continual learning training script
  - `Training.md` - Ready-to-copy training commands

### Models

**ResNet18**: For 2D spectrograms (DMM, CI4R, DIAT)
- Supports 1-channel (grayscale) or 3-channel (RGB colormap) inputs
- Uses pretrained ImageNet weights with grayscale adaptation
- Methods: `freeze_backbone()`, `unfreeze_backbone()`, `get_features()` (512-dim)

**RadarTransformer** (~11M params): For 4D volumetric radar data
- `spatial_encoder='linear'`: For DRC radar cubes (T, V, H, R)
- `spatial_encoder='conv3d'`: For RadHAR voxels (T, D, H, W)
- Dual encoder: spatial transformer/conv + PrefixTransformerEncoder for temporal
- Uses **prefix tuning** for prompt injection (prompts prepended to K/V in attention)
- Methods: `freeze_backbone()`, `unfreeze_backbone()`, `get_features()`, `get_query()`
- Prompts format: `{'key_prefixes': [...], 'value_prefixes': [...]}` with per-layer lists

**RangeAlignedFusionModel** (~892K params): For fused DMM+DRC data
- Fuses DMM Doppler features with DRC angular tokens at matching range bins
- DMM → CNN → per-range embeddings (B, 25, embed_dim)
- DRC → per-range angular projection (B, T, 25, embed_dim)
- Spatial transformer over range tokens + temporal transformer
- Methods: `freeze_backbone()`, `unfreeze_backbone()`, `get_features()` (128-dim)
- Variant: `GatedRangeAlignedFusionModel` adds learnable per-range gating

### Dataset-Model Mapping

| Dataset | Model | Input Shape | Classes |
|---------|-------|-------------|---------|
| DMM | ResNet18(in_channels=1) | (B, 1, 21, 77) | 4 |
| DMM (image) | ResNet18(in_channels=3) | (B, 3, 224, 224) | 4 |
| DRC | RadarTransformer(spatial_encoder='linear') | (B, 21, 5, 5, 25) | 4 |
| DMM+DRC | RangeAlignedFusionModel | DMM: (B, 1, 21, 77), DRC: (B, 21, 5, 5, 25) | 4 |
| CI4R | ResNet18(in_channels=1) | (B, 1, 128, 128) | 11 |
| RadHAR | RadarTransformer(spatial_encoder='conv3d') | (B, 60, 10, 32, 32) | 5 |
| DIAT | ResNet18(in_channels=1) | (B, 1, 224, 224) | 6 |

## Key Conventions

### Data Path Configuration
`DATA_ROOT` in `datasets/single.py` points to `../../radar/datasets/` - configure this for your environment.

### File Naming Pattern
DMM/DRC files follow: `{participant_id}_{class_label}_{c}_{timestep}.npy`
- Class labels are 1-indexed in filenames, converted to 0-indexed internally

### Normalization Options
- `log_zscore` (default): log1p + Z-score per sample
- `minmax`: Scale to [0, 1]
- `raw_batchnorm`: Let model BatchNorm handle normalization
- `image`: Convert to 3-channel colormap (224x224) for pretrained models

### DataLoader Factories
- `create_dataloaders()`: DMM/DRC with participant or random split
- `create_fusion_dataloaders()`: DMM+DRC fusion (scene 1 only) from `datasets/fusion.py`
- `create_ci4r_dataloaders()`: CI4R multi-frequency data
- `create_radhar_dataloaders()`: RadHAR voxel data
- `create_diat_dataloaders()`: DIAT JPG images

## Continual Learning Module

The `cl/` module provides 10 CL algorithms with a unified interface.

### Algorithms

| Category | Algorithm | Description |
|----------|-----------|-------------|
| Regularization | `ewc` | Elastic Weight Consolidation (Online EWC with Fisher accumulation) |
| Regularization | `lwf` | Learning without Forgetting |
| Replay | `replay` | Experience Replay with balanced sampling |
| Replay | `derpp` | Dark Experience Replay++ |
| Contrastive | `co2l` | Contrastive Continual Learning |
| Adapter | `ease` | Expandable Subspace Ensemble (class-incremental only) |
| Prompt | `l2p` | Learning to Prompt with prefix tuning (transformer only) |
| Prompt | `coda` | CODA-Prompt with prefix tuning (transformer only) |
| Prompt | `dualprompt` | DualPrompt with prefix tuning, G-Prompt for early layers, E-Prompt pool for later layers (transformer only) |

### Usage

```python
from cl import CLTrainer, CLEvaluator, ALGORITHMS

# Initialize trainer
trainer = CLTrainer(model, 'ewc', device, config={'ewc_importance': 1000, 'ewc_decay': 0.9})

# Train on each task
for task_id, (train_loader, val_loader, task_classes) in enumerate(tasks):
    trainer.train_task(task_id, train_loader, val_loader, task_classes, epochs=30)
    trainer.after_task(train_loader, task_classes)
```

### Training Script

```bash
# Run CL benchmark
python trainings/train_cl.py --dataset drc --setting scene --algorithm ewc --epochs 30
python trainings/train_cl.py --dataset ci4r --setting class --algorithm ease --epochs 50
```

## Dependencies

numpy, torch, torchvision, scipy, PIL, matplotlib
