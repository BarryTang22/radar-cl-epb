"""Test RangeAlignedFusionModel implementation."""

import sys
sys.path.insert(0, '.')

import torch
from datasets import create_fusion_dataloaders
from models import RangeAlignedFusionModel, GatedRangeAlignedFusionModel

print("=" * 60)
print("Testing RangeAlignedFusionModel")
print("=" * 60)

# Create model
model = RangeAlignedFusionModel(num_classes=4)
print(f"\nModel created successfully")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test with synthetic data
print("\n--- Testing with synthetic data ---")
batch_size = 4
dmm = torch.randn(batch_size, 1, 21, 77)  # DMM: (B, 1, Range=21, Doppler=77)
drc = torch.randn(batch_size, 21, 5, 5, 25)  # DRC: (B, T=21, V=5, H=5, R=25)

# Forward pass
logits = model(dmm, drc)
print(f"Input DMM shape: {dmm.shape}")
print(f"Input DRC shape: {drc.shape}")
print(f"Output logits shape: {logits.shape}")  # Expected: (4, 4)

# Feature extraction
features = model.get_features(dmm, drc)
print(f"Features shape: {features.shape}")  # Expected: (4, 128)

# Test freeze/unfreeze
print("\n--- Testing freeze/unfreeze ---")
model.freeze_backbone()
frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params after freeze: {frozen_trainable:,}")

model.unfreeze_backbone()
unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params after unfreeze: {unfrozen_trainable:,}")

# Test GatedRangeAlignedFusionModel
print("\n" + "=" * 60)
print("Testing GatedRangeAlignedFusionModel")
print("=" * 60)

gated_model = GatedRangeAlignedFusionModel(num_classes=4)
gated_params = sum(p.numel() for p in gated_model.parameters())
print(f"Total parameters: {gated_params:,}")

gated_logits = gated_model(dmm, drc)
print(f"Output logits shape: {gated_logits.shape}")

gated_features = gated_model.get_features(dmm, drc)
print(f"Features shape: {gated_features.shape}")

# Test with real dataloader
print("\n" + "=" * 60)
print("Testing with real dataloader (Scene 1)")
print("=" * 60)

train_loader, val_loader, test_loader = create_fusion_dataloaders(scene='1', batch_size=4)
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Get one batch
dmm_batch, drc_batch, labels = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  DMM: {dmm_batch.shape}")
print(f"  DRC: {drc_batch.shape}")
print(f"  Labels: {labels.shape}")

# Forward pass with real data
model.eval()
with torch.no_grad():
    logits = model(dmm_batch, drc_batch)
    features = model.get_features(dmm_batch, drc_batch)

print(f"\nForward pass results:")
print(f"  Logits shape: {logits.shape}")
print(f"  Features shape: {features.shape}")
print(f"  Predicted classes: {logits.argmax(dim=1).tolist()}")
print(f"  True labels: {labels.tolist()}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
