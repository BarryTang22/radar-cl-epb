"""Radar models for continual learning research.

This module provides 2 model architectures:
- ResNet18: Unified ResNet18 for DMM/CI4R/DIAT (1 or 3 channels, pretrained)
- RadarTransformer: Unified transformer for 4D radar data (DRC and RadHAR)
  - spatial_encoder='linear': For DRC radar cubes (T, V, H, R)
  - spatial_encoder='conv3d': For RadHAR voxels (T, D, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pos_encoding(seq_len, embed_dim):
    """Generate sinusoidal positional encoding for variable sequence lengths."""
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(seq_len, embed_dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ResNet18(nn.Module):
    """Unified ResNet18 for radar spectrograms.

    Supports both grayscale (1-channel) and RGB (3-channel) inputs.
    Uses pretrained weights with proper initialization for grayscale.

    Args:
        num_classes: Number of output classes
        in_channels: Input channels (1 for CI4R/DIAT grayscale, 3 for DMM colormap images)
        pretrained: Whether to use pretrained ImageNet weights
    """

    def __init__(self, num_classes, in_channels=1, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.in_channels = in_channels

        if in_channels == 1:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                self.backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def freeze_backbone(self):
        """Freeze all layers except final fc."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_features(self, x):
        """Extract features before classifier (512-dim)."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, prompts=None, adapter=None):
        """Forward pass. prompts and adapter args for CL method compatibility."""
        return self.backbone(x)


class RadarTransformer(nn.Module):
    """Unified transformer for 4D radar data (DRC radar cubes and RadHAR voxels).

    Supports two spatial encoding modes:
    - 'linear': For structured grids like DRC (T, V, H, R) -> linear projection + spatial transformer
    - 'conv3d': For volumetric data like RadHAR (T, D, H, W) -> 3D convolution

    Architecture:
    1. Spatial encoder (configurable: linear+transformer or conv3d)
    2. Temporal transformer with sinusoidal positional encoding
    3. Prompt injection support for continual learning methods
    4. Mean pooling + classifier

    Scaled to ~11M parameters (comparable to ResNet18) with default settings.
    """

    def __init__(self, num_classes, spatial_encoder='linear',
                 input_dim=25, spatial_shape=(5, 5),
                 depth_dim=10, height_dim=32, width_dim=32,
                 embed_dim=384, nhead=6, spatial_layers=2, temporal_layers=4, dropout=0.1):
        """Initialize RadarTransformer.

        Args:
            num_classes: Number of output classes
            spatial_encoder: 'linear' for DRC, 'conv3d' for voxels
            input_dim: Range dimension for DRC (default 25)
            spatial_shape: Angular grid shape for DRC (default (5,5))
            depth_dim: Depth dimension for voxels (default 10)
            height_dim: Height dimension for voxels (default 32)
            width_dim: Width dimension for voxels (default 32)
            embed_dim: Embedding dimension (default 384 for ~11M params)
            nhead: Number of attention heads (default 6)
            spatial_layers: Number of spatial transformer layers (default 2)
            temporal_layers: Number of temporal transformer layers (default 4)
            dropout: Dropout rate (default 0.1)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.spatial_encoder_type = spatial_encoder
        self.spatial_shape = spatial_shape

        if spatial_encoder == 'linear':
            self.range_proj = nn.Linear(input_dim, embed_dim)
            num_spatial_tokens = spatial_shape[0] * spatial_shape[1]
            self.spatial_pos = nn.Parameter(torch.randn(1, num_spatial_tokens, embed_dim) * 0.02)

            spatial_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True
            )
            self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)

        elif spatial_encoder == 'conv3d':
            self.spatial_conv = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
            )

            with torch.no_grad():
                dummy = torch.zeros(1, 1, depth_dim, height_dim, width_dim)
                dummy_out = self.spatial_conv(dummy)
                conv_out_size = dummy_out.view(1, -1).size(1)

            self.spatial_fc = nn.Linear(conv_out_size, embed_dim)
        else:
            raise ValueError(f"Unknown spatial_encoder: {spatial_encoder}")

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes),
        )

    def freeze_backbone(self):
        """Freeze all parameters except classifier."""
        for name, param in self.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_query(self, x):
        """Get query vector for prompt selection (mean of projected tokens)."""
        if self.spatial_encoder_type == 'linear':
            B, T, V, H, R = x.shape
            x = x.view(B, T, V * H, R)
            x = self.range_proj(x)
            return x.mean(dim=(1, 2))
        else:
            B, T, D, H, W = x.shape
            x_first = x[:, 0, :, :, :].unsqueeze(1)
            x_first = self.spatial_conv(x_first)
            x_first = x_first.flatten(1)
            return self.spatial_fc(x_first)

    def _encode_spatial_linear(self, x):
        """Spatial encoding for DRC (linear + transformer)."""
        B, T, V, H, R = x.shape
        N = V * H

        x = x.view(B, T, N, R)
        x = self.range_proj(x)
        x = x.view(B * T, N, self.embed_dim)
        x = x + self.spatial_pos
        x = self.spatial_transformer(x)
        x = x.mean(dim=1)
        x = x.view(B, T, self.embed_dim)
        return x

    def _encode_spatial_conv3d(self, x):
        """Spatial encoding for voxels (3D conv)."""
        B, T, D, H, W = x.shape

        x = x.view(B * T, 1, D, H, W)
        x = self.spatial_conv(x)
        x = x.flatten(1)
        x = self.spatial_fc(x)
        x = x.view(B, T, self.embed_dim)
        return x

    def get_features(self, x, prompts=None, adapter=None):
        """Extract features with optional prompt injection.

        Args:
            x: Input tensor
               - For 'linear': (B, T, V, H, R) radar cube
               - For 'conv3d': (B, T, D, H, W) voxels
            prompts: Optional (B, P, embed_dim) prompts for temporal transformer
            adapter: Optional adapter module for EASE
        """
        if self.spatial_encoder_type == 'linear':
            x = self._encode_spatial_linear(x)
        else:
            x = self._encode_spatial_conv3d(x)

        B, T, _ = x.shape

        temporal_pe = sinusoidal_pos_encoding(T, self.embed_dim).to(x.device)
        x = x + temporal_pe.unsqueeze(0)

        if prompts is not None:
            num_prompts = prompts.size(1)
            x = torch.cat([prompts, x], dim=1)

        x = self.temporal_transformer(x)

        if prompts is not None:
            x = x[:, num_prompts:]

        x = x.mean(dim=1)

        if adapter is not None:
            x = adapter(x)

        return x

    def forward(self, x, prompts=None, adapter=None):
        """Forward pass with optional prompt injection.

        Args:
            x: Input tensor
               - For 'linear': (B, T, V, H, R) radar cube
               - For 'conv3d': (B, T, D, H, W) voxels
            prompts: Optional (B, P, embed_dim) prompts for temporal transformer
            adapter: Optional adapter module for EASE
        """
        features = self.get_features(x, prompts, adapter)
        return self.classifier(features)
