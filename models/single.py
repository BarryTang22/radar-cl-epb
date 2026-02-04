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


class PrefixAttention(nn.Module):
    """Multi-head attention with prefix tuning support.

    Implements attention where prompts are prepended to Keys and Values,
    following the DualPrompt paper's prefix tuning approach.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, key_prefix=None, value_prefix=None):
        """Forward with optional K/V prefix prompts.

        Args:
            x: Input tensor (B, seq_len, embed_dim)
            key_prefix: Optional key prefix (B, num_heads, prefix_len, head_dim)
            value_prefix: Optional value prefix (B, num_heads, prefix_len, head_dim)
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if key_prefix is not None:
            k = torch.cat([key_prefix, k], dim=2)
        if value_prefix is not None:
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dim)
        return self.out_proj(out)


class PrefixTransformerLayer(nn.Module):
    """Transformer encoder layer with prefix tuning support."""

    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = PrefixAttention(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_prefix=None, value_prefix=None):
        x = x + self.dropout1(self.self_attn(self.norm1(x), key_prefix, value_prefix))
        x = x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))
        return x


class PrefixTransformerEncoder(nn.Module):
    """Transformer encoder with layer-wise prefix tuning support."""

    def __init__(self, embed_dim, num_heads, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PrefixTransformerLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self, x, key_prefixes=None, value_prefixes=None):
        """Forward with optional layer-wise K/V prefixes.

        Args:
            x: Input (B, seq_len, embed_dim)
            key_prefixes: Optional list of key prefixes per layer, each (B, num_heads, prefix_len, head_dim)
            value_prefixes: Optional list of value prefixes per layer
        """
        for i, layer in enumerate(self.layers):
            k_prefix = key_prefixes[i] if key_prefixes is not None else None
            v_prefix = value_prefixes[i] if value_prefixes is not None else None
            x = layer(x, k_prefix, v_prefix)
        return x


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
        self.backbone.fc = nn.Identity()  # Remove backbone's fc
        self.fc_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(in_features, num_classes)

    def freeze_backbone(self):
        """Freeze all layers except classifier."""
        for name, param in self.named_parameters():
            if 'classifier' not in name:
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
        features = self.get_features(x)
        x = self.fc_dropout(features)
        return self.classifier(x)


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

        self.nhead = nhead
        self.temporal_layers = temporal_layers
        self.temporal_transformer = PrefixTransformerEncoder(
            embed_dim=embed_dim, num_heads=nhead, num_layers=temporal_layers,
            dim_feedforward=embed_dim * 4, dropout=dropout
        )

        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def freeze_backbone(self):
        """Freeze all parameters except classifier."""
        for name, param in self.named_parameters():
            # Only keep classifier trainable, not pre_classifier
            if name.startswith('classifier.'):
                param.requires_grad = True
            else:
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
        """Extract features with optional prefix tuning prompt injection.

        Args:
            x: Input tensor
               - For 'linear': (B, T, V, H, R) radar cube
               - For 'conv3d': (B, T, D, H, W) voxels
            prompts: Optional dict with 'key_prefixes' and 'value_prefixes' for prefix tuning.
                     Each is a list of (B, num_heads, prefix_len, head_dim) tensors per layer.
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
            key_prefixes = prompts.get('key_prefixes')
            value_prefixes = prompts.get('value_prefixes')
            x = self.temporal_transformer(x, key_prefixes, value_prefixes)
        else:
            x = self.temporal_transformer(x)

        x = x.mean(dim=1)

        if adapter is not None:
            x = adapter(x)

        return x

    def forward(self, x, prompts=None, adapter=None):
        """Forward pass with optional prefix tuning prompt injection.

        Args:
            x: Input tensor
               - For 'linear': (B, T, V, H, R) radar cube
               - For 'conv3d': (B, T, D, H, W) voxels
            prompts: Optional dict with 'key_prefixes' and 'value_prefixes' for prefix tuning.
            adapter: Optional adapter module for EASE
        """
        features = self.get_features(x, prompts, adapter)
        features = self.pre_classifier(features)
        return self.classifier(features)
