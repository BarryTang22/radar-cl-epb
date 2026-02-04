"""Range-Aligned Fusion Model for DMM + DRC radar data.

This module implements a fusion architecture that aligns DMM Doppler information
with DRC angular tokens at matching range bins.
"""

import torch
import torch.nn as nn

from .single import sinusoidal_pos_encoding


class RangeAlignedFusionModel(nn.Module):
    """Range-aligned fusion: DMM Doppler features fused with DRC angular tokens per range.

    Architecture:
    1. DMM -> CNN -> per-range Doppler embeddings (B, 25, embed_dim)
    2. DRC -> per-range angular projection -> (B, T, 25, embed_dim)
    3. Fuse: Add DMM embeddings to DRC at each range (broadcast across time)
    4. Spatial transformer over 25 range tokens per frame
    5. Temporal transformer over 21 frame embeddings
    6. Mean pool + classifier -> logits

    Args:
        num_classes: Number of output classes
        embed_dim: Embedding dimension for transformers
        nhead: Number of attention heads
        spatial_layers: Number of spatial transformer layers
        temporal_layers: Number of temporal transformer layers
        dropout: Dropout rate
    """

    def __init__(self, num_classes=4, embed_dim=128, nhead=4,
                 spatial_layers=2, temporal_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # DMM Range Encoder: Extract per-range Doppler features
        # Input: (B, 1, 21, 77) where 21=Range, 77=Doppler
        # Output: (B, 25, embed_dim) - one embedding per range bin (resampled to 25)
        self.dmm_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((25, 1)),  # Resample range: 21->25, pool Doppler->1
        )
        self.dmm_proj = nn.Linear(128, embed_dim)

        # DRC Angular Projection: Project 5x5=25 angular grid per range
        # Input reshape: (B, T=21, 5, 5, R=25) -> (B, T, R=25, Angular=25)
        # Output: (B, T, 25, embed_dim)
        self.drc_proj = nn.Linear(25, embed_dim)  # 5x5=25 angular values -> embed_dim

        # Learnable positional encoding for 25 range tokens
        self.range_pos = nn.Parameter(torch.randn(1, 25, embed_dim) * 0.02)

        # Spatial Transformer: attention over 25 range tokens per frame
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)

        # Temporal Transformer: attention over T=21 frame embeddings
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

        # Classifier (CL-compatible structure)
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _encode_dmm(self, dmm):
        """Encode DMM to per-range Doppler embeddings.

        Args:
            dmm: (B, 1, 21, 77) DMM input

        Returns:
            (B, 25, embed_dim) per-range embeddings
        """
        x = self.dmm_encoder(dmm)  # (B, 128, 25, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, 25, 128)
        x = self.dmm_proj(x)  # (B, 25, embed_dim)
        return x

    def _encode_drc(self, drc):
        """Encode DRC to per-range-per-frame embeddings.

        Args:
            drc: (B, T=21, V=5, H=5, R=25) DRC input

        Returns:
            (B, T, 25, embed_dim) per-range-per-frame embeddings
        """
        B, T, V, H, R = drc.shape
        # Reshape: (B, T, V, H, R) -> (B, T, R, V*H)
        # This reorders so we have R range bins, each with 25 angular measurements
        x = drc.permute(0, 1, 4, 2, 3)  # (B, T, R, V, H)
        x = x.reshape(B, T, R, V * H)  # (B, T, 25, 25)
        x = self.drc_proj(x)  # (B, T, 25, embed_dim)
        return x

    def get_features(self, dmm, drc):
        """Extract features (B, embed_dim) before classifier.

        Args:
            dmm: (B, 1, 21, 77) DMM input
            drc: (B, T=21, V=5, H=5, R=25) DRC input

        Returns:
            (B, embed_dim) feature vectors
        """
        # 1. Encode DMM per range
        dmm_emb = self._encode_dmm(dmm)  # (B, 25, embed_dim)

        # 2. Encode DRC per range per frame
        drc_emb = self._encode_drc(drc)  # (B, T, 25, embed_dim)

        # 3. Fuse: add DMM to each frame (broadcast across time)
        combined = drc_emb + dmm_emb.unsqueeze(1)  # (B, T, 25, embed_dim)

        # 4. Spatial transformer per frame
        B, T, R, E = combined.shape
        combined = combined.view(B * T, R, E)  # (B*T, 25, embed_dim)
        combined = combined + self.range_pos
        combined = self.spatial_transformer(combined)  # (B*T, 25, embed_dim)
        frame_emb = combined.mean(dim=1)  # (B*T, embed_dim)
        frame_emb = frame_emb.view(B, T, E)  # (B, T, embed_dim)

        # 5. Temporal transformer
        temporal_pe = sinusoidal_pos_encoding(T, self.embed_dim).to(frame_emb.device)
        frame_emb = frame_emb + temporal_pe.unsqueeze(0)
        frame_emb = self.temporal_transformer(frame_emb)  # (B, T, embed_dim)

        # 6. Mean pool over time
        return frame_emb.mean(dim=1)  # (B, embed_dim)

    def forward(self, dmm, drc):
        """Forward pass returning logits.

        Args:
            dmm: (B, 1, 21, 77) DMM input
            drc: (B, T=21, V=5, H=5, R=25) DRC input

        Returns:
            (B, num_classes) logits
        """
        features = self.get_features(dmm, drc)
        features = self.pre_classifier(features)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze all parameters except classifier."""
        for name, param in self.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class GatedRangeAlignedFusionModel(RangeAlignedFusionModel):
    """Range-aligned fusion with learnable gating for DMM contribution.

    Same as RangeAlignedFusionModel but adds a per-range gate that controls
    how much DMM information is fused with DRC at each range bin.
    """

    def __init__(self, num_classes=4, embed_dim=128, nhead=4,
                 spatial_layers=2, temporal_layers=2, dropout=0.1):
        super().__init__(num_classes, embed_dim, nhead, spatial_layers, temporal_layers, dropout)

        # Per-range gating network
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def get_features(self, dmm, drc):
        """Extract features with gated fusion."""
        # 1. Encode DMM per range
        dmm_emb = self._encode_dmm(dmm)  # (B, 25, embed_dim)

        # 2. Compute per-range gates
        gate = self.gate_net(dmm_emb)  # (B, 25, 1)
        gated_dmm = gate * dmm_emb  # (B, 25, embed_dim)

        # 3. Encode DRC per range per frame
        drc_emb = self._encode_drc(drc)  # (B, T, 25, embed_dim)

        # 4. Fuse: add gated DMM to each frame
        combined = drc_emb + gated_dmm.unsqueeze(1)  # (B, T, 25, embed_dim)

        # 5. Spatial transformer per frame
        B, T, R, E = combined.shape
        combined = combined.view(B * T, R, E)
        combined = combined + self.range_pos
        combined = self.spatial_transformer(combined)
        frame_emb = combined.mean(dim=1).view(B, T, E)

        # 6. Temporal transformer
        temporal_pe = sinusoidal_pos_encoding(T, self.embed_dim).to(frame_emb.device)
        frame_emb = frame_emb + temporal_pe.unsqueeze(0)
        frame_emb = self.temporal_transformer(frame_emb)

        return frame_emb.mean(dim=1)
