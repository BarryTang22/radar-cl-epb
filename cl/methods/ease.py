"""EASE - Expandable Subspace Ensemble."""
from collections import defaultdict

import torch
import torch.nn as nn

from ..utils import CosineLinear, EASEAdapter


class EASE(nn.Module):
    """EASE - Expandable Subspace Ensemble for Continual Learning.

    Paper: Zhou et al. "Expandable Subspace Ensemble for Pre-Trained Model-Based
           Class-Incremental Learning" (CVPR 2024)
    Official: https://github.com/sun-hailong/CVPR24-Ease

    Key features:
    - CONCATENATES features from all adapters (not sequential)
    - Uses CosineLinear classifier (prototype-based)
    - Feature reweighting with alpha for previous tasks
    - Growing feature dimension: embed_dim * num_tasks

    Args:
        embed_dim: Backbone embedding dimension
        bottleneck_dim: Adapter bottleneck dimension
        alpha: Reweight factor for previous task features
        use_init_ptm: Include original backbone features
        beta: Scale for backbone features if use_init_ptm
    """

    def __init__(self, embed_dim=128, bottleneck_dim=32, alpha=0.1, use_init_ptm=False, beta=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha
        self.use_init_ptm = use_init_ptm
        self.beta = beta
        self.adapters = nn.ModuleList()
        self.classifier = None
        self.task_id = -1
        self.num_tasks = 0
        self.prototypes = {}

    def to(self, device):
        """Move module to device."""
        return super().to(device)

    def get_output_dim(self):
        """Feature dimension grows with tasks."""
        base = self.embed_dim if self.use_init_ptm else 0
        return base + self.num_tasks * self.embed_dim

    def add_task(self):
        """Add new adapter for new task and freeze old ones."""
        self.task_id += 1
        self.num_tasks += 1
        adapter = EASEAdapter(self.embed_dim, self.bottleneck_dim)
        self.adapters.append(adapter)
        for i, adapt in enumerate(self.adapters[:-1]):
            for p in adapt.parameters():
                p.requires_grad = False

    def create_or_expand_classifier(self, num_classes, device):
        """Create or expand CosineLinear classifier for new classes."""
        output_dim = self.get_output_dim()
        if self.classifier is None:
            self.classifier = CosineLinear(output_dim, num_classes).to(device)
        else:
            if num_classes > self.classifier.num_classes:
                self.classifier.expand_classes(num_classes, device)
            if output_dim > self.classifier.in_features:
                old_weight = self.classifier.weight.data
                new_weight = torch.zeros(self.classifier.num_classes, output_dim, device=device)
                new_weight[:, :old_weight.size(1)] = old_weight
                self.classifier.weight = nn.Parameter(new_weight)
                self.classifier.in_features = output_dim

    def parameters(self):
        """Return trainable parameters (current adapter + classifier)."""
        params = []
        if self.task_id >= 0 and self.task_id < len(self.adapters):
            params.extend(self.adapters[self.task_id].parameters())
        if self.classifier is not None:
            params.extend(self.classifier.parameters())
        return params

    def forward_features(self, x, training=True):
        """CONCATENATE features from all adapters."""
        features_list = []

        if self.use_init_ptm:
            features_list.append(self.beta * x)

        for i, adapter in enumerate(self.adapters):
            adapter_feat = adapter(x)
            if i < self.task_id:
                adapter_feat = self.alpha * adapter_feat
            features_list.append(adapter_feat)

        if not features_list:
            return x

        return torch.cat(features_list, dim=1)

    def forward(self, x, training=True):
        """Full forward pass: features -> classifier."""
        features = self.forward_features(x, training)
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized. Call create_or_expand_classifier first.")
        return self.classifier(features)

    def extract_prototypes(self, dataloader, model, device, task_classes):
        """Extract class mean features as prototypes."""
        self.eval()
        model.eval()

        class_features = defaultdict(list)

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                backbone_features = model.get_features(data)
                concat_features = self.forward_features(backbone_features, training=False)

                for i, label in enumerate(labels):
                    class_features[label.item()].append(concat_features[i].cpu())

        for class_id, feats in class_features.items():
            if feats:
                prototype = torch.stack(feats).mean(dim=0)
                self.prototypes[class_id] = prototype

                if self.classifier is not None and class_id < self.classifier.num_classes:
                    self.classifier.weight.data[class_id] = prototype.to(device)

    def train(self, mode=True):
        """Set training mode."""
        if self.task_id >= 0 and self.task_id < len(self.adapters):
            self.adapters[self.task_id].train(mode)
        if self.classifier is not None:
            self.classifier.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        for adapter in self.adapters:
            adapter.eval()
        if self.classifier is not None:
            self.classifier.eval()
        return self
