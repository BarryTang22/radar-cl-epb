"""Base classes and utilities for Continual Learning.

Contains common components used across CL methods:
- IncrementalClassifier: Dynamic classifier that expands for new classes
- TaskTracker: Track which classes belong to which task
- CosineLinear: Prototype-based classifier for EASE
- EASEAdapter: Lightweight adapter for EASE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IncrementalClassifier(nn.Module):
    """Dynamic classifier that expands when new classes are encountered.

    Following Avalanche's IncrementalClassifier implementation.
    Reference: https://github.com/ContinualAI/avalanche

    This classifier automatically expands its output layer when new classes
    are encountered, preserving weights for previously learned classes.
    """

    def __init__(self, in_features, initial_classes=0):
        super().__init__()
        self.in_features = in_features
        self.classifier = nn.Linear(in_features, max(initial_classes, 1))
        self.active_classes = set()
        self.mask_value = -1e9

    def adaptation(self, new_classes):
        """Expand classifier if new classes encountered. Returns classes for this task."""
        new_classes = set(new_classes) if not isinstance(new_classes, set) else new_classes
        unseen = new_classes - self.active_classes

        if unseen:
            all_classes = self.active_classes | new_classes
            new_nclasses = max(all_classes) + 1

            if new_nclasses > self.classifier.out_features:
                old_weight = self.classifier.weight.data
                old_bias = self.classifier.bias.data
                device = old_weight.device

                self.classifier = nn.Linear(self.in_features, new_nclasses).to(device)
                with torch.no_grad():
                    self.classifier.weight.data[:old_weight.size(0)] = old_weight
                    self.classifier.bias.data[:old_bias.size(0)] = old_bias

            self.active_classes = all_classes

        return list(new_classes)

    def forward(self, x, task_classes=None):
        """Forward pass with optional masking of inactive classes."""
        logits = self.classifier(x)
        return logits

    def get_masked_logits(self, logits, active_classes):
        """Mask logits for classes not in active_classes."""
        mask = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
        for c in active_classes:
            if c < mask.size(0):
                mask[c] = False
        return logits.masked_fill(mask.unsqueeze(0), self.mask_value)


class TaskTracker:
    """Track which classes belong to which task for class-incremental learning.

    This class maintains a mapping of task IDs to class sets, enabling:
    - Tracking of all seen classes
    - Retrieval of classes for specific tasks
    - Identification of old classes for distillation
    """

    def __init__(self):
        self.task_classes = {}
        self.seen_classes = set()

    def register_task(self, task_id, classes):
        """Register classes for a task."""
        if isinstance(classes, (list, tuple)):
            classes = set(classes)
        self.task_classes[task_id] = classes
        self.seen_classes.update(classes)

    def get_active_classes(self):
        """All classes seen so far."""
        return sorted(self.seen_classes)

    def get_task_classes(self, task_id):
        """Classes for a specific task."""
        return sorted(self.task_classes.get(task_id, set()))

    def get_old_classes(self, current_task_id):
        """Get classes from all tasks BEFORE the current task (for LwF distillation)."""
        old_classes = set()
        for task_id in range(current_task_id):
            if task_id in self.task_classes:
                old_classes.update(self.task_classes[task_id])
        return sorted(old_classes)

    def is_class_incremental(self):
        """Check if this is a class-incremental scenario (different classes per task)."""
        if len(self.task_classes) < 2:
            return False
        class_sets = list(self.task_classes.values())
        return class_sets[0] != class_sets[1]


class CosineLinear(nn.Module):
    """Prototype-based classifier using cosine similarity.

    Used in EASE for classification based on distance to class prototypes.
    L2-normalizes both features and weights, computes cosine similarity,
    with learnable temperature scaling (sigma).

    Reference: CVPR 2024 EASE paper
    """

    def __init__(self, in_features, num_classes, sigma=True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(num_classes, in_features) * 0.01)
        if sigma:
            self.sigma = nn.Parameter(torch.ones(1) * 10.0)
        else:
            self.register_buffer('sigma', torch.ones(1) * 10.0)
        self.use_sigma = sigma

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        out = F.linear(x_norm, w_norm)
        return self.sigma * out

    def expand_classes(self, new_num_classes, device):
        """Expand weight matrix to accommodate new classes."""
        if new_num_classes <= self.num_classes:
            return
        old_weight = self.weight.data
        new_weight = torch.randn(new_num_classes, self.in_features, device=device) * 0.01
        new_weight[:self.num_classes] = old_weight
        self.weight = nn.Parameter(new_weight)
        self.num_classes = new_num_classes


class EASEAdapter(nn.Module):
    """Lightweight adapter for EASE method.

    Projects features into a task-specific subspace.
    Returns SUBSPACE FEATURES (not residual) for concatenation.

    Reference: CVPR 2024 EASE paper - down-up projection structure
    """

    def __init__(self, embed_dim=128, bottleneck_dim=32):
        super().__init__()
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return self.scale * self.up(F.relu(self.down(x)))
