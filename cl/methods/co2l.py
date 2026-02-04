"""Contrastive Continual Learning.

Official implementation reference: https://github.com/chaht01/Co2L
Paper: Cha et al. "Co2L: Contrastive Continual Learning" (ICCV 2021)
"""
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Co2L:
    """Contrastive Continual Learning.

    Combines supervised contrastive learning with instance-wise relation
    distillation (IRD) to learn discriminative representations while preserving
    knowledge from previous tasks.

    Key components:
    1. Supervised Contrastive Loss: Learns discriminative features within each task
    2. Instance-wise Relation Distillation (IRD): Preserves pairwise similarity
       structure from old model using KL divergence
    3. Memory Buffer: Stores samples for replay (asymmetric loss handling)

    Args:
        feature_dim: Dimension of input features from backbone
        proj_dim: Dimension of projected features (default: 128)
        temperature: Temperature for SupCon loss (default: 0.07, per official code)
        current_temp: Temperature for current task similarities in IRD (default: 0.2)
        past_temp: Temperature for past task similarities in IRD (default: 0.01)
        distill_power: Weight for IRD loss (default: 1.0)
        buffer_size: Size of replay buffer (default: 500)
    """

    def __init__(self, feature_dim, proj_dim=128, temperature=0.07,
                 current_temp=0.2, past_temp=0.01, distill_power=1.0,
                 supcon_weight=0.1, buffer_size=500):
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.current_temp = current_temp
        self.past_temp = past_temp
        self.distill_power = distill_power
        self.supcon_weight = supcon_weight
        self.buffer_size = buffer_size

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.old_model = None
        self.old_projector = None

        # Memory buffer for replay
        self.class_buffers = {}

    def to(self, device):
        """Move projector (and old projector if exists) to device."""
        self.projector = self.projector.to(device)
        if self.old_projector is not None:
            self.old_projector = self.old_projector.to(device)
        return self

    def update_old_model(self, model):
        """Freeze a copy of current model/projector as distillation target."""
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
        self.old_projector = copy.deepcopy(self.projector)
        self.old_projector.eval()
        for p in self.old_projector.parameters():
            p.requires_grad = False

    def add_to_buffer(self, dataloader, task_classes):
        """Add samples from current task to replay buffer.

        Args:
            dataloader: DataLoader for current task
            task_classes: List of class indices in current task
        """
        class_samples = {c: [] for c in task_classes}

        for data, labels in dataloader:
            for i in range(len(labels)):
                c = labels[i].item()
                if c in class_samples:
                    class_samples[c].append((data[i].cpu().clone(), c))

        total_classes = len(self.class_buffers) + len([c for c in task_classes if c not in self.class_buffers])
        per_class = max(1, self.buffer_size // max(total_classes, 1))

        # Shrink existing buffers
        for c in self.class_buffers:
            if len(self.class_buffers[c]) > per_class:
                self.class_buffers[c] = random.sample(self.class_buffers[c], per_class)

        # Add new samples
        for c, samples in class_samples.items():
            if samples:
                selected = random.sample(samples, min(len(samples), per_class))
                if c in self.class_buffers:
                    self.class_buffers[c].extend(selected)
                    if len(self.class_buffers[c]) > per_class:
                        self.class_buffers[c] = random.sample(self.class_buffers[c], per_class)
                else:
                    self.class_buffers[c] = selected

    def get_replay_batch(self, batch_size, device):
        """Sample replay batch from buffer.

        Returns:
            (data, labels) tensors or (None, None) if buffer is empty
        """
        if not self.class_buffers:
            return None, None

        per_class = max(1, batch_size // len(self.class_buffers))
        samples = []

        for c, buffer in self.class_buffers.items():
            if buffer:
                class_samples = random.choices(buffer, k=min(per_class, len(buffer)))
                samples.extend(class_samples)

        if not samples:
            return None, None

        random.shuffle(samples)
        samples = samples[:batch_size]
        data = torch.stack([s[0] for s in samples]).to(device)
        labels = torch.tensor([s[1] for s in samples]).to(device)
        return data, labels

    def get_combined_batch(self, current_data, current_labels, device):
        """Combine current batch with replay samples.

        Args:
            current_data: Current task data tensor
            current_labels: Current task labels tensor
            device: Device to move tensors to

        Returns:
            combined_data: Combined data tensor
            combined_labels: Combined labels tensor
            current_mask: Boolean mask indicating current task samples (for asymmetric loss)
        """
        replay_data, replay_labels = self.get_replay_batch(current_data.size(0), device)

        if replay_data is None:
            # No replay data, all samples are current task
            mask = torch.ones(current_data.size(0), dtype=torch.bool, device=device)
            return current_data, current_labels, mask

        combined_data = torch.cat([current_data, replay_data], dim=0)
        combined_labels = torch.cat([current_labels, replay_labels], dim=0)

        # Mask: True for current task samples (anchors), False for replay (negatives only)
        current_mask = torch.zeros(combined_data.size(0), dtype=torch.bool, device=device)
        current_mask[:current_data.size(0)] = True

        return combined_data, combined_labels, current_mask

    def supcon_loss(self, features, labels, current_task_mask=None):
        """Supervised Contrastive Loss with replay augmentation.

        Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)

        When current_task_mask is provided:
        - Only current task samples serve as anchors (compute loss for these)
        - Positive pairs include same-class samples from both current and replay
        - Replay samples provide additional negatives (different class samples)

        Args:
            features: Feature tensor (B, D)
            labels: Label tensor (B,)
            current_task_mask: Boolean mask where True = current task sample (anchor)
                               If None, all samples are treated as anchors
        """
        z = F.normalize(self.projector(features), dim=1)
        batch_size = z.size(0)
        device = z.device

        # Label mask: same class pairs
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_mask = label_mask.float()

        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Self-mask (exclude diagonal)
        self_mask = torch.eye(batch_size, device=device)
        exp_sim = torch.exp(sim) * (1 - self_mask)

        # Positive mask: same class, exclude self
        # Allow positive pairs between current and replay samples of same class
        pos_mask = label_mask * (1 - self_mask)

        # Log probability
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Compute loss only for valid anchors
        pos_sum = pos_mask.sum(dim=1)
        pos_sum = torch.clamp(pos_sum, min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / pos_sum

        # If using mask, only compute loss for current task samples (anchors)
        if current_task_mask is not None:
            loss = loss[current_task_mask]

        if loss.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss.mean()

    def ird_loss(self, old_features, new_features):
        """Instance-wise Relation Distillation loss using KL divergence.

        Preserves the pairwise similarity structure from the old model.
        Uses KL divergence between softmax-normalized similarity distributions
        (not MSE as previously implemented).

        Args:
            old_features: Features from old (frozen) model
            new_features: Features from current model
        """
        if self.old_model is None or self.old_projector is None:
            return torch.tensor(0.0, device=new_features.device, requires_grad=True)

        with torch.no_grad():
            old_z = F.normalize(self.old_projector(old_features), dim=1)
            old_sim = torch.matmul(old_z, old_z.T) / self.past_temp
            # Softmax normalization (teacher distribution)
            old_probs = F.softmax(old_sim, dim=1)

        new_z = F.normalize(self.projector(new_features), dim=1)
        new_sim = torch.matmul(new_z, new_z.T) / self.current_temp
        # Log-softmax for student (for KL divergence)
        new_log_probs = F.log_softmax(new_sim, dim=1)

        # KL divergence: KL(old || new) = sum(old * log(old/new))
        loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')

        return loss * self.distill_power

    def parameters(self):
        """Return projector parameters for optimizer."""
        return self.projector.parameters()

    def buffer_size_total(self):
        """Return total number of samples in buffer."""
        return sum(len(b) for b in self.class_buffers.values())
