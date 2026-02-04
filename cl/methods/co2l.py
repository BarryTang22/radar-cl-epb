"""Contrastive Continual Learning."""
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Co2L:
    """Contrastive Continual Learning.

    Paper: Cha et al. "Co2L: Contrastive Continual Learning" (ICCV 2021)
    Official: https://github.com/chaht01/Co2L

    Combines supervised contrastive learning with instance-wise relation
    distillation to learn discriminative representations.

    Args:
        feature_dim: Dimension of input features from backbone
        proj_dim: Dimension of projected features (default: 128)
        temperature: Temperature for SupCon loss (default: 0.5)
        current_temp: Temperature for current task similarities
        past_temp: Temperature for past task similarities in IRD
        distill_power: Weight for IRD loss (default: 1.0)
    """

    def __init__(self, feature_dim, proj_dim=128, temperature=0.5,
                 current_temp=None, past_temp=None, distill_power=1.0):
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.current_temp = current_temp if current_temp is not None else temperature
        self.past_temp = past_temp if past_temp is not None else temperature
        self.distill_power = distill_power
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.old_model = None
        self.old_projector = None

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

    def supcon_loss(self, features, labels):
        """Supervised contrastive loss.

        Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
        """
        z = F.normalize(self.projector(features), dim=1)
        batch_size = z.size(0)

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = mask.float().fill_diagonal_(0)

        sim = torch.matmul(z, z.T) / self.current_temp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim * (1 - torch.eye(batch_size, device=z.device))

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1)
        loss = -(mask * log_prob).sum(dim=1) / mask_sum
        return loss.mean()

    def ird_loss(self, old_features, new_features):
        """Instance-wise Relation Distillation loss."""
        if self.old_model is None:
            return 0

        with torch.no_grad():
            old_z = F.normalize(self.old_projector(old_features), dim=1)
            old_sim = torch.matmul(old_z, old_z.T) / self.past_temp

        new_z = F.normalize(self.projector(new_features), dim=1)
        new_sim = torch.matmul(new_z, new_z.T) / self.current_temp

        return F.mse_loss(new_sim, old_sim) * self.distill_power

    def classifier_distill_loss(self, old_logits, new_logits, temperature=2.0):
        """Distill old classifier outputs to prevent forgetting."""
        if self.old_model is None:
            return 0

        old_probs = F.softmax(old_logits / temperature, dim=1)
        new_log_probs = F.log_softmax(new_logits / temperature, dim=1)

        min_classes = min(old_logits.size(1), new_logits.size(1))
        loss = F.kl_div(new_log_probs[:, :min_classes],
                        old_probs[:, :min_classes],
                        reduction='batchmean') * (temperature ** 2)
        return loss

    def parameters(self):
        """Return projector parameters for optimizer."""
        return self.projector.parameters()
