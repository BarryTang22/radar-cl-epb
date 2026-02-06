"""EPB - Elastic Prompt-Backbone for Continual Learning.

A method combining:
1. Hierarchical Elastic Consolidation (HEC) - Layer-wise importance scaling
2. Prompt-Conditioned Fisher (PCF) - Fisher computed with prompt injection
3. Feature Anchoring Loss (FAL) - Preserve feature space structure

Designed for small radar datasets without pretrained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


DEFAULT_LAYER_SCALES = {
    'range_proj': 2.0,
    'spatial_transformer': 1.5,
    'spatial_conv': 1.5,
    'spatial_fc': 1.5,
    'temporal_transformer': 0.5,
    'classifier': 0.3,
}

UNIFORM_LAYER_SCALES = {
    'range_proj': 1.0,
    'spatial_transformer': 1.0,
    'spatial_conv': 1.0,
    'spatial_fc': 1.0,
    'temporal_transformer': 1.0,
    'classifier': 1.0,
}

INVERSE_LAYER_SCALES = {
    'range_proj': 0.3,
    'spatial_transformer': 0.5,
    'spatial_conv': 0.5,
    'spatial_fc': 0.5,
    'temporal_transformer': 1.5,
    'classifier': 2.0,
}


class HierarchicalEWC:
    """Hierarchical Elastic Weight Consolidation.

    Unlike standard EWC (uniform lambda), HEC applies domain-informed
    layer-wise protection. Early layers (feature extraction) are protected
    strongly, while later layers (task-specific) adapt more freely.
    """

    def __init__(self, layer_scales=None, ewc_lambda=500, fisher_ema=0.7):
        self.layer_scales = layer_scales or DEFAULT_LAYER_SCALES
        self.ewc_lambda = ewc_lambda
        self.fisher_ema = fisher_ema
        self.fisher = {}
        self.params = {}
        self.param_scales = {}

    def _get_layer_scale(self, param_name):
        """Get importance scale for a parameter based on its layer."""
        for layer_key, scale in self.layer_scales.items():
            if layer_key in param_name:
                return scale
        return 1.0

    def compute_fisher(self, model, dataloader, device, prompt_method=None):
        """Compute Fisher Information with optional prompt conditioning."""
        was_training = model.training
        model.eval()
        new_fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                      if p.requires_grad}

        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            model.zero_grad()

            if prompt_method is not None and hasattr(model, 'get_query'):
                query = model.get_query(data)
                if hasattr(prompt_method, 'select_prompts'):
                    prompts = prompt_method.select_prompts(query)
                else:
                    prompts = prompt_method.get_prompt(query)
                outputs = model(data, prompts=prompts)
            else:
                outputs = model(data)

            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new_fisher[n] += p.grad.pow(2)

        for n in new_fisher:
            new_fisher[n] /= len(dataloader)

        if self.fisher:
            for n in new_fisher:
                if n in self.fisher:
                    if self.fisher[n].shape == new_fisher[n].shape:
                        self.fisher[n] = (self.fisher_ema * self.fisher[n] +
                                         (1 - self.fisher_ema) * new_fisher[n])
                    else:
                        self.fisher[n] = new_fisher[n]
                else:
                    self.fisher[n] = new_fisher[n]
        else:
            self.fisher = new_fisher

        self.params = {n: p.clone().detach() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.param_scales = {n: self._get_layer_scale(n) for n in self.params}

        if was_training:
            model.train()

    def penalty(self, model):
        """Compute hierarchical EWC penalty."""
        if not self.fisher:
            return 0

        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                stored = self.params[n]
                fisher = self.fisher[n]
                scale = self.param_scales.get(n, 1.0)

                if p.shape != stored.shape:
                    if p.dim() == 1:
                        min_size = min(p.shape[0], stored.shape[0])
                        loss += scale * (fisher[:min_size] *
                                        (p[:min_size] - stored[:min_size]).pow(2)).sum()
                    elif p.dim() == 2:
                        min_out = min(p.shape[0], stored.shape[0])
                        loss += scale * (fisher[:min_out] *
                                        (p[:min_out] - stored[:min_out]).pow(2)).sum()
                else:
                    loss += scale * (fisher * (p - stored).pow(2)).sum()

        return self.ewc_lambda * loss


class FeatureAnchor:
    """Feature Anchoring for continual learning.

    Preserves feature space structure by storing representative anchors
    for each class and penalizing drift from these anchors.
    """

    def __init__(self, num_anchors_per_class=10, margin=0.5, fal_lambda=0.1):
        self.num_anchors = num_anchors_per_class
        self.margin = margin
        self.fal_lambda = fal_lambda
        self.anchors = {}
        self.anchor_labels = {}

    def store_anchors(self, model, dataloader, device, prompt_method=None):
        """Store representative features for each class."""
        was_training = model.training
        model.eval()
        class_features = defaultdict(list)

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)

                if prompt_method is not None and hasattr(model, 'get_query'):
                    query = model.get_query(data)
                    if hasattr(prompt_method, 'select_prompts'):
                        prompts = prompt_method.select_prompts(query)
                    else:
                        prompts = prompt_method.get_prompt(query)
                    features = model.get_features(data, prompts=prompts)
                else:
                    features = model.get_features(data)

                for i, label in enumerate(labels):
                    class_features[label.item()].append(features[i].cpu())

        for class_id, feats in class_features.items():
            feats = torch.stack(feats)
            n_samples = len(feats)
            n_anchors = min(self.num_anchors, n_samples)
            indices = torch.randperm(n_samples)[:n_anchors]
            self.anchors[class_id] = feats[indices]
            self.anchor_labels[class_id] = class_id

        if was_training:
            model.train()

    def anchor_loss(self, model, data, labels, device, prompt_method=None):
        """Compute feature anchoring loss."""
        if not self.anchors:
            return torch.tensor(0.0, device=device)

        if prompt_method is not None and hasattr(model, 'get_query'):
            query = model.get_query(data)
            if hasattr(prompt_method, 'select_prompts'):
                prompts = prompt_method.select_prompts(query)
            else:
                prompts = prompt_method.get_prompt(query)
            features = model.get_features(data, prompts=prompts)
        else:
            features = model.get_features(data)

        loss = torch.tensor(0.0, device=device)
        count = 0

        for class_id, anchors in self.anchors.items():
            mask = (labels == class_id)
            if not mask.any():
                continue
            anchors = anchors.to(device)
            class_features = features[mask]
            dists = torch.cdist(class_features, anchors)
            min_dists = dists.min(dim=1)[0]
            loss = loss + F.relu(min_dists - self.margin).mean()
            count += 1

        return self.fal_lambda * loss / max(count, 1)

    def get_num_anchored_classes(self):
        """Return number of classes with stored anchors."""
        return len(self.anchors)


class EPB:
    """Elastic Prompt-Backbone for Continual Learning.

    Combines:
    - HEC: Hierarchical EWC with layer-wise importance
    - PCF: Prompt-conditioned Fisher computation
    - FAL: Feature anchoring loss
    - Optional replay buffer for stronger forgetting prevention

    Args:
        model: Neural network model
        prompt_method: L2P or CODAPrompt instance
        embed_dim: Feature embedding dimension
        use_hec: Enable hierarchical EWC (vs uniform)
        layer_scales: Dict mapping layer names to importance scales
        ewc_lambda: EWC penalty strength
        fisher_ema: EMA coefficient for Fisher accumulation
        use_pcf: Compute Fisher with prompt conditioning
        use_fal: Enable feature anchoring loss
        num_anchors_per_class: Number of anchor features per class
        anchor_margin: Hinge loss margin for anchors
        fal_lambda: Feature anchor loss weight
        use_replay: Enable replay buffer for experience replay
        replay_buffer_size: Maximum size of replay buffer
    """

    def __init__(
        self,
        model,
        prompt_method,
        embed_dim=128,
        use_hec=True,
        layer_scales=None,
        ewc_lambda=500,
        fisher_ema=0.7,
        use_pcf=True,
        use_fal=False,
        num_anchors_per_class=10,
        anchor_margin=0.5,
        fal_lambda=0.1,
        use_replay=False,
        replay_buffer_size=500,
    ):
        self.model = model
        self.prompt_method = prompt_method
        self.embed_dim = embed_dim

        self.use_hec = use_hec
        self.use_pcf = use_pcf
        self.use_fal = use_fal

        if use_hec:
            scales = layer_scales or DEFAULT_LAYER_SCALES
        else:
            scales = UNIFORM_LAYER_SCALES
        self.hec = HierarchicalEWC(
            layer_scales=scales,
            ewc_lambda=ewc_lambda,
            fisher_ema=fisher_ema
        )

        self.fal = FeatureAnchor(
            num_anchors_per_class=num_anchors_per_class,
            margin=anchor_margin,
            fal_lambda=fal_lambda
        ) if use_fal else None

        self.use_replay = use_replay
        if use_replay:
            from .replay import BalancedReplayBuffer
            self.replay_buffer = BalancedReplayBuffer(buffer_size=replay_buffer_size)
        else:
            self.replay_buffer = None

        self.task_count = 0

    def compute_losses(self, data, labels, device, query=None):
        """Compute EPB losses during training.

        Args:
            data: Input data tensor
            labels: Ground truth labels
            device: Computation device
            query: Pre-computed query from forward(); avoids redundant get_query call

        Returns dict with:
        - hec_loss: Hierarchical EWC penalty
        - fal_loss: Feature anchoring loss
        - prompt_loss: Prompt pull loss
        """
        losses = {}

        if self.task_count > 0:
            losses['hec_loss'] = self.hec.penalty(self.model)
        else:
            losses['hec_loss'] = torch.tensor(0.0, device=device)

        if self.use_fal and self.fal is not None and self.task_count > 0:
            losses['fal_loss'] = self.fal.anchor_loss(
                self.model, data, labels, device,
                self.prompt_method if self.use_pcf else None
            )
        else:
            losses['fal_loss'] = torch.tensor(0.0, device=device)

        if query is None and hasattr(self.model, 'get_query'):
            query = self.model.get_query(data)

        if query is not None and hasattr(self.prompt_method, 'get_prompt_loss'):
            losses['prompt_loss'] = self.prompt_method.get_prompt_loss(query)
        else:
            losses['prompt_loss'] = torch.tensor(0.0, device=device)

        return losses

    def forward(self, data, device):
        """Forward pass with prompts.

        Returns:
            outputs: Model outputs
            query: Pre-computed query (reusable in compute_losses), or None for CNN models
        """
        if hasattr(self.model, 'get_query'):
            query = self.model.get_query(data)
            if hasattr(self.prompt_method, 'select_prompts'):
                prompts = self.prompt_method.select_prompts(query)
            else:
                prompts = self.prompt_method.get_prompt(query)
            outputs = self.model(data, prompts=prompts)
            return outputs, query
        else:
            outputs = self.model(data)
            return outputs, None

    def consolidate(self, dataloader, device):
        """Consolidate knowledge after training on a task."""
        prompt_for_fisher = self.prompt_method if self.use_pcf else None
        self.hec.compute_fisher(self.model, dataloader, device, prompt_for_fisher)

        if self.use_fal and self.fal is not None:
            prompt_for_anchors = self.prompt_method if self.use_pcf else None
            self.fal.store_anchors(self.model, dataloader, device, prompt_for_anchors)

        if hasattr(self.prompt_method, 'update_task'):
            self.prompt_method.update_task()

        self.task_count += 1

    def add_to_replay(self, dataloader, task_classes):
        """Add current task data to replay buffer."""
        if self.replay_buffer is not None:
            self.replay_buffer.add_task_data(dataloader, task_classes)

    def get_replay_batch(self, batch_size, device):
        """Sample from replay buffer and forward with prompts.

        Returns:
            Tuple of (outputs, labels, data) or (None, None, None) if empty
        """
        if self.replay_buffer is None or len(self.replay_buffer) == 0:
            return None, None, None

        replay_x, replay_y = self.replay_buffer.sample(batch_size)
        if replay_x is None:
            return None, None, None

        replay_x, replay_y = replay_x.to(device), replay_y.to(device)

        if hasattr(self.model, 'get_query'):
            query = self.model.get_query(replay_x)
            if hasattr(self.prompt_method, 'select_prompts'):
                prompts = self.prompt_method.select_prompts(query)
            else:
                prompts = self.prompt_method.get_prompt(query)
            replay_out = self.model(replay_x, prompts=prompts)
        else:
            replay_out = self.model(replay_x)

        return replay_out, replay_y, replay_x

    def get_stats(self):
        """Return statistics about the method state."""
        stats = {
            'task_count': self.task_count,
            'use_hec': self.use_hec,
            'use_pcf': self.use_pcf,
            'use_fal': self.use_fal,
            'use_replay': self.use_replay,
        }

        if self.use_fal and self.fal is not None:
            stats['anchored_classes'] = self.fal.get_num_anchored_classes()

        if self.use_replay and self.replay_buffer is not None:
            stats['replay_buffer_size'] = len(self.replay_buffer)

        if hasattr(self.prompt_method, 'get_usage_stats'):
            stats.update(self.prompt_method.get_usage_stats())

        return stats
