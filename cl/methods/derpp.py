"""Dark Experience Replay++."""
import random

import torch
import torch.nn.functional as F


class BalancedDERPlusPlus:
    """Dark Experience Replay++ with balanced class sampling.

    Paper: Buzzega et al. "Dark Experience for General Continual Learning" (NeurIPS 2020)
    Official: https://github.com/aimagelab/mammoth

    Stores (data, label, logits) tuples. During replay, combines:
    - Cross-entropy loss on labels (like standard replay)
    - MSE loss on logits (distillation from past model states)

    Args:
        buffer_size: Maximum total size of the buffer (default: 500)
        alpha: Weight for MSE loss on logits (default: 0.5)
        beta: Weight for cross-entropy loss on labels (default: 0.5)
    """

    def __init__(self, buffer_size=500, alpha=0.5, beta=0.5):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.class_buffers = {}

    def add_task_data(self, dataloader, classes, model, device):
        """Add data from current task with logits."""
        model.eval()
        class_samples = {c: [] for c in classes}

        with torch.no_grad():
            for data, labels in dataloader:
                data_dev = data.to(device)
                logits = model(data_dev)
                for i in range(len(labels)):
                    c = labels[i].item()
                    if c in class_samples:
                        class_samples[c].append((data[i].cpu().clone(), c, logits[i].cpu().clone()))

        total_classes = len(self.class_buffers) + len([c for c in classes if c not in self.class_buffers])
        per_class = max(1, self.buffer_size // max(total_classes, 1))

        for c in self.class_buffers:
            if len(self.class_buffers[c]) > per_class:
                self.class_buffers[c] = random.sample(self.class_buffers[c], per_class)

        for c, samples in class_samples.items():
            if samples:
                selected = random.sample(samples, min(len(samples), per_class))
                if c in self.class_buffers:
                    self.class_buffers[c].extend(selected)
                    if len(self.class_buffers[c]) > per_class:
                        self.class_buffers[c] = random.sample(self.class_buffers[c], per_class)
                else:
                    self.class_buffers[c] = selected

    def sample(self, batch_size):
        """Sample with balanced class distribution."""
        if not self.class_buffers:
            return None, None, None

        per_class = max(1, batch_size // len(self.class_buffers))
        samples = []

        for c, buffer in self.class_buffers.items():
            if buffer:
                class_samples = random.choices(buffer, k=min(per_class, len(buffer)))
                samples.extend(class_samples)

        if not samples:
            return None, None, None

        random.shuffle(samples)
        samples = samples[:batch_size]
        data = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples])

        logit_list = [s[2] for s in samples]
        max_classes = max(l.size(0) for l in logit_list)
        padded_logits = []
        for l in logit_list:
            if l.size(0) < max_classes:
                pad = torch.full((max_classes - l.size(0),), float('-inf'))
                l = torch.cat([l, pad])
            padded_logits.append(l)
        logits = torch.stack(padded_logits)
        return data, labels, logits

    def replay_loss(self, model, batch_size, device, active_classes=None):
        """Compute replay loss with symmetric masking."""
        data, labels, old_logits = self.sample(batch_size)
        if data is None:
            return 0

        data = data.to(device)
        labels = labels.to(device)
        old_logits = old_logits.to(device)

        new_logits = model(data)

        valid_mask = ~torch.isinf(old_logits)
        min_classes = min(new_logits.size(1), old_logits.size(1))
        mse_new = new_logits[:, :min_classes]
        mse_old = old_logits[:, :min_classes]
        mask = valid_mask[:, :min_classes]

        if mask.any():
            mse_loss = F.mse_loss(mse_new[mask], mse_old[mask])
        else:
            mse_loss = 0

        if active_classes is not None:
            mask = torch.ones(new_logits.size(-1), dtype=torch.bool, device=device)
            for c in active_classes:
                if c < mask.size(0):
                    mask[c] = False
            new_logits_masked = new_logits.masked_fill(mask.unsqueeze(0), -1e9)
        else:
            new_logits_masked = new_logits

        ce_loss = F.cross_entropy(new_logits_masked, labels)
        return self.alpha * mse_loss + self.beta * ce_loss

    def __len__(self):
        return sum(len(b) for b in self.class_buffers.values())
