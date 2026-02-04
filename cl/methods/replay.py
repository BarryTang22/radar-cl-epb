"""Experience Replay Buffer."""
import random

import torch


class BalancedReplayBuffer:
    """Replay buffer with balanced sampling across all seen classes.

    Reference: https://github.com/ContinualAI/avalanche

    Maintains separate buffers for each class and samples equally from all
    classes during replay.

    Args:
        buffer_size: Maximum total size of the buffer (default: 500)
    """

    def __init__(self, buffer_size=500):
        self.buffer_size = buffer_size
        self.class_buffers = {}

    def add_task_data(self, dataloader, classes):
        """Add data from current task, maintaining class balance."""
        class_samples = {c: [] for c in classes}
        for data, labels in dataloader:
            for i in range(len(labels)):
                c = labels[i].item()
                if c in class_samples:
                    class_samples[c].append((data[i].cpu().clone(), c))

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
        """Sample equally from all classes."""
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
        data = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples])
        return data, labels

    def get_batch(self, batch_size):
        """Alias for sample() for compatibility."""
        return self.sample(batch_size)

    def __len__(self):
        return sum(len(b) for b in self.class_buffers.values())
