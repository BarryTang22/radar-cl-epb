"""Elastic Weight Consolidation."""
import torch
import torch.nn.functional as F


class EWC:
    """Elastic Weight Consolidation.

    Paper: Kirkpatrick et al. "Overcoming catastrophic forgetting" (PNAS 2017)
    Official: https://github.com/ContinualAI/avalanche

    After training on each task, computes the Fisher Information Matrix
    to estimate parameter importance. During training on new tasks,
    adds a penalty proportional to the squared change in important parameters.

    Args:
        model: The neural network model
        dataloader: DataLoader for the current task (used to compute Fisher)
        device: Device to run computations on
        importance: Lambda coefficient for the penalty term (default: 1000)
    """

    def __init__(self, model, dataloader, device, importance=1000):
        self.model = model
        self.device = device
        self.importance = importance
        self.params = {}
        self.fisher = {}
        self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        """Compute Fisher Information Matrix using empirical Fisher approximation."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        for data, labels in dataloader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)

        for n in fisher:
            fisher[n] /= len(dataloader)

        self.fisher = fisher
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self):
        """Compute EWC penalty term.

        Handles size mismatch for class-incremental learning where classifier expands.
        """
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                stored = self.params[n]
                fisher = self.fisher[n]
                if p.shape != stored.shape:
                    if p.dim() == 1:
                        min_size = min(p.shape[0], stored.shape[0])
                        loss += (fisher[:min_size] * (p[:min_size] - stored[:min_size]).pow(2)).sum()
                    elif p.dim() == 2:
                        min_out = min(p.shape[0], stored.shape[0])
                        loss += (fisher[:min_out] * (p[:min_out] - stored[:min_out]).pow(2)).sum()
                    else:
                        continue
                else:
                    loss += (fisher * (p - stored).pow(2)).sum()
        return self.importance * loss
