"""Learning without Forgetting."""
import copy

import torch
import torch.nn.functional as F


class LwF:
    """Learning without Forgetting.

    Paper: Li & Hoiem "Learning without Forgetting" (TPAMI 2017)
    Official: https://github.com/lizhitwo/LearningWithoutForgetting

    Maintains a frozen copy of the model from the previous task.
    Uses knowledge distillation on old task classes.

    Args:
        model: The neural network model (will be copied and frozen)
        temperature: Temperature for softmax in distillation (default: 2.0)
        alpha: Weight for distillation loss (default: 1.0)
    """

    def __init__(self, model, temperature=2.0, alpha=1.0):
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, new_outputs, inputs, device, old_classes=None):
        """Compute distillation loss using KL divergence.

        Args:
            new_outputs: Logits from current model
            inputs: Input data
            device: Device to use
            old_classes: List of class indices to distill on (classes from previous tasks)

        Returns:
            Distillation loss (scalar tensor)
        """
        with torch.no_grad():
            old_outputs = self.old_model(inputs.to(device))

        if old_classes is not None and len(old_classes) > 0:
            valid_old_classes = [c for c in old_classes if c < old_outputs.size(1)]
            if valid_old_classes:
                old_outputs = old_outputs[:, valid_old_classes]
                new_outputs = new_outputs[:, valid_old_classes]
            else:
                return torch.tensor(0.0, device=device)
        else:
            old_num_classes = old_outputs.size(1)
            if new_outputs.size(1) > old_num_classes:
                new_outputs = new_outputs[:, :old_num_classes]

        old_probs = F.softmax(old_outputs / self.temperature, dim=1)
        new_log_probs = F.log_softmax(new_outputs / self.temperature, dim=1)

        return self.alpha * F.kl_div(new_log_probs, old_probs, reduction='batchmean') * (self.temperature ** 2)
