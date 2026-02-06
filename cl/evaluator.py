"""Continual Learning Evaluation Utilities.

Provides CLEvaluator class and metrics computation for CL experiments.
"""

import numpy as np
import torch
import torch.nn as nn

from .utils import IncrementalClassifier


def get_incremental_classifier(model):
    """Get the IncrementalClassifier from a model."""
    if hasattr(model, 'fc') and isinstance(model.fc, IncrementalClassifier):
        return model.fc
    if hasattr(model, 'head') and isinstance(model.head, IncrementalClassifier):
        return model.head
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'classifier'):
            if isinstance(model.backbone.classifier, IncrementalClassifier):
                return model.backbone.classifier
            if isinstance(model.backbone.classifier, nn.Sequential):
                for layer in model.backbone.classifier:
                    if isinstance(layer, IncrementalClassifier):
                        return layer
        if hasattr(model.backbone, 'fc') and isinstance(model.backbone.fc, IncrementalClassifier):
            return model.backbone.fc
    return None


def compute_cl_metrics(acc_matrix):
    """Compute standard Continual Learning metrics from accuracy matrix.

    Args:
        acc_matrix: 2D array where acc_matrix[i][j] = accuracy on task j after training on task i

    Returns:
        dict with:
            - final_acc: Mean accuracy on all tasks after all training (normalized to [0,1])
            - avg_acc: Mean of per-task accuracies throughout training (normalized to [0,1])
            - forgetting: Average accuracy drop on previously learned tasks (normalized to [0,1])
            - fwd_transfer: Forward transfer metric
    """
    acc_matrix = np.array(acc_matrix)
    num_tasks = len(acc_matrix)

    if num_tasks == 0:
        return {'final_acc': 0, 'avg_acc': 0, 'forgetting': 0, 'fwd_transfer': 0}

    # Final accuracy: mean accuracy on all tasks after all training
    final_acc = np.mean(acc_matrix[-1])

    # Average accuracy: mean of per-task accuracies throughout training
    avg_acc = np.mean([acc_matrix[i][i] for i in range(num_tasks)])

    # Forgetting: average accuracy drop on previously learned tasks
    forgetting = 0
    if num_tasks > 1:
        forgetting_vals = []
        for j in range(num_tasks - 1):
            max_acc = max(acc_matrix[i][j] for i in range(j, num_tasks))
            final_task_acc = acc_matrix[-1][j]
            forgetting_vals.append(max(0, max_acc - final_task_acc))
        forgetting = np.mean(forgetting_vals) if forgetting_vals else 0

    # Forward transfer: benefit to new tasks from prior learning
    fwd_transfer = 0

    return {
        'final_acc': final_acc / 100,
        'avg_acc': avg_acc / 100,
        'forgetting': forgetting / 100,
        'fwd_transfer': fwd_transfer
    }


class CLEvaluator:
    """Evaluator for Continual Learning experiments.

    Maintains accuracy matrix and computes CL metrics.

    Args:
        num_tasks: Expected number of tasks
    """

    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.acc_matrix = np.zeros((num_tasks, num_tasks))
        self.current_task = -1

    def evaluate_task(self, model, dataloader, task_id_trained, task_id_eval,
                      device, task_classes=None, ease=None, prompt_module=None, prompt_method=None):
        """Evaluate model on a specific task.

        Args:
            model: The backbone model
            dataloader: Data to evaluate on
            task_id_trained: Task index that was just trained
            task_id_eval: Task index being evaluated
            device: Device to use
            task_classes: Optional list of classes to mask outputs to
            ease: Optional EASERadar module for EASE algorithm
            prompt_module: Optional prompt module (L2P, CODAPrompt, or DualPrompt)
            prompt_method: Name of prompt method ('l2p', 'coda', or 'dualprompt')

        Returns:
            Accuracy on the evaluated task
        """
        model.eval()
        if ease is not None:
            ease.eval()
        if prompt_module is not None and hasattr(prompt_module, 'eval'):
            prompt_module.eval()

        correct, total = 0, 0
        inc_classifier = get_incremental_classifier(model)

        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)

                # Handle EASE: use adapter features + CosineLinear classifier
                if ease is not None and hasattr(model, 'get_features'):
                    backbone_features = model.get_features(data)
                    outputs = ease(backbone_features, training=False)

                # Handle prompt-based methods
                elif prompt_module is not None and hasattr(model, 'get_query'):
                    query = model.get_query(data)
                    if prompt_method == 'l2p':
                        prompts = prompt_module.select_prompts(query)
                    elif prompt_method == 'coda':
                        prompts = prompt_module.get_prompt(query, train=False)
                    elif prompt_method == 'dualprompt':
                        prompts = prompt_module.get_prompt(query)
                    elif prompt_method == 'epb':
                        if hasattr(prompt_module, 'select_prompts'):
                            prompts = prompt_module.select_prompts(query)
                        else:
                            prompts = prompt_module.get_prompt(query, train=False)
                    else:
                        prompts = None
                    outputs = model(data, prompts=prompts)

                else:
                    outputs = model(data)

                if task_classes is not None and len(task_classes) > 0:
                    task_classes_sorted = sorted(task_classes)
                    task_outputs = outputs[:, task_classes_sorted]
                    _, predicted_idx = task_outputs.max(1)
                    predicted = torch.tensor([task_classes_sorted[i] for i in predicted_idx], device=device)
                else:
                    _, predicted = outputs.max(1)

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = 100. * correct / total if total > 0 else 0
        self.acc_matrix[task_id_trained][task_id_eval] = accuracy
        self.current_task = max(self.current_task, task_id_trained)

        return accuracy

    def get_metrics(self):
        """Compute and return CL metrics based on current accuracy matrix.

        Returns:
            dict with final_acc, avg_acc, forgetting, fwd_transfer
        """
        return compute_cl_metrics(self.acc_matrix[:self.current_task + 1, :self.current_task + 1])

    def get_accuracy_matrix(self):
        """Get the full accuracy matrix.

        Returns:
            numpy array of shape (num_tasks, num_tasks)
        """
        return self.acc_matrix.copy()

    def print_summary(self):
        """Print summary of evaluation results."""
        metrics = self.get_metrics()
        print(f"\nCL Evaluation Summary:")
        print(f"  Final Accuracy: {metrics['final_acc']:.4f}")
        print(f"  Average Accuracy: {metrics['avg_acc']:.4f}")
        print(f"  Forgetting: {metrics['forgetting']:.4f}")
        print(f"\nAccuracy Matrix:")
        for i in range(self.current_task + 1):
            row = " ".join([f"{self.acc_matrix[i][j]:6.2f}" for j in range(self.current_task + 1)])
            print(f"  Task {i}: {row}")

    def to_dict(self):
        """Convert results to dictionary for serialization.

        Returns:
            dict with acc_matrix and metrics
        """
        metrics = self.get_metrics()
        return {
            'acc_matrix': self.acc_matrix[:self.current_task + 1, :self.current_task + 1].tolist(),
            'num_tasks_completed': self.current_task + 1,
            **metrics
        }
