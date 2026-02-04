"""Unified Continual Learning Trainer.

Provides CLTrainer class for training with all 10 CL algorithms:
- naive, ewc, lwf, replay, derpp, co2l, ease, l2p, coda, dualprompt
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .utils import IncrementalClassifier, TaskTracker
from .methods import (EWC, LwF, BalancedReplayBuffer, BalancedDERPlusPlus,
                      Co2L, EASE, L2P, CODAPrompt, DualPrompt)


# Models that support prompt-based methods (have transformer architecture)
TRANSFORMER_MODELS = ['RadarCubeTransformer', 'PromptVoxelTransformer']


def get_incremental_classifier(model):
    """Get the IncrementalClassifier from a model."""
    # Primary: model.classifier (ResNet18 and RadarTransformer)
    if hasattr(model, 'classifier') and isinstance(model.classifier, IncrementalClassifier):
        return model.classifier
    # Fallbacks for other architectures
    if hasattr(model, 'fc') and isinstance(model.fc, IncrementalClassifier):
        return model.fc
    if hasattr(model, 'head') and isinstance(model.head, IncrementalClassifier):
        return model.head
    return None


def get_feature_dim(model):
    """Get the feature dimension before the classifier."""
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            return model.classifier.in_features
        if isinstance(model.classifier, IncrementalClassifier):
            return model.classifier.in_features
    if hasattr(model, 'embed_dim'):
        return model.embed_dim
    return 512  # ResNet18 default


def get_classifier_module(model):
    """Get the classifier module from a model."""
    inc_classifier = get_incremental_classifier(model)
    if inc_classifier is not None:
        return inc_classifier
    if hasattr(model, 'classifier'):
        return model.classifier
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        return model.head
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'classifier'):
            return model.backbone.classifier
        if hasattr(model.backbone, 'fc'):
            return model.backbone.fc
    return None


def make_incremental_model(model, device):
    """Replace the final classifier with IncrementalClassifier."""
    # Primary: model.classifier as nn.Linear (ResNet18 and RadarTransformer)
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = IncrementalClassifier(in_features, initial_classes=0).to(device)
        return model
    # Fallbacks for other model architectures
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = IncrementalClassifier(in_features, initial_classes=0).to(device)
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = IncrementalClassifier(in_features, initial_classes=0).to(device)
    return model


def freeze_backbone(model):
    """Freeze all model parameters except the classifier."""
    classifier = get_classifier_module(model)
    for name, param in model.named_parameters():
        param.requires_grad = False
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True


class CLTrainer:
    """Unified Continual Learning Trainer.

    Supports all 10 CL algorithms through a single interface.

    Args:
        model: The neural network model
        algorithm: CL algorithm name ('naive', 'ewc', 'lwf', 'replay', 'derpp',
                   'co2l', 'ease', 'l2p', 'coda', 'dualprompt')
        device: Device to run computations on
        config: Optional configuration dict with hyperparameters
    """

    def __init__(self, model, algorithm, device, config=None):
        self.model = model
        self.algorithm = algorithm
        self.device = device
        self.config = config or {}

        self.feature_dim = get_feature_dim(model)
        self.task_tracker = TaskTracker()
        self.cl_params = {}
        self.current_task_id = -1

        self._init_cl_method()

    def _init_cl_method(self):
        """Initialize CL method before training."""
        buffer_size = self.config.get('buffer_size', 500)

        if self.algorithm == 'replay':
            self.cl_params['buffer'] = BalancedReplayBuffer(buffer_size=buffer_size)
        elif self.algorithm == 'derpp':
            alpha = self.config.get('derpp_alpha', 0.5)
            beta = self.config.get('derpp_beta', 0.5)
            self.cl_params['buffer'] = BalancedDERPlusPlus(buffer_size=buffer_size, alpha=alpha, beta=beta)
        elif self.algorithm == 'co2l':
            proj_dim = self.config.get('proj_dim', 128)
            temperature = self.config.get('temperature', 0.07)  # Official default
            current_temp = self.config.get('current_temp', 0.2)  # Official default
            past_temp = self.config.get('past_temp', 0.01)  # Official default
            distill_power = self.config.get('distill_power', 1.0)
            supcon_weight = self.config.get('supcon_weight', 0.1)
            self.cl_params['co2l'] = Co2L(
                self.feature_dim, proj_dim=proj_dim, temperature=temperature,
                current_temp=current_temp, past_temp=past_temp,
                distill_power=distill_power, supcon_weight=supcon_weight,
                buffer_size=buffer_size
            )
            self.cl_params['co2l'].to(self.device)
        elif self.algorithm == 'ease':
            bottleneck_dim = self.config.get('bottleneck_dim', 32)
            alpha = self.config.get('ease_alpha', 0.1)
            self.cl_params['ease'] = EASE(embed_dim=self.feature_dim, bottleneck_dim=bottleneck_dim, alpha=alpha)
            self.cl_params['ease'].to(self.device)
        elif self.algorithm == 'l2p':
            pool_size = self.config.get('pool_size', 20)
            prompt_length = self.config.get('prompt_length', 5)
            top_k = self.config.get('top_k', 5)
            self.cl_params['l2p'] = L2P(pool_size=pool_size, prompt_length=prompt_length,
                                        embed_dim=self.feature_dim, top_k=top_k)
            self.cl_params['l2p'].to(self.device)
        elif self.algorithm == 'coda':
            pool_size = self.config.get('pool_size', 100)
            prompt_length = self.config.get('prompt_length', 8)
            ortho_weight = self.config.get('ortho_weight', 0.1)
            self.cl_params['coda'] = CODAPrompt(pool_size=pool_size, prompt_length=prompt_length,
                                                 embed_dim=self.feature_dim, ortho_weight=ortho_weight)
            self.cl_params['coda'].to(self.device)
        elif self.algorithm == 'dualprompt':
            g_prompt_length = self.config.get('g_prompt_length', 5)
            e_pool_size = self.config.get('e_pool_size', 10)
            top_k = self.config.get('top_k', 5)
            self.cl_params['dualprompt'] = DualPrompt(g_prompt_length=g_prompt_length,
                                                       e_pool_size=e_pool_size,
                                                       embed_dim=self.feature_dim, top_k=top_k)
            self.cl_params['dualprompt'].to(self.device)

    def _setup_optimizer(self, lr):
        """Setup optimizer with algorithm-specific parameters."""
        if self.algorithm == 'co2l' and 'co2l' in self.cl_params:
            params = list(self.model.parameters()) + list(self.cl_params['co2l'].projector.parameters())
        elif self.algorithm == 'ease' and 'ease' in self.cl_params:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            params = trainable_params + list(self.cl_params['ease'].parameters())
        elif self.algorithm == 'l2p' and 'l2p' in self.cl_params:
            params = list(self.model.parameters()) + list(self.cl_params['l2p'].parameters())
        elif self.algorithm == 'coda' and 'coda' in self.cl_params:
            params = list(self.model.parameters()) + list(self.cl_params['coda'].parameters())
        elif self.algorithm == 'dualprompt' and 'dualprompt' in self.cl_params:
            params = list(self.model.parameters()) + list(self.cl_params['dualprompt'].parameters())
        else:
            params = self.model.parameters()

        return optim.Adam(params, lr=lr)

    def _train_epoch(self, dataloader, optimizer, criterion, active_classes, old_classes):
        """Train for one epoch with class-incremental masking."""
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        inc_classifier = get_incremental_classifier(self.model)

        if self.algorithm == 'coda' and 'coda' in self.cl_params:
            self.cl_params['coda'].train()

        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            # Handle EASE forward pass
            if self.algorithm == 'ease' and 'ease' in self.cl_params and hasattr(self.model, 'get_features'):
                backbone_features = self.model.get_features(data)
                if self.cl_params['ease'].task_id > 0:
                    backbone_features = backbone_features.detach()
                outputs = self.cl_params['ease'](backbone_features, training=True)

            # Handle prompt-based methods
            elif self.algorithm in ['l2p', 'coda', 'dualprompt'] and hasattr(self.model, 'get_query'):
                with torch.no_grad():
                    query = self.model.get_query(data)

                if self.algorithm == 'l2p':
                    prompts = self.cl_params['l2p'].select_prompts(query)
                elif self.algorithm == 'coda':
                    prompts = self.cl_params['coda'].get_prompt(query)
                elif self.algorithm == 'dualprompt':
                    prompts = self.cl_params['dualprompt'].get_prompt(query)

                outputs = self.model(data, prompts=prompts)
            else:
                outputs = self.model(data)

            # Apply output masking
            if active_classes is not None and inc_classifier is not None:
                outputs_masked = inc_classifier.get_masked_logits(outputs, active_classes)
            else:
                outputs_masked = outputs

            loss = criterion(outputs_masked, labels)

            # Add CL-specific losses
            if self.algorithm == 'ewc' and 'ewc' in self.cl_params:
                loss += self.cl_params['ewc'].penalty()
            elif self.algorithm == 'lwf' and 'lwf' in self.cl_params:
                loss += self.cl_params['lwf'].distillation_loss(outputs, data, self.device, old_classes=old_classes)
            elif self.algorithm == 'replay' and 'buffer' in self.cl_params:
                replay_data, replay_labels = self.cl_params['buffer'].get_batch(data.size(0))
                if replay_data is not None:
                    replay_data = replay_data.to(self.device)
                    replay_labels = replay_labels.to(self.device)
                    replay_outputs = self.model(replay_data)
                    if active_classes is not None and inc_classifier is not None:
                        replay_outputs_masked = inc_classifier.get_masked_logits(replay_outputs, active_classes)
                    else:
                        replay_outputs_masked = replay_outputs
                    loss += criterion(replay_outputs_masked, replay_labels)
            elif self.algorithm == 'derpp' and 'buffer' in self.cl_params:
                loss += self.cl_params['buffer'].replay_loss(self.model, data.size(0), self.device, active_classes)
            elif self.algorithm == 'co2l' and 'co2l' in self.cl_params:
                co2l = self.cl_params['co2l']
                # Combine current batch with replay samples
                combined_data, combined_labels, current_mask = co2l.get_combined_batch(data, labels, self.device)

                # Get features for combined batch
                if hasattr(self.model, 'get_features'):
                    combined_features = self.model.get_features(combined_data)
                else:
                    combined_features = self.model(combined_data)

                # Asymmetric SupCon loss (current samples as anchors, replay as negatives)
                loss += co2l.supcon_weight * co2l.supcon_loss(combined_features, combined_labels, current_mask)

                # IRD loss (only on current batch data for efficiency)
                if co2l.old_model is not None:
                    if hasattr(self.model, 'get_features'):
                        current_features = self.model.get_features(data)
                    else:
                        current_features = outputs
                    with torch.no_grad():
                        if hasattr(co2l.old_model, 'get_features'):
                            old_features = co2l.old_model.get_features(data)
                        else:
                            old_features = co2l.old_model(data)
                    loss += co2l.ird_loss(old_features, current_features)

            # Add prompt losses
            elif self.algorithm == 'l2p' and 'l2p' in self.cl_params:
                prompt_loss = self.cl_params['l2p'].get_prompt_loss(query)
                loss += 0.1 * prompt_loss
            elif self.algorithm == 'coda' and 'coda' in self.cl_params:
                prompt_loss = self.cl_params['coda'].get_prompt_loss(query)
                loss += prompt_loss
            elif self.algorithm == 'dualprompt' and 'dualprompt' in self.cl_params:
                prompt_loss = self.cl_params['dualprompt'].get_prompt_loss(query)
                loss += 0.1 * prompt_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs_masked.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0

    def train_task(self, task_id, train_loader, val_loader, task_classes, epochs=30, lr=0.001):
        """Train model on a single task.

        Args:
            task_id: Task identifier
            train_loader: Training data loader
            val_loader: Validation data loader
            task_classes: List of class indices in this task
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Best validation accuracy
        """
        self.current_task_id = task_id
        self.task_tracker.register_task(task_id, task_classes)

        active_classes = self.task_tracker.get_active_classes()
        old_classes = self.task_tracker.get_old_classes(task_id)

        # Adapt incremental classifier
        inc_classifier = get_incremental_classifier(self.model)
        if inc_classifier is not None and self.algorithm != 'ease':
            inc_classifier.adaptation(task_classes)

        # Initialize EASE adapter for new task
        if self.algorithm == 'ease' and 'ease' in self.cl_params:
            ease = self.cl_params['ease']
            ease.add_task()
            ease.to(self.device)
            num_classes = max(active_classes) + 1
            ease.create_or_expand_classifier(num_classes, self.device)
            if ease.task_id > 0:
                freeze_backbone(self.model)

        optimizer = self._setup_optimizer(lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_val_acc = 0
        best_model_state = None
        best_ease_state = None
        best_prompt_state = None

        ease = self.cl_params.get('ease') if self.algorithm == 'ease' else None
        prompt_module = None
        if self.algorithm in ['l2p', 'coda', 'dualprompt']:
            prompt_module = self.cl_params.get(self.algorithm)

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion, active_classes, old_classes
            )

            # Evaluate
            val_acc = self._evaluate(val_loader, task_classes, ease, prompt_module)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                if ease is not None:
                    best_ease_state = copy.deepcopy(ease.state_dict())
                if prompt_module is not None and hasattr(prompt_module, 'state_dict'):
                    best_prompt_state = copy.deepcopy(prompt_module.state_dict())

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        if best_ease_state is not None and ease is not None:
            ease.load_state_dict(best_ease_state)
        if best_prompt_state is not None and prompt_module is not None:
            prompt_module.load_state_dict(best_prompt_state)

        return best_val_acc

    def _evaluate(self, dataloader, task_classes, ease=None, prompt_module=None):
        """Evaluate model on dataloader."""
        self.model.eval()
        if ease is not None:
            ease.eval()
        if prompt_module is not None and hasattr(prompt_module, 'eval'):
            prompt_module.eval()

        correct, total = 0, 0

        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)

                if ease is not None and hasattr(self.model, 'get_features'):
                    backbone_features = self.model.get_features(data)
                    outputs = ease(backbone_features, training=False)
                elif prompt_module is not None and hasattr(self.model, 'get_query'):
                    query = self.model.get_query(data)
                    if self.algorithm == 'l2p':
                        prompts = prompt_module.select_prompts(query)
                    elif self.algorithm == 'coda':
                        prompts = prompt_module.get_prompt(query)
                    elif self.algorithm == 'dualprompt':
                        prompts = prompt_module.get_prompt(query)
                    else:
                        prompts = None
                    outputs = self.model(data, prompts=prompts)
                else:
                    outputs = self.model(data)

                if task_classes is not None and len(task_classes) > 0:
                    task_classes_sorted = sorted(task_classes)
                    task_outputs = outputs[:, task_classes_sorted]
                    _, predicted_idx = task_outputs.max(1)
                    predicted = torch.tensor([task_classes_sorted[i] for i in predicted_idx], device=self.device)
                else:
                    _, predicted = outputs.max(1)

                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return 100. * correct / total if total > 0 else 0

    def after_task(self, train_loader, task_classes):
        """Update CL method after training on a task.

        Args:
            train_loader: Training data loader for the completed task
            task_classes: Classes in the completed task
        """
        # Update replay buffer
        if self.algorithm == 'replay' and 'buffer' in self.cl_params:
            self.cl_params['buffer'].add_task_data(train_loader, task_classes)
        elif self.algorithm == 'derpp' and 'buffer' in self.cl_params:
            self.cl_params['buffer'].add_task_data(train_loader, task_classes, self.model, self.device)

        # Extract prototypes for EASE
        if self.algorithm == 'ease' and 'ease' in self.cl_params:
            self.cl_params['ease'].extract_prototypes(train_loader, self.model, self.device, task_classes)

        # Update EWC Fisher with Online EWC accumulation
        if self.algorithm == 'ewc':
            importance = self.config.get('ewc_importance', 1000)
            decay_factor = self.config.get('ewc_decay', 0.9)
            new_ewc = EWC(self.model, train_loader, self.device, importance=importance)

            if 'ewc' in self.cl_params:
                old_ewc = self.cl_params['ewc']
                # Accumulate Fisher with decay: F_new = decay * F_old + F_current
                for n in new_ewc.fisher:
                    if n in old_ewc.fisher and new_ewc.fisher[n].shape == old_ewc.fisher[n].shape:
                        new_ewc.fisher[n] = decay_factor * old_ewc.fisher[n] + new_ewc.fisher[n]
                # Keep the oldest parameters as anchor point
                for n in new_ewc.params:
                    if n in old_ewc.params and new_ewc.params[n].shape == old_ewc.params[n].shape:
                        new_ewc.params[n] = old_ewc.params[n]

            self.cl_params['ewc'] = new_ewc

        # Update LwF old model
        if self.algorithm == 'lwf':
            temperature = self.config.get('lwf_temperature', 2.0)
            alpha = self.config.get('lwf_alpha', 1.0)
            self.cl_params['lwf'] = LwF(self.model, temperature=temperature, alpha=alpha)

        # Update Co2L: save model and add to buffer
        if self.algorithm == 'co2l' and 'co2l' in self.cl_params:
            self.cl_params['co2l'].add_to_buffer(train_loader, task_classes)
            self.cl_params['co2l'].update_old_model(self.model)

        # Update prompt methods
        if self.algorithm == 'coda' and 'coda' in self.cl_params:
            self.cl_params['coda'].update_task()
        elif self.algorithm == 'dualprompt' and 'dualprompt' in self.cl_params:
            self.cl_params['dualprompt'].update_task()
        elif self.algorithm == 'l2p' and 'l2p' in self.cl_params:
            self.cl_params['l2p'].update_task()

    def evaluate_all_tasks(self, test_loaders, task_classes_list):
        """Evaluate on all seen tasks.

        Args:
            test_loaders: List of test data loaders for each task
            task_classes_list: List of class indices for each task

        Returns:
            List of accuracies for each task
        """
        ease = self.cl_params.get('ease') if self.algorithm == 'ease' else None
        prompt_module = None
        if self.algorithm in ['l2p', 'coda', 'dualprompt']:
            prompt_module = self.cl_params.get(self.algorithm)

        accuracies = []
        for test_loader, task_classes in zip(test_loaders, task_classes_list):
            acc = self._evaluate(test_loader, task_classes, ease, prompt_module)
            accuracies.append(acc)

        return accuracies
