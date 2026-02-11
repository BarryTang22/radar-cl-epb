"""Prompt-Guided Selective Update (PGSU) — Paper-aligned controller.

Implements the novelty-coupled controller from Sections III-B and III-C:
- Transformer path: LoRA rank-masked adapters with frozen backbone (Eq. 13-15)
- CNN path: bottleneck adapters with width masking and alpha gating (Eq. 26-31)
- Novelty-driven replay ratio (Eq. 16-17)
- Concatenated mixed CE loss (Eq. 19-20)
"""

import torch
import torch.nn.functional as F

from .replay import BalancedReplayBuffer
from .pgsu_modules import mask_first


class PGSU:
    """Prompt-Guided Selective Update controller.

    Allocates adapter capacity (LoRA rank or CNN adapter width/gate) based on
    task novelty. Backbone weights are frozen; only adapters, prompts, and
    classifier are trained.

    Args:
        k_min: Minimum LoRA rank (transformer path)
        k_max: Maximum LoRA rank (transformer path)
        r_adp_min: Minimum CNN adapter width
        r_adp_max: Maximum CNN adapter width
        alpha_min: Minimum adapter gate (CNN path)
        alpha_max: Maximum adapter gate (CNN path)
        rho_min: Minimum replay ratio (novel tasks)
        rho_max: Maximum replay ratio (familiar tasks)
        lambda_p: Prompt loss weight
        buffer_size: Replay buffer capacity
    """

    def __init__(self, k_min=2, k_max=16, r_adp_min=8, r_adp_max=64,
                 alpha_min=0.1, alpha_max=1.0, rho_min=0.1, rho_max=0.5,
                 lambda_p=0.1, buffer_size=300):
        # LoRA params (transformer)
        self.k_min = k_min
        self.k_max = k_max
        # CNN adapter params
        self.r_adp_min = r_adp_min
        self.r_adp_max = r_adp_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # Replay
        self.rho_min = rho_min
        self.rho_max = rho_max
        # Prompt loss
        self.lambda_p = lambda_p
        # State
        self.task_centroids = []
        self.current_novelty = 1.0
        self.deployment_path = None  # Set to 'transformer' or 'cnn' by trainer
        # Balanced replay buffer
        self.replay_buffer = BalancedReplayBuffer(buffer_size=buffer_size)
        # References set by trainer
        self.lora_injector = None
        self.cnn_wrapper = None

    # ------------------------------------------------------------------
    # Novelty computation (Eq. 11) — unchanged from original
    # ------------------------------------------------------------------

    def compute_task_novelty(self, query_centroid):
        """Compute how novel this task is compared to previous tasks."""
        if not self.task_centroids:
            self.current_novelty = 1.0
            return 1.0

        similarities = [
            F.cosine_similarity(query_centroid.unsqueeze(0), c.unsqueeze(0), dim=1).item()
            for c in self.task_centroids
        ]
        max_sim = max(similarities)
        novelty = 1.0 - max_sim
        self.current_novelty = max(0.0, min(1.0, novelty))
        return self.current_novelty

    def add_task_centroid(self, centroid):
        """Store centroid for the current task."""
        self.task_centroids.append(centroid.detach().clone())

    # ------------------------------------------------------------------
    # Transformer path: LoRA rank and mask (Eq. 13-15)
    # ------------------------------------------------------------------

    def get_lora_rank(self, nu=None):
        """Map novelty to LoRA rank k_t (Eq. 13-14: affine + round + clip)."""
        if nu is None:
            nu = self.current_novelty
        k_t = self.k_min + nu * (self.k_max - self.k_min)
        k_t = int(round(k_t))
        return max(self.k_min, min(self.k_max, k_t))

    def get_lora_mask(self, nu=None):
        """Compute MASKFIRST binary mask for LoRA (Eq. 15)."""
        k_t = self.get_lora_rank(nu)
        return mask_first(k_t, self.k_max)

    # ------------------------------------------------------------------
    # CNN path: adapter width and gate (Eq. 30-31)
    # ------------------------------------------------------------------

    def get_adapter_width(self, nu=None):
        """Map novelty to CNN adapter width r_adp_t (Eq. 30)."""
        if nu is None:
            nu = self.current_novelty
        r = self.r_adp_min + nu * (self.r_adp_max - self.r_adp_min)
        r = int(round(r))
        return max(self.r_adp_min, min(self.r_adp_max, r))

    def get_adapter_alpha(self, nu=None):
        """Map novelty to adapter gate alpha_t (Eq. 31)."""
        if nu is None:
            nu = self.current_novelty
        alpha = self.alpha_min + nu * (self.alpha_max - self.alpha_min)
        return max(self.alpha_min, min(self.alpha_max, alpha))

    # ------------------------------------------------------------------
    # Replay (Eq. 16-17)
    # ------------------------------------------------------------------

    def get_replay_ratio(self, novelty=None):
        """Get adaptive replay ratio (Eq. 16).

        High novelty -> rho_min (focus on new data)
        Low novelty  -> rho_max (replay more to prevent forgetting)
        """
        if novelty is None:
            novelty = self.current_novelty
        return self.rho_max - novelty * (self.rho_max - self.rho_min)

    def get_replay_batch_size(self, batch_size, nu=None):
        """Compute replay batch size (Eq. 17). Returns 0 when buffer is empty."""
        if len(self.replay_buffer) == 0:
            return 0
        rho = self.get_replay_ratio(nu)
        if rho <= 0:
            return 0
        return max(1, int(batch_size * rho))

    # ------------------------------------------------------------------
    # Apply novelty controls to active modules
    # ------------------------------------------------------------------

    def apply_novelty_controls(self):
        """Set masks/alpha on active modules after novelty is computed."""
        nu = self.current_novelty
        if self.deployment_path == 'transformer' and self.lora_injector is not None:
            mask = self.get_lora_mask(nu)
            self.lora_injector.set_rank_mask(mask.to(
                next(self.lora_injector.parameters()).device
            ))
        elif self.deployment_path == 'cnn' and self.cnn_wrapper is not None:
            # Width mask
            r_t = self.get_adapter_width(nu)
            width_mask = mask_first(r_t, self.r_adp_max)
            self.cnn_wrapper.adapter.set_width_mask(width_mask.to(
                next(self.cnn_wrapper.parameters()).device
            ))
            # Alpha gate
            self.cnn_wrapper.adapter.alpha_t = self.get_adapter_alpha(nu)


def compute_task_centroid(model, dataloader, device, cnn_wrapper=None):
    """Compute mean feature vector for a task's data.

    For CNN path: uses model.get_features() (frozen 512-dim backbone features)
    so all centroids live in the same fixed space across tasks.
    For transformer path: uses model.get_query() (part of frozen backbone).

    Args:
        model: The base model
        dataloader: Task data loader
        device: Computation device
        cnn_wrapper: If not None, signals CNN path (uses backbone features)
    """
    was_training = model.training
    model.eval()
    all_queries = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            if cnn_wrapper is not None:
                # CNN path: use frozen backbone features (consistent space across tasks)
                query = model.get_features(data)
            elif hasattr(model, 'get_query'):
                query = model.get_query(data)
            else:
                query = model.get_features(data)
            all_queries.append(query)

    all_queries = torch.cat(all_queries, dim=0)
    centroid = all_queries.mean(dim=0)
    if was_training:
        model.train()
    return centroid
