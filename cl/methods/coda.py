"""CODA-Prompt - Official Implementation.

Based on GT-RIPL/CODA-Prompt (CVPR 2023)
https://github.com/GT-RIPL/CODA-Prompt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CODAPrompt(nn.Module):
    """CODA-Prompt - COntinual Decomposed Attention-based Prompting.

    Official design with three learnable components (K, A, P) per layer,
    task-based pool partitioning, and gradient isolation for old components.

    Args:
        pool_size: Number of prompt components in the pool
        prompt_length: Length of each prompt component
        embed_dim: Embedding dimension
        n_tasks: Number of tasks (for pool partitioning)
        ortho_weight: Weight for orthogonality regularization loss
    """

    def __init__(self, pool_size=100, prompt_length=8, embed_dim=128,
                 n_tasks=3, ortho_weight=0.1):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks
        self.ortho_weight = ortho_weight
        self.task_count = 0

        # Three learnable components (official design)
        # P: Prompt components (pool_size, prompt_length, embed_dim)
        # K: Keys for matching (pool_size, embed_dim)
        # A: Attention weights for query modulation (pool_size, embed_dim)
        self.prompt_components = nn.Parameter(torch.randn(pool_size, prompt_length, embed_dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)
        self.attention = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)

        # Tracking statistics
        self.register_buffer('key_freq', torch.zeros(pool_size))
        self.register_buffer('key_freq_per_task', torch.zeros(1, pool_size))

        # Initialize with Gram-Schmidt orthogonalization
        self._gram_schmidt_keys()

    def _gram_schmidt_keys(self):
        """Apply Gram-Schmidt orthogonalization to keys."""
        with torch.no_grad():
            for k in range(self.pool_size):
                v = self.keys[k].clone()
                for j in range(k):
                    u = self.keys[j]
                    v = v - (torch.dot(v, u) / (torch.dot(u, u) + 1e-8)) * u
                norm = v.norm()
                if norm > 1e-8:
                    self.keys.data[k] = v / norm

    def _get_task_indices(self):
        """Get start and end indices for current task's components."""
        pt = self.pool_size // self.n_tasks
        s = self.task_count * pt
        f = min((self.task_count + 1) * pt, self.pool_size)
        return s, f, pt

    def get_prompt(self, query, train=False):
        """Get attention-weighted combination of prompt components.

        Official attention mechanism: query * A (element-wise broadcast),
        then dot product with normalized K, NO softmax.

        Args:
            query: Query tensor of shape (B, embed_dim)
            train: Whether in training mode (affects which components are used)

        Returns:
            prompt: Weighted prompt of shape (B, prompt_length, embed_dim)
        """
        s, f, pt = self._get_task_indices()

        # Select components based on train/eval mode
        if train:
            if self.task_count > 0:
                # Freeze old components (detach), train new ones
                K = torch.cat([self.keys[:s].detach(), self.keys[s:f]], dim=0)
                A = torch.cat([self.attention[:s].detach(), self.attention[s:f]], dim=0)
                P = torch.cat([self.prompt_components[:s].detach(), self.prompt_components[s:f]], dim=0)
            else:
                # First task: only use current task's components
                K = self.keys[s:f]
                A = self.attention[s:f]
                P = self.prompt_components[s:f]
        else:
            # Evaluation: use all components up to current task
            K = self.keys[:f]
            A = self.attention[:f]
            P = self.prompt_components[:f]

        # Official attention mechanism (NO softmax)
        # Step 1: Element-wise query-attention modulation
        # a_query shape: (B, num_components, embed_dim)
        a_query = torch.einsum('bd,kd->bkd', query, A)

        # Step 2: Normalize
        n_K = F.normalize(K, dim=1)  # Normalize keys
        q = F.normalize(a_query, dim=2)  # Normalize modulated queries

        # Step 3: Compute attention scores (NO softmax - raw dot products)
        aq_k = torch.einsum('bkd,kd->bk', q, n_K)  # (B, num_components)

        # Step 4: Weighted combination of prompts
        prompt = torch.einsum('bk,kld->bld', aq_k, P)  # (B, prompt_length, embed_dim)

        # Track key usage during training
        if self.training and train:
            with torch.no_grad():
                # Accumulate attention weights for current task's components only
                freq_update = aq_k.abs().sum(dim=0)
                num_active = f - s
                if self.task_count > 0:
                    # freq_update has size f (old + new), extract current task portion
                    self.key_freq[s:f] += freq_update[s:f]
                else:
                    # First task: freq_update has size pt
                    self.key_freq[s:f] += freq_update[:num_active]

        return prompt

    def get_attention_weights(self, query):
        """Get raw attention scores for analysis/debugging."""
        s, f, pt = self._get_task_indices()
        K = self.keys[:f]
        A = self.attention[:f]

        a_query = torch.einsum('bd,kd->bkd', query, A)
        n_K = F.normalize(K, dim=1)
        q = F.normalize(a_query, dim=2)
        aq_k = torch.einsum('bkd,kd->bk', q, n_K)

        return aq_k

    def _ortho_penalty(self, t):
        """Compute orthogonality penalty for a tensor."""
        # t shape: (num_components, dim)
        n = t.shape[0]
        if n <= 1:
            return torch.tensor(0.0, device=t.device)
        gram = t @ t.T
        eye = torch.eye(n, device=t.device)
        return ((gram - eye) ** 2).mean()

    def orthogonality_loss(self):
        """Compute orthogonality regularization loss on K, A, and P."""
        s, f, pt = self._get_task_indices()

        # Only penalize current task's components
        K = self.keys[s:f]
        A = self.attention[s:f]
        P = self.prompt_components[s:f].view(f - s, -1)  # Flatten prompt

        loss = self._ortho_penalty(K)
        loss = loss + self._ortho_penalty(A)
        loss = loss + self._ortho_penalty(P)

        return self.ortho_weight * loss

    def get_prompt_loss(self, query):
        """Get combined prompt-related losses."""
        return self.orthogonality_loss()

    def process_task_count(self):
        """Called after each task to update task count and re-orthogonalize."""
        # Save current task's key frequencies
        if self.task_count >= self.key_freq_per_task.size(0):
            new_buffer = torch.zeros(self.task_count + 1, self.pool_size, device=self.key_freq.device)
            new_buffer[:self.key_freq_per_task.size(0)] = self.key_freq_per_task
            self.key_freq_per_task = new_buffer

        self.key_freq_per_task[self.task_count] = self.key_freq.clone()
        self.key_freq.zero_()

        # Increment task count
        self.task_count += 1

        # Re-orthogonalize keys for next task (official behavior)
        self._gram_schmidt_keys()

    def update_task(self):
        """Alias for process_task_count for backward compatibility."""
        self.process_task_count()

    def get_key_usage_stats(self):
        """Get statistics about key usage across tasks."""
        total_freq = self.key_freq_per_task.sum(dim=0)
        active_keys = (total_freq > 0).sum().item()

        return {
            'total_keys': self.pool_size,
            'active_keys': active_keys,
            'key_freq_per_task': self.key_freq_per_task.clone(),
            'total_key_freq': total_freq,
            'task_count': self.task_count,
            'n_tasks': self.n_tasks
        }

