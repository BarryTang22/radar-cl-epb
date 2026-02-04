"""CODA-Prompt with prefix tuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CODAPrompt(nn.Module):
    """CODA-Prompt - COntinual Decomposed Attention-based Prompting with prefix tuning.

    Paper: Smith et al. "CODA-Prompt" (CVPR 2023)
    Official: https://github.com/GT-RIPL/CODA-Prompt

    Uses attention mechanism to compose prompts from a pool of components,
    allowing soft selection that adapts during learning. Implements prefix tuning
    where prompts are prepended to Keys and Values in attention.

    Args:
        pool_size: Number of prompt components in the pool
        prompt_length: Length of each prompt component
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers to attach prompts
        ortho_weight: Weight for orthogonality regularization loss
    """

    def __init__(self, pool_size=100, prompt_length=8, embed_dim=128,
                 num_heads=6, num_layers=4, ortho_weight=0.1):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads
        self.ortho_weight = ortho_weight

        # Prompt components: (pool_size, num_layers, 2, prompt_length, num_heads, head_dim)
        # 2 = key and value prefixes
        self.prompt_components = nn.Parameter(
            torch.randn(pool_size, num_layers, 2, prompt_length, num_heads, self.head_dim) * 0.02
        )
        self.keys = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)

        self.task_count = 0
        self.register_buffer('key_freq', torch.zeros(pool_size))
        self.register_buffer('key_freq_per_task', torch.zeros(1, pool_size))

    def to(self, device):
        return super().to(device)

    def get_prompt(self, query):
        """Get attention-weighted combination of prompt components.

        Returns:
            Dict with 'key_prefixes' and 'value_prefixes' lists for all layers.
        """
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        attn_weights = F.softmax(torch.matmul(query, keys.T), dim=-1)

        if self.training:
            with torch.no_grad():
                self.key_freq += attn_weights.sum(dim=0)

        batch_size = query.size(0)
        # prompt_components: (pool_size, num_layers, 2, prompt_length, num_heads, head_dim)
        # attn_weights: (B, pool_size)

        # Weighted combination: (B, num_layers, 2, prompt_length, num_heads, head_dim)
        prompt = torch.einsum('bp,plknhd->blknhd', attn_weights, self.prompt_components)

        key_prefixes = []
        value_prefixes = []
        for layer_idx in range(self.num_layers):
            # (B, prompt_length, num_heads, head_dim) -> (B, num_heads, prompt_length, head_dim)
            k = prompt[:, layer_idx, 0].permute(0, 2, 1, 3)
            v = prompt[:, layer_idx, 1].permute(0, 2, 1, 3)
            key_prefixes.append(k)
            value_prefixes.append(v)

        return {
            'key_prefixes': key_prefixes,
            'value_prefixes': value_prefixes
        }

    def get_attention_weights(self, query):
        """Get attention weights for analysis/debugging."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        return F.softmax(torch.matmul(query, keys.T), dim=-1)

    def orthogonality_loss(self):
        """Compute orthogonality regularization loss."""
        # Flatten each prompt component for orthogonality computation
        B = self.prompt_components.view(self.pool_size, -1)
        I = torch.eye(self.pool_size, device=B.device)
        ortho_loss = ((B @ B.T - I) ** 2).mean()
        return self.ortho_weight * ortho_loss

    def get_prompt_loss(self, query):
        """Get combined prompt-related losses."""
        return self.orthogonality_loss()

    def gram_schmidt_init(self, start_idx):
        """Initialize components from start_idx orthogonal to previous."""
        with torch.no_grad():
            for k in range(start_idx, self.pool_size):
                v = self.prompt_components[k].view(-1).clone()
                for j in range(k):
                    u = self.prompt_components[j].view(-1)
                    v = v - (torch.dot(v, u) / (torch.dot(u, u) + 1e-8)) * u
                v = v / (v.norm() + 1e-8)
                self.prompt_components.data[k] = v.view(
                    self.num_layers, 2, self.prompt_length, self.num_heads, self.head_dim
                )

    def update_task(self):
        """Called after each task to update statistics."""
        self.task_count += 1

        if self.task_count > self.key_freq_per_task.size(0):
            new_buffer = torch.zeros(self.task_count, self.pool_size, device=self.key_freq.device)
            new_buffer[:self.task_count - 1] = self.key_freq_per_task
            self.key_freq_per_task = new_buffer

        self.key_freq_per_task[self.task_count - 1] = self.key_freq.clone()
        self.key_freq.zero_()

    def get_key_usage_stats(self):
        """Get statistics about key usage across tasks."""
        total_freq = self.key_freq_per_task.sum(dim=0)
        active_keys = (total_freq > 0).sum().item()

        return {
            'total_keys': self.pool_size,
            'active_keys': active_keys,
            'key_freq_per_task': self.key_freq_per_task.clone(),
            'total_key_freq': total_freq
        }
