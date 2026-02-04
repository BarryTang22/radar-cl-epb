"""CODA-Prompt."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CODAPrompt(nn.Module):
    """CODA-Prompt - COntinual Decomposed Attention-based Prompting.

    Paper: Smith et al. "CODA-Prompt" (CVPR 2023)
    Official: https://github.com/GT-RIPL/CODA-Prompt

    Uses attention mechanism to compose prompts from a pool of components,
    allowing soft selection that adapts during learning.

    Args:
        pool_size: Number of prompt components in the pool
        prompt_length: Length of each prompt component
        embed_dim: Embedding dimension
        ortho_weight: Weight for orthogonality regularization loss
    """

    def __init__(self, pool_size=100, prompt_length=8, embed_dim=128, ortho_weight=0.1):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.ortho_weight = ortho_weight

        self.prompt_components = nn.Parameter(torch.randn(pool_size, prompt_length, embed_dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)

        self.task_count = 0
        self.register_buffer('key_freq', torch.zeros(pool_size))
        self.register_buffer('key_freq_per_task', torch.zeros(1, pool_size))

    def to(self, device):
        return super().to(device)

    def get_prompt(self, query):
        """Get attention-weighted combination of prompt components."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        attn_weights = F.softmax(torch.matmul(query, keys.T), dim=-1)
        prompt = torch.einsum('bp,ple->ble', attn_weights, self.prompt_components)

        if self.training:
            with torch.no_grad():
                self.key_freq += attn_weights.sum(dim=0)

        return prompt

    def get_attention_weights(self, query):
        """Get attention weights for analysis/debugging."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        return F.softmax(torch.matmul(query, keys.T), dim=-1)

    def orthogonality_loss(self):
        """Compute orthogonality regularization loss."""
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
                self.prompt_components.data[k] = v.view(self.prompt_length, self.embed_dim)

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
