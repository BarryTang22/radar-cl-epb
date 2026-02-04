"""Learning to Prompt with prefix tuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2P(nn.Module):
    """Learning to Prompt for continual learning with prefix tuning.

    Paper: Wang et al. "Learning to Prompt for Continual Learning" (CVPR 2022)
    Official: https://github.com/google-research/l2p

    Uses a learnable prompt pool with key-query matching to select
    task-relevant prompts. Implements prefix tuning where prompts are
    prepended to Keys and Values in attention.

    Args:
        pool_size: Number of prompts in the pool (M in paper)
        prompt_length: Length of each prompt (L_p in paper)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers to attach prompts
        top_k: Number of prompts to select
    """

    def __init__(self, pool_size=10, prompt_length=5, embed_dim=128,
                 num_heads=6, num_layers=4, top_k=5):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k

        # Prompt pool: (pool_size, num_layers, 2, prompt_length, num_heads, head_dim)
        # 2 = key and value prefixes
        self.prompt_pool = nn.Parameter(
            torch.randn(pool_size, num_layers, 2, prompt_length, num_heads, self.head_dim) * 0.02
        )
        self.prompt_keys = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)

        self.register_buffer('prompt_freq', torch.zeros(pool_size))
        self.task_count = 0

    def to(self, device):
        return super().to(device)

    def select_prompts(self, query):
        """Select top-k prompts based on query similarity.

        Returns:
            Dict with 'key_prefixes' and 'value_prefixes' lists for all layers.
        """
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.prompt_keys, dim=-1)
        similarities = torch.matmul(query, keys.T)
        _, indices = similarities.topk(self.top_k, dim=-1)

        if self.training:
            with torch.no_grad():
                for idx in indices.view(-1).unique():
                    self.prompt_freq[idx] += 1

        batch_size = query.size(0)
        # prompt_pool: (pool_size, num_layers, 2, prompt_length, num_heads, head_dim)
        selected = self.prompt_pool[indices]
        # selected: (B, top_k, num_layers, 2, prompt_length, num_heads, head_dim)

        key_prefixes = []
        value_prefixes = []
        for layer_idx in range(self.num_layers):
            # (B, top_k, prompt_length, num_heads, head_dim)
            k = selected[:, :, layer_idx, 0]
            v = selected[:, :, layer_idx, 1]
            # Reshape to (B, top_k * prompt_length, num_heads, head_dim)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)
            # Permute to (B, num_heads, total_prompt_length, head_dim)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            key_prefixes.append(k)
            value_prefixes.append(v)

        return {
            'key_prefixes': key_prefixes,
            'value_prefixes': value_prefixes
        }

    def get_prompt_loss(self, query):
        """Pull loss: encourage selected prompts to match query."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.prompt_keys, dim=-1)
        similarities = torch.matmul(query, keys.T)
        _, indices = similarities.topk(self.top_k, dim=-1)
        selected_sims = torch.gather(similarities, 1, indices)
        return (1 - selected_sims).mean()

    def get_usage_stats(self):
        """Get statistics about prompt usage."""
        active_prompts = (self.prompt_freq > 0).sum().item()
        return {
            'total_prompts': self.pool_size,
            'active_prompts': active_prompts,
            'prompt_freq': self.prompt_freq.clone()
        }

    def update_task(self):
        """Called after each task completes."""
        self.task_count += 1
