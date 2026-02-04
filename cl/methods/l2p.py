"""Learning to Prompt."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2P(nn.Module):
    """Learning to Prompt for continual learning.

    Paper: Wang et al. "Learning to Prompt for Continual Learning" (CVPR 2022)
    Official: https://github.com/google-research/l2p

    Uses a learnable prompt pool with key-query matching to select
    task-relevant prompts without explicit task identity.

    Args:
        pool_size: Number of prompts in the pool (M in paper)
        prompt_length: Length of each prompt (L_p in paper)
        embed_dim: Embedding dimension
        top_k: Number of prompts to select
    """

    def __init__(self, pool_size=10, prompt_length=5, embed_dim=128, top_k=5):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.prompt_pool = nn.Parameter(torch.randn(pool_size, prompt_length, embed_dim) * 0.02)
        self.prompt_keys = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)

        self.register_buffer('prompt_freq', torch.zeros(pool_size))
        self.task_count = 0

    def to(self, device):
        return super().to(device)

    def select_prompts(self, query):
        """Select top-k prompts based on query similarity."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.prompt_keys, dim=-1)
        similarities = torch.matmul(query, keys.T)
        _, indices = similarities.topk(self.top_k, dim=-1)

        if self.training:
            with torch.no_grad():
                for idx in indices.view(-1).unique():
                    self.prompt_freq[idx] += 1

        batch_size = query.size(0)
        selected = self.prompt_pool[indices]
        return selected.view(batch_size, self.top_k * self.prompt_length, self.embed_dim)

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
