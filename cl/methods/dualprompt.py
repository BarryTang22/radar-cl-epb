"""DualPrompt."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPrompt(nn.Module):
    """DualPrompt - Complementary Prompting for Rehearsal-free Continual Learning.

    Paper: Wang et al. "DualPrompt" (ECCV 2022)
    Official: https://github.com/google-research/l2p

    Combines general (G-Prompt) and expert (E-Prompt) prompts:
    - G-Prompt: Shared across all tasks, captures general knowledge
    - E-Prompt: Task-specific pool with key-query selection

    Args:
        g_prompt_length: Length of general prompts
        e_prompt_length: Length of expert prompts per key
        e_pool_size: Number of expert prompts in the pool
        embed_dim: Embedding dimension
        top_k: Number of E-prompts to select
    """

    def __init__(self, g_prompt_length=5, e_prompt_length=5, e_pool_size=10,
                 embed_dim=128, top_k=5):
        super().__init__()
        self.g_prompt_length = g_prompt_length
        self.e_prompt_length = e_prompt_length
        self.e_pool_size = e_pool_size
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.g_prompt = nn.Parameter(torch.randn(1, g_prompt_length, embed_dim) * 0.02)
        self.e_prompt_pool = nn.Parameter(torch.randn(e_pool_size, e_prompt_length, embed_dim) * 0.02)
        self.e_prompt_keys = nn.Parameter(torch.randn(e_pool_size, embed_dim) * 0.02)

        self.register_buffer('e_prompt_freq', torch.zeros(e_pool_size))
        self.task_count = 0

    def to(self, device):
        return super().to(device)

    def get_g_prompt(self, batch_size):
        """Get general prompts expanded to batch size."""
        return self.g_prompt.expand(batch_size, -1, -1)

    def select_e_prompts(self, query):
        """Select top-k expert prompts based on query similarity."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.e_prompt_keys, dim=-1)
        similarities = torch.matmul(query, keys.T)
        _, indices = similarities.topk(self.top_k, dim=-1)

        if self.training:
            with torch.no_grad():
                for idx in indices.view(-1).unique():
                    self.e_prompt_freq[idx] += 1

        batch_size = query.size(0)
        selected = self.e_prompt_pool[indices]
        return selected.view(batch_size, self.top_k * self.e_prompt_length, self.embed_dim)

    def get_prompt(self, query):
        """Get combined G-Prompt and selected E-Prompts."""
        batch_size = query.size(0)
        g = self.get_g_prompt(batch_size)
        e = self.select_e_prompts(query)
        return torch.cat([g, e], dim=1)

    def get_prompt_loss(self, query):
        """Pull loss for E-prompts: encourage selected prompts to match query."""
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.e_prompt_keys, dim=-1)
        similarities = torch.matmul(query, keys.T)
        _, indices = similarities.topk(self.top_k, dim=-1)
        selected_sims = torch.gather(similarities, 1, indices)
        return (1 - selected_sims).mean()

    def get_usage_stats(self):
        """Get statistics about E-prompt usage."""
        active_prompts = (self.e_prompt_freq > 0).sum().item()
        return {
            'total_e_prompts': self.e_pool_size,
            'active_e_prompts': active_prompts,
            'e_prompt_freq': self.e_prompt_freq.clone()
        }

    def update_task(self):
        """Called after each task completes."""
        self.task_count += 1
