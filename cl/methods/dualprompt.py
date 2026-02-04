"""DualPrompt with prefix tuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPrompt(nn.Module):
    """DualPrompt - Complementary Prompting for Rehearsal-free Continual Learning.

    Paper: Wang et al. "DualPrompt" (ECCV 2022)
    Official: https://github.com/google-research/l2p

    Implements prefix tuning where prompts are prepended to Keys and Values in attention,
    following the official DualPrompt implementation.

    Combines general (G-Prompt) and expert (E-Prompt) prompts:
    - G-Prompt: Shared across all tasks, captures general knowledge, attached to earlier layers
    - E-Prompt: Task-specific pool with key-query selection, attached to later layers

    Args:
        g_prompt_length: Length of general prompts per layer
        e_prompt_length: Length of expert prompts per layer
        e_pool_size: Number of expert prompts in the pool
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        g_layers: Layers to attach G-Prompt (default: first half)
        e_layers: Layers to attach E-Prompt (default: second half)
        top_k: Number of E-prompts to select
    """

    def __init__(self, g_prompt_length=5, e_prompt_length=5, e_pool_size=10,
                 embed_dim=128, num_heads=6, num_layers=4, g_layers=None,
                 e_layers=None, top_k=5):
        super().__init__()
        self.g_prompt_length = g_prompt_length
        self.e_prompt_length = e_prompt_length
        self.e_pool_size = e_pool_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k

        if g_layers is None:
            g_layers = list(range(num_layers // 2))
        if e_layers is None:
            e_layers = list(range(num_layers // 2, num_layers))
        self.g_layers = g_layers
        self.e_layers = e_layers

        # G-Prompt: shared prompts for general knowledge (attached to earlier layers)
        # Shape: (num_g_layers, 2, g_prompt_length, num_heads, head_dim)
        # 2 = key and value prefixes
        num_g_layers = len(g_layers)
        self.g_prompt = nn.Parameter(
            torch.randn(num_g_layers, 2, g_prompt_length, num_heads, self.head_dim) * 0.02
        )

        # E-Prompt pool: task-specific prompts (attached to later layers)
        # Shape: (e_pool_size, num_e_layers, 2, e_prompt_length, num_heads, head_dim)
        num_e_layers = len(e_layers)
        self.e_prompt_pool = nn.Parameter(
            torch.randn(e_pool_size, num_e_layers, 2, e_prompt_length, num_heads, self.head_dim) * 0.02
        )

        # Keys for E-prompt selection
        self.e_prompt_keys = nn.Parameter(torch.randn(e_pool_size, embed_dim) * 0.02)

        self.register_buffer('e_prompt_freq', torch.zeros(e_pool_size))
        self.task_count = 0

    def to(self, device):
        return super().to(device)

    def get_g_prompt(self, batch_size):
        """Get general prompts expanded to batch size.

        Returns:
            Tuple of (key_prefixes, value_prefixes) lists for G-prompt layers.
            Each element is (B, num_heads, g_prompt_length, head_dim).
        """
        # g_prompt: (num_g_layers, 2, g_prompt_length, num_heads, head_dim)
        # Expand to batch and reshape for prefix attention
        g = self.g_prompt.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)
        # g: (B, num_g_layers, 2, g_prompt_length, num_heads, head_dim)

        key_prefixes = []
        value_prefixes = []
        for layer_idx in range(g.size(1)):
            # (B, g_prompt_length, num_heads, head_dim) -> (B, num_heads, g_prompt_length, head_dim)
            k = g[:, layer_idx, 0].permute(0, 2, 1, 3)
            v = g[:, layer_idx, 1].permute(0, 2, 1, 3)
            key_prefixes.append(k)
            value_prefixes.append(v)

        return key_prefixes, value_prefixes

    def select_e_prompts(self, query):
        """Select top-k expert prompts based on query similarity.

        Returns:
            Tuple of (key_prefixes, value_prefixes) lists for E-prompt layers.
            Each element is (B, num_heads, total_prompt_length, head_dim).
        """
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.e_prompt_keys, dim=-1)
        similarities = torch.matmul(query, keys.T)
        _, indices = similarities.topk(self.top_k, dim=-1)

        if self.training:
            with torch.no_grad():
                for idx in indices.view(-1).unique():
                    self.e_prompt_freq[idx] += 1

        batch_size = query.size(0)
        # e_prompt_pool: (e_pool_size, num_e_layers, 2, e_prompt_length, num_heads, head_dim)
        # indices: (B, top_k)

        selected = self.e_prompt_pool[indices]
        # selected: (B, top_k, num_e_layers, 2, e_prompt_length, num_heads, head_dim)

        num_e_layers = selected.size(2)

        key_prefixes = []
        value_prefixes = []
        for layer_idx in range(num_e_layers):
            # Concatenate top_k prompts along prompt length dimension
            # (B, top_k, e_prompt_length, num_heads, head_dim)
            k = selected[:, :, layer_idx, 0]
            v = selected[:, :, layer_idx, 1]
            # Reshape to (B, top_k * e_prompt_length, num_heads, head_dim)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)
            # Permute to (B, num_heads, total_prompt_length, head_dim)
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()
            key_prefixes.append(k)
            value_prefixes.append(v)

        return key_prefixes, value_prefixes, similarities, indices

    def get_prompt(self, query):
        """Get combined G-Prompt and selected E-Prompts as prefix tuning format.

        Returns:
            Dict with 'key_prefixes' and 'value_prefixes' lists for all layers.
        """
        batch_size = query.size(0)

        g_key_prefixes, g_value_prefixes = self.get_g_prompt(batch_size)
        e_key_prefixes, e_value_prefixes, _, _ = self.select_e_prompts(query)

        # Combine G and E prompts for all layers
        all_key_prefixes = [None] * self.num_layers
        all_value_prefixes = [None] * self.num_layers

        # Attach G-prompts to designated layers
        for i, layer_idx in enumerate(self.g_layers):
            all_key_prefixes[layer_idx] = g_key_prefixes[i]
            all_value_prefixes[layer_idx] = g_value_prefixes[i]

        # Attach E-prompts to designated layers
        for i, layer_idx in enumerate(self.e_layers):
            all_key_prefixes[layer_idx] = e_key_prefixes[i]
            all_value_prefixes[layer_idx] = e_value_prefixes[i]

        return {
            'key_prefixes': all_key_prefixes,
            'value_prefixes': all_value_prefixes
        }

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
