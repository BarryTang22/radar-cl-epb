"""Continual Learning module for radar datasets.

Provides 10 CL algorithms:
- Regularization: EWC, LwF
- Replay: BalancedReplayBuffer, BalancedDERPlusPlus
- Contrastive: Co2L
- Adapter-based: EASE
- Prompt-based: L2P, CODAPrompt, DualPrompt
"""

from .utils import IncrementalClassifier, TaskTracker, CosineLinear, EASEAdapter
from .methods import (EWC, LwF, BalancedReplayBuffer, BalancedDERPlusPlus,
                      Co2L, EASE, L2P, CODAPrompt, DualPrompt)
from .trainer import CLTrainer
from .evaluator import CLEvaluator, compute_cl_metrics

ALGORITHMS = ['naive', 'ewc', 'lwf', 'replay', 'derpp', 'co2l', 'ease', 'l2p', 'coda', 'dualprompt']

__all__ = [
    'IncrementalClassifier', 'TaskTracker', 'CosineLinear', 'EASEAdapter',
    'EWC', 'LwF', 'BalancedReplayBuffer', 'BalancedDERPlusPlus',
    'Co2L', 'EASE', 'L2P', 'CODAPrompt', 'DualPrompt',
    'CLTrainer', 'CLEvaluator', 'compute_cl_metrics',
    'ALGORITHMS'
]
