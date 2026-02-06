"""Continual Learning module for radar datasets.

Provides 11 CL algorithms:
- Regularization: EWC, LwF
- Replay: BalancedReplayBuffer, BalancedDERPlusPlus
- Contrastive: Co2L
- Adapter-based: EASE
- Prompt-based: L2P, CODAPrompt, DualPrompt
- Hybrid: EPB (Elastic Prompt-Backbone)
"""

from .utils import IncrementalClassifier, TaskTracker, CosineLinear, EASEAdapter
from .methods import (EWC, LwF, BalancedReplayBuffer, BalancedDERPlusPlus,
                      Co2L, EASE, L2P, CODAPrompt, DualPrompt,
                      EPB, HierarchicalEWC, FeatureAnchor)
from .trainer import CLTrainer
from .evaluator import CLEvaluator, compute_cl_metrics

ALGORITHMS = ['naive', 'ewc', 'lwf', 'replay', 'derpp', 'co2l', 'ease', 'l2p', 'coda', 'dualprompt', 'epb']

__all__ = [
    'IncrementalClassifier', 'TaskTracker', 'CosineLinear', 'EASEAdapter',
    'EWC', 'LwF', 'BalancedReplayBuffer', 'BalancedDERPlusPlus',
    'Co2L', 'EASE', 'L2P', 'CODAPrompt', 'DualPrompt',
    'EPB', 'HierarchicalEWC', 'FeatureAnchor',
    'CLTrainer', 'CLEvaluator', 'compute_cl_metrics',
    'ALGORITHMS'
]
