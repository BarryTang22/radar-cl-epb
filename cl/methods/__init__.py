"""Continual Learning Methods."""
from .ewc import EWC
from .lwf import LwF
from .replay import BalancedReplayBuffer
from .derpp import BalancedDERPlusPlus
from .co2l import Co2L
from .ease import EASE
from .l2p import L2P
from .coda import CODAPrompt
from .dualprompt import DualPrompt
from .epb import EPB, HierarchicalEWC, FeatureAnchor

__all__ = [
    'EWC', 'LwF', 'BalancedReplayBuffer', 'BalancedDERPlusPlus',
    'Co2L', 'EASE', 'L2P', 'CODAPrompt', 'DualPrompt',
    'EPB', 'HierarchicalEWC', 'FeatureAnchor',
]
