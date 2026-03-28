"""
SuperModel 学习模块
==================

自监督学习与在线强化学习
"""

from .self_supervised import (
    LearningConfig,
    ContrastiveLoss,
    WorldModel,
    IntrinsicCuriosity,
    AutonomousLearning,
    get_learning_spec
)

__all__ = [
    'LearningConfig',
    'ContrastiveLoss',
    'WorldModel',
    'IntrinsicCuriosity',
    'AutonomousLearning',
    'get_learning_spec'
]
