"""
SuperModel 融合模块
==================

跨模态感知融合
"""

from .cross_modal_fusion import (
    MultimodalInput,
    FusionConfig,
    CrossModalFusion,
    UnifiedRepresentation,
    FusionStrategy,
    create_multimodal_input
)

__all__ = [
    'MultimodalInput',
    'FusionConfig',
    'CrossModalFusion',
    'UnifiedRepresentation',
    'FusionStrategy',
    'create_multimodal_input'
]
