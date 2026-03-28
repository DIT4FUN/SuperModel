"""
SuperModel 感知模块
===================

多模态感知融合与特征提取
"""

from .cross_modal_fusion import (
    MultimodalInput,
    FusionConfig,
    CrossModalFusion,
    UnifiedRepresentation,
    create_multimodal_input
)

__all__ = [
    'MultimodalInput',
    'FusionConfig', 
    'CrossModalFusion',
    'UnifiedRepresentation',
    'create_multimodal_input'
]
