"""
SuperModel 感知模块
===================

多模态感知融合与特征提取
"""

# 从 fusion 模块导入 (实际实现在 fusion/cross_modal_fusion.py)
from fusion.cross_modal_fusion import (
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
